"""QAT fine-tuning entrypoint for Qwen3-VL.

Group-wise Learned Step-Size Quantization (W4 / W8) on the LLM backbone, with
the vision tower, visual.merger, lm_head, embed_tokens and norms kept in bf16.
Compatible with DeepSpeed ZeRO-2/3 multi-node training.
"""

from __future__ import annotations

import os
import logging
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import Qwen3VLForConditionalGeneration
from transformers import AutoProcessor, Trainer, TrainerCallback

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.qat_modules import (
    DEFAULT_QAT_SKIP_KEYWORDS,
    bake_qat_weights_inplace,
    collect_qat_param_groups,
    materialize_quantized_state_dict,
    replace_linears_with_qat,
    strip_log_scale_from_checkpoint_dir,
)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class QATArguments:
    qat_enable: bool = field(default=True)
    qat_bits: int = field(default=4, metadata={"help": "Weight bit-width: 4 or 8."})
    qat_group_size: int = field(default=128)
    qat_scale_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for log-scale params. Defaults to base learning_rate."},
    )
    qat_skip_extra: str = field(
        default="",
        metadata={"help": "Comma-separated extra module-name keywords to skip."},
    )
    qat_save_quantized: bool = field(
        default=True,
        metadata={"help": "Also dump a fake-quantized state dict + scales at the end."},
    )


class SaveProcessorCallback(TrainerCallback):
    """Writes the AutoProcessor (preprocessor + video preprocessor configs)
    into every checkpoint dir. Trainer only persists the tokenizer when
    `processing_class=tokenizer`, so without this callback the saved
    checkpoint-* dirs are not a complete model dir for downstream loading."""

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            self.processor.save_pretrained(ckpt_dir)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _qwen3vl_submodule(model, attr: str):
    """transformers >=4.57 nested ``visual`` / ``language_model`` one level
    deeper (``model.model.visual``); older releases exposed them directly on
    ``Qwen3VLForConditionalGeneration``. Try both."""
    if hasattr(model, attr):
        return getattr(model, attr)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, attr):
        return getattr(inner, attr)
    raise AttributeError(
        f"Could not locate '{attr}' on {type(model).__name__} (checked top "
        f"level and .model). Did the transformers Qwen3-VL layout change again?"
    )


def set_model(model_args, model):
    visual = _qwen3vl_submodule(model, "visual")
    language_model = _qwen3vl_submodule(model, "language_model")

    flag = bool(model_args.tune_mm_vision)
    for _, p in visual.named_parameters():
        p.requires_grad = flag

    flag = bool(model_args.tune_mm_mlp)
    for _, p in visual.merger.named_parameters():
        p.requires_grad = flag

    flag = bool(model_args.tune_mm_llm)
    for _, p in language_model.named_parameters():
        p.requires_grad = flag
    # lm_head sometimes sits at the wrapper, sometimes inside the inner model.
    # Set requires_grad on its parameters (the prior `module.requires_grad = ...`
    # form silently no-ops on nn.Module — only Parameter has that property).
    lm_head = getattr(model, "lm_head", None) or getattr(model.model, "lm_head", None)
    if lm_head is not None:
        for _, p in lm_head.named_parameters():
            p.requires_grad = flag


def _build_model(model_args, training_args, attn_implementation):
    compute_dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=compute_dtype,
    )
    return model, "qwen3vl"


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, QATArguments)
    )
    model_args, data_args, training_args, qat_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if qat_args.qat_enable and qat_args.qat_bits not in (4, 8):
        raise ValueError(f"--qat_bits must be 4 or 8, got {qat_args.qat_bits}")

    # ---- model ----
    model, model_type = _build_model(model_args, training_args, attn_implementation)
    data_args.model_type = model_type
    rank0_print(
        f"the initlized model is {model_args.model_name_or_path} "
        f"the class is {model.__class__.__name__}"
    )

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _make_grad(module, inp, out):
                out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_make_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # ---- trainable params (QAT path is incompatible with LoRA) ----
    if training_args.lora_enable:
        raise ValueError("QAT training does not support --lora_enable. Use full fine-tune.")
    set_model(model_args, model)
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        _qwen3vl_submodule(model, "visual").print_trainable_parameters()
        # `print_trainable_parameters` on the inner Qwen3VLModel reads
        # ``self.language_model.{embed_tokens,layers}`` which exists in 4.57+.
        model.model.print_trainable_parameters()

    # ---- QAT swap (BEFORE the trainer / DeepSpeed engine is built) ----
    if qat_args.qat_enable:
        skip = list(DEFAULT_QAT_SKIP_KEYWORDS)
        if qat_args.qat_skip_extra:
            skip.extend([s for s in qat_args.qat_skip_extra.split(",") if s.strip()])
        replaced = replace_linears_with_qat(
            model,
            bits=qat_args.qat_bits,
            group_size=qat_args.qat_group_size,
            skip_keywords=skip,
        )
        model.config.qat = {
            "bits": qat_args.qat_bits,
            "group_size": qat_args.qat_group_size,
            "skip_keywords": skip,
            "num_replaced": len(replaced),
        }
        rank0_print(
            f"[QAT] LLM linears quantized: {len(replaced)} layers, "
            f"W{qat_args.qat_bits}, group={qat_args.qat_group_size}"
        )

    # ---- data ----
    data_module = make_supervised_data_module(processor, data_args=data_args)

    # ---- optimizer with separate group for log_scale ----
    base_lr = training_args.learning_rate
    scale_lr = qat_args.qat_scale_lr if qat_args.qat_scale_lr is not None else base_lr

    class _QATTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            param_groups = collect_qat_param_groups(
                self.model,
                base_lr=base_lr,
                scale_lr=scale_lr,
                weight_decay=self.args.weight_decay,
            )
            # Optional projector LR override (mirror upstream Trainer monkey-patch).
            mm_lr = getattr(self.args, "mm_projector_lr", None)
            if mm_lr:
                proj_params = [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and "merger" in n
                ]
                if proj_params:
                    proj_set = {id(p) for p in proj_params}
                    for g in param_groups:
                        g["params"] = [p for p in g["params"] if id(p) not in proj_set]
                    param_groups.append({
                        "params": proj_params,
                        "lr": mm_lr,
                        "weight_decay": self.args.weight_decay,
                    })
            self.optimizer = opt_cls(param_groups, **opt_kwargs)
            return self.optimizer

    trainer = _QATTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )
    trainer.add_callback(SaveProcessorCallback(processor))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # ---- bake fake-quantized weights into master params, then save ----
    if qat_args.qat_enable:
        is_zero3 = bool(getattr(trainer, "deepspeed", None)) and \
            getattr(trainer.args, "deepspeed", None) is not None
        try:
            n = bake_qat_weights_inplace(model, is_zero3=is_zero3)
            rank0_print(f"[QAT] Baked fake-quantized weights into {n} linear layers.")
        except Exception as e:
            rank0_print(f"[QAT] Weight baking failed: {e}")

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if qat_args.qat_enable and (training_args.local_rank in (-1, 0)):
        try:
            removed = strip_log_scale_from_checkpoint_dir(training_args.output_dir)
            rank0_print(f"[QAT] Stripped {removed} log_scale entries from saved checkpoint.")
        except Exception as e:
            rank0_print(f"[QAT] log_scale strip failed: {e}")

    if qat_args.qat_enable and qat_args.qat_save_quantized and \
            (training_args.local_rank in (-1, 0)):
        try:
            qsd = materialize_quantized_state_dict(model)
            out = os.path.join(training_args.output_dir, "qat_quantized_weights.bin")
            torch.save(qsd, out)
            rank0_print(f"[QAT] Saved fake-quantized weights + scales sidecar to {out}")
        except Exception as e:
            rank0_print(f"[QAT] Quantized state-dict export failed: {e}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
