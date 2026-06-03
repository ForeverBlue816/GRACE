"""QAT fine-tuning entrypoint for LLaVA 1.5 / 1.6.

Group-wise Learned Step-Size Quantization (W4 / W8) on the LLM backbone, with
all other parts (vision tower, mm projector, lm_head, norms, embeddings) kept
in bf16. Compatible with DeepSpeed ZeRO-3 multi-node training.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

from llava.train import train as base_train
from llava.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)
from llava.constants import IGNORE_INDEX
from llava import conversation as conversation_lib
from llava.model import (
    LlavaLlamaForCausalLM,
    LlavaMistralForCausalLM,
    LlavaMptForCausalLM,
)
from llava.train.llava_trainer import LLaVATrainer
from llava.train.qat_modules import (
    QuantizedLinear,
    bake_qat_weights_inplace,
    collect_qat_param_groups,
    materialize_quantized_state_dict,
    replace_linears_with_qat,
    strip_log_scale_from_checkpoint_dir,
)


@dataclass
class QATArguments:
    qat_enable: bool = field(default=True)
    qat_bits: int = field(default=4, metadata={"help": "Weight bit-width: 4 or 8."})
    qat_group_size: int = field(default=128)
    qat_scale_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for log-scale params. Defaults to base learning_rate."},
    )
    qat_learn_scales: bool = field(
        default=True,
        metadata={"help": "Optimize LSQ log-scales. Disable for a fixed-scale stability baseline."},
    )
    qat_debug_grad_check_steps: int = field(
        default=1,
        metadata={"help": "Scan all gradients for NaN/Inf during the first N optimizer steps."},
    )
    qat_skip_extra: str = field(
        default="",
        metadata={"help": "Comma-separated extra module-name keywords to skip."},
    )
    qat_save_quantized: bool = field(
        default=True,
        metadata={"help": "Also dump a fake-quantized state dict at the end."},
    )


def _rank0(*a):
    rank = int(os.environ.get("RANK", "0"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank == 0:
        print(*a)


def _qat_scale_summary(model) -> str:
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        log_scale = module.log_scale.detach()
        invalid = (~torch.isfinite(log_scale)).sum().item()
        if invalid:
            return f"{name}.log_scale has {invalid}/{log_scale.numel()} non-finite values"
    return "all QAT log_scale tensors are finite"


def _qat_sanitized_scale_grad_count(model) -> int:
    counts = [
        module._nonfinite_scale_grad_count
        for module in model.modules()
        if isinstance(module, QuantizedLinear)
    ]
    if not counts:
        return 0
    return int(torch.stack(counts).sum().item())


def _first_nonfinite_qat_weight_gradient(model) -> Optional[str]:
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        invalid = int(module._nonfinite_weight_grad_count.item())
        if invalid:
            return f"{name}.weight observed {invalid} non-finite gradient values"
    return None


def _first_nonfinite_gradient(model) -> Optional[str]:
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        invalid = (~torch.isfinite(grad)).sum().item()
        if invalid:
            return f"{name} has {invalid}/{grad.numel()} non-finite gradient values"
    return None


def _first_nonfinite_parameter(model) -> Optional[str]:
    for name, param in model.named_parameters():
        invalid = (~torch.isfinite(param.detach())).sum().item()
        if invalid:
            return f"{name} has {invalid}/{param.numel()} non-finite values"
    return None


def _tensor_summary(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=unavailable"
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    invalid = (~finite).sum().item()
    if invalid:
        return f"{name} has {invalid}/{detached.numel()} non-finite values"
    return f"{name} is finite"


def _select_model_cls(model_name_or_path: str):
    name = model_name_or_path.lower()
    if "mpt" in name:
        return LlavaMptForCausalLM
    if "mistral" in name:
        return LlavaMistralForCausalLM
    return LlavaLlamaForCausalLM


def train(attn_implementation: Optional[str] = None):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, QATArguments)
    )
    model_args, data_args, training_args, qat_args = parser.parse_args_into_dataclasses()
    base_train.local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if training_args.bits in (4, 8):
        raise ValueError(
            "Use --qat_enable for QAT; do NOT pass --bits 4/8 (those flip on bnb-LoRA)."
        )
    if qat_args.qat_bits not in (4, 8):
        raise ValueError(f"--qat_bits must be 4 or 8, got {qat_args.qat_bits}")

    # ---- model ----
    model_cls = _select_model_cls(model_args.model_name_or_path)
    is_mpt = model_cls is LlavaMptForCausalLM

    if is_mpt:
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        config.attn_config["attn_impl"] = training_args.mpt_attn_impl
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=compute_dtype,
        )
    else:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=compute_dtype,
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _make_grad(module, inp, out):
                out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_make_grad)

    # ---- tokenizer ----
    if is_mpt:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            legacy=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # ---- vision tower / mm projector ----
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # ---- QAT swap (BEFORE the trainer/DeepSpeed engine is built) ----
    if qat_args.qat_enable:
        skip = list(__import__("llava.train.qat_modules", fromlist=["DEFAULT_QAT_SKIP_KEYWORDS"])
                    .DEFAULT_QAT_SKIP_KEYWORDS)
        if qat_args.qat_skip_extra:
            skip.extend([s for s in qat_args.qat_skip_extra.split(",") if s.strip()])
        # Restrict QAT to LLM linears only (skip vision tower + projector).
        replaced = replace_linears_with_qat(
            model.get_model(),
            bits=qat_args.qat_bits,
            group_size=qat_args.qat_group_size,
            skip_keywords=skip,
        )
        if not qat_args.qat_learn_scales:
            for module in model.modules():
                if isinstance(module, QuantizedLinear):
                    module.log_scale.requires_grad_(False)
        # Persist QAT config on the model for downstream loaders.
        model.config.qat = {
            "bits": qat_args.qat_bits,
            "group_size": qat_args.qat_group_size,
            "learn_scales": qat_args.qat_learn_scales,
            "skip_keywords": skip,
            "num_replaced": len(replaced),
        }
        _rank0(f"[QAT] LLM linears quantized: {len(replaced)} layers, "
               f"W{qat_args.qat_bits}, group={qat_args.qat_group_size}, "
               f"learn_scales={qat_args.qat_learn_scales}")

    # cast remaining bf16 params (norms etc. should stay fp32 for stability)
    if training_args.bf16:
        for name, p in model.named_parameters():
            if p.requires_grad and "log_scale" not in name and "norm" not in name.lower():
                if p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)

    # ---- data ----
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # ---- optimizer with separate group for log_scale ----
    base_lr = training_args.learning_rate
    scale_lr = qat_args.qat_scale_lr if qat_args.qat_scale_lr is not None else base_lr
    _rank0(
        f"[QAT] Optimizer LR: weights={base_lr:g}, "
        f"log_scale={'frozen' if not qat_args.qat_learn_scales else f'{scale_lr:g}'}"
    )

    class _QATTrainer(LLaVATrainer):
        _last_sanitized_scale_grad_count = 0

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            valid_targets = None
            if labels is not None:
                valid_targets = int(labels.ne(IGNORE_INDEX).sum().item())
                if valid_targets == 0:
                    raise ValueError(
                        "Batch has no supervised tokens: every label is IGNORE_INDEX. "
                        "Check tokenization-mismatch warnings and the mixed JSON sources."
                    )

            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            if not torch.isfinite(loss).all():
                invalid_param = _first_nonfinite_parameter(model)
                raise FloatingPointError(
                    f"Non-finite training loss (valid_targets={valid_targets}); "
                    f"{_qat_scale_summary(model)}; "
                    f"{invalid_param or 'all model parameters are finite'}; "
                    f"{_tensor_summary('logits', getattr(outputs, 'logits', None))}."
                )
            if return_outputs:
                return loss, outputs
            return loss

        def training_step(self, model, inputs):
            loss = super().training_step(model, inputs)
            invalid_qat_weight_grad = _first_nonfinite_qat_weight_gradient(model)
            if invalid_qat_weight_grad is not None:
                raise FloatingPointError(
                    f"Non-finite QAT weight gradient before optimizer step: "
                    f"{invalid_qat_weight_grad}; {_qat_scale_summary(model)}."
                )

            sanitized = _qat_sanitized_scale_grad_count(model)
            should_check_gradients = (
                self.state.global_step < qat_args.qat_debug_grad_check_steps
                or sanitized > self._last_sanitized_scale_grad_count
            )
            if should_check_gradients:
                invalid_grad = _first_nonfinite_gradient(model)
                if invalid_grad is not None:
                    raise FloatingPointError(
                        f"Non-finite gradient before optimizer step: {invalid_grad}; "
                        f"{_qat_scale_summary(model)}."
                    )

            if sanitized > self._last_sanitized_scale_grad_count:
                _rank0(
                    f"[QAT] Sanitized {sanitized - self._last_sanitized_scale_grad_count} "
                    "non-finite log_scale gradient values before optimizer step."
                )
                self._last_sanitized_scale_grad_count = sanitized
            return loss

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            opt_cls, opt_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
            param_groups = collect_qat_param_groups(
                self.model,
                base_lr=base_lr,
                scale_lr=scale_lr,
                weight_decay=self.args.weight_decay,
            )
            # mm_projector_lr override (mirror LLaVATrainer behaviour)
            if getattr(self.args, "mm_projector_lr", None):
                proj_params = [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and "mm_projector" in n
                ]
                if proj_params:
                    proj_set = {id(p) for p in proj_params}
                    for g in param_groups:
                        g["params"] = [p for p in g["params"] if id(p) not in proj_set]
                    param_groups.append({
                        "params": proj_params,
                        "lr": self.args.mm_projector_lr,
                        "weight_decay": self.args.weight_decay,
                    })
            self.optimizer = opt_cls(param_groups, **opt_kwargs)
            return self.optimizer

    trainer = _QATTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # ---- bake fake-quantized weights into master params, then save ----
    # After this step the saved checkpoint is a vanilla LLaVA checkpoint whose
    # `weight` tensors lie on the integer grid; running official LLaVA eval
    # against it requires no code change and yields the QAT inference numbers.
    if qat_args.qat_enable:
        # ZeRO-3 needs GatheredParameters + rank-0 write-back; ZeRO-0/1/2 have
        # replicated weights so plain in-place copy on every rank is correct.
        # The previous check only tested "deepspeed enabled", which mis-routed
        # ZeRO-2 jobs into the ZeRO-3 path and left non-rank-0 weights stale.
        is_zero3 = False
        ds_engine = getattr(trainer, "deepspeed", None)
        if ds_engine is not None:
            try:
                ds_cfg = ds_engine.config if hasattr(ds_engine, "config") else {}
                stage = (ds_cfg or {}).get("zero_optimization", {}).get("stage", 0)
                is_zero3 = (int(stage) == 3)
            except Exception:
                is_zero3 = False
        try:
            n = bake_qat_weights_inplace(model, is_zero3=is_zero3)
            _rank0(f"[QAT] Baked fake-quantized weights into {n} linear layers "
                   f"(zero3={is_zero3}).")
        except Exception as e:
            _rank0(f"[QAT] Weight baking failed: {e}")

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # rank-0 cleanup: drop log_scale keys so HF / LLaVA loaders see a stock model.
    if qat_args.qat_enable and trainer.is_world_process_zero():
        try:
            removed = strip_log_scale_from_checkpoint_dir(training_args.output_dir)
            _rank0(f"[QAT] Stripped {removed} log_scale entries from saved checkpoint.")
        except Exception as e:
            _rank0(f"[QAT] log_scale strip failed: {e}")

    # Sidecar with raw scales for downstream true-INT4/8 packing.
    if qat_args.qat_save_quantized and trainer.is_world_process_zero():
        try:
            qsd = materialize_quantized_state_dict(model)
            out = os.path.join(training_args.output_dir, "qat_quantized_weights.bin")
            torch.save(qsd, out)
            _rank0(f"[QAT] Saved fake-quantized weights + scales sidecar to {out}")
        except Exception as e:
            _rank0(f"[QAT] Quantized state-dict export failed: {e}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
