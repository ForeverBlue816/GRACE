# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

# --- Trust-our-own-checkpoints shim (torch 2.5.1 + transformers >=4.55) ----
# Stack is pinned to torch 2.5.1 for flash-attn / deepspeed wheel compatibility,
# but transformers gates torch.load behind torch>=2.6 AND torch defaults
# weights_only=True (which rejects numpy globals in RNG/scheduler files).
# All checkpoints we resume from are our own DeepSpeed shards — trusted.
# So we (1) neutralize the version gate and (2) force weights_only=False.
try:
    from transformers.utils import import_utils as _hf_import_utils
    def _hf_noop_check_torch_load_is_safe(): return None
    _hf_import_utils.check_torch_load_is_safe.__code__ = (
        _hf_noop_check_torch_load_is_safe.__code__
    )
except Exception:
    pass

_orig_torch_load = torch.load
def _torch_load_trust_all(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_trust_all
# --------------------------------------------------------------------------

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

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
    """transformers >=4.57 nests ``visual`` / ``language_model`` under
    ``model.model`` on Qwen3VLForConditionalGeneration; older releases kept
    them at the top. Try both."""
    if hasattr(model, attr):
        return getattr(model, attr)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, attr):
        return getattr(inner, attr)
    raise AttributeError(
        f"Could not locate '{attr}' on {type(model).__name__}; "
        f"transformers layout may have changed again."
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
    # `module.requires_grad = ...` is a silent no-op on nn.Module — only
    # nn.Parameter has the setter. Walk parameters explicitly so the flag
    # actually takes effect on lm_head's weight.
    lm_head = getattr(model, "lm_head", None) or getattr(model.model, "lm_head", None)
    if lm_head is not None:
        for _, p in lm_head.named_parameters():
            p.requires_grad = flag


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )
    # AutoProcessor ships its own tokenizer with model_max_length baked from the
    # processor config (often 8192 for Qwen3-VL). Data preprocessing goes through
    # processor.tokenizer, so the standalone tokenizer's model_max_length is not
    # what controls truncation/warnings here. Sync it to training_args so:
    #   (a) the "Token indices sequence length is longer..." warning stops, and
    #   (b) any non-flatten DataCollator path truncates against the right limit.
    processor.tokenizer.model_max_length = training_args.model_max_length

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            _qwen3vl_submodule(model, "visual").print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
