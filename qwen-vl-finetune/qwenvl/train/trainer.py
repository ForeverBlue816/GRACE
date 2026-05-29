from typing import Dict, List, Optional, Sequence, Tuple, Callable

import torch
from torch.utils.data import Sampler
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    
    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    # batch, head, seq_len, dim
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # batch, seqlen, head, dim

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    attn_output = attn_output.unsqueeze(0)

    return attn_output, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(
    config=None,
    input_embeds=None,
    attention_mask=None,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
    *args,
    **kwargs,
):
    """Monkey-patch replacement for ``create_causal_mask`` / its sliding-window
    variant. We bypass the dense [B, 1, S, S] causal mask entirely because the
    flatten collator's varlen flash-attn path consumes ``attention_mask`` as a
    1-D ``cu_seqlens`` tensor directly.

    Signature is fully defaulted + ``*args, **kwargs`` because transformers
    has churned the ``create_causal_mask`` calling convention across releases
    (4.57 dropped ``input_embeds`` / ``cache_position`` from the call site).
    We don't care about any field except ``attention_mask`` and just hand it
    back unchanged.
    """
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils


    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = (
        return_mask
    )    
    ## qwen2_5_vl
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = (
        return_mask
    )
    ## qwen3vl
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = (
        return_mask
    )
    ## qwen3vl moe
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = (
        return_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# Apply monkey patches
Trainer.create_optimizer = create_optimizer

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

Qwen3VLVisionModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters


# ============================================================================
# Modality-aware length-grouped sampler (LLaVA-style)
# ----------------------------------------------------------------------------
# Why: ShareGPT4V mix665k mixes ~6% text-only samples. With text-only batches
# the visual merger does not run, so on that rank its parameter grads are
# None. Under ZeRO-3 / DDP this produces different reduce-bucket shapes across
# ranks and the NCCL all-reduce hangs until watchdog timeout. Grouping samples
# by sign(modality_length) ensures every global step is uniformly all-MM or
# all-text across all ranks; merger then either runs everywhere or nowhere.
# ============================================================================


def _split_to_even_chunks(indices: List[int], lengths: List[int], num_chunks: int):
    """Distribute indices into num_chunks roughly-equal-total-length chunks."""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    per_chunk = len(indices) // num_chunks
    chunks: List[List[int]] = [[] for _ in range(num_chunks)]
    chunk_lens: List[float] = [0.0 for _ in range(num_chunks)]
    for idx in indices:
        target = chunk_lens.index(min(chunk_lens))
        chunks[target].append(idx)
        chunk_lens[target] += abs(lengths[idx])
        if len(chunks[target]) == per_chunk:
            chunk_lens[target] = float("inf")
    return chunks


def _get_length_grouped_indices(
    lengths: List[int], batch_size: int, world_size: int, generator=None
) -> List[int]:
    indices = torch.randperm(len(lengths), generator=generator).tolist()
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size]
        for i in range(0, len(indices), megabatch_size)
    ]
    megabatches = [
        sorted(mb, key=lambda i: abs(lengths[i]), reverse=True) for mb in megabatches
    ]
    megabatches = [
        _split_to_even_chunks(mb, lengths, world_size) for mb in megabatches
    ]
    return [i for mb in megabatches for chunk in mb for i in chunk]


def _get_modality_length_grouped_indices(
    lengths: List[int], batch_size: int, world_size: int, generator=None
) -> List[int]:
    # Treat 0 as MM to avoid degenerate samples breaking sign-based partitioning.
    safe_lengths = [l if l != 0 else 1 for l in lengths]

    if all(l > 0 for l in safe_lengths) or all(l < 0 for l in safe_lengths):
        return _get_length_grouped_indices(
            safe_lengths, batch_size, world_size, generator=generator
        )

    mm_pairs = [(i, l) for i, l in enumerate(safe_lengths) if l > 0]
    lang_pairs = [(i, l) for i, l in enumerate(safe_lengths) if l < 0]
    mm_indices = [i for i, _ in mm_pairs]
    mm_lengths = [l for _, l in mm_pairs]
    lang_indices = [i for i, _ in lang_pairs]
    lang_lengths = [l for _, l in lang_pairs]

    mm_order = _get_length_grouped_indices(
        mm_lengths, batch_size, world_size, generator=generator
    )
    lang_order = _get_length_grouped_indices(
        lang_lengths, batch_size, world_size, generator=generator
    )
    mm_shuffle = [mm_indices[i] for i in mm_order]
    lang_shuffle = [lang_indices[i] for i in lang_order]

    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    # Drop any short tail megabatch in each modality. Pooling the two tails into
    # a single trailing megabatch would put samples of different modalities into
    # the same global step, so the vision merger runs on some ranks but not
    # others — that triggers NCCL all-reduce shape mismatches in ZeRO and hangs
    # the last 1–2 steps until watchdog timeout. Pair with
    # `--dataloader_drop_last True` so the DataLoader does not re-introduce a
    # partial batch at the very end.
    if mm_megabatches and len(mm_megabatches[-1]) < megabatch_size:
        mm_megabatches.pop()
    if lang_megabatches and len(lang_megabatches[-1]) < megabatch_size:
        lang_megabatches.pop()

    megabatches = mm_megabatches + lang_megabatches
    perm = torch.randperm(len(megabatches), generator=generator).tolist()
    megabatches = [megabatches[i] for i in perm]

    return [i for mb in megabatches for i in mb]


class ModalityLengthGroupedSampler(Sampler):
    """Samples indices so that each global megabatch is single-modality.

    A megabatch covers `world_size * batch_size` samples; the dataloader's
    BatchSamplerShard then dispatches one micro-batch per rank from each
    megabatch, so all ranks see the same modality at every global step.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: List[int],
        generator=None,
    ):
        if lengths is None or len(lengths) == 0:
            raise ValueError("ModalityLengthGroupedSampler requires non-empty lengths.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = list(lengths)
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        return iter(
            _get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        )


# Patch Trainer._get_train_sampler so any subclass (including the QAT trainer)
# inherits modality-aware grouping when group_by_modality_length=True.
_orig_get_train_sampler = Trainer._get_train_sampler


def _qwenvl_get_train_sampler(self, *args, **kwargs):
    if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
        return _orig_get_train_sampler(self, *args, **kwargs)

    if not getattr(self.args, "group_by_modality_length", False):
        return _orig_get_train_sampler(self, *args, **kwargs)

    if not hasattr(self.train_dataset, "modality_lengths"):
        logger.warning(
            "group_by_modality_length=True but train_dataset has no "
            "`modality_lengths` property; falling back to the default sampler."
        )
        return _orig_get_train_sampler(self, *args, **kwargs)

    lengths = self.train_dataset.modality_lengths

    # Megabatch must cover *every* rank's micro-step within a gradient
    # accumulation window so consecutive accumulation slots on the same rank
    # stay in the same modality as well.
    grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
    world_size = max(1, int(getattr(self.args, "world_size", 1)))
    megabatch_world = world_size * grad_accum

    return ModalityLengthGroupedSampler(
        batch_size=self.args.train_batch_size,
        world_size=megabatch_world,
        lengths=lengths,
    )


Trainer._get_train_sampler = _qwenvl_get_train_sampler