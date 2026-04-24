# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Patch configuration for Qwen3-Omni-MoE transformers>=5.2.0 code generation.

Covers the thinker training path (text + vision + audio + MoE):
  - Vision SP slicing with pad_scale=4 + varlen-aware attention
  - Audio tower with SP gather/slice of (mel, time) features
  - Thinker text model FSDP-safe deepstack process + MoeModelOutputWithPast
  - Qwen3OmniMoeThinkerTextExperts fused-MoE replacement (drops the
    @use_experts_implementation decorator which would otherwise bypass our
    fused kernel)
  - Thinker ForConditionalGeneration: pre-computed image/video/audio masks,
    async-Ulysses-aware embed gather+scatter, deepstack SP selection, fused
    loss via self.loss_function, precomputed multimodal position-ids
  - Qwen3OmniMoeForConditionalGeneration: propagate _moe_implementation
    down to thinker.text_config, forward-to-thinker only (skip talker),
    VeOmni parallel plan.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_omni_moe.qwen3_omni_moe_npu_patch_gen_config -o veomni/models/transformers/qwen3_omni_moe/generated --diff
"""

import torch
from veomni.patchgen.patch_spec import PatchConfig

from .qwen3_omni_moe_gpu_patch_gen_config import (
    qwen3_omni_moe_pretrained_init_weights_patched,
    qwen3_omni_moe_get_rope_index_patched,
    qwen3_omni_moe_vision_attention_forward_patched,
    qwen3_omni_moe_vision_forward_patched,
    qwen3_omni_moe_vision_dummy_forward_patched,
    qwen3_omni_moe_audio_forward_patched,
    qwen3_omni_moe_audio_dummy_forward_patched,
    qwen3_omni_moe_thinker_text_model_forward_patched,
    qwen3_omni_moe_thinker_text_deepstack_process_patched,
    PatchedQwen3OmniMoeThinkerTextExperts,
    qwen3_omni_moe_thinker_get_audio_features_patched,
    qwen3_omni_moe_thinker_get_position_id_func_patched,
    qwen3_omni_moe_thinker_forward_patched,
    qwen3_omni_moe_for_conditional_generation_init_patched,
    qwen3_omni_moe_enable_talker_patched,
    qwen3_omni_moe_for_conditional_generation_forward_patched,
    qwen3_omni_moe_top_get_position_id_func_patched,
    qwen3_omni_moe_get_parallel_plan_patched,
)


config = PatchConfig(
    source_module="transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe",
    target_file="patched_modeling_qwen3_omni_moe_npu.py",
    description="Qwen3-Omni-MoE thinker with VeOmni v5 compatibility (SP + FSDP + fused MoE + fused loss)",
)


# ================================================================
# Additional imports needed by the patched methods in the generated file
# ================================================================
config.add_import("torch_npu", is_from_import=False)
config.add_import("copy", is_from_import=False)
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.nn.functional", alias="F", is_from_import=False)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=[
        "gather_heads_scatter_seq",
        "gather_outputs",
        "gather_seq_scatter_heads",
        "slice_input_tensor",
        "sp_pad_and_slice",
        "unpad_tensor",
    ],
)
config.add_import("veomni.distributed.sequence_parallel.ulysses", names=["_Gather"])
config.add_import("veomni.models.transformers.attention_utils", names=["VARLEN_ATTENTION_TYPES"])
config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import(
    "veomni.utils.constants",
    names=["AUDIO_INPUT_INDEX", "IGNORE_INDEX", "IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"],
)


# ================================================================
# Drop talker + code2wav from generated output.
#
# The patched `Qwen3OmniMoeForConditionalGeneration.__init__` forces
# `has_talker=False` and never calls `enable_talker()`, so the talker /
# code2wav modules are never instantiated on the training path. But HF's
# `PreTrainedModel.post_init` aggregates `_no_split_modules` by walking the
# class hierarchy, so leaving `Qwen3OmniMoeTalkerDecoderLayer`,
# `Qwen3OmniMoeCode2WavTransformerLayer`, etc. in the generated module
# still lets them contribute to the aggregation and can pull dead layer
# names into the FSDP no-split set. Excluding them here also trims the
# generated file (~1500 lines) and removes an import-time footprint that
# isn't exercised.
#
# The top-level package `__init__.py` imports `Qwen3OmniMoeTalkerModel` /
# `Qwen3OmniMoeTalkerForConditionalGeneration` directly from upstream
# transformers (not from the generated file), so excluding them here is
# safe for the registry.
#
# The remaining methods on `Qwen3OmniMoeForConditionalGeneration`
# (`enable_talker`, `_get_talker_*`, `generate`, `token2wav`) reference
# these classes by name but only at call time (Python late binding) — the
# training forward never reaches them.
# ================================================================
config.exclude_from_output(
    # Talker
    "Qwen3OmniMoeTalkerResizeMLP",
    "Qwen3OmniMoeTalkerCodePredictorOutputWithPast",
    "Qwen3OmniMoeTalkerCodePredictorAttention",
    "Qwen3OmniMoeTalkerCodePredictorDecoderLayer",
    "Qwen3OmniMoeTalkerCodePredictorModel",
    "Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration",
    "Qwen3OmniMoeTalkerOutputWithPast",
    "Qwen3OmniMoeTalkerRotaryEmbedding",
    "Qwen3OmniMoeTalkerTextMLP",
    "Qwen3OmniMoeTalkerTextTopKRouter",
    "Qwen3OmniMoeTalkerTextExperts",
    "Qwen3OmniMoeTalkerTextSparseMoeBlock",
    "Qwen3OmniMoeTalkerDecoderLayer",
    "Qwen3OmniMoeTalkerModel",
    "Qwen3OmniMoeTalkerForConditionalGeneration",
    # Shared by talker + code2wav (not referenced by thinker)
    "Qwen3OmniMoeRMSNorm",
    "Qwen3OmniMoeMLP",
    "Qwen3OmniMoeRotaryEmbedding",
    # Code2Wav
    "Qwen3OmniMoeCausalConvNet",
    "Qwen3OmniMoeCausalTransConvNet",
    "Qwen3OmniMoeConvNeXtBlock",
    "Qwen3OmniMoeCode2WavAttention",
    "Qwen3OmniMoeCode2WavMlp",
    "Qwen3OmniMoeCode2WavRMSNorm",
    "Qwen3OmniMoeCode2WavLayerScale",
    "Qwen3OmniMoeCode2WavTransformerLayer",
    "Qwen3OmniMoeCode2WavTransformerModel",
    "Qwen3OmniMoeCode2WavDecoderResidualUnit",
    "Qwen3OmniMoeCode2WavDecoderBlock",
    "Qwen3OmniMoeCode2Wav",
    # SnakeBeta activation is only referenced inside the excluded Code2Wav
    # residual blocks, so exclude it too to avoid generating dead code.
    "SnakeBeta",
)


# ================================================================
# Module-level helper emitted into the generated file so multiprocessing
# dataloaders can pickle the per-sample position-id closure.
# ================================================================
config.add_post_import_block(
    '''
def get_position_id(main_func, self, **kwargs):
    """Per-sample position-ids for VeOmni dataloader workers.

    Invoked inside the data pipeline (bs=1 per sample). Wraps the HF
    get_rope_index so it can be partial-bound with a SimpleNamespace carrying
    the model config and helpers, then shipped across multiprocessing workers.
    """
    position_ids, rope_deltas = main_func(self, **kwargs)
    assert len(position_ids.shape) == 3 and position_ids.shape[1] == 1
    assert len(rope_deltas.shape) == 2 and rope_deltas.shape[0] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}
'''
)

# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
torch_npu = None


@config.override_method(
    "Qwen3OmniMoeThinkerTextRMSNorm.forward",
    description="Use fused rmsnorm to impl zero-centered rmsnorm (1+weight centered formulation)",
)
def qwen3_omni_moe_thinker_text_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]


@config.replace_function(
    "apply_rotary_pos_emb",
    description="Use fused rope to impl rotary postion embedding",
)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


@config.replace_function(
    "apply_rotary_pos_emb_vision", description="Use fused rope to impl rotary postion embedding in vit"
)
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = q.unsqueeze(0), k.unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(2).float()
    sin = sin.unsqueeze(0).unsqueeze(2).float()
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    q_embed, k_embed = q_embed.squeeze(0), k_embed.squeeze(0)
    return q_embed, k_embed


# ================================================================
# Patch: Qwen3OmniMoePreTrainedModel._init_weights
# Upstream branches on `isinstance(module, Qwen3OmniMoeCode2Wav)`; that
# class is excluded from the generated file (training never instantiates
# code2wav), so the name lookup fails at runtime during `post_init`. Drop
# the Code2Wav branch — everything else matches upstream verbatim.
# ================================================================
# These names (`init`, `np`, `SinusoidsPositionEmbedding`,
# `Qwen3OmniMoeThinkerTextSparseMoeBlock`, `Qwen3OmniMoeVisionRotaryEmbedding`)
# are resolved from the generated file's module namespace at import time,
# not from this patch config — patchgen lifts the function body verbatim
# into the generated class.
config.override_method(
    "Qwen3OmniMoePreTrainedModel._init_weights",
    replacement=qwen3_omni_moe_pretrained_init_weights_patched,
    description="Drop Qwen3OmniMoeCode2Wav branch since the class is excluded from the generated file",
)


# ================================================================
# Patch: Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index
# 1. [PosID] support interleaved video-with-audio vs video-without-audio in
#    one batch. v5 upstream uses a single `use_audio_in_video` boolean flag
#    applied globally; we use `audio_seqlens[audio_idx] == 0` (inherited from
#    the Qwen2.5-Omni convention) to decide per-video, consuming the zero
#    placeholder to keep `audio_idx` aligned.
# 2. [Mask] tolerate `attention_mask=None` at the data boundary.
# ================================================================
config.override_method(
    "Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index",
    replacement=qwen3_omni_moe_get_rope_index_patched,
    description="Per-video use_audio_in_video via audio_seqlens + None attention_mask tolerance",
)


# ================================================================
# Patch: Qwen3OmniMoeVisionAttention.forward
# 1. [SP] Dispatch via VARLEN_ATTENTION_TYPES (covers veomni_flash_attention_*
#    custom names) instead of v5 upstream `is_flash_attention_requested`,
#    which only recognizes HF built-in flash attention names. Without this
#    dispatch the SP-appended cu_seqlens-padding entry would run through the
#    non-varlen split branch and size-mismatch.
# ================================================================
config.override_method(
    "Qwen3OmniMoeVisionAttention.forward",
    replacement=qwen3_omni_moe_vision_attention_forward_patched,
    description="Route through VARLEN_ATTENTION_TYPES so veomni_flash_attention_* with cu_seqlens works",
)


# ================================================================
# Patch: Qwen3OmniMoeVisionEncoder.forward
# 1. [SP] Slice pos_embeds and rotary cos/sin to match the SP-sharded
#    hidden_states (pad_scale=4 matches the patch-embed's padding scale).
# 2. [SP] Append an extra cu_seqlens entry for the padded tail when the
#    total seq length is not divisible by sp_size, so the varlen kernel
#    attends within the padding chunk rather than across samples.
# 3. Return v5 BaseModelOutputWithDeepstackFeatures (with pooler_output =
#    merged_hidden_states and deepstack_features list).
# ================================================================
config.override_method(
    "Qwen3OmniMoeVisionEncoder.forward",
    replacement=qwen3_omni_moe_vision_forward_patched,
    description="SP-slice pos/rotary, extend cu_seqlens for SP pad tail, return BaseModelOutputWithDeepstackFeatures",
)


# ================================================================
# Patch: Qwen3OmniMoeVisionEncoder.dummy_forward (NEW method)
# [FSDP] Drive a synthetic forward so reduce-scatter stays in sync when
# some ranks get pixel_values=None.
# ================================================================
config.override_method(
    "Qwen3OmniMoeVisionEncoder.dummy_forward",
    replacement=qwen3_omni_moe_vision_dummy_forward_patched,
    description="FSDP dummy forward with patch-embed dtype lookup to stay bf16-safe under MixedPrecision",
)


# ================================================================
# Patch: Qwen3OmniMoeAudioEncoder.forward
# 1. [data] VeOmni ships input_features as (len, num_mel_bins); permute to
#    (num_mel_bins, len) to match the HF contract.
# 2. [SP] Gather along time, strip SP-padding so chunking is deterministic.
# 3. [SP] Slice hidden_states along seq dim before encoder layers; extend
#    cu_seqlens for the padded tail on the last rank.
# ================================================================
config.override_method(
    "Qwen3OmniMoeAudioEncoder.forward",
    replacement=qwen3_omni_moe_audio_forward_patched,
    description="Permute VeOmni (len, mel) input + SP gather/strip + SP slice + extend cu_seqlens",
)


# ================================================================
# Patch: Qwen3OmniMoeAudioEncoder.dummy_forward (NEW method)
# [FSDP] Synthetic forward so reduce-scatter stays in sync on ranks with
# no audio data. Minimum valid shape is one chunk of length n_window*2.
# Dtype is looked up from `self.conv2d1.weight.dtype` at call time — under
# FSDP2 + MixedPrecision, `self.dtype` (via `next(self.parameters()).dtype`)
# may still report the sharded full precision (fp32) while the per-module
# compute dtype has already been cast to bf16; using the conv weight's
# live dtype keeps the dummy inputs matched to whatever the conv actually
# runs in. We do NOT cache (`_dummy_data`) because the cached tensor
# would stay at first-call dtype and break on subsequent calls that
# enter a different mixed-precision context.
# ================================================================
config.override_method(
    "Qwen3OmniMoeAudioEncoder.dummy_forward",
    replacement=qwen3_omni_moe_audio_dummy_forward_patched,
    description="FSDP dummy forward with conv-weight dtype lookup (no caching) to stay bf16-safe",
)


# ================================================================
# Patch: Qwen3OmniMoeThinkerTextModel.forward
# Same outer shape as v5 upstream but:
# 1. Preserve v4's explicit 4-axis position_ids handling (`[4, bs, L]` where
#    index 0 is the text_position_ids used for the causal mask).
# 2. Return MoeModelOutputWithPast so @capture_outputs can inject
#    router_logits via the registered OutputRecorder.
# NOTE: the @merge_with_config_defaults / @capture_outputs / @auto_docstring
# decorators on the upstream method are preserved by patchgen.
# ================================================================
config.override_method(
    "Qwen3OmniMoeThinkerTextModel.forward",
    replacement=qwen3_omni_moe_thinker_text_model_forward_patched,
    description="Preserve explicit [4, bs, L] position_ids handling and MoeModelOutputWithPast return",
)


# ================================================================
# Patch: Qwen3OmniMoeThinkerTextModel._deepstack_process
# 1. [FSDP] If visual_pos_masks is None (no visual input on this rank) still
#    touch visual_embeds so FSDP reduce-scatter stays in sync across ranks.
# 2. [Mask] Squeeze trailing dim when mask is still 3D (legacy path).
# ================================================================
config.override_method(
    "Qwen3OmniMoeThinkerTextModel._deepstack_process",
    replacement=qwen3_omni_moe_thinker_text_deepstack_process_patched,
    description="Handle visual_pos_masks=None by adding 0.0 so FSDP reduce-scatter stays in sync",
)


# ================================================================
# Patch: Qwen3OmniMoeThinkerTextExperts (replace_class)
# 1. Drop the upstream `@use_experts_implementation` decorator — routing
#    through ALL_EXPERTS_FUNCTIONS bypasses our fused MoE kernel.
# 2. Add VeOmni fused-MoE dispatch via the `_moe_implementation` flag; pass
#    `gate_up_proj` directly as `fc1_1_2_weight` (v5 already stores it in
#    the fused `[E, 2*I, H]` layout).
# ================================================================
config.replace_class(
    "Qwen3OmniMoeThinkerTextExperts",
    replacement=PatchedQwen3OmniMoeThinkerTextExperts,
    description="Drop @use_experts_implementation and add VeOmni fused MoE dispatch",
)


# ================================================================
# Patch: Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_features
# Simplified to the VeOmni training path: input_features is already the
# flat (len, num_mel_bins) tensor (after the collator strips feature
# padding), and feature_attention_mask is not carried in training.
# Return the raw last_hidden_state to keep the forward body terse.
# ================================================================
config.override_method(
    "Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_features",
    replacement=qwen3_omni_moe_thinker_get_audio_features_patched,
    description="Simplify get_audio_features for VeOmni flat (len, mel) inputs — no feature_attention_mask",
)


# ================================================================
# Patch: Qwen3OmniMoeThinkerForConditionalGeneration.get_position_id_func
# Returns a per-sample closure that converts VeOmni's multimodal tokens
# (IMAGE_INPUT_INDEX / VIDEO_INPUT_INDEX / AUDIO_INPUT_INDEX) into 3D
# position_ids at data-preprocessing time. SimpleNamespace + unbound
# methods avoid pickling the full model across dataloader workers.
# ================================================================
config.override_method(
    "Qwen3OmniMoeThinkerForConditionalGeneration.get_position_id_func",
    replacement=qwen3_omni_moe_thinker_get_position_id_func_patched,
    description="Multiprocessing-safe per-sample position-id closure with VeOmni multimodal token ids",
)


# ================================================================
# Patch: Qwen3OmniMoeThinkerForConditionalGeneration.forward
# 1. [Constants] Use VeOmni data constants for multimodal token indices (via
#    get_position_id_func); precomputed masks arrive via kwargs.
# 2. [Mask] Pop pre-computed image/video/audio masks — avoids extra all_gather
#    for full mask information when using SP.
# 3. [ViT] Pop flash-attention kwargs before ViT forward so ViT computes its
#    own cu_seqlens from grid_thw.
# 4. [SP] gather_seq_scatter_heads on input/image/video/audio embeddings to
#    do the multimodal fill-back on the full sequence.
# 5. [FSDP] Dummy ViT/audio forward when pixel_values/input_features is None
#    on this rank.
# 6. [SP] gather_heads_scatter_seq to restore seq-parallel layout.
# 7. [SP] all_gather deepstack embeddings then select per-rank slice.
# 8. [Loss] Delegate loss to `self.loss_function` for fused CE.
# 9. [PosIDs] Transpose precomputed position_ids from (bs, 3, L) to (3, bs, L).
# 10.[Data] Filter zero-length audio_feature_lengths (placeholder entries for
#    videos without audio) before forwarding the audio tower.
# ================================================================
config.override_method(
    "Qwen3OmniMoeThinkerForConditionalGeneration.forward",
    replacement=qwen3_omni_moe_thinker_forward_patched,
    description="VeOmni SP + FSDP + precomputed masks + fused loss + precomputed multimodal position-ids",
)


# ================================================================
# Patch: Qwen3OmniMoeForConditionalGeneration.__init__
# 1. [MoE] Propagate `_moe_implementation` down to `config.thinker_config`
#    and `config.thinker_config.text_config` so the fused SparseMoeBlock
#    picks up the correct mode before sub-module construction.
# 2. [Talker] Force `has_talker=False` — VeOmni's training path only
#    forwards through `thinker` (see the patched `forward` below), so
#    constructing the talker and code2wav would only add unused
#    parameters and drag `Qwen3OmniMoeTalker*Layer` into the FSDP
#    `_no_split_modules` aggregation (HF recursively merges children's
#    `_no_split_modules` at `post_init`, see transformers
#    `modeling_utils.PreTrainedModel.post_init`). Unused talker layers
#    FSDP-wrapped but never forwarded cause a rank-desync hang during
#    asymmetric-modality forward.
# 3. [FSDP] After `post_init`, replace the aggregated
#    `self._no_split_modules` with the exact VeOmni target set. The
#    upstream top-level `Qwen3OmniMoePreTrainedModel._no_split_modules`
#    lists `Qwen3OmniMoeDecoderLayer` (a typo — no such class exists),
#    which `post_init` seeds into the aggregation. Resetting here
#    removes the phantom entry and pins the set to the three real
#    training targets.
# ================================================================
config.override_method(
    "Qwen3OmniMoeForConditionalGeneration.__init__",
    replacement=qwen3_omni_moe_for_conditional_generation_init_patched,
    description="Propagate _moe_implementation, skip talker for training, pin _no_split_modules to real training targets",
)


# ================================================================
# Patch: Qwen3OmniMoeForConditionalGeneration.enable_talker
# The talker + code2wav classes are excluded from the generated file
# (training never instantiates them), so the upstream body
# `self.talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(...)`
# would fail at import-time static analysis and at call time. Replace
# with an explicit NotImplementedError so the reason is clear if anything
# reaches here.
# ================================================================
config.override_method(
    "Qwen3OmniMoeForConditionalGeneration.enable_talker",
    replacement=qwen3_omni_moe_enable_talker_patched,
    description="Disable talker/code2wav path in the training modeling (excluded classes)",
)


# ================================================================
# Patch: Qwen3OmniMoeForConditionalGeneration.forward
# Simplified training path: only forward through thinker; talker +
# code2wav are skipped (only used in the TTS generate path).
# ================================================================
config.override_method(
    "Qwen3OmniMoeForConditionalGeneration.forward",
    replacement=qwen3_omni_moe_for_conditional_generation_forward_patched,
    description="Forward through thinker only (talker/code2wav not trained via this path)",
)


# ================================================================
# Patch: Qwen3OmniMoeForConditionalGeneration.get_position_id_func (NEW)
# Delegate to the thinker's closure; data pipeline calls the top-level
# model's get_position_id_func.
# ================================================================
config.override_method(
    "Qwen3OmniMoeForConditionalGeneration.get_position_id_func",
    replacement=qwen3_omni_moe_top_get_position_id_func_patched,
    description="Delegate position-id computation to the thinker submodule",
)


# ================================================================
# Patch: Qwen3OmniMoeForConditionalGeneration.get_parallel_plan (NEW)
# Register the VeOmni EP plan for thinker.model.layers.*.mlp.experts.*
# on the generated v5 modeling.
# ================================================================
config.override_method(
    "Qwen3OmniMoeForConditionalGeneration.get_parallel_plan",
    replacement=qwen3_omni_moe_get_parallel_plan_patched,
    description="Register Qwen3-Omni-MoE thinker expert parallel plan for v5 generated modeling",
)
