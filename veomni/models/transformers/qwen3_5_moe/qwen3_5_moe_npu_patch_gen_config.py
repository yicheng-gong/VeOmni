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
Patch configuration for Qwen3_5Moe NPU/SP patched modeling generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_5_moe.qwen3_5_moe_npu_patch_gen_config -o veomni/models/transformers/qwen3_5_moe/generated --diff

Patches applied:
1. Fused MoE expert replacement (merged gate_up_proj layout).
2. Device-agnostic GatedDeltaNet init and varlen FLA forward.
3. DecoderLayer forward with cu_seq_lens_q passthrough.
4. Fused loss + aux_loss in ForConditionalGeneration.
"""

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from veomni.models.transformers.qwen3_5.qwen3_5_npu_patch_gen_config import (
    PatchedQwen3_5RMSNorm,
    qwen3_5_gated_deltanet_forward_patched,
)
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    qwen3_5_gated_deltanet_get_local_conv1d_weight,
    qwen3_5_gated_deltanet_init_patched,
    qwen3_5_model_get_image_features,
    qwen3_5_model_get_placeholder_mask,
    qwen3_5_vision_model_dummy_forward,
    qwen3_5_vision_model_fast_pos_embed_interpolate,
    qwen3_5_vision_model_forward,
)
from veomni.models.transformers.qwen3_5_moe.qwen3_5_moe_gpu_patch_gen_config import (
    qwen3_5_moe_model_init_patched,
    qwen3_5_moe_sparse_moe_block_forward_patched,
    qwen3_5_moe_model_forward_patched,
    qwen3_5_moe_forconditional_generation_get_position_id_func,
    qwen3_5_moe_decoder_layer_forward_patched,
    qwen3_5_moe_forconditional_generation_forward_patched,
    qwen3_5_moe_get_parallel_plan_patched,
)
from veomni.ops import fused_moe_forward
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    target_file="patched_modeling_qwen3_5_moe_npu.py",
    description="Qwen3_5Moe with mojo_opset NPU replacements, fused MoE, and VeOmni SP/fused loss patches",
)

config.add_import("copy", names=["copy"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import("veomni.utils.device", names=["get_device_id"])
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_seq_scatter_heads", "gather_heads_scatter_seq"],
)
config.add_import("veomni.distributed.sequence_parallel", names=["sp_pad_and_slice"])
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
config.drop_import_names(
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)
config.add_post_import_block(
    """
    # Modification: We are not using https://github.com/Dao-AILab/causal-conv1d now
    # we are using the triton impl of causal_conv1d from fla.
    # TODO: Evaluate Tridao's impl in the future.
    from functools import wraps

    from mojo_opset import MojoRMSNormFunction, mojo_rope
    from mojo_opset import mojo_causal_conv1d as causal_conv1d_fn
    # from mojo_opset_ext import mojo_chunk_gated_delta_rule as chunk_gated_delta_rule
    from xpu_models.ops.attn_impl import chunk_gated_delta_rule
    def causal_conv1d_fn_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.pop("seq_idx")
            kwargs.pop("backend")
            return func(*args, **kwargs)

        return wrapper

    causal_conv1d_fn = causal_conv1d_fn_wrapper(causal_conv1d_fn)

    FusedRMSNormGated = None
    fused_recurrent_gated_delta_rule = None
    causal_conv1d_update = None
    """
)


# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None


config.replace_class(
    "Qwen3_5MoeRMSNorm",
    replacement=PatchedQwen3_5RMSNorm,
    name_map={"PatchedQwen3_5RMSNorm": "Qwen3_5MoeRMSNorm"},
    description="Use eager Qwen3Next-style RMSNorm (1+weight centered formulation) for NPU patchgen",
)


# NOTE: apply_rotary_pos_emb is NOT replaced with LigerKernel rotary because
# Qwen3_5Moe uses partial_rotary_factor=0.25 with mrope_interleaved=True.
# The HF implementation correctly handles partial rotary (applying RoPE only
# to the first `rotary_dim` dims and passing through the rest), while
# liger_rotary_pos_emb applies RoPE to the full head_dim, producing incorrect
# results and NaN in attention output.
# @config.replace_function(
#     "apply_rotary_pos_emb",
#     description="Use fused rope for NPU patchgen",
# )
# def patched_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Removes the interleaving of cos and sin from GLM

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     from mojo_opset import mojo_rope
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)

#     # Keep half or full tensor for later concatenation
#     rotary_dim = cos.shape[-1]
#     q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
#     k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

#     # Apply rotary embeddings on the first half or full tensor
#     q_embed, k_embed = mojo_rope(q_rot, k_rot, cos, sin)

#     # Concatenate back to full shape
#     q_embed = torch.cat([q_embed, q_pass], dim=-1)
#     k_embed = torch.cat([k_embed, k_pass], dim=-1)
#     return q_embed, k_embed

# ── Propagate _moe_implementation from top-level config to text_config ────────


config.override_method(
    "Qwen3_5MoeModel.__init__",
    replacement=qwen3_5_moe_model_init_patched,
    description="Propagate _moe_implementation from top-level config to text_config",
)


# ── SparseMoeBlock forward (avoid in-place op on autograd Function output) ────


config.override_method(
    "Qwen3_5MoeSparseMoeBlock.forward",
    replacement=qwen3_5_moe_sparse_moe_block_forward_patched,
    description="Avoid in-place += on custom autograd Function output",
)


# ── ViT patches ───────────────────────────────────────────────────────────────

config.override_method(
    "Qwen3_5MoeModel.get_image_features",
    replacement=qwen3_5_model_get_image_features,
    description="Remove unnecessary split operation to maintain contiguous memory layout.",
)

config.override_method(
    "Qwen3_5MoeModel.get_placeholder_mask",
    replacement=qwen3_5_model_get_placeholder_mask,
    description="Extract multimodal placeholder masks from input_ids using self-defined placeholder IDs.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.fast_pos_embed_interpolate",
    replacement=qwen3_5_vision_model_fast_pos_embed_interpolate,
    description="Optimized bilinear interpolation for high-resolution vision embeddings, adapted from vLLM.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.forward",
    replacement=qwen3_5_vision_model_forward,
    description="Optimized vision forward with Sequence Parallel (SP) support and padded cu_seqlens.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.dummy_forward",
    replacement=qwen3_5_vision_model_dummy_forward,
    description="Add dummy_forward to prevent FSDP reduce-scatter hang on uneven multimodal batches.",
)


config.override_method(
    "Qwen3_5MoeModel.forward",
    replacement=qwen3_5_moe_model_forward_patched,
    description=(
        "Optimized multimodal forward supporting Ulysses SP (multimodal scattering), "
        "FSDP-safe dummy vision processing, position_ids shape alignment, and "
        "CPU-GPU sync avoidance via pre-computed metadata."
    ),
)


config.add_post_import_block("""
def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}
""")


config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_position_id_func",
    replacement=qwen3_5_moe_forconditional_generation_get_position_id_func,
    description="Expose get_position_id_func to pre-computes position IDs per sample during data preprocessing in worker processes.",
)


# ── MoE Expert replacement (merged gate_up_proj layout) ─────────────────────────


@config.replace_class(
    "Qwen3_5MoeExperts",
    description="Remove @use_experts_implementation decorator and add VeOmni fused MoE dispatch path",
)
class PatchedQwen3_5MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    Replaces the HF class to remove the @use_experts_implementation decorator
    (which routes to grouped_mm and bypasses our fused MoE path) and to add
    VeOmni fused MoE dispatch via _moe_implementation config flag.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        # Modification: read _moe_implementation to switch between eager and fused MoE paths.
        self._moe_implementation = getattr(config, "_moe_implementation", "eager")

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        # Modification: dispatch to fused MoE when _moe_implementation is set.
        # NPU fused MoE kernels are more stable with explicit split+contiguous fc1 weights.
        if self._moe_implementation == "fused":
            intermediate_dim = self.intermediate_dim
            fc1_1_weight = self.gate_up_proj[:, :intermediate_dim, :].contiguous()
            fc1_2_weight = self.gate_up_proj[:, intermediate_dim:, :].contiguous()
            selected_experts = top_k_index.to(torch.int32)
            routing_weights = top_k_weights.to(final_hidden_states.dtype)
            final_hidden_states = fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                hidden_states=hidden_states,
                fc1_1_weight=fc1_1_weight,
                fc1_2_weight=fc1_2_weight,
                fc2_weight=self.down_proj,
                fc1_1_2_weight=None,
            )
        elif self._moe_implementation == "eager":
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        else:
            raise ValueError(f"Invalid moe implementation: {self._moe_implementation}")

        return final_hidden_states


# ── GatedDeltaNet patches (shared with qwen3_5 via name_map) ─────────────────

_NAME_MAP = {"Qwen3_5": "Qwen3_5Moe"}

config.override_method(
    "Qwen3_5MoeGatedDeltaNet.__init__",
    replacement=qwen3_5_gated_deltanet_init_patched,
    name_map=_NAME_MAP,
    description="Use device-agnostic get_device_id() for FusedRMSNormGated init",
)

config.override_method(
    "Qwen3_5MoeGatedDeltaNet._get_local_conv1d_weight",
    replacement=qwen3_5_gated_deltanet_get_local_conv1d_weight,
    name_map=_NAME_MAP,
    description="Shard depthwise conv1d weights for local heads under Ulysses SP",
)

config.override_method(
    "Qwen3_5MoeGatedDeltaNet.forward",
    replacement=qwen3_5_gated_deltanet_forward_patched,
    name_map=_NAME_MAP,
    description="Support varlen flash linear attention and Ulysses SP in Qwen3_5MoeGatedDeltaNet.forward",
)


# ── DecoderLayer forward ────────────────────────────────────────────────────────


config.override_method(
    "Qwen3_5MoeDecoderLayer.forward",
    replacement=qwen3_5_moe_decoder_layer_forward_patched,
    description="Extract and pass cu_seq_lens_q for varlen linear attention in Qwen3_5MoeDecoderLayer.forward",
)


# ── ForConditionalGeneration forward (fused loss + aux_loss) ─────────────────────


config.override_method(
    "Qwen3_5MoeForConditionalGeneration.forward",
    replacement=qwen3_5_moe_forconditional_generation_forward_patched,
    description="Support fused cross entropy path in Qwen3_5MoeForConditionalGeneration.forward",
)



# ── Expert parallel plan ─────────────────────────────────────────────────────


config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_parallel_plan",
    replacement=qwen3_5_moe_get_parallel_plan_patched,
    description="Register Qwen3_5Moe expert parallel plan for v5 generated modeling",
)
