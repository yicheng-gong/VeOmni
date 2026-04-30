import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import (
    use_kernelized_func,
)
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    ModelOutput,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeConfig,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    auto_docstring,
    logging,
    torch_compilable_check,
)
from transformers.utils.generic import (
    TransformersKwargs,
    can_return_tuple,
    is_flash_attention_requested,
    maybe_autocast,
    merge_with_config_defaults,
)
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from xpu_models.ops.attn_impl import chunk_gated_delta_rule_func as chunk_gated_delta_rule
from xpu_models.ops.mojo_impl.conv1d import causal_conv1d_fn

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad_and_slice, unpad_tensor
from veomni.distributed.sequence_parallel.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads
from veomni.ops import fused_moe_forward
from veomni.utils.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.device import get_device_id

from .configuration_qwen3_5_omni_moe import Qwen3_5OmniMoeConfig, Qwen3_5OmniMoeThinkerConfig


FusedRMSNormGated = None
fused_recurrent_gated_delta_rule = None
causal_conv1d_update = None


logger = logging.get_logger(__name__)


class Qwen3_5MoeVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3_5MoeTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3_5MoeTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3_5MoeTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3_5Moe has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Qwen3_5MoeDynamicCache:
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the linear attention
    cache (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for gated deltanet cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For linear attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `recurrent_states` represents the recurrent state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    is_compileable = False

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        # Initialize everything to None -> will be lazy initialized to allow multi-gpu (device_map) inference
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        return self.conv_states[self.last_linear_layer] is not None


class Qwen3_5MoeRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        hidden_states = torch.cat([gate, hidden_states], dim=-1)
        hidden_states = torch_npu.npu_swiglu(hidden_states, dim=-1)

        return hidden_states


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5MoeGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
            Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                # Modification: use device-agnostic get_device_id() instead of hardcoded device
                device=get_device_id(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3_5MoeDynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # Modification: plumb varlen sequence metadata to FLA kernels.
        cu_seq_lens_q: torch.Tensor | None = None,
    ):
        cu_seq_lens_q = cu_seq_lens_q.pin_memory().to(
            device=torch.npu.current_device(), non_blocking=True
        )
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Modification: Ulysses SP all-to-all for linear attention heads.
        ulysses_enabled = get_parallel_state().ulysses_enabled
        if ulysses_enabled:
            ulysses_group = get_parallel_state().ulysses_group
            ulysses_size = get_parallel_state().ulysses_size
            ulysses_rank = get_parallel_state().ulysses_rank
            assert self.num_k_heads % ulysses_size == 0 and self.num_v_heads % ulysses_size == 0, (
                f"SP size ({ulysses_size}) must divide num_k_heads ({self.num_k_heads}) "
                f"and num_v_heads ({self.num_v_heads}) for gated deltanet LASP"
            )

            local_num_k_heads = self.num_k_heads // ulysses_size
            local_num_v_heads = self.num_v_heads // ulysses_size
            local_key_dim = self.head_k_dim * local_num_k_heads
            local_value_dim = self.head_v_dim * local_num_v_heads

            # Reshape mixed_qkv to head layout for all-to-all: [B, S_local, D] -> split+reshape to heads
            q_proj, k_proj, v_proj = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q_proj = q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
            k_proj = k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
            v_proj = v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

            # All-to-all: gather full sequence, scatter heads -> [B, S_full, local_heads, head_dim]
            q_proj = gather_seq_scatter_heads(q_proj, seq_dim=1, head_dim=2, group=ulysses_group)
            k_proj = gather_seq_scatter_heads(k_proj, seq_dim=1, head_dim=2, group=ulysses_group)
            v_proj = gather_seq_scatter_heads(v_proj, seq_dim=1, head_dim=2, group=ulysses_group)

            b = b.reshape(batch_size, seq_len, self.num_v_heads)
            a = a.reshape(batch_size, seq_len, self.num_v_heads)
            b = gather_seq_scatter_heads(b, seq_dim=1, head_dim=2, group=ulysses_group)
            a = gather_seq_scatter_heads(a, seq_dim=1, head_dim=2, group=ulysses_group)

            # Flatten heads back to channels and concat for conv1d: [B, S_full, local_dim]
            q_proj = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], -1)
            k_proj = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], -1)
            v_proj = v_proj.reshape(v_proj.shape[0], v_proj.shape[1], -1)
            mixed_qkv = torch.cat((q_proj, k_proj, v_proj), dim=-1)
        else:
            local_num_k_heads = self.num_k_heads
            local_num_v_heads = self.num_v_heads
            local_key_dim = self.key_dim
            local_value_dim = self.value_dim

        if use_precomputed_states:
            # Modification: keep this disabled until FLA causal_conv1d_update decode path is validated.
            raise NotImplementedError("use_precomputed_states=True is not supported yet for causal_conv1d_update now.")
        else:
            if cache_params is not None:
                mixed_qkv_t = mixed_qkv.transpose(1, 2)
                conv_state = F.pad(mixed_qkv_t, (self.conv_kernel_size - mixed_qkv_t.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                # Modification: shard conv1d weights per Ulysses rank to match head-sharded channels.
                if ulysses_enabled:
                    conv_weight = self._get_local_conv1d_weight(
                        ulysses_rank=ulysses_rank,
                        local_key_dim=local_key_dim,
                        local_value_dim=local_value_dim,
                    )
                else:
                    conv_weight = self.conv1d.weight.squeeze(1)
                # mixed_qkv is [B, S, D] — FLA causal_conv1d expects [B, S, D].
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=conv_weight,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                    backend="triton",
                    cu_seqlens=cu_seq_lens_q.npu(),
                )[0]
            else:
                raise NotImplementedError("This path is not supported yet because it can't process varlen now.")

        query, key, value = torch.split(
            mixed_qkv,
            [
                local_key_dim,
                local_key_dim,
                local_value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(query.shape[0], query.shape[1], local_num_k_heads, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], local_num_k_heads, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], local_num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        # Modification: slice A_log/dt_bias for local V-heads under Ulysses SP.
        if ulysses_enabled:
            v_head_offset = ulysses_rank * local_num_v_heads
            v_head_slice = slice(v_head_offset, v_head_offset + local_num_v_heads)
            g = -self.A_log[v_head_slice].float().exp() * F.softplus(a.float() + self.dt_bias[v_head_slice])
        else:
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            if self.chunk_gated_delta_rule is torch_chunk_gated_delta_rule:
                raise RuntimeError(
                    "Varlen training requires FLA. Install flash-linear-attention so "
                    "chunk_gated_delta_rule supports cu_seqlens."
                )
            else:
                # Modification: use direct args and pass cu_seqlens for varlen FLA attention.
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seq_lens_q.npu(),
                )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # Modification: gather attention output back to sequence-sharded layout before gated norm.
        if ulysses_enabled:
            core_attn_out = gather_heads_scatter_seq(
                core_attn_out, head_dim=2, seq_dim=1, group=get_parallel_state().ulysses_group
            )

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output

    def _get_local_conv1d_weight(self, ulysses_rank: int, local_key_dim: int, local_value_dim: int) -> torch.Tensor:
        # Modification: shard depthwise conv1d weights to match head-sharded mixed_qkv channels.
        w_full = self.conv1d.weight.squeeze(1)
        assert w_full.shape[0] == self.key_dim * 2 + self.value_dim, (
            f"conv1d weight dim ({w_full.shape[0]}) must match "
            f"(2 * key_dim + value_dim) ({self.key_dim * 2 + self.value_dim})"
        )
        k_off = ulysses_rank * local_key_dim
        v_off = ulysses_rank * local_value_dim
        w_q = w_full[k_off : k_off + local_key_dim]
        w_k = w_full[self.key_dim + k_off : self.key_dim + k_off + local_key_dim]
        w_v = w_full[2 * self.key_dim + v_off : 2 * self.key_dim + v_off + local_value_dim]
        return torch.cat([w_q, w_k, w_v], dim=0)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3_5MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3_5MoeRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
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
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5MoeMLP(nn.Module):
    def __init__(self, config: Qwen3_5MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3_5MoeExperts(nn.Module):
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
        # Pass gate_up_proj directly as fc1_1_2_weight to avoid chunk + contiguous overhead.
        if self._moe_implementation == "fused":
            final_hidden_states = fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights.to(final_hidden_states.dtype),
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=None,
                fc1_2_weight=None,
                fc2_weight=self.down_proj,
                fc1_1_2_weight=self.gate_up_proj,
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


class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    # ── SparseMoeBlock forward (avoid in-place op on autograd Function output) ────
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        # Modification: use out-of-place add instead of `expert_output += shared_expert_output`
        # to avoid "Output of MergedFc1TritonFusedMoeExpertFunctionBackward is a view and is
        # being modified inplace" RuntimeError from PyTorch autograd.
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.eps)[0]

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ── DecoderLayer forward ────────────────────────────────────────────────────────
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Modification: read varlen metadata from kwargs and enforce it for linear-attention varlen kernels.
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        assert cu_seq_lens_q is not None, (
            "cu_seq_lens_q must be provided to support varlen Flash Linear Attention, varlen Conv1D,"
            "and to remove the full Flash Attention CPU-GPU sync."
        )

        # Token Mixer
        if self.layer_type == "linear_attention":
            # Modification: pass cu_seq_lens_q through to Qwen3_5MoeGatedDeltaNet.forward.
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5MoePreTrainedModel(PreTrainedModel):
    config: Qwen3_5MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionModel"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3_5MoeTopKRouter, index=0),
        "hidden_states": Qwen3_5MoeDecoderLayer,
        "attentions": Qwen3_5MoeAttention,
    }
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5MoeGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3_5MoeRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3_5MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_5MoeSparseMoeBlock):
            init.normal_(module.gate.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_5MoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Qwen3_5MoeVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3_5MoeVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3_5MoeVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3_5MoeVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


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


class Qwen3_5MoeVisionAttention(nn.Module):
    def __init__(self, config: Qwen3_5MoeVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5MoeVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5MoeVisionAttention(config=config)
        self.mlp = Qwen3_5MoeVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5MoeVisionModel(Qwen3_5MoePreTrainedModel):
    config: Qwen3_5MoeVisionConfig
    _no_split_modules = ["Qwen3_5MoeVisionModel"]
    _can_record_outputs = {
        "hidden_states": Qwen3_5MoeVisionBlock,
        "attentions": Qwen3_5MoeVisionAttention,
    }

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3_5MoeVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3_5MoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5MoeVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.gradient_checkpointing = False

        self.post_init()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        """
        Efficient implementation adapted from vLLM's Qwen-VL optimization.

        Key optimizations over standard Transformers implementation:
        1. Computational Efficiency: Reduces bilinear interpolation multiplications from 4 to 1
           per patch using algebraic simplification (w11=dh*dw; w10=dh-w11; w01=dw-w11; w00=1-dh-w01).
        2. Vectorization: Uses torch.meshgrid to compute indices and weights for the entire
           grid at once, avoiding expensive Python loops.

        Original source: https://github.com/vllm-project/vllm/blob/95c0f92/vllm/model_executor/models/qwen3_vl.py#L470
        """

        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        dtype = self.pos_embed.weight.dtype
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h, device=self.device, dtype=torch.float64)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w, device=self.device, dtype=torch.float64)

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=dtype)

            embeds = self.pos_embed(indices) * weights
            combined = embeds[0] + embeds[1] + embeds[2] + embeds[3]
            combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)

            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)

            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        # --- Patch.1: Sequence parallel padding and slicing for position embeddings ---
        if get_parallel_state().sp_enabled:
            # Note: grid_thw records the original, unpadded visual shapes. However, the data collator
            # pads the visual sequence (hidden_states) to a multiple of (sp_size * pad_scale)
            # to support Sequence Parallelism and subsequent spatial merging.
            #
            # pad_scale=4 matches the 4-to-1 spatial merge (2x2 pooling) ratio in the Qwen-VL Vision Tower.
            # We must manually pad and slice the generated position embeddings to ensure they
            # correctly align with the padded and sharded hidden states.
            pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)
        # --- Patch.1 ---

        hidden_states = hidden_states + pos_embeds

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        # --- Patch.2: Flatten full-sequence rotary embeddings using the actual total sequence length ---
        # In Sequence Parallelism, hidden_states.size(0) only represents the local shard length.
        # We must use cu_seqlens[-1] (derived from unpadded grid_thw) to flatten the global
        # rotary_pos_emb. This ensures the embeddings cover the entire original sequence
        # before they are padded and sliced in Patch 3 to match the sharded hidden_states.
        total_seq_len = cu_seqlens[-1]
        rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
        # --- Patch.2 ---

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        if get_parallel_state().sp_enabled:
            # --- Patch.3: Sequence parallel padding and slicing for sin/cos rotary embeddings ---
            cos, sin = position_embeddings
            # Similar to Patch.1, we pad and slice the rotary embeddings to align with the
            # padded hidden states, using pad_scale=4 to match the 4-to-1 spatial merge ratio.
            cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
            sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
            position_embeddings = (cos, sin)
            # --- Patch.3 ---

            # --- Patch.4: Pad cu_seqlens to align with the padded hidden_states buffer under SP ---
            # The Data Collator pads hidden_states to a multiple of (sp_size * pad_scale),
            # but cu_seqlens (derived from grid_thw) only covers the original unpadded sequence.
            # We must extend cu_seqlens to cover the entire padded buffer by treating the
            # padding region as an additional "virtual sample". This ensures that varlen
            # kernels (like FlashAttention) process the full buffer, preventing shape
            # mismatches or collective communication hangs during subsequent Sequence
            # Parallel operations (e.g., All-to-All).
            sp_size = get_parallel_state().sp_size
            # Calculate global padding: (local_seq_len * num_ranks) - original_total_len
            pad_seq_len = seq_len * sp_size - total_seq_len.item()
            if pad_seq_len > 0:
                # Append a new entry to cu_seqlens to include the padding tokens as a final segment
                new_cumsum = cu_seqlens[-1] + pad_seq_len
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
            # --- Patch.4 ---

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    def dummy_forward(self):
        """
        # Run a fake ViT forward so every FSDP rank touches the vision tower.
        # This prevents reduce-scatter hangs when some ranks have no real images/videos.
        """
        if get_parallel_state().sp_enabled:
            sp_size = get_parallel_state().sp_size

            # Fake patch sequence for one local rank:
            # 16 patch tokens, each token flattened from:
            #   3 channels * 2 temporal patches * 16 * 16 spatial patch
            pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
            # grid_thw describes the *global* pre-sharded vision grid, not the local shard.
            # Here:
            #   T = 1
            #   H = 4 * sp_size
            #   W = 4
            # so total global patch tokens = 1 * (4 * sp_size) * 4 = 16 * sp_size.
            grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
            dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        else:
            pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
            # Non-SP case: a minimal valid 4x4 patch grid.
            # Total patch tokens = 1 * 4 * 4 = 16.
            grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
            dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        return self(**dummy_data)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen3_5MoeModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen3_5Moe causal language model (or autoregressive) outputs.
    """
)
class Qwen3_5MoeCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    router_logits: tuple[torch.FloatTensor] | None = None
    aux_loss: torch.FloatTensor | None = None


class Qwen3_5MoeTextModel(Qwen3_5MoePreTrainedModel):
    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5MoeDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # mrope: the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for _, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask


# Additional import blocks for patches
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


@dataclass
@auto_docstring
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    r"""
    deepstack_features (`List[torch.FloatTensor]`, *optional*):
        List of hidden-states (feature maps) from deepstack layers.
    """

    deepstack_features: list[torch.FloatTensor] | None = None


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


# ======================================================================
# [MODIFIED CLASS] Qwen3OmniMoePreTrainedModel
# Methods patched: _init_weights
# ======================================================================


@auto_docstring
class Qwen3OmniMoePreTrainedModel(PreTrainedModel):
    config: Qwen3OmniMoeConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionModel", "Qwen3OmniMoeAudioEncoder"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True

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
    @torch.no_grad()
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5MoeGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3_5MoeRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3_5MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_5MoeSparseMoeBlock):
            init.normal_(module.gate.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_5MoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)
        elif isinstance(module, SinusoidsPositionEmbedding):  # noqa: F821
            log_timescale_increment = np.log(module.max_timescale) / (module.channels // 2 - 1)  # noqa: F821
            inv_timescales = torch.exp(-log_timescale_increment * torch.arange(module.channels // 2).float())
            scaled_time = torch.arange(module.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]  # noqa: F821
            init.copy_(module.positional_embedding, torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1))  # noqa: F821


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


# ======================================================================
# [MODIFIED CLASS] Qwen3OmniMoePreTrainedModelForConditionalGeneration
# Methods patched: get_rope_index
# ======================================================================


class Qwen3OmniMoePreTrainedModelForConditionalGeneration(Qwen3OmniMoePreTrainedModel):
    input_modalities = ("image", "video", "audio", "text")

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`torch.Tensor` of shape `(seq_len, )`): A monotonically increasing list of
                                token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
            remove_index (`int`) An index id to subtract from `token_indices` before chunking

        Returns:
            `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    # ================================================================
    # Patch: Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index
    # 1. [PosID] support interleaved video-with-audio vs video-without-audio in
    #    one batch. v5 upstream uses a single `use_audio_in_video` boolean flag
    #    applied globally; we use `audio_seqlens[audio_idx] == 0` (inherited from
    #    the Qwen2.5-Omni convention) to decide per-video, consuming the zero
    #    placeholder to keep `audio_idx` aligned.
    # 2. [Mask] tolerate `attention_mask=None` at the data boundary.
    # ================================================================
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # --- Patch.1 ---
        # use_audio_in_video removed; decided per-video below.
        # --- Patch.1 ---
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            # --- Patch.2 ---
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            attention_mask = attention_mask == 1
            # --- Patch.2 ---
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids_i in enumerate(total_input_ids):
                # --- Patch.2 ---
                input_ids_i = input_ids_i[attention_mask[i]]
                # --- Patch.2 ---
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_i[vision_start_indices + 1]

                # --- Patch.1 ---
                audio_start_indices = torch.argwhere(input_ids_i == audio_start_token_id).squeeze(1)
                audio_nums = torch.sum(
                    input_ids_i[audio_start_indices - 1] != vision_start_token_id
                )  # audio but not in <video><audio>
                # --- Patch.1 ---

                image_nums = (vision_tokens == image_token_id).sum()
                # --- Patch.1 ---
                video_nums = (vision_tokens == audio_start_token_id).sum() + (vision_tokens == video_token_id).sum()
                # --- Patch.1 ---

                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums

                # --- Patch.1 ---
                multimodal_nums = image_nums + video_nums + audio_nums
                # --- Patch.1 ---

                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx += text_len
                    # audio-in-video shares bos (vision_start immediately followed by audio_start)
                    if min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == audio_start_token_id:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += bos_len
                    # Audio only
                    if min_ed == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image only
                    elif min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == image_token_id:
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video only — audio track determined per-video via audio_seqlens
                    elif min_ed == ed_vision_start:
                        # --- Patch.1 ---
                        if audio_seqlens[audio_idx] == 0:
                            use_audio_in_video = False
                            audio_idx += 1  # consume zero-length placeholder
                        else:
                            use_audio_in_video = True
                        # --- Patch.1 ---

                        if not use_audio_in_video:
                            assert input_ids_i[ed_vision_start + 1] == video_token_id

                            grid_t = video_grid_thw[video_idx][0]
                            grid_hs = video_grid_thw[:, 1]
                            grid_ws = video_grid_thw[:, 2]
                            t_index = (
                                torch.arange(grid_t)
                                * second_per_grids[video_idx].cpu().float()
                                * position_id_per_seconds
                            ).float()
                            llm_pos_ids = self.get_llm_pos_ids_for_vision(
                                st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                            )
                            video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                            llm_pos_ids_list.append(llm_pos_ids)

                            st += int(text_len + bos_len + video_len + eos_len)
                            video_idx += 1
                            remain_videos -= 1
                        else:
                            assert input_ids_i[ed_vision_start + 1] == audio_start_token_id
                            audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                            audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                            grid_t = video_grid_thw[video_idx][0]
                            grid_hs = video_grid_thw[:, 1]
                            grid_ws = video_grid_thw[:, 2]
                            t_index = (
                                torch.arange(grid_t)
                                * second_per_grids[video_idx].cpu().float()
                                * position_id_per_seconds
                            ).float()
                            video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                                st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                            )
                            video_data_index, audio_data_index = 0, 0
                            while (
                                video_data_index < video_llm_pos_ids.shape[-1]
                                and audio_data_index < audio_llm_pos_ids.shape[-1]
                            ):
                                if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                                    llm_pos_ids_list.append(
                                        video_llm_pos_ids[:, video_data_index : video_data_index + 1]
                                    )
                                    video_data_index += 1
                                else:
                                    llm_pos_ids_list.append(
                                        audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1]
                                    )
                                    audio_data_index += 1
                            if video_data_index < video_llm_pos_ids.shape[-1]:
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                                )
                            if audio_data_index < audio_llm_pos_ids.shape[-1]:
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                                )
                            video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                            st += int(text_len + bos_len + audio_len + video_len + eos_len)
                            audio_idx += 1
                            video_idx += 1
                            remain_videos -= 1

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_i))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas


class Qwen3OmniMoeAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""

        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,  # pass cu seq lens for FA2
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class Qwen3OmniMoeAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3OmniMoeAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


# ======================================================================
# [MODIFIED CLASS] Qwen3OmniMoeAudioEncoder
# Methods patched: forward, dummy_forward
# ======================================================================


@auto_docstring(
    custom_intro="""
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen3OmniMoeAudioEncoderLayer`].
    """
)
class Qwen3OmniMoeAudioEncoder(Qwen3OmniMoePreTrainedModel):
    config: Qwen3OmniMoeAudioEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Qwen3OmniMoeAudioEncoder"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": Qwen3OmniMoeAudioEncoderLayer,
        "attentions": Qwen3OmniMoeAudioAttention,
    }

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([Qwen3OmniMoeAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv2d1

    def set_input_embeddings(self, value):
        self.conv2d1 = value

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
        # NOTE: the created attention masl only approximates the ragged FA2 attention by
        # allowing bidirectional attention within `cu_seqlens` blocks, and not attending between
        # blocks. Though it will not be a 100% match for FA2's `varlen` path
        if is_flash_attention_requested(self.config):
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    # ================================================================
    # Patch: Qwen3OmniMoeAudioEncoder.forward
    # 1. [data] VeOmni ships input_features as (len, num_mel_bins); permute to
    #    (num_mel_bins, len) to match the HF contract.
    # 2. [SP] Gather along time, strip SP-padding so chunking is deterministic.
    # 3. [SP] Slice hidden_states along seq dim before encoder layers; extend
    #    cu_seqlens for the padded tail on the last rank.
    # ================================================================
    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        **kwargs,
    ):
        # --- Patch.1 ---
        input_features = input_features.permute(1, 0)  # (len, num_mel_bins) -> (num_mel_bins, len)
        # --- Patch.1 ---

        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.full((chunk_num.sum(),), self.n_window * 2, dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        # --- Patch.2 ---
        if get_parallel_state().sp_enabled:
            unpadded_input_len = torch.sum(chunk_lengths)
            input_features = gather_outputs(input_features, gather_dim=1, group=get_parallel_state().sp_group)
            sp_input_padding = input_features.size(1) - unpadded_input_len
            if sp_input_padding > 0:
                input_features = unpad_tensor(input_features, dim=1, padding_size=sp_input_padding)
        # --- Patch.2 ---

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            unpadded_hidden_len = cu_seqlens[-1]
            hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
            pad_seq_len = hidden_states.size(0) * get_parallel_state().sp_size - unpadded_hidden_len
            if pad_seq_len > 0:
                cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + pad_seq_len).unsqueeze(0)], dim=0)
        # --- Patch.3 ---

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

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
    def dummy_forward(self):
        min_len = self.n_window * 2
        dtype = self.conv2d1.weight.dtype
        input_features = torch.zeros((min_len, self.num_mel_bins), dtype=dtype, device=self.device)
        feature_lens = torch.tensor([min_len], dtype=torch.long, device=self.device)
        return self(input_features=input_features, feature_lens=feature_lens)


@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    r"""
    Args:
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    rope_deltas: torch.LongTensor | None = None


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# ======================================================================
# [MODIFIED CLASS] Qwen3OmniMoeThinkerForConditionalGeneration
# Methods patched: get_audio_features, get_position_id_func, forward
# ======================================================================


@auto_docstring(
    custom_intro="""
    The Qwen2.5OmniThinker model which consists of a audio backbone and a language model.
    """
)
class Qwen3_5OmniMoeThinkerForConditionalGeneration(
    Qwen3OmniMoePreTrainedModelForConditionalGeneration, GenerationMixin
):
    config: Qwen3_5OmniMoeThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _no_split_modules = [
        "Qwen3OmniMoeAudioEncoder",
        "Qwen3_5MoeDecoderLayer",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.audio_tower = Qwen3OmniMoeAudioEncoder._from_config(config.audio_config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen3_5MoeTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.num_experts = config.text_config.num_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok
        self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        return self.visual(pixel_values_videos, grid_thw=video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)

    # ================================================================
    # Patch: Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_features
    # Simplified to the VeOmni training path: input_features is already the
    # flat (len, num_mel_bins) tensor (after the collator strips feature
    # padding), and feature_attention_mask is not carried in training.
    # Return the raw last_hidden_state to keep the forward body terse.
    # ================================================================
    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features,
        audio_feature_lengths=None,
        **kwargs,
    ):
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
        )
        return audio_outputs.last_hidden_state

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

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
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits: Optional[bool] = None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        **kwargs,
    ) -> tuple | Qwen3OmniMoeThinkerCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # --- Patch.2 ---
        assert "image_mask" in kwargs, "image_mask should have already been computed in process_sample"
        assert "video_mask" in kwargs, "video_mask should have already been computed in process_sample"
        assert "audio_mask" in kwargs, "audio_mask should have already been computed in process_sample"
        image_mask = kwargs.pop("image_mask")
        video_mask = kwargs.pop("video_mask")
        audio_mask = kwargs.pop("audio_mask")
        # --- Patch.2 ---

        # --- Patch.3 ---
        flash_attn_kwargs = {}
        for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
            if key in kwargs:
                flash_attn_kwargs[key] = kwargs.pop(key)
        # --- Patch.3 ---

        # --- Patch.4 ---
        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_seq_scatter_heads(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            )
        # --- Patch.4 ---

        # --- Patch.10 ---
        if input_features is not None:
            valid_mask = audio_feature_lengths != 0
            audio_feature_lengths = audio_feature_lengths[valid_mask]
            if input_features.shape[0] == 0:
                input_features = None
        # --- Patch.10 ---

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            # --- Patch.4 ---
            if self.training and get_parallel_state().sp_enabled:
                audio_features = gather_seq_scatter_heads(
                    audio_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                )
            # --- Patch.4 ---
            n_audio_tokens = audio_mask.sum().long().item()
            audio_features = audio_features[:n_audio_tokens]
            audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_expanded, audio_features)
        elif get_parallel_state().fsdp_enabled:
            # --- Patch.5 ---
            fake_audio = self.audio_tower.dummy_forward().last_hidden_state.mean() * 0.0
            fake_audio = fake_audio.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_audio
            # --- Patch.5 ---

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output

            # --- Patch.1: Shard image_embeds for sequence parallel scatter ---
            if get_parallel_state().sp_enabled:
                # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
                image_embeds = gather_seq_scatter_heads(
                    image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )
            n_image_tokens = image_mask.sum().long().item()
            embeds_image_mask = (
                image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
            )
            # Slice tensor to drop any padded image tokens
            image_embeds = image_embeds[:n_image_tokens]
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

            # sequence parallel patch for image_mask
            if get_parallel_state().sp_enabled:
                seq_len = image_mask.shape[1]

                seq_per_rank = seq_len // get_parallel_state().sp_size
                rank_start = get_parallel_state().sp_rank * seq_per_rank
                rank_end = rank_start + seq_per_rank

                image_mask = image_mask[:, rank_start:rank_end]
            # --- Patch.1 ---
        elif get_parallel_state().fsdp_enabled:
            # --- Patch.2: Dummy forward to prevent FSDP reduce-scatter hang on uneven multimodal batches ---
            # add dummy ViT forward to avoid FSDP reduce-scatter hang
            # when some ranks get None pixel_values while others get valid pixel_values
            vision_output = self.visual.dummy_forward()
            fake_embeds = vision_output.pooler_output
            fake_embeds = fake_embeds.mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.2 ---

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output

            # --- Patch.1: Shard video_embeds for sequence parallel scatter ---
            # sequence parallel patch for video embeds
            if get_parallel_state().sp_enabled:
                # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
                video_embeds = gather_seq_scatter_heads(
                    video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )
            n_video_tokens = video_mask.sum().long().item()
            embeds_video_mask = (
                video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
            )

            # Slice tensor to drop any padded video tokens
            video_embeds = video_embeds[:n_video_tokens]
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

            # sequence parallel patch for video_mask
            if get_parallel_state().sp_enabled:
                seq_len = video_mask.shape[1]

                seq_per_rank = seq_len // get_parallel_state().sp_size
                rank_start = get_parallel_state().sp_rank * seq_per_rank
                rank_end = rank_start + seq_per_rank

                video_mask = video_mask[:, rank_start:rank_end]
            # --- Patch.1 ---
        elif get_parallel_state().fsdp_enabled:
            # --- Patch.2: Dummy forward for video encoder to avoid FSDP hang ---
            # add dummy ViT forward to avoid FSDP reduce-scatter hang
            # when some ranks get None pixel_values_videos while others get valid pixel_values_videos
            vision_output = self.visual.dummy_forward()
            fake_embeds = vision_output.pooler_output
            fake_embeds = fake_embeds.mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.2 ---

        # --- Patch.6 ---
        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(
                inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
            )

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids is not None:
            # --- Patch.9 ---
            if position_ids.ndim == 3 and position_ids.shape[1] == 3:
                position_ids = position_ids.transpose(0, 1).contiguous()
            # --- Patch.9 ---

        # --- Patch.3 ---
        kwargs.update(flash_attn_kwargs)
        # --- Patch.3 ---

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # --- Patch.8 ---
        loss = None
        logits = None
        if labels is not None:
            loss, logits = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                ignore_index=IGNORE_INDEX,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
        # --- Patch.8 ---

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None and isinstance(aux_loss, torch.Tensor):
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            router_logits=getattr(outputs, "router_logits", None),
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None

        return model_inputs

    # ================================================================
    # Patch: Qwen3OmniMoeThinkerForConditionalGeneration.get_position_id_func
    # Returns a per-sample closure that converts VeOmni's multimodal tokens
    # (IMAGE_INPUT_INDEX / VIDEO_INPUT_INDEX / AUDIO_INPUT_INDEX) into 3D
    # position_ids at data-preprocessing time. SimpleNamespace + unbound
    # methods avoid pickling the full model across dataloader workers.
    # ================================================================
    def get_position_id_func(self):
        fake_config = copy.copy(self.config)
        fake_config.image_token_id = IMAGE_INPUT_INDEX
        fake_config.video_token_id = VIDEO_INPUT_INDEX
        fake_config.audio_token_id = AUDIO_INPUT_INDEX
        fake_model = SimpleNamespace(
            config=fake_config,
            spatial_merge_size=self.spatial_merge_size,
            get_llm_pos_ids_for_vision=partial(type(self).get_llm_pos_ids_for_vision, None),
            get_chunked_index=partial(type(self).get_chunked_index, None),
        )
        return partial(
            get_position_id,  # noqa: F821 — defined at module scope via add_post_import_block
            type(self).get_rope_index,
            fake_model,
        )


# ======================================================================
# [MODIFIED CLASS] Qwen3OmniMoeForConditionalGeneration
# Methods patched: __init__, enable_talker, forward, get_position_id_func, get_parallel_plan
# ======================================================================


class Qwen3_5OmniMoeForConditionalGeneration(Qwen3OmniMoePreTrainedModel, GenerationMixin):
    config_class = Qwen3_5OmniMoeConfig
    output_modalities = ("text", "audio")

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
    def __init__(self, config):
        # --- Patch.1 ---
        moe_implementation = getattr(config, "_moe_implementation", "eager")
        config.thinker_config._moe_implementation = moe_implementation
        config.thinker_config.text_config._moe_implementation = moe_implementation
        # --- Patch.1 ---
        super().__init__(config)
        self.thinker = Qwen3_5OmniMoeThinkerForConditionalGeneration._from_config(config.thinker_config)
        # --- Patch.2 ---
        self.has_talker = False
        # --- Patch.2 ---
        self.post_init()
        # --- Patch.3 ---
        self._no_split_modules = {
            "Qwen3_5MoeDecoderLayer",
            "Qwen3_5MoeVisionModel",
            "Qwen3OmniMoeAudioEncoder",
        }
        # --- Patch.3 ---

    # ================================================================
    # Patch: Qwen3OmniMoeForConditionalGeneration.enable_talker
    # The talker + code2wav classes are excluded from the generated file
    # (training never instantiates them), so the upstream body
    # `self.talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(...)`
    # would fail at import-time static analysis and at call time. Replace
    # with an explicit NotImplementedError so the reason is clear if anything
    # reaches here.
    # ================================================================
    def enable_talker(self):
        raise NotImplementedError(
            "talker / code2wav are not available in the VeOmni training modeling. "
            "Use the upstream transformers implementation for TTS generation."
        )

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "code2wav"):
            del self.code2wav
        self.has_talker = False

    def _get_talker_user_parts(
        self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
    ):
        user_talker_part = torch.empty(
            (1, segment_end_index - im_start_index, self.config.talker_config.text_config.hidden_size),
            device=thinker_hidden.device,
            dtype=self.talker.dtype,
        )

        user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]

        # Multimodal data exists
        if user_mm_mask.any():
            user_thinker_hidden_mm = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
            mm_hidden = self.talker.hidden_projection(user_thinker_hidden_mm).to(thinker_hidden.device)
            user_talker_part[user_mm_mask] = mm_hidden
        user_thinker_embed = thinker_embed[:, im_start_index:segment_end_index][~user_mm_mask]
        user_text_hidden = self.talker.text_projection(user_thinker_embed).to(thinker_hidden.device)
        user_talker_part[~user_mm_mask] = user_text_hidden
        return user_talker_part

    def _get_talker_assistant_parts(
        self, im_start_index, segment_end_index, speaker_id, thinker_embed, tts_pad_embed, tts_bos_embed, tts_eos_embed
    ):
        assistant_hidden = self.talker.text_projection(thinker_embed[:, im_start_index:segment_end_index]).to(
            tts_pad_embed.device
        )  # [1 t d]
        assistant_text_hidden = torch.cat(
            (
                assistant_hidden[:, :3],
                tts_pad_embed.expand(-1, 4, -1),
                tts_bos_embed,
                assistant_hidden[:, 3:4],  # First text
            ),
            dim=1,
        )
        codec_special_tokens = torch.tensor(
            [
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    speaker_id,
                    self.config.talker_config.codec_pad_id,
                    self.config.talker_config.codec_bos_id,
                ]
            ],
            device=tts_pad_embed.device,
            dtype=torch.long,
        )
        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (1, 3, self.config.talker_config.text_config.hidden_size),
                    device=tts_pad_embed.device,
                    dtype=self.talker.dtype,
                ),
                self.talker.get_input_embeddings()(codec_special_tokens).to(tts_pad_embed.device),
            ),
            dim=1,
        )
        trailing_text_hidden = torch.cat(
            (
                assistant_hidden[:, 4:],
                tts_eos_embed,
            ),
            dim=1,
        )

        inputs_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = torch.full(
            (1, assistant_text_hidden.shape[1]),
            fill_value=self.config.tts_pad_token_id,
            dtype=torch.long,
            device=assistant_text_hidden.device,
        )
        return inputs_embeds, input_ids, trailing_text_hidden

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        speaker: str = "Ethan",
        use_audio_in_video: bool = False,
        return_audio: bool | None = None,
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 50,
        talker_top_p: float = 1.0,
        talker_temperature: float = 0.9,
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
            "eos_token_id": thinker_eos_token_id,
        }

        talker_kwargs = {}
        token2wav_kwargs = {}
        if return_audio:
            speaker_id = self.config.talker_config.speaker_id.get(speaker.lower())
            if speaker_id is None:
                raise NotImplementedError(f"Speaker {speaker} not implemented")
            if input_ids.shape[0] != 1:
                raise NotImplementedError("Qwen3-Omni currently does not support batched inference with audio output")
            talker_supppressed_tokens = [
                i
                for i in range(
                    self.config.talker_config.text_config.vocab_size - 1024,
                    self.config.talker_config.text_config.vocab_size,
                )
                if i != self.config.talker_config.codec_eos_token_id
            ]  # Suppress additional special tokens, should not be predicted
            talker_kwargs = {
                "max_new_tokens": talker_max_new_tokens,
                "do_sample": talker_do_sample,
                "top_k": talker_top_k,
                "top_p": talker_top_p,
                "temperature": talker_temperature,
                "eos_token_id": self.config.talker_config.codec_eos_token_id,
                "repetition_penalty": talker_repetition_penalty,
                "suppress_tokens": talker_supppressed_tokens,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }
            token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs and key in ["image_grid_thw", "video_grid_thw", "video_second_per_grid"]:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result

        # 2. Prepare talker input
        thinker_embed = torch.cat([hidden_states[0] for hidden_states in thinker_result.hidden_states], dim=1).to(
            input_ids.device
        )  # [1 t d]
        thinker_hidden = torch.cat(
            [
                hidden_states[self.config.talker_config.accept_hidden_layer]
                for hidden_states in thinker_result.hidden_states
            ],
            dim=1,
        ).to(input_ids.device)  # [1 t d]
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([thinker_result.sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result.sequences == self.config.thinker_config.audio_token_id) |
            (thinker_result.sequences == self.config.thinker_config.image_token_id) |
            (thinker_result.sequences == self.config.thinker_config.video_token_id)
        ).to(input_ids.device)  # [1 t] # fmt: skip

        talker_special_tokens = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            device=self.thinker.device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            self.talker.text_projection(self.thinker.get_input_embeddings()(talker_special_tokens))
            .to(input_ids.device)
            .chunk(3, dim=1)
        )  # 3 * [1 1 d]

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i]
            segment_end_index = im_start_indexes[i + 1]
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if role_token == self.config.system_token_id:
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif role_token == self.config.user_token_id:
                talker_user_part = self._get_talker_user_parts(
                    im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(thinker_result.sequences[:, im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2:
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
            # History assistant output (ignore for now)
            elif role_token == self.config.assistant_token_id and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat([embed.to(input_ids.device) for embed in talker_input_embeds], dim=1)
        talker_input_id = torch.cat([embed.to(input_ids.device) for embed in talker_input_ids], dim=1)
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetation penalty out of bound
            **talker_kwargs,
        )
        talker_codes = (
            torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
            .transpose(1, 2)
            .to(self.code2wav.device)
        )
        talker_wavs = self.code2wav.chunked_decode(talker_codes, chunk_size=300, left_context_size=25)

        return thinker_result.sequences, talker_wavs.float()

    # ================================================================
    # Patch: Qwen3OmniMoeForConditionalGeneration.forward
    # Simplified training path: only forward through thinker; talker +
    # code2wav are skipped (only used in the TTS generate path).
    # ================================================================
    def forward(
        self,
        **kwargs,
    ) -> tuple | Qwen3OmniMoeThinkerCausalLMOutputWithPast:
        return self.thinker(**kwargs)

    # ================================================================
    # Patch: Qwen3OmniMoeForConditionalGeneration.get_position_id_func (NEW)
    # Delegate to the thinker's closure; data pipeline calls the top-level
    # model's get_position_id_func.
    # ================================================================
    def get_position_id_func(self):
        return self.thinker.get_position_id_func()

    # ================================================================
    # Patch: Qwen3OmniMoeForConditionalGeneration.get_parallel_plan (NEW)
    # Register the VeOmni EP plan for thinker.model.layers.*.mlp.experts.*
    # on the generated v5 modeling.
    # ================================================================
    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan as _get_parallel_plan

        return _get_parallel_plan()


__all__ = [
    "Qwen3_5OmniMoeForConditionalGeneration",
    "Qwen3_5MoeTextModel",
    "Qwen3_5OmniMoeThinkerForConditionalGeneration",
    "Qwen3OmniMoePreTrainedModel",
    "Qwen3OmniMoePreTrainedModelForConditionalGeneration",
    "Qwen3_5MoePreTrainedModel",
]
