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

from typing import Optional

import torch

from .eager import eager_cross_entropy


try:
    import torch_npu
    import torch_npu.utils.custom_ops  # noqa: F401 - registers torch_npu custom op aliases
except Exception:
    torch_npu = None


class NpuFusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        normalizer: torch.Tensor,
        label_smoothing: float,
    ) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        weights = weights.contiguous()
        labels = labels.contiguous()

        (
            logits_max,
            sum_exp_logits,
            predicted_logits,
            target_mask,
            masked_target,
            _,
        ) = torch_npu.fused_linear_online_max_sum(
            hidden_states,
            weights,
            labels,
            0,
            weights.size(0) - 1,
            False,
        )
        loss, _ = torch_npu.fused_cross_entropy_loss_with_max_sum(
            logits_max,
            sum_exp_logits,
            predicted_logits,
            label_smoothing=label_smoothing,
            input=hidden_states,
            weight=weights,
        )

        ctx.save_for_backward(
            hidden_states,
            weights,
            target_mask,
            masked_target,
            logits_max,
            sum_exp_logits,
            normalizer,
        )
        ctx.label_smoothing = label_smoothing
        return loss.sum() / normalizer.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, weights, target_mask, masked_target, logits_max, sum_exp_logits, normalizer = ctx.saved_tensors
        grad = grad_output.to(torch.float32).expand(hidden_states.size(0)) / normalizer.clamp_min(1)
        grad_hidden_states, grad_weights = torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad(
            grad,
            hidden_states,
            weights,
            target_mask,
            masked_target,
            ctx.label_smoothing,
            logits_max=logits_max,
            sum_exp_logits=sum_exp_logits,
        )
        return grad_hidden_states, grad_weights, None, None, None


def npu_fused_linear_cross_entropy(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)
    label_smoothing = float(kwargs.pop("label_smoothing", 0.0))

    if (
        torch_npu is None
        or hidden_states is None
        or weights is None
        or hidden_states.device.type != "npu"
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or weights.dtype != hidden_states.dtype
    ):
        return eager_cross_entropy(
            logits,
            labels,
            vocab_size,
            num_items_in_batch,
            ignore_index,
            shift_labels,
            hidden_states=hidden_states,
            weights=weights,
            **kwargs,
        )

    labels = labels.to(hidden_states.device)
    valid_mask = labels.ne(ignore_index)
    hidden_states = hidden_states[valid_mask]
    labels = labels[valid_mask]

    if hidden_states.size(0) == 0:
        loss = hidden_states.sum() * 0.0 + weights.sum() * 0.0
        return loss, None

    if num_items_in_batch is None:
        normalizer = torch.tensor(hidden_states.size(0), device=hidden_states.device, dtype=torch.float32)
    else:
        normalizer = torch.as_tensor(num_items_in_batch, device=hidden_states.device, dtype=torch.float32)

    loss = NpuFusedLinearCrossEntropy.apply(
        hidden_states,
        weights,
        labels.to(torch.int64),
        normalizer,
        label_smoothing,
    )
    return loss, None
