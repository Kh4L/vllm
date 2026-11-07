# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static hot/cold expert placement for MuseSpark NVFP4 TRT-LLM MoE layers.

A layer's hot experts stay in GPU memory and its cold experts move to CPU
memory, read through UVA. The router is permuted so that the hot experts take
IDs ``[0, H)``: every forward runs the TRT-LLM kernel once per bank, each one
local to its ID range, and finalizes the merged output once.
"""

import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

if TYPE_CHECKING:
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xMoE
    from vllm.models.muse_spark.nvidia.muse_moe_runner import MuseMoERunner

logger = init_logger(__name__)


@dataclass
class ExpertBank:
    """The experts ``[offset, offset + num_experts)`` of a layer, in kernel format."""

    offset: int
    num_experts: int
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    g1_alphas: torch.Tensor
    g1_scale_c: torch.Tensor
    g2_alphas: torch.Tensor


@functools.cache
def _load_hot_experts(path: str) -> dict[int, tuple[int, ...]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(layer): tuple(int(e) for e in ids) for layer, ids in raw.items()}


def hot_experts_for_layer(
    vllm_config: VllmConfig, layer_idx: int, num_experts: int
) -> tuple[int, ...] | None:
    path = vllm_config.offload_config.uva.cpu_offload_hot_experts
    if path is None:
        return None
    hot = _load_hot_experts(path).get(layer_idx)
    if hot is not None and (
        len(set(hot)) != len(hot) or not all(0 <= e < num_experts for e in hot)
    ):
        raise ValueError(
            f"{path}: layer {layer_idx} needs distinct expert IDs in "
            f"[0, {num_experts})."
        )
    return hot


def validate_hot_experts(moe: "MuseSpark1xMoE") -> None:
    from vllm.models.muse_spark.nvidia.muse_moe_runner import (
        MuseSparkModelOptNvFp4FusedMoE,
        MuseSparkTrtLlmNvFp4Experts,
    )

    routed_experts = getattr(moe.experts, "routed_experts", None)
    quant_method = getattr(routed_experts, "quant_method", None)
    if not (
        isinstance(quant_method, MuseSparkModelOptNvFp4FusedMoE)
        and quant_method.experts_cls is MuseSparkTrtLlmNvFp4Experts
    ):
        raise NotImplementedError(
            f"--cpu-offload-hot-experts lists layer {moe.layer_idx}, which does "
            "not run the MuseSpark NVFP4 FlashInfer TRT-LLM MoE (checkpoint BF16 "
            "or online-quantized layers, MegaMoE, Marlin and all2all backends are "
            "not supported)."
        )
    if moe.experts.moe_config.moe_parallel_config.ep_size != 1:
        raise NotImplementedError(
            "--cpu-offload-hot-experts does not support expert parallel."
        )


def merge_unfinalized(outputs: list[UnfinalizedMoEOutput]) -> UnfinalizedMoEOutput:
    """One output from the banks' outputs; each route is local to one bank."""
    merged = outputs[0]
    for out in outputs[1:]:
        idx, other = (
            merged.expanded_idx_to_permuted_idx,
            out.expanded_idx_to_permuted_idx,
        )
        rows = merged.gemm2_permuted.shape[0]
        local = idx >= 0
        merged = UnfinalizedMoEOutput(
            gemm2_permuted=torch.cat([merged.gemm2_permuted, out.gemm2_permuted]),
            expert_weights=torch.where(
                local, merged.expert_weights, out.expert_weights
            ),
            expanded_idx_to_permuted_idx=torch.where(
                local, idx, torch.where(other >= 0, other + rows, -1)
            ),
        )
    return merged


def _select_rows(
    tensor: torch.Tensor, ids: torch.Tensor, cpu: bool, pin_memory: bool
) -> torch.Tensor:
    """``tensor[ids]`` in GPU memory, or in CPU memory read through UVA."""
    src = tensor.view(torch.uint8)
    shape = (ids.numel(), *src.shape[1:])
    if cpu and ids.numel() > 0:
        host = torch.empty(
            shape, dtype=torch.uint8, device="cpu", pin_memory=pin_memory
        )
        # A device-side copy into the UVA view: no DtoH into pageable memory.
        out = get_accelerator_view_from_cpu_tensor(host)
    else:
        out = torch.empty(shape, dtype=torch.uint8, device=tensor.device)
    torch.index_select(src, 0, ids, out=out)
    return out.view(tensor.dtype)


def split_hot_cold_experts(
    runner: "MuseMoERunner",
    hot_ids: Sequence[int],
    routing_tensors: Sequence[torch.Tensor],
) -> int:
    """Move all but ``hot_ids`` of the runner's experts to CPU memory.

    The experts are renumbered so that ``hot_ids`` take IDs ``[0, H)``;
    ``routing_tensors`` (router rows, expert bias) are permuted to match, in place:
    the routers and the experts share them.

    Returns:
        The bytes moved to CPU memory.

    """
    from vllm.models.muse_spark.nvidia.muse_moe_runner import (
        MuseSparkTrtLlmNvFp4Experts,
    )

    layer = runner.routed_experts
    moe_kernel = layer.quant_method.moe_kernel
    assert moe_kernel is not None
    experts = moe_kernel.fused_experts
    assert isinstance(experts, MuseSparkTrtLlmNvFp4Experts)
    if experts.expert_banks is not None:
        raise NotImplementedError(
            "MuseSpark hot experts do not support reloading the weights."
        )

    num_experts = layer.global_num_experts
    num_hot = len(hot_ids)
    cold_ids = sorted(set(range(num_experts)) - set(hot_ids))
    perm = torch.tensor([*hot_ids, *cold_ids], device=layer.w13_weight.device)
    hot, cold = perm[:num_hot], perm[num_hot:]

    for routing_tensor in routing_tensors:
        routing_tensor.data.copy_(routing_tensor.data[perm])
    quant_config = experts.quant_config
    g1_alphas, g1_scale_c = quant_config.g1_alphas, experts.g1_scale_c
    g2_alphas = quant_config.g2_alphas
    assert g1_alphas is not None and g2_alphas is not None
    for scale in (g1_alphas, g1_scale_c, g2_alphas):
        scale.copy_(scale[perm])

    pin_memory = (
        is_pin_memory_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
    )
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    hot_weights = [
        _select_rows(getattr(layer, name), hot, False, pin_memory) for name in names
    ]
    cold_weights = [
        _select_rows(getattr(layer, name), cold, True, pin_memory) for name in names
    ]
    for name, weight in zip(names, hot_weights):
        # Rebind instead of replace_parameter, which would copy into the old
        # storage and keep it allocated.
        param = getattr(layer, name)
        param.data = weight
        param._vllm_is_uva_offloaded = False
    # Drop the last references to the full-size scales.
    quant_config._w1.scale = layer.w13_weight_scale
    quant_config._w2.scale = layer.w2_weight_scale

    banks = []
    for offset, count, weights in (
        (0, num_hot, hot_weights),
        (num_hot, num_experts - num_hot, cold_weights),
    ):
        if count == 0:
            continue
        rows = slice(offset, offset + count)
        w13, w13_scale, w2, w2_scale = weights
        banks.append(
            ExpertBank(
                offset,
                count,
                w13,
                w13_scale,
                w2,
                w2_scale,
                g1_alphas[rows],
                g1_scale_c[rows],
                g2_alphas[rows],
            )
        )
    experts.expert_banks = tuple(banks)
    return sum(w.numel() * w.element_size() for w in cold_weights)
