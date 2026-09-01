# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for --data-parallel-replicate-moe: MoE DP ranks run as
fully independent replicas (complete expert weights, no cross-DP collectives,
no lockstep waves or dummy forward passes) behind one API endpoint with
stats-based internal load balancing."""

import asyncio
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from tests.v1.utils import check_request_balancing
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

MODEL_NAME = "ibm-research/PowerMoE-3b"
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))


def _check_replica_independence(worker: Any) -> bool:
    """Runs inside every engine's worker; raises on any violation so failures
    surface regardless of which engine's return value is gathered."""
    import torch.distributed as dist

    parallel_config = worker.vllm_config.parallel_config
    assert parallel_config.data_parallel_size == 1, (
        "replicated-MoE engine must be localized to DP=1, got "
        f"{parallel_config.data_parallel_size}"
    )
    assert parallel_config.data_parallel_replicate_moe
    # The torch distributed world must be replica-local (no cross-DP group).
    assert dist.get_world_size() == parallel_config.world_size, (
        f"torch world size {dist.get_world_size()} spans more than one "
        f"replica (world_size={parallel_config.world_size})"
    )
    return True


def _check_complete_experts(worker: Any) -> bool:
    """Every MoE layer must hold ALL experts locally (no EP, no DP folding)."""
    checked = 0
    for module in worker.get_model().modules():
        moe_config = getattr(module, "moe_config", None)
        if moe_config is None or not hasattr(moe_config, "num_local_experts"):
            continue
        assert moe_config.num_local_experts == moe_config.num_experts, (
            f"expected complete experts per replica, got "
            f"{moe_config.num_local_experts}/{moe_config.num_experts}"
        )
        assert not moe_config.moe_parallel_config.use_ep
        assert moe_config.moe_parallel_config.dp_size == 1
        checked += 1
    assert checked > 0, "no MoE layers found in model"
    return True


def _forbid_dummy_batches(worker: Any) -> bool:
    def poisoned_dummy_batch() -> None:
        raise RuntimeError(
            "execute_dummy_batch must never run in replicated-MoE DP mode"
        )

    worker.execute_dummy_batch = poisoned_dummy_batch
    return True


@pytest.mark.asyncio
async def test_replicated_moe_dp_independent_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    """One AsyncLLM launch covering: complete per-replica experts, replica-
    local distributed world, no dummy forwards, unequal work progressing
    independently, request balancing, and per-replica prefix cache reuse."""
    # The probes below ship functions through collective_rpc.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    stats_loggers = {}

    @dataclass
    class SimpleStatsLogger(StatLoggerBase):
        finished_req_count: int = 0

        def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
            stats_loggers[engine_index] = self

        def record(
            self,
            scheduler_stats: SchedulerStats | None,
            iteration_stats: IterationStats | None,
            mm_cache_stats: MultiModalCacheStats | None = None,
            engine_idx: int = 0,
        ):
            if iteration_stats:
                self.finished_req_count += len(iteration_stats.finished_requests)

        def log_engine_initialized(self):
            pass

    with ExitStack() as after:
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            enforce_eager=True,
            tensor_parallel_size=TP_SIZE,
            data_parallel_size=DP_SIZE,
            data_parallel_replicate_moe=True,
        )
        engine = AsyncLLM.from_engine_args(
            engine_args, stat_loggers=[SimpleStatsLogger]
        )
        after.callback(engine.shutdown)

        assert isinstance(engine.engine_core, DPLBAsyncMPClient)
        assert all(
            await engine.engine_core.collective_rpc_async(_check_replica_independence)
        )
        assert all(
            await engine.engine_core.collective_rpc_async(_check_complete_experts)
        )
        # Any dummy forward from here on kills the engine and fails the test.
        assert all(await engine.engine_core.collective_rpc_async(_forbid_dummy_batches))

        async def run(request_id: str, dp_rank: int | None = None) -> int:
            final = None
            async for out in engine.generate(
                request_id=request_id,
                prompt="This is a test of replicated MoE data parallel",
                sampling_params=SamplingParams(
                    max_tokens=10, ignore_eos=True, temperature=0
                ),
                data_parallel_rank=dp_rank,
            ):
                final = out
            assert final is not None and final.finished
            return len(final.outputs[0].token_ids)

        # Unequal work must progress independently: every request pinned to
        # rank 0 while rank 1 stays idle. Under lockstep DP (waves + dummy
        # forwards) this topology is impossible without cross-DP coordination.
        pinned = await asyncio.wait_for(
            asyncio.gather(*(run(f"pinned-{i}", dp_rank=0) for i in range(8))),
            timeout=180,
        )
        assert all(count == 10 for count in pinned)

        # Unpinned traffic is balanced across the replicas by queue stats.
        NUM_REQUESTS = 50
        tasks = []
        for i in range(NUM_REQUESTS):
            tasks.append(asyncio.create_task(run(f"request-{i}")))
            await asyncio.sleep(0.01)
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=300)
        assert all(count == 10 for count in results)

        assert len(stats_loggers) == DP_SIZE
        for engine_index, logger in stats_loggers.items():
            assert logger.finished_req_count > NUM_REQUESTS // (DP_SIZE + 1), (
                f"requests are imbalanced: engine {engine_index} finished "
                f"{logger.finished_req_count}"
            )

        # Each replica keeps a private prefix cache: a repeated prompt pinned
        # to the same rank must hit it on the second pass.
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 40
        cached_counts = []
        for i in range(2):
            final = None
            async for out in engine.generate(
                request_id=f"cache-{i}",
                prompt=long_prompt,
                sampling_params=SamplingParams(max_tokens=5, temperature=0),
                data_parallel_rank=1,
            ):
                final = out
            assert final is not None
            cached_counts.append(final.num_cached_tokens or 0)
        assert cached_counts[1] > 0, (
            f"expected a prefix cache hit on the pinned replica, got {cached_counts=}"
        )


def test_replicated_moe_dp_server():
    """Single-endpoint server smoke test: completions succeed and are
    balanced across the independent replicas."""
    server_args = [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--data-parallel-size",
        str(DP_SIZE),
        "--data-parallel-size-local",
        str(DP_SIZE),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--data-parallel-replicate-moe",
    ]
    env_dict = {
        current_platform.device_control_env_var: ",".join(
            str(current_platform.device_id_to_physical_device_id(i))
            for i in range(DP_SIZE * TP_SIZE)
        ),
    }

    with RemoteOpenAIServer(MODEL_NAME, server_args, env_dict=env_dict) as server:
        for i in range(40):
            response = requests.post(
                server.url_for("v1/completions"),
                json={
                    "model": MODEL_NAME,
                    "prompt": f"Replicated MoE data parallel test {i}",
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()

        check_request_balancing(server, DP_SIZE)
