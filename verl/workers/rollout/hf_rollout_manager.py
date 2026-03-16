# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# ── [seek-apps fork] HFAgentLoopManager — HF rollout without async HTTP server ────────────────
# Replaces AgentLoopManager for rollout.name=hf. Dispatches generate_sequences directly
# to the FSDP worker group via registered Ray remote calls — no vLLM/sglang server needed.
#
# Usage (in smoke config):
#   actor_rollout_ref.rollout.agent.agent_loop_manager_class:
#     verl.workers.rollout.hf_rollout_manager.HFAgentLoopManager
#
# Related: verl issue #1940 (HF rollout broken after async rollout refactor).

from omegaconf import DictConfig

from verl import DataProto
from verl.utils.ray_utils import auto_await


class HFAgentLoopManager:
    """Minimal AgentLoopManager for HF rollout — no HTTP inference server.

    Dispatches generate_sequences directly to the FSDP worker group.
    Weight synchronization is not needed: HFRollout uses actor_module_fsdp directly,
    so any actor update is immediately reflected in the next rollout step.

    rollout_replicas is empty: CheckpointEngineManager with backend=naive only calls
    trainer.update_weights() which is safe with HF rollout (see AsyncActorRolloutRefWorker).
    """

    def __init__(self, config: DictConfig, worker_group):
        self.config = config
        self.worker_group = worker_group
        self.rollout_replicas = []  # no server replicas — checkpoint_manager no-ops on empty list

    @classmethod
    @auto_await
    async def create(
        cls,
        config: DictConfig,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
    ):
        """Create HFAgentLoopManager — no server initialization needed."""
        return cls(config, worker_group)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Dispatch generate_sequences to FSDP workers directly."""
        return self.worker_group.generate_sequences(prompts)

    def start_profile(self, **kwargs):
        pass

    def stop_profile(self):
        pass
