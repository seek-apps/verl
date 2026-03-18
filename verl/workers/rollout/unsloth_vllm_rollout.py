# Copyright 2024 Bytedance Ltd. and/or its affiliates
# [seek-apps fork] Unsloth vLLM rollout — uses Unsloth's fast_generate (vLLM backend)
# for 3-5x faster rollout generation compared to HF model.generate().
#
# Weight sync: LoRA weights are synced from PyTorch training model to vLLM
# via save_lora/load_lora (in-memory with load_tensors=True, ~12MB for rank 8).
#
# Requires: fast_inference=True in Unsloth from_pretrained + UNSLOTH_VLLM_STANDBY=1

import os
import logging
from typing import Optional

import torch
from tensordict import TensorDict
from vllm import SamplingParams

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

logger = logging.getLogger(__name__)


class UnslothVLLMRollout:
    """Rollout using Unsloth's vLLM-backed fast_generate for fast inference.

    Replaces HFRollout when the model is loaded with fast_inference=True.
    The vLLM engine shares GPU memory with PyTorch via Unsloth's Standby mode.
    """

    def __init__(self, module, tokenizer, config, lora_sync_path: str = "/tmp/verl_lora_sync"):
        self.module = module          # PeftModel with .fast_generate, .save_lora, .load_lora
        self.tokenizer = tokenizer
        self.config = config
        self.lora_sync_path = lora_sync_path
        self._lora_initialized = False

        # Verify fast_generate is available
        if not hasattr(module, "fast_generate"):
            raise RuntimeError(
                "UnslothVLLMRollout requires model loaded with fast_inference=True. "
                "model.fast_generate not found."
            )

        # One-time: write adapter_config.json so load_lora(load_tensors=True) works later
        try:
            self.module.save_lora(self.lora_sync_path)
            self._lora_initialized = True
            logger.info(f"[UnslothVLLMRollout] Initial LoRA saved to {self.lora_sync_path}")
        except Exception as e:
            logger.warning(f"[UnslothVLLMRollout] Initial save_lora failed: {e}")

    def _sync_lora_weights(self):
        """Sync current LoRA weights from PyTorch model to vLLM engine."""
        try:
            # In-memory sync: reads tensors from model.state_dict(), avoids disk I/O
            lora_request = self.module.load_lora(self.lora_sync_path)
            return lora_request
        except Exception as e:
            logger.warning(f"[UnslothVLLMRollout] load_lora failed: {e}, trying save+load")
            try:
                self.module.save_lora(self.lora_sync_path)
                lora_request = self.module.load_lora(self.lora_sync_path)
                return lora_request
            except Exception as e2:
                logger.error(f"[UnslothVLLMRollout] LoRA sync failed completely: {e2}")
                return None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate completions using Unsloth's vLLM-backed fast_generate.

        Input: DataProto with left-padded input_ids, attention_mask, position_ids
        Output: DataProto with prompts, responses, input_ids, attention_mask, position_ids
               (same contract as HFRollout)
        """
        # Extract config
        temperature = prompts.meta_info.get("temperature", 1.0)
        response_length = self.config.response_length
        do_sample = getattr(self.config, "do_sample", True)
        top_p = getattr(self.config, "top_p", 0.95)
        top_k = getattr(self.config, "top_k", 0)
        pad_token_id = prompts.meta_info.get("pad_token_id", self.tokenizer.pad_token_id)
        eos_token_id = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]

        # Build vLLM SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if top_k > 0 else -1,  # vLLM uses -1 for disabled
            max_tokens=response_length,
            n=1,  # verl handles repeats via rollout.n in ray_trainer
        )

        # Convert left-padded input_ids to list of token ID lists (strip padding)
        token_prompts = []
        for i in range(batch_size):
            mask = attention_mask[i]  # (prompt_length,)
            valid_ids = input_ids[i][mask.bool()].tolist()
            token_prompts.append({"prompt_token_ids": valid_ids})

        # Sync LoRA weights from training model to vLLM
        lora_request = self._sync_lora_weights()

        # Generate via vLLM
        vllm_outputs = self.module.fast_generate(
            token_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        # Convert vLLM RequestOutput → tensors matching HFRollout contract
        responses = torch.full(
            (batch_size, response_length),
            pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        for i, req_output in enumerate(vllm_outputs):
            output_ids = list(req_output.outputs[0].token_ids)
            # Truncate to response_length
            output_ids = output_ids[:response_length]
            responses[i, :len(output_ids)] = torch.tensor(
                output_ids, dtype=input_ids.dtype, device=input_ids.device
            )

        # Build full sequence: prompt + response
        seq = torch.cat([input_ids, responses], dim=1)

        # Position IDs for response tokens
        response_len = responses.shape[1]
        delta_position_id = torch.arange(1, response_len + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        full_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # Attention mask for response tokens
        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": responses,
                "input_ids": seq,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            },
            batch_size=batch_size,
        )

        # Clear KV cache
        torch.cuda.empty_cache()

        return DataProto(batch=batch)
