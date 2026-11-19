# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash sparse and sliding-window MLA on SM12x through FlashInfer.

SM12x GPUs (RTX 50, RTX PRO 6000 Blackwell, DGX Spark) have no FlashMLA.
FlashInfer's SM120 sparse MLA decode kernel reads the same packed
``fp8_ds_mla`` rows (latent 512 + rotary 64) for top-k 512 and 2048 and accepts
a per-head FP32 sink, which the shared ``FLASHINFER_MLA_SPARSE_SM120`` impl does
not forward. The causal window of sliding layers comes from the window
indexer's indices, so the kernel itself runs without a window.
"""

import torch

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)


class FlashSM120Impl(FlashInferMLASparseSM120Impl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        *args,
        **mla_args,
    ) -> None:
        sinks = mla_args.pop("sinks", None)
        super().__init__(
            num_heads, head_size, scale, num_kv_heads, alibi_slopes, None,
            *args, **mla_args,
        )  # fmt: skip
        if sinks is not None and (
            sinks.dtype != torch.float32 or tuple(sinks.shape) != (num_heads,)
        ):
            raise ValueError(
                f"Flash SM120 sinks must be float32 ({num_heads},), got "
                f"{sinks.dtype} {tuple(sinks.shape)}"
            )
        self.sinks = sinks

    def _run_mqa_kernel(self, q, kv_cache, topk_indices_physical):
        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        output = q.new_empty((q.shape[0], self.num_heads, self.kv_lora_rank))
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        capacity = topk_indices_physical.shape[1]
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=capacity,
            kv_scale_format=self.kv_scale_format,
            sinks=None if self.sinks is None else [self.sinks],
        )
        return out.squeeze(1)


class FlashSM120SparseBackend(FlashInferMLASparseSM120Backend):
    @staticmethod
    def get_impl_cls():
        return FlashSM120Impl

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_dcp(cls) -> bool:
        return False


class FlashSM120SWABackend(FlashSM120SparseBackend):
    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True
