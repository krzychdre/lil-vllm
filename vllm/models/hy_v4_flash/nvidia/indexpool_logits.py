# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexPool paged FP8 MQA logits without DeepGEMM (SM12x).

DeepGEMM's `fp8_fp4_paged_mqa_logits` has no SM120 build in this tree. This
Triton kernel computes the same next_n == 1 logits over IndexPool pages:

    logits[t, j] = kscale[j] * sum_h weights[t, h] * relu(q[t, h] . k[j])

Each page holds `page_size` rows of 128 FP8 values followed by `page_size`
FP32 scales, the layout `write_pool` fills. Columns past a row's length are
left untouched, as with DeepGEMM.
"""

import torch

from vllm.triton_utils import tl, triton

INDEX_DIM = 128


@triton.jit
def _paged_logits_kernel(
    q_ptr,  # fp8 [T, H, D]
    cache_ptr,  # uint8 [pages, page_size * (D + 4)]
    weights_ptr,  # fp32 [T, H]
    lengths_ptr,  # int32 [T]
    table_ptr,  # int32 [T, max_pages]
    out_ptr,  # fp32 [T, width]
    q_stride_t,
    q_stride_h,
    w_stride_t,
    table_stride_t,
    out_stride_t,
    page_bytes,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
):
    t = tl.program_id(0)
    page = tl.program_id(1)
    length = tl.load(lengths_ptr + t)
    start = page * PAGE
    if start >= length:
        return
    physical = tl.load(table_ptr + t * table_stride_t + page).to(tl.int64)
    rows = tl.arange(0, PAGE)
    dims = tl.arange(0, D)
    heads = tl.arange(0, H)
    base = cache_ptr + physical * page_bytes
    k = tl.load(base + rows[:, None] * D + dims[None, :]).to(tl.float8e4nv)
    kscale = tl.load((base + PAGE * D).to(tl.pointer_type(tl.float32)) + rows)
    q = tl.load(
        q_ptr + t * q_stride_t + heads[:, None] * q_stride_h + dims[None, :]
    ).to(tl.float8e4nv, bitcast=True)
    scores = tl.dot(k.to(tl.float16), tl.trans(q.to(tl.float16)), out_dtype=tl.float32)
    w = tl.load(weights_ptr + t * w_stride_t + heads)
    logits = tl.sum(tl.maximum(scores, 0.0) * w[None, :], axis=1) * kscale
    cols = start + rows
    tl.store(out_ptr + t * out_stride_t + cols, logits, mask=cols < length)


def paged_mqa_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    table: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """FP32 logits [T, width] for FP8 q [T, 1, H, 128] over IndexPool pages."""
    tokens, next_n, heads, dim = q.shape
    if next_n != 1 or dim != INDEX_DIM:
        raise ValueError("IndexPool logits take next_n == 1 and 128-dim heads")
    page_size = cache.shape[1]
    out = torch.empty((tokens, width), dtype=torch.float32, device=q.device)
    if tokens == 0:
        return out
    q = q.view(torch.uint8).reshape(tokens, heads, dim)
    weights = weights.float().contiguous()
    lengths = lengths.reshape(tokens).int().contiguous()
    table = table.int()
    if table.stride(-1) != 1:
        table = table.contiguous()
    max_pages = min(table.shape[1], triton.cdiv(width, page_size))
    _paged_logits_kernel[(tokens, max_pages)](
        q,
        cache,
        weights,
        lengths,
        table,
        out,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        table.stride(0),
        out.stride(0),
        cache.stride(0),
        H=heads,
        D=dim,
        PAGE=page_size,
        num_warps=4,
    )
    return out
