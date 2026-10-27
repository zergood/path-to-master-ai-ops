"""Тесты к attention.py. Запуск: pytest week-01/attention -v

Эталон — torch.nn.functional.scaled_dot_product_attention и nn.MultiheadAttention.
Своя реализация вызывается внутри `no_builtin_attention()`: если она дёрнет
torch.softmax / F.softmax / Tensor.softmax / F.scaled_dot_product_attention, тест упадёт.
"""

from __future__ import annotations

import contextlib
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import KVCache, MultiHeadAttention, scaled_dot_product_attention, softmax

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ATOL = 1e-5
RTOL = 1e-4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@contextlib.contextmanager
def no_builtin_attention():
    def banned(name):
        def fn(*args, **kwargs):
            raise AssertionError(f"{name} запрещён — реализуй сам")

        return fn

    targets = [
        (torch, "softmax"),
        (F, "softmax"),
        (torch.Tensor, "softmax"),
        (F, "scaled_dot_product_attention"),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name in targets]
    try:
        for obj, name in targets:
            setattr(obj, name, banned(f"{getattr(obj, '__name__', obj)}.{name}"))
        yield
    finally:
        for obj, name, orig in saved:
            setattr(obj, name, orig)


def causal_mask(L: int, S: int, device=None) -> torch.Tensor:
    """bool (L, S), выравнивание по правому нижнему углу: i видит j <= i + S - L."""
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    return j <= i + (S - L)


def ref_mha(mha: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    """Эталонный causal MHA/GQA на весах mha."""
    B, T, _ = x.shape
    H, Hkv, Dh = mha.n_heads, mha.n_kv_heads, mha.head_dim
    q = mha.wq(x).view(B, T, H, Dh).transpose(1, 2)
    k = mha.wk(x).view(B, T, Hkv, Dh).transpose(1, 2)
    v = mha.wv(x).view(B, T, Hkv, Dh).transpose(1, 2)
    k = k.repeat_interleave(H // Hkv, dim=1)
    v = v.repeat_interleave(H // Hkv, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return mha.wo(out.transpose(1, 2).reshape(B, T, H * Dh))


# ---------------------------------------------------------------------------
# 1. softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_1_softmax_matches_torch(device, dim):
    x = torch.randn(4, 5, 6, device=device)
    with no_builtin_attention():
        out = softmax(x, dim=dim)
    torch.testing.assert_close(out, torch.softmax(x, dim=dim), atol=ATOL, rtol=RTOL)


def test_1_softmax_large_values_stable():
    x = torch.tensor([[1000.0, 1000.0, 999.0], [-1000.0, -1000.0, -1001.0]])
    with no_builtin_attention():
        out = softmax(x)
    assert torch.isfinite(out).all(), "переполнение exp: вычти максимум"
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_1_softmax_with_neg_inf():
    x = torch.tensor([[0.5, float("-inf"), float("-inf")], [1.0, 2.0, float("-inf")]])
    with no_builtin_attention():
        out = softmax(x)
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_1_softmax_rows_sum_to_one():
    x = torch.randn(8, 33) * 20
    with no_builtin_attention():
        out = softmax(x)
    torch.testing.assert_close(out.sum(-1), torch.ones(8))
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# 2. scaled dot-product attention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(7, 16), (2, 7, 16), (2, 4, 7, 16)])
def test_2_sdpa_no_mask(device, shape):
    q = torch.randn(*shape, device=device)
    k = torch.randn(*shape, device=device)
    v = torch.randn(*shape, device=device)
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=ATOL, rtol=RTOL)


def test_2_sdpa_scale_is_sqrt_head_dim():
    # Одинаковые ключи кроме одного: результат чувствителен к масштабу логитов.
    D = 64
    q = torch.ones(1, 1, D)
    k = torch.zeros(1, 2, D)
    k[0, 0] = 0.1
    v = torch.tensor([[[1.0], [0.0]]])
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v)
    expected = torch.sigmoid(torch.tensor(0.1 * D / math.sqrt(D)))
    torch.testing.assert_close(out.flatten(), expected.reshape(1))


def test_2_sdpa_different_lengths_and_value_dim():
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 11, 8)
    v = torch.randn(2, 3, 11, 4)
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 5, 4)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("device", DEVICES)
def test_2_sdpa_bool_mask(device):
    q = torch.randn(2, 4, 6, 16, device=device)
    k = torch.randn(2, 4, 9, 16, device=device)
    v = torch.randn(2, 4, 9, 16, device=device)
    mask = torch.rand(2, 1, 6, 9, device=device) > 0.4
    mask[..., 0] = True  # в каждой строке хоть один ключ разрешён
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_2_sdpa_masked_values_are_ignored():
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 5, 8)
    v = torch.randn(1, 5, 8)
    mask = torch.tensor([True, True, False, True, False])
    v_poisoned = v.clone()
    v_poisoned[:, ~mask] = 1e6
    with no_builtin_attention():
        a = scaled_dot_product_attention(q, k, v, mask=mask)
        b = scaled_dot_product_attention(q, k, v_poisoned, mask=mask)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_2_sdpa_causal_square(device):
    q, k, v = (torch.randn(2, 4, 10, 16, device=device) for _ in range(3))
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("L,S", [(1, 10), (3, 10), (10, 10)])
def test_2_sdpa_causal_bottom_right(L, S):
    # torch is_causal выравнивает по левому верхнему углу, поэтому эталон — явная маска.
    q = torch.randn(2, L, 16)
    k = torch.randn(2, S, 16)
    v = torch.randn(2, S, 16)
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(L, S))
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_2_sdpa_causal_and_mask_combined():
    q, k, v = (torch.randn(1, 2, 6, 8) for _ in range(3))
    pad = torch.tensor([True, True, True, True, False, False])  # паддинг справа
    mask = pad[None, :].expand(6, 6).clone()
    mask[:, 0] = True
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v, mask=mask, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask & causal_mask(6, 6))
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_2_sdpa_gradients_flow():
    q, k, v = (torch.randn(2, 5, 8, requires_grad=True) for _ in range(3))
    with no_builtin_attention():
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
    out.sum().backward()
    grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
    for g, t in zip(grads, (q, k, v)):
        torch.testing.assert_close(g, t.grad, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# 3. Multi-head attention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d_model,n_heads", [(32, 1), (32, 4), (64, 8)])
def test_3_mha_matches_reference(device, d_model, n_heads):
    mha = MultiHeadAttention(d_model, n_heads).to(device)
    x = torch.randn(2, 9, d_model, device=device)
    with no_builtin_attention():
        out = mha(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, ref_mha(mha, x), atol=ATOL, rtol=RTOL)


def test_3_mha_matches_nn_multihead_attention():
    d_model, n_heads, T = 32, 4, 7
    mha = MultiHeadAttention(d_model, n_heads)
    torch_mha = nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(torch.cat([mha.wq.weight, mha.wk.weight, mha.wv.weight]))
        torch_mha.out_proj.weight.copy_(mha.wo.weight)
    x = torch.randn(3, T, d_model)
    blocked = ~causal_mask(T, T)  # у nn.MultiheadAttention True = «нельзя»
    ref, _ = torch_mha(x, x, x, attn_mask=blocked, need_weights=False)
    with no_builtin_attention():
        out = mha(x)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_3_mha_is_causal():
    # Изменение будущих токенов не должно влиять на прошлые выходы.
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(1, 8, 32)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 32)
    with no_builtin_attention():
        a, b = mha(x), mha(x2)
    torch.testing.assert_close(a[:, :5], b[:, :5])
    assert not torch.allclose(a[:, 5:], b[:, 5:])


def test_3_mha_batch_independent():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(4, 6, 32)
    with no_builtin_attention():
        full = mha(x)
        single = torch.cat([mha(x[i : i + 1]) for i in range(4)])
    torch.testing.assert_close(full, single, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# 4. KV-cache
# ---------------------------------------------------------------------------


def test_4_kvcache_update_returns_prefix():
    cache = KVCache(batch=2, n_kv_heads=3, max_len=10, head_dim=4)
    k1, v1 = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    k2, v2 = torch.randn(2, 3, 1, 4), torch.randn(2, 3, 1, 4)
    k, v = cache.update(k1, v1)
    assert cache.length == 4 and k.shape == (2, 3, 4, 4)
    torch.testing.assert_close(k, k1)
    torch.testing.assert_close(v, v1)
    k, v = cache.update(k2, v2)
    assert cache.length == 5 and k.shape == (2, 3, 5, 4)
    torch.testing.assert_close(k, torch.cat([k1, k2], dim=2))
    torch.testing.assert_close(v, torch.cat([v1, v2], dim=2))


def test_4_kvcache_no_reallocation():
    cache = KVCache(batch=1, n_kv_heads=2, max_len=8, head_dim=4)
    k_ptr, v_ptr = cache.k.data_ptr(), cache.v.data_ptr()
    for _ in range(8):
        k, v = cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
        assert k.data_ptr() == k_ptr and v.data_ptr() == v_ptr, "верни view на буфер, а не копию"
    assert cache.k.data_ptr() == k_ptr and cache.v.data_ptr() == v_ptr


def test_4_kvcache_overflow():
    cache = KVCache(batch=1, n_kv_heads=1, max_len=4, head_dim=2)
    cache.update(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2))
    with pytest.raises(ValueError):
        cache.update(torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))
    assert cache.length == 3


def _run_chunks(mha, x, chunks, max_len):
    B = x.shape[0]
    cache = KVCache(B, mha.n_kv_heads, max_len, mha.head_dim, device=x.device)
    outs, pos = [], 0
    with no_builtin_attention():
        for n in chunks:
            outs.append(mha(x[:, pos : pos + n], cache=cache))
            pos += n
    assert cache.length == x.shape[1]
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("device", DEVICES)
def test_4_prefill_then_decode_matches_full(device):
    mha = MultiHeadAttention(32, 4).to(device)
    x = torch.randn(2, 12, 32, device=device)
    out = _run_chunks(mha, x, chunks=[5] + [1] * 7, max_len=16)
    torch.testing.assert_close(out, ref_mha(mha, x), atol=ATOL, rtol=RTOL)


def test_4_chunked_prefill_matches_full():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 12, 32)
    out = _run_chunks(mha, x, chunks=[3, 4, 1, 4], max_len=12)
    torch.testing.assert_close(out, ref_mha(mha, x), atol=ATOL, rtol=RTOL)


def test_4_cache_stores_k_v_projections():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(1, 6, 32)
    cache = KVCache(1, 4, 8, 8)
    with no_builtin_attention():
        mha(x, cache=cache)
    k_ref = mha.wk(x).view(1, 6, 4, 8).transpose(1, 2)
    v_ref = mha.wv(x).view(1, 6, 4, 8).transpose(1, 2)
    torch.testing.assert_close(cache.k[:, :, :6], k_ref)
    torch.testing.assert_close(cache.v[:, :, :6], v_ref)


# ---------------------------------------------------------------------------
# 5. GQA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 4), (8, 2), (8, 1)])
def test_5_gqa_matches_reference(device, n_heads, n_kv_heads):
    mha = MultiHeadAttention(64, n_heads, n_kv_heads).to(device)
    assert mha.wk.out_features == n_kv_heads * mha.head_dim
    x = torch.randn(2, 9, 64, device=device)
    with no_builtin_attention():
        out = mha(x)
    torch.testing.assert_close(out, ref_mha(mha, x), atol=ATOL, rtol=RTOL)


def test_5_gqa_head_grouping():
    # Обнуляем wv у kv-головы 0: при group=2 это должно занулить выход только query-голов 0 и 1.
    mha = MultiHeadAttention(32, n_heads=4, n_kv_heads=2)
    with torch.no_grad():
        mha.wv.weight[: mha.head_dim] = 0
        mha.wo.weight.copy_(torch.eye(32))
    x = torch.randn(1, 5, 32)
    with no_builtin_attention():
        out = mha(x)
    Dh = mha.head_dim
    assert torch.all(out[..., : 2 * Dh] == 0), "головы 0,1 должны читать kv-голову 0"
    assert not torch.all(out[..., 2 * Dh :] == 0)


def test_5_gqa_decode_with_small_cache():
    mha = MultiHeadAttention(64, n_heads=8, n_kv_heads=2)
    x = torch.randn(2, 10, 64)
    out = _run_chunks(mha, x, chunks=[4] + [1] * 6, max_len=10)
    torch.testing.assert_close(out, ref_mha(mha, x), atol=ATOL, rtol=RTOL)
