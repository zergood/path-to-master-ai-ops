"""Attention с нуля на чистом PyTorch.

Заполни все места с `raise NotImplementedError`. Проверка:

    pytest week-01/attention -v

Задания идут по возрастанию, тесты сгруппированы так же (test_1_*, test_2_*, ...):

1. softmax                      — численно устойчивый softmax
2. scaled_dot_product_attention — attention для одной/многих голов, маска, causal
3. MultiHeadAttention           — проекции Q/K/V/O, разбиение на головы, causal
4. KVCache + decode             — prefill/decode с кэшем, результат = полному проходу
5. GQA                          — n_kv_heads < n_heads (Llama-3, Mistral)

Нельзя: torch.softmax / F.softmax / Tensor.softmax и F.scaled_dot_product_attention
внутри своей реализации — тесты это ловят. Можно: matmul, exp, max, sum, masked_fill,
view/reshape/transpose, repeat_interleave и прочая элементарщина.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. softmax
# ---------------------------------------------------------------------------


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Численно устойчивый softmax вдоль `dim`.

    Должен корректно работать при больших значениях (x = 1000) и на строках,
    где всё, кроме одного элемента, равно -inf.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. scaled dot-product attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """softmax(q @ k^T / sqrt(D) + mask) @ v.

    Формы:
        q: (..., L, D)   — L запросов
        k: (..., S, D)   — S ключей
        v: (..., S, Dv)
        mask: bool, бродкастится к (..., L, S). True = «можно смотреть».
        return: (..., L, Dv)

    is_causal: запрос i видит ключи j <= i + (S - L). Это выравнивание по
    правому нижнему углу: при L == S получается обычная нижнетреугольная маска,
    а при декоде с кэшем (L=1, S=длина контекста) последний запрос видит всё.
    Если заданы и mask, и is_causal — применяются обе.

    Можно считать, что в каждой строке разрешён хотя бы один ключ.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. KV-cache (нужен в 3-м задании только как тип аргумента)
# ---------------------------------------------------------------------------


class KVCache:
    """Предвыделенный KV-кэш одного слоя attention.

    Буферы self.k, self.v формы (batch, n_kv_heads, max_len, head_dim) создаются
    один раз в __init__ и дальше не переаллоцируются — только пишем в срез.
    self.length — сколько позиций уже заполнено.
    """

    def __init__(
        self,
        batch: int,
        n_kv_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        self.max_len = max_len
        self.length = 0
        self.k = torch.zeros(batch, n_kv_heads, max_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(batch, n_kv_heads, max_len, head_dim, dtype=dtype, device=device)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Дописать T новых позиций и вернуть весь накопленный контекст.

        k_new, v_new: (batch, n_kv_heads, T, head_dim)
        return: (k, v) формы (batch, n_kv_heads, length + T, head_dim) — view на
        буферы, без копирования. После вызова self.length увеличивается на T.
        Если места не хватает — ValueError, кэш не меняется.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3 + 5. Multi-head attention (с GQA)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Causal self-attention декодера (как в Llama, без RoPE).

    Раскладка (на неё завязаны тесты):
        wq: d_model -> n_heads * head_dim
        wk, wv: d_model -> n_kv_heads * head_dim
        wo: n_heads * head_dim -> d_model
        все Linear без bias; head_dim = d_model // n_heads.
        Головы лежат подряд по последней оси: (B, T, H*Dh).view(B, T, H, Dh).

    GQA (задание 5): n_kv_heads делит n_heads, группа из n_heads // n_kv_heads
    подряд идущих query-голов использует одну и ту же kv-голову
    (query-голова h -> kv-голова h // group).
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model).

        cache=None: обычный causal проход по всей последовательности.
        cache задан: x — это T новых токенов, идущих сразу после cache.length
        уже обработанных. K/V новых токенов дописываются в кэш, attention
        считается по всему контексту из кэша (causal с учётом смещения).
        В кэше хранятся K/V в форме (B, n_kv_heads, S, head_dim) — до повтора голов для GQA.
        """
        raise NotImplementedError
