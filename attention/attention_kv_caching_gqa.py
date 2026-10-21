import time

import torch
from torch import nn


class Attention(nn.Module):
    """Grouped-query attention: each K/V head is shared by `queries_per_group` query heads."""

    def __init__(
        self, dim: int, num_heads: int = 8, queries_per_group: int = 2
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % queries_per_group == 0
        self.num_heads = num_heads
        self.queries_per_group = queries_per_group
        self.num_kv_heads = num_heads // queries_per_group
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.kv_proj = nn.Linear(dim, 2 * self.num_kv_heads * self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None, kv_cache=None, use_cache=False):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = (
            self.kv_proj(x)
            .reshape(B, N, 2, self.num_kv_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # shape: (B, num_kv_heads, N, head_dim)
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        if use_cache:
            kv_cache = {"k": k, "v": v}  # cache stores only num_kv_heads

        # expand each kv head to serve its group of queries: (B, num_heads, T, head_dim)
        k = k.repeat_interleave(self.queries_per_group, dim=1)
        v = v.repeat_interleave(self.queries_per_group, dim=1)

        attn = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return (x, kv_cache) if use_cache else x


def main():
    prompt = torch.randn(1, 4, 2048)  # shape: (B, N, C)
    num_new_tokens = 3
    attn = Attention(dim=2048, num_heads=4)

    with torch.no_grad():
        # prefill: the whole prompt goes through in one pass and fills the cache. A
        # lower triangular mask prevents prompt tokens from attending to later ones.
        N = prompt.shape[1]
        prefill_mask = torch.ones(N, N).tril()
        output, kv_cache = attn(prompt, attn_mask=prefill_mask, use_cache=True)
        outputs = [output]

        # decode: generate one token at a time. The last output stands in for the
        # next token's embedding, as there is no vocab/embedding layer here. With a
        # single token, no mask is needed because the new query should attend to
        # every cached position.
        inputs = [
            prompt
        ]  # this will store all inputs for the full causal pass comparison
        for _ in range(num_new_tokens):
            token = output[:, -1:]
            inputs.append(token)
            output, kv_cache = attn(token, kv_cache=kv_cache, use_cache=True)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        print(outputs.shape)

        # a full causal pass (no cache) over the same inputs should match
        x = torch.cat(inputs, dim=1)
        T = x.shape[1]
        full = attn(x, attn_mask=torch.ones(T, T).tril())
        print("prefill + decode == full:", torch.allclose(full, outputs, atol=1e-4))


def benchmark(seq_len: int = 256, dim: int = 512, num_heads: int = 8) -> None:
    """Autoregressive decoding: recompute the whole prefix each step vs use the cache,
    for MHA (1 query per group) and GQA (2 queries per group)."""
    x = torch.randn(1, seq_len, dim)

    print(f"seq_len={seq_len} dim={dim} num_heads={num_heads}")
    for name, queries_per_group in [("MHA", 1), ("GQA", 2)]:
        attn = Attention(dim, num_heads, queries_per_group)

        with torch.no_grad():
            start = time.perf_counter()
            for t in range(1, seq_len + 1):
                attn(x[:, :t], attn_mask=torch.ones(t, t).tril())
            no_cache = time.perf_counter() - start

            start = time.perf_counter()
            kv_cache = None
            for token in x.split(1, dim=1):
                _, kv_cache = attn(token, kv_cache=kv_cache, use_cache=True)
            cached = time.perf_counter() - start

        cache_kb = sum(t.numel() * t.element_size() for t in kv_cache.values()) / 1024
        print(
            f"  {name} (kv_heads={attn.num_kv_heads}): "
            f"no cache {no_cache:.3f}s, kv cache {cached:.3f}s, "
            f"speedup {no_cache / cached:.1f}x, cache size {cache_kb:.0f} KiB"
        )


if __name__ == "__main__":
    main()
    benchmark()
