import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None, kv_cache=None, use_cache=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        if use_cache:
            kv_cache = {"k": k, "v": v}

        attn = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return (x, kv_cache) if use_cache else x


def main():
    x = torch.randn(1, 7, 2048)
    attn = Attention(dim=2048, num_heads=4)

    with torch.no_grad():
        kv_cache = None
        outputs = []
        for token in x.split(1, dim=1):
            # with kv caching, a mask is not needed because the new query should attend
            # to every past position.
            (output, kv_cache) = attn(token, kv_cache=kv_cache, use_cache=True)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        print(outputs.shape)


if __name__ == "__main__":
    main()
