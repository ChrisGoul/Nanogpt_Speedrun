"""Parameter allocation for ~170M configs. No training — just counts where the
budget goes (embedding lookup table vs the transformer that actually computes).
Matches our architecture: attn = 4*d^2/layer (or GQA), MLP = 8*d^2/layer,
embed = vocab*d (x2 if untied)."""

def breakdown(vocab, dim, layers, tied, kv_heads=None, head_dim=64):
    heads = dim // head_dim
    if kv_heads:                       # grouped-query attention: fewer k/v heads
        attn_per = dim * dim + 2 * dim * (kv_heads * head_dim) + dim * dim
    else:                              # full multi-head attention
        attn_per = 4 * dim * dim
    mlp_per = 8 * dim * dim
    attn, mlp = layers * attn_per, layers * mlp_per
    embed = vocab * dim * (1 if tied else 2)
    total = attn + mlp + embed
    return embed, attn, mlp, total

configs = [
    ("char vocab 257 (reason168)", 257, 1024, 12, False, None),
    ("16K vocab, untied",          16384, 1024, 12, False, None),
    ("16K vocab, TIED",            16384, 1024, 12, True, None),
    ("49K vocab, untied",          49152, 1024, 12, False, None),
    ("49K vocab, TIED",            49152, 1024, 12, True, None),
    ("49K tied, DEEP 768x20",      49152, 768, 20, True, None),
    ("49K tied, WIDE 1536x7",      49152, 1536, 7, True, None),
    ("32K tied, DEEP 768x22",      32000, 768, 22, True, None),
    ("32K tied, 896x16 + GQA(4)",  32000, 896, 16, True, 4),
]

print(f"{'config':30} {'total':>7} | {'embed':>10} {'attn':>10} {'mlp':>10}")
print("-" * 76)
for name, v, d, l, t, kv in configs:
    e, a, m, tot = breakdown(v, d, l, t, kv_heads=kv)
    print(f"{name:30} {tot/1e6:6.0f}M | {e/1e6:5.1f}M {100*e/tot:3.0f}%  "
          f"{a/1e6:4.1f}M {100*a/tot:3.0f}%  {m/1e6:5.1f}M {100*m/tot:3.0f}%")
