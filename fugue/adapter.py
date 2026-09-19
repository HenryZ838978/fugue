"""Fugue latent adapter: YuE2-Vae latent (64-d @ 25 Hz) -> MiniMax-Music3 DiT condition (2048-d @ 25 Hz).

Architecture (66.6M params): Linear(64->d) -> 2x Conv1d(k=5) local front-end -> depthwise Conv1d(k=63) relative position
-> N-layer bidirectional pre-LN Transformer -> LayerNorm -> Linear(d->2048). No absolute positions, so any length works;
inference is chunked (win=768, pad=128) to match the training crop.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, d_in=64, d=768, d_out=2048, layers=8, heads=12, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.front = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2))
        self.pos = nn.Sequential(nn.Conv1d(d, d, 63, padding=31, groups=16), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_out))

    def forward(self, x, mask=None):
        h = self.inp(x).transpose(1, 2)
        h = h + self.front(h)
        h = h + self.pos(h)
        h = h.transpose(1, 2)
        kpm = None if mask is None else (mask < 0.5)
        h = self.blocks(h, src_key_padding_mask=kpm)
        return self.out(h)


class FugueAdapter:
    """Loads adapter.safetensors + stats.npz + config.json (HF repo layout) and maps latents to c25."""

    def __init__(self, path, device="cuda"):
        from safetensors.torch import load_file
        path = Path(path)
        if not path.exists():                                   # HF repo id
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(path)))
        cfg = json.load(open(path / "config.json"))
        self.model = Adapter(d_in=cfg["d_in"], d=cfg["d"], d_out=cfg["d_out"], layers=cfg["layers"], heads=cfg["heads"]).to(device).eval()
        self.model.load_state_dict(load_file(str(path / "adapter.safetensors")))
        st = np.load(path / "stats.npz")
        self.stats = {k: torch.from_numpy(st[k]).to(device) for k in ("x_mean", "x_std", "y_mean", "y_std")}
        self.win, self.pad = cfg["chunked_inference"]["win"], cfg["chunked_inference"]["pad"]
        self.device = device

    @torch.no_grad()
    def __call__(self, z):
        """z: (T, 64) numpy or tensor, YuE2-Vae latent at 25 Hz -> (T, 2048) float32 c25 in MM3 condition space."""
        x = torch.as_tensor(np.asarray(z, dtype=np.float32)).to(self.device)
        x = (x - self.stats["x_mean"]) / self.stats["x_std"]
        T = x.shape[0]
        out = torch.empty(T, self.stats["y_mean"].shape[0], device=self.device)
        for s in range(0, T, self.win):
            e = min(s + self.win, T)
            a, b = max(0, s - self.pad), min(T, e + self.pad)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                p = self.model(x[a:b][None])[0].float()
            out[s:e] = p[s - a: s - a + (e - s)]
        return out * self.stats["y_std"] + self.stats["y_mean"]
