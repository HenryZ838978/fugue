"""SCION 阶段 0:线性探针。YuE2 latent(±w 帧上下文)→ c25 的 ridge 回归,held-out 歌上算 R²。

扫描三件事:
  shift ∈ [-3..3]        整体移位 —— 对齐核验(峰应在 0)
  stitched vs naive      拼接时间轴修正是否有用(长歌尾部差 1-2 帧)
  w ∈ {0,1,2,4,8}        上下文宽度 —— 局部线性能解释多少
R² 报两种:var-weighted(1 - Σ残差²/Σ总方差,由大方差通道主导,= DiT 实际看到的尺度)
          与 per-channel mean(每通道标准化后的平均 R²)。
流式实现:逐歌累加 XᵀX / XᵀY(float64),不把整块 Y 放显存。

用法: python probe_linear.py [--n_train 300 --n_val 100]
"""
import argparse
import json
import time

import numpy as np
import torch

from data import PAIRS, align_index, compute_stats, load_meta

dev = "cuda" if torch.cuda.is_available() else "cpu"


def featurize(x, w):
    """(T, 64) → (T, (2w+1)*64 + 1):前后 w 帧拼接(边界重复)+ 常数项。"""
    T = x.shape[0]
    if w > 0:
        idx = (torch.arange(T, device=x.device)[:, None] + torch.arange(-w, w + 1, device=x.device)[None]).clamp(0, T - 1)
        x = x[idx].reshape(T, -1)
    return torch.cat([x, torch.ones(T, 1, device=x.device, dtype=x.dtype)], 1)


class Songs:
    """RAM 缓存:y (T,2048) f32、z 整曲 latent (T',64) f32、T。"""

    def __init__(self, rows, stats):
        self.items = []
        for r in rows:
            y = np.load(PAIRS / "c25" / f"{r['id']}.npy").astype(np.float32)
            z = (np.load(PAIRS / "yue2lat" / f"{r['id']}.npy").astype(np.float32) - stats["x_mean"]) / stats["x_std"]
            self.items.append((torch.from_numpy(z), torch.from_numpy(y)))

    def iter(self, w, shift, stitched, step=1):
        for z, y in self.items:
            T = y.shape[0]
            m = torch.from_numpy(align_index(T, z.shape[0], shift, stitched))
            x = featurize(z[m].to(dev), w)
            yield x[::step], y[::step].to(dev)


def fit(songs, w, shift, stitched, lam, step=2):
    xtx = xty = None
    n = 0
    for x, y in songs.iter(w, shift, stitched, step):
        x, y = x.double(), y.double()
        xtx = x.T @ x if xtx is None else xtx + x.T @ x
        xty = x.T @ y if xty is None else xty + x.T @ y
        n += x.shape[0]
    W = torch.linalg.solve(xtx + lam * torch.eye(xtx.shape[0], dtype=xtx.dtype, device=dev), xty)
    return W, n


def score(songs, W, w, shift, stitched, y_mean):
    res = tot = None
    for x, y in songs.iter(w, shift, stitched):
        p = (x.double() @ W).float()
        d = ((y - p) ** 2).sum(0).double(); t = ((y - y_mean) ** 2).sum(0).double()
        res = d if res is None else res + d
        tot = t if tot is None else tot + t
    r2_ch = 1 - res / tot
    return dict(r2_var=float(1 - res.sum() / tot.sum()), r2_chmean=float(r2_ch.mean()),
                r2_ch_p10=float(r2_ch.quantile(0.1)), r2_ch_p90=float(r2_ch.quantile(0.9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()
    rows = load_meta("val")
    print(f"val songs with both sides: {len(rows)}", flush=True)
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    tr_rows, va_rows = rows[: args.n_train], rows[args.n_train: args.n_train + args.n_val]
    stats = compute_stats(tr_rows, n=min(200, len(tr_rows)), path=PAIRS / "stats_probe.npz")
    y_mean = torch.from_numpy(stats["y_mean"]).to(dev)
    t0 = time.time()
    tr, va = Songs(tr_rows, stats), Songs(va_rows, stats)
    print(f"cached {len(tr.items)}+{len(va.items)} songs in {time.time() - t0:.0f}s", flush=True)
    out = {}

    def run(name, w, shift, stitched):
        t0 = time.time()
        W, n = fit(tr, w, shift, stitched, args.lam)
        m = score(va, W, w, shift, stitched, y_mean)
        m_tr = score(tr, W, w, shift, stitched, y_mean)["r2_var"]
        out[name] = dict(w=w, shift=shift, stitched=stitched, n_frames_train=n, r2_var_train=round(m_tr, 4),
                         **{k: round(v, 4) for k, v in m.items()})
        print(f"{name:28s} R²var={m['r2_var']:.4f} (train {m_tr:.4f}) R²ch={m['r2_chmean']:.4f} [p10 {m['r2_ch_p10']:.3f} p90 {m['r2_ch_p90']:.3f}] {time.time() - t0:.0f}s", flush=True)

    for s in range(-3, 4):
        run(f"shift{s:+d}_w1_stitched", 1, s, True)
    run("shift0_w1_naive", 1, 0, False)
    for w in (0, 2, 4, 8):
        run(f"shift0_w{w}_stitched", w, 0, True)
    json.dump(out, open(PAIRS.parent / "probe_linear.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
