"""SCION 诊断:R² 天花板 —— 用 MM3 自家 DAV latent(zip 内 vae.npy,128 维 @86Hz)代替 YuE2 latent 做同一个线性探针。

DAV latent 是同一套 VAE 的编码,若它 → c25 的 R² 与 YuE2 latent 相当,说明 c25 里不可由音频预测的那部分
(prompt、LM 因果长上下文)才是上限,接口选 YuE2 latent 并无损失;若 DAV 高出很多,则 YuE2 latent 丢了 MM3 需要的信息。
池化:帧 j 的 latent 区间 [start_j, start_{j+1}),start 按拼接时间轴 345k + (j-100k)·3.4453 算,区间内取均值。

用法: python probe_dav.py [--n_train 300 --n_val 100]
"""
import argparse
import io
import json
import time
import zipfile

import numpy as np
import torch

from data import PAIRS, HOP_LATENT, LAT_PER_FRAME, chunk_of_frame, load_meta
from probe_linear import dev, featurize

ZIPS = "/cache/zhangjing/musicodec_data/mm3-rvq-distill-corpus-8k/data"


def frame_starts(T):
    j = np.arange(T + 1)
    k = chunk_of_frame(np.minimum(j, T - 1), T)
    return np.floor(HOP_LATENT * k + (j - 100 * k) * LAT_PER_FRAME).astype(np.int64)


def pooled_dav(row):
    with zipfile.ZipFile(f"{ZIPS}/{row['shard']}.zip") as zf:
        lat = np.load(io.BytesIO(zf.read(f"{row['id']}/{row['id']}.vae.npy"))).astype(np.float32)   # (L, 128)
    T = row["T"] - 1
    st = np.clip(frame_starts(T), 0, lat.shape[0])
    cs = np.concatenate([np.zeros((1, lat.shape[1]), np.float32), np.cumsum(lat, 0)])
    n = np.maximum(st[1:] - st[:-1], 1)[:, None]
    return (cs[st[1:]] - cs[st[:-1]]) / n                                                             # (T, 128)


class Songs:
    def __init__(self, rows, w_pool):
        self.items = []
        for r in rows:
            y = torch.from_numpy(np.load(PAIRS / "c25" / f"{r['id']}.npy").astype(np.float32))
            x = torch.from_numpy(pooled_dav(r))
            T = min(len(x), len(y))
            self.items.append((x[:T], y[:T]))
        xs = torch.cat([x[::4] for x, _ in self.items])
        self.xm, self.xs = xs.mean(0), xs.std(0) + 1e-6

    def iter(self, w, step=1, norm=None):
        xm, xs = norm or (self.xm, self.xs)
        for x, y in self.items:
            yield featurize(((x - xm) / xs).to(dev), w)[::step], y[::step].to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()
    rows = load_meta("val")
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    tr_rows, va_rows = rows[: args.n_train], rows[args.n_train: args.n_train + args.n_val]
    t0 = time.time()
    tr, va = Songs(tr_rows, 0), Songs(va_rows, 0)
    norm = (tr.xm, tr.xs)
    y_mean = torch.cat([y[::4] for _, y in tr.items]).mean(0).to(dev)
    print(f"cached in {time.time() - t0:.0f}s; dav pooled std/dim min {tr.xs.min():.3f} max {tr.xs.max():.3f}", flush=True)
    out = {}
    for w in (0, 1, 2, 4, 8):
        t0 = time.time()
        xtx = xty = None
        for x, y in tr.iter(w, 2, norm):
            x, y = x.double(), y.double()
            xtx = x.T @ x if xtx is None else xtx + x.T @ x
            xty = x.T @ y if xty is None else xty + x.T @ y
        W = torch.linalg.solve(xtx + args.lam * torch.eye(xtx.shape[0], dtype=xtx.dtype, device=dev), xty)
        res = tot = None
        for x, y in va.iter(w, 1, norm):
            p = (x.double() @ W).float()
            d = ((y - p) ** 2).sum(0).double(); t = ((y - y_mean) ** 2).sum(0).double()
            res = d if res is None else res + d; tot = t if tot is None else tot + t
        r2c = 1 - res / tot
        out[f"dav_w{w}"] = dict(r2_var=round(float(1 - res.sum() / tot.sum()), 4), r2_chmean=round(float(r2c.mean()), 4))
        print(f"dav_w{w}: R²var={out[f'dav_w{w}']['r2_var']:.4f} R²ch={out[f'dav_w{w}']['r2_chmean']:.4f} {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(PAIRS.parent / "probe_dav.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
