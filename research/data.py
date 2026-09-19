"""SCION 配对数据:YuE2 latent (T', 64) @25Hz  ↔  MM3 c25 (T, 2048) @25Hz,含拼接时间轴对齐。

MM3 的音频不是均匀时间轴:200 帧窗 / 100 帧 hop,每窗 int(200×3.4453)=689 latent,拼接后每 100 帧占 345 latent
(名义 344.53),所以第 k 窗的帧比名义时刻晚 0.47k latent ≈ 0.136k 个 YuE2 帧。align_index() 把每个 MM3 帧 j
映射到它在音频里真实时刻对应的 YuE2 帧 m_j = round(j + 0.1358·k(j))。

标准化统计(均值/方差)从 train 子集算一次,存 stats.npz。
"""
import json
from pathlib import Path

import numpy as np
import torch

PAIRS = Path("/cache/zhangjing/fugue/scion/pairs")
META = "/cache/zhangjing/fugue/ss2/pairs8k/meta.jsonl"
LAT_PER_FRAME = 441 / 128            # 3.4453 latent / 帧
HOP_LATENT = 345                      # 100 帧 hop 在拼接时间轴上实际占的 latent 数


def chunk_of_frame(j, T):
    """帧 j(0..T-1,T=发射帧数)归属的窗 k(裁剪后由哪个窗渲染)。"""
    n_chunks = 1 if T <= 200 else len(range(0, T - 100, 100))
    k = np.where(j < 125, 0, (j - 25) // 100)
    return np.minimum(k, n_chunks - 1)


def align_index(T, T_yue2, shift=0, stitched=True):
    """MM3 帧 j → YuE2 帧索引 m_j(nearest),clip 到 [0, T_yue2-1]。shift 为整体平移(扫描用)。"""
    j = np.arange(T)
    if stitched:
        k = chunk_of_frame(j, T)
        lat = HOP_LATENT * k + (j - 100 * k) * LAT_PER_FRAME
        m = np.rint(lat / LAT_PER_FRAME).astype(np.int64)
    else:
        m = j.copy()
    return np.clip(m + shift, 0, T_yue2 - 1)


def load_meta(split=None):
    rows = [json.loads(l) for l in open(META)]
    if split:
        rows = [r for r in rows if r["split"] == split]
    rows = [r for r in rows if (PAIRS / "c25" / f"{r['id']}.npy").exists() and (PAIRS / "yue2lat" / f"{r['id']}.npy").exists()]
    return sorted(rows, key=lambda r: r["id"])


def load_pair(row, shift=0, stitched=True, mmap=True):
    """返回 x (T, 64) float32 已对齐, y (T, 2048) float32。"""
    y = np.load(PAIRS / "c25" / f"{row['id']}.npy", mmap_mode="r" if mmap else None)
    z = np.load(PAIRS / "yue2lat" / f"{row['id']}.npy", mmap_mode="r" if mmap else None)
    T = y.shape[0]
    m = align_index(T, z.shape[0], shift, stitched)
    return np.asarray(z[m], dtype=np.float32), np.asarray(y, dtype=np.float32)


def compute_stats(rows, n=400, path=PAIRS / "stats.npz"):
    xs, ys = [], []
    for r in rows[:n]:
        x, y = load_pair(r)
        xs.append(x[::4]); ys.append(y[::4])
    x, y = np.concatenate(xs), np.concatenate(ys)
    st = dict(x_mean=x.mean(0), x_std=x.std(0) + 1e-6, y_mean=y.mean(0), y_std=y.std(0) + 1e-6, n_songs=n, n_frames=len(x))
    np.savez(path, **st)
    return st


def load_stats(path=PAIRS / "stats.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


class CropDataset(torch.utils.data.Dataset):
    """随机截 crop 帧的训练集;每次 __getitem__ 随机选歌、随机起点。"""

    def __init__(self, rows, stats, crop=512, size=100000):
        self.rows, self.crop, self.size = rows, crop, size
        self.xm, self.xs = torch.from_numpy(stats["x_mean"]), torch.from_numpy(stats["x_std"])
        self.ym, self.ys = torch.from_numpy(stats["y_mean"]), torch.from_numpy(stats["y_std"])

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        rng = np.random.default_rng()
        r = self.rows[rng.integers(len(self.rows))]
        x, y = load_pair(r)
        T = y.shape[0]
        if T > self.crop:
            s = rng.integers(T - self.crop + 1)
            x, y = x[s: s + self.crop], y[s: s + self.crop]
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x = (x - self.xm) / self.xs
        y = (y - self.ym) / self.ys
        if T < self.crop:
            pad = self.crop - T
            x = torch.cat([x, torch.zeros(pad, x.shape[1])]); y = torch.cat([y, torch.zeros(pad, y.shape[1])])
            mask = torch.cat([torch.ones(T), torch.zeros(pad)])
        else:
            mask = torch.ones(self.crop)
        return x, y, mask
