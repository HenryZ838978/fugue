"""SCION 内容保持度量:两段音频的 chroma(CQT 12 维)逐帧余弦相似度 + 时间平移扫描 + 打乱基线。

用来回答"嫁接换了音色,旋律/和声骨架守住了吗":同一 YuE2 latent 的 YuE2-VAE 解码 vs graft 解码,
或 graft 重建 vs 原 flac。报:best-lag 的均值余弦、shuffle 基线(帧顺序打乱)、以及超出基线的幅度。
两路各自 resample 到 22.05k mono,hop 2048(≈93ms,粗于 25Hz 帧,容忍 ±1 帧对齐误差)。

用法: python chroma_agree.py a.wav b.wav [--max_lag 3]
"""
import argparse
import json

import librosa
import numpy as np

SR = 22050
HOP = 2048


def chroma(path, dur=None):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=dur)
    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, n_chroma=12, bins_per_octave=36)
    c = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-8)
    return c.T                                                     # (F, 12)


def agree(a, b, max_lag=3, seed=0):
    n = min(len(a), len(b)) - max_lag
    best = (-1, 0)
    for lag in range(-max_lag, max_lag + 1):
        aa = a[max_lag: max_lag + n]
        bb = b[max_lag + lag: max_lag + lag + n]
        s = float((aa * bb).sum(1).mean())
        if s > best[0]:
            best = (s, lag)
    rng = np.random.default_rng(seed)
    aa = a[max_lag: max_lag + n]
    base = float(np.mean([(aa * b[rng.permutation(len(b))][:n]).sum(1).mean() for _ in range(20)]))
    return dict(cos=round(best[0], 4), lag_frames=best[1], shuffle_base=round(base, 4), above_base=round(best[0] - base, 4), frames=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--max_lag", type=int, default=3)
    ap.add_argument("--dur", type=float, default=None)
    args = ap.parse_args()
    r = agree(chroma(args.a, args.dur), chroma(args.b, args.dur), args.max_lag)
    print(json.dumps(dict(a=args.a, b=args.b, **r)))


if __name__ == "__main__":
    main()
