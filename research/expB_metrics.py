"""SCION 实验 B 度量:同一段音频经 YuE2 VAE / MM3 Flow-VAE 重建,vs 原始的 log-mel L1(全带 + 低/中/高三段)与 SI-SDR。
两路输出统一重采样到 44.1k 后比较。mm3 env。

用法: python expB_metrics.py
"""
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

OUT = Path("/cache/zhangjing/fugue/scion/expB")
SR = 44100


def load(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(a.T)
    return torchaudio.functional.resample(x, sr, SR) if sr != SR else x


def best_lag(a, b, max_lag=2048):
    """VAE 重建可能整体平移几十个样本,用互相关找最佳对齐(mono)。"""
    am, bm = a.mean(0)[: SR * 10], b.mean(0)[: SR * 10]
    am, bm = am - am.mean(), bm - bm.mean()
    xc = torch.nn.functional.conv1d(bm[None, None], am[None, None, max_lag:-max_lag])[0, 0]
    return int(xc.argmax()) - max_lag


def metrics(rec, ref):
    lag = best_lag(ref, rec)
    if lag > 0:
        rec = rec[:, lag:]
    elif lag < 0:
        ref = ref[:, -lag:]
    n = min(rec.shape[-1], ref.shape[-1])
    rec, ref = rec[..., :n], ref[..., :n]
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=2048, hop_length=512, n_mels=128, f_max=SR / 2)
    freqs = torchaudio.functional.melscale_fbanks(1025, 0, SR / 2, 128, SR).argmax(0) * SR / 2 / 1024
    la, lb = torch.log(mel(rec.mean(0)) + 1e-5), torch.log(mel(ref.mean(0)) + 1e-5)
    d = (la - lb).abs().mean(1)
    bands = {"lo<4k": d[freqs < 4000].mean(), "mid4-10k": d[(freqs >= 4000) & (freqs < 10000)].mean(), "hi>10k": d[freqs >= 10000].mean()}
    x, y = ref.mean(0), rec.mean(0)
    x, y = x - x.mean(), y - y.mean()
    s = (x * y).sum() / (x * x).sum() * x
    sisdr = 10 * torch.log10((s * s).sum() / ((y - s) ** 2).sum())
    return dict(lag_samples=lag, logmel_l1=round(float(d.mean()), 4), **{k: round(float(v), 4) for k, v in bands.items()}, si_sdr_db=round(float(sisdr), 2))


def main():
    res = {}
    for seg in ["mf-1-000415", "real_s-ave"]:
        ref = load(OUT / f"{seg}.ref441.wav")
        for sys_ in ["yue2vae", "mm3vae"]:
            res[f"{seg}/{sys_}"] = metrics(load(OUT / f"{seg}.{sys_}.wav"), ref)
            print(f"{seg:14s} {sys_:8s}", res[f"{seg}/{sys_}"], flush=True)
    (OUT / "expB_metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
