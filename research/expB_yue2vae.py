"""SCION 实验 B(接穗侧声学):YuE2 VAE encode→decode 重建。graphtokenizer env(装了 yue2 包)。

输入两段 30s:MM3 自生成 flac(与实验 A 同一首,前 30s)+ 一首真实录音(refs_full,20s-50s)。
44.1k → 48k 重采样 → encode(posterior mean)→ decode_tiled → 48k wav。另把 MM3 整曲的 latent 存下来
(pairs/yue2lat/{id}.npy,(T',64) fp16),顺带看 T' 与 MM3 帧数 T-1 是否对齐。

用法: CUDA_VISIBLE_DEVICES=0 python expB_yue2vae.py --a mf-1-000000
"""
import argparse
import io
import json
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from yue2.modeling_vae import YuE2VAE

V = "/cache/zhangjing/models/_ms/models/m-a-p--YuE2-Vae/snapshots/master"
ZIPS = Path("/cache/zhangjing/musicodec_data/mm3-rvq-distill-corpus-8k/data")
META = "/cache/zhangjing/fugue/ss2/pairs8k/meta.jsonl"
REAL = "/cache/zhangjing/fugue/refs_full/02_004-s-ave.flac"
OUT = Path("/cache/zhangjing/fugue/scion/expB")
LAT = Path("/cache/zhangjing/fugue/scion/pairs/yue2lat")
SR = 48000


def load_48k(data_or_path, start_s=0.0, dur_s=None):
    audio, sr = sf.read(data_or_path, dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.T)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    a = int(start_s * SR)
    b = x.shape[1] if dur_s is None else min(x.shape[1], a + int(dur_s * SR))
    return x[:, a:b].contiguous()


@torch.inference_mode()
def roundtrip(vae, x, device):
    z = vae.encode(x[None].to(device))                                  # (1, 64, T')
    y = vae.decode_tiled(z, output_device="cpu")                        # (1, 2, S)
    return z[0].T.cpu(), y[0].clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--dur", type=float, default=30.0)
    args = ap.parse_args()
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    LAT.mkdir(parents=True, exist_ok=True)
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(META)}
    ra = meta[args.a]

    t0 = time.time()
    vae = YuE2VAE.from_pretrained(V, device=device, local_files_only=True).eval()
    print(f"vae loaded {time.time() - t0:.1f}s  params={sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M", flush=True)

    with zipfile.ZipFile(ZIPS / f"{ra['shard']}.zip") as zf:
        flac = zf.read(f"{args.a}/{args.a}.flac")
    segs = {f"{args.a}": load_48k(io.BytesIO(flac), 0, args.dur), "real_s-ave": load_48k(REAL, 20, args.dur)}
    res = {}
    for name, x in segs.items():
        t0 = time.time()
        z, y = roundtrip(vae, x, device)
        sf.write(OUT / f"{name}.ref48.wav", x.T.numpy(), SR)
        sf.write(OUT / f"{name}.yue2vae.wav", y.T.numpy(), SR)
        res[name] = dict(frames=int(z.shape[0]), in_samples=int(x.shape[1]), out_samples=int(y.shape[1]),
                         lat_std=round(float(z.std()), 4), lat_absmax=round(float(z.abs().max()), 3), t=round(time.time() - t0, 1))
        print(name, res[name], flush=True)

    # 整曲 latent(adapter 训练输入的样子)
    x_full = load_48k(io.BytesIO(flac))
    with torch.inference_mode():
        z_full = vae.encode(x_full[None].to(device))[0].T.cpu()
    np.save(LAT / f"{args.a}.npy", z_full.numpy().astype(np.float16))
    res["full"] = dict(T_yue2=int(z_full.shape[0]), T_mm3_emitted=int(ra["T"]) - 1, dur=ra["dur"],
                       per_dim_std_min=round(float(z_full.std(0).min()), 4), per_dim_std_max=round(float(z_full.std(0).max()), 4))
    print("full", res["full"], flush=True)
    (OUT / f"{args.a}.expB_yue2.json").write_text(json.dumps(res, indent=1))
    for w in sorted(OUT.glob("*.wav")):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", "160k", str(w.with_suffix(".mp3"))], check=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
