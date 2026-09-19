"""SCION 实验 B(砧木侧声学):MM3 Flow-VAE 的 encode→decode 重建,与 YuE2 VAE 同段对照。mm3 env。

dav.pth 的 encoder + mean_proj(社区 reference adapter 的 DAVEncoderOnly 类)把 44.1k 立体声编成 (128, L) latent,
MM3 vocoder 解回来。同一段真实录音走这里 vs 走 YuE2 VAE,就是 "MM3 声学侧是否明显优于 YuE2 声学侧" 的直接对照。
自生成那首另有 zip 里现成的 vae.npy(实验 A 的 davrecon 已用),这里再用 encoder 现场编一遍做一致性核对。

用法: CUDA_VISIBLE_DEVICES=7 python expB_mm3vae.py --a mf-1-000000
"""
import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/cache/zhangjing/models/open-rvq-encoder-minimax-music3--SimpleTuner")
from minimax_music3_reference_adapter import DAVEncoderOnly  # noqa: E402
from diffusers.models import MiniMaxMusic3Vocoder  # noqa: E402

MODEL = "/cache/zhangjing/models/MiniMax-Music3"
DAV = f"{MODEL}/dav.pth"
ZIPS = Path("/cache/zhangjing/musicodec_data/mm3-rvq-distill-corpus-8k/data")
META = "/cache/zhangjing/fugue/ss2/pairs8k/meta.jsonl"
REAL = "/cache/zhangjing/fugue/refs_full/02_004-s-ave.flac"
OUT = Path("/cache/zhangjing/fugue/scion/expB")
SR = 44100


def load_441(data_or_path, start_s=0.0, dur_s=None):
    audio, sr = sf.read(data_or_path, dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.T)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    a = int(start_s * SR)
    b = x.shape[1] if dur_s is None else min(x.shape[1], a + int(dur_s * SR))
    return x[:, a:b].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--dur", type=float, default=30.0)
    args = ap.parse_args()
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(META)}
    ra = meta[args.a]

    enc = DAVEncoderOnly()
    ck = torch.load(DAV, map_location="cpu", weights_only=True)
    enc.load_state_dict({k: v for k, v in ck.items() if k.startswith(("encoder.", "mean_proj."))}, strict=True)
    enc = enc.to(device).eval()
    vocoder = MiniMaxMusic3Vocoder.from_pretrained(MODEL, subfolder="vocoder", torch_dtype=torch.bfloat16).to(device).eval()

    with zipfile.ZipFile(ZIPS / f"{ra['shard']}.zip") as zf:
        flac = zf.read(f"{args.a}/{args.a}.flac")
        vae_npy = np.load(io.BytesIO(zf.read(f"{args.a}/{args.a}.vae.npy"))).astype(np.float32)
    segs = {f"{args.a}": load_441(io.BytesIO(flac), 0, args.dur), "real_s-ave": load_441(REAL, 20, args.dur)}
    res = {}
    for name, x in segs.items():
        t0 = time.time()
        with torch.inference_mode():
            z = enc(x[None].to(device))                                              # (1, 128, L)
            y = vocoder(z.to(vocoder.dtype)).float().clamp(-1, 1)[0].cpu()
        sf.write(OUT / f"{name}.ref441.wav", x.T.numpy(), SR)
        sf.write(OUT / f"{name}.mm3vae.wav", y.T.numpy(), SR)
        res[name] = dict(latents=int(z.shape[-1]), in_samples=int(x.shape[1]), out_samples=int(y.shape[1]),
                         lat_std=round(float(z.std()), 4), t=round(time.time() - t0, 1))
        if name == args.a:
            n = min(z.shape[-1], vae_npy.shape[0])
            d = (z[0, :, :n].cpu().T - torch.from_numpy(vae_npy[:n])).abs().mean()
            res[name]["vs_zip_vae_npy_mean_abs_diff"] = round(float(d), 4)
            res[name]["zip_vae_npy_std"] = round(float(vae_npy.std()), 4)
        print(name, res[name], flush=True)
    (OUT / f"{args.a}.expB_mm3.json").write_text(json.dumps(res, indent=1))
    for w in sorted(OUT.glob("*.wav")):
        if not w.with_suffix(".mp3").exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", "160k", str(w.with_suffix(".mp3"))], check=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
