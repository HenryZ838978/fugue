"""SCION 实验 A:条件消融 —— DiT + vocoder 到底承担了多少"音质"。

同一首自生成歌的 teacher-forcing hidden(mm3_tf.py --save_hidden 产物),取前 30s,condition 做五种处理喂 DiT:
  orig     原样(应≈复现原 flac;这同时是 teacher-forcing 提取器的正确性检验)
  zero     condition 置零(= 训练时 cond-dropout 的无条件分支)→ 纯 DiT 先验
  swap     换成另一首歌的 hidden
  noise10  c25 加噪,噪声方差 = 每通道方差的 10%(模拟 adapter 回归 R²≈0.9 的残差)
  noise30  同上 30%(R²≈0.7)
  davrecon 不走 DiT:zip 里 vae.npy(dav 编码器对渲染音频编出的 128 维 latent)直接过 vocoder —— MM3 自家 VAE 的重建上限
另存 ref = 原 flac 同段。对每个输出算 log-mel L1(vs ref),wav 转 mp3 供回传耳判。

用法: CUDA_VISIBLE_DEVICES=7 python expA.py --a mf-1-000000 --b mf-1-000001
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
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, "/cache/zhangjing/fugue/selfdistill/v2")
sys.path.insert(0, "/cache/zhangjing/fugue/scion")
from denoise_decode import (CHUNK_FRAMES, CHUNK_HOP, CROP_LEFT_LATENT, CROP_RIGHT_LATENT, GUIDANCE,  # noqa: E402
                            MODEL, OVERLAP_LATENT)
from mm3_tf import cond25, load_meta  # noqa: E402
from diffusers.models import MiniMaxMusic3ConditionEncoder, MiniMaxMusic3Transformer1DModel, MiniMaxMusic3Vocoder  # noqa: E402
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402

ZIPS = Path("/cache/zhangjing/musicodec_data/mm3-rvq-distill-corpus-8k/data")
HID = Path("/cache/zhangjing/fugue/scion/pairs/hidden")
OUT = Path("/cache/zhangjing/fugue/scion/expA")


def latent_len(cond_enc, frames):
    c = cond_enc.config
    return max(1, int(frames * c.output_sampling_rate / c.input_sampling_rate * c.input_hop_length / c.output_hop_length))


def out_latents(cond_enc, T):
    """T 帧走 200/100 窗拼接后输出的 latent 总数(与 denoise 的裁剪逻辑一致;T=1500 → 5174,与 zip json 吻合)。"""
    starts = [0] if T <= CHUNK_FRAMES else list(range(0, T - CHUNK_HOP, CHUNK_HOP))
    n = 0
    for i, s in enumerate(starts):
        n += latent_len(cond_enc, min(s + CHUNK_FRAMES, T) - s)
        n -= (0 if i == 0 else CROP_LEFT_LATENT) + (0 if i == len(starts) - 1 else CROP_RIGHT_LATENT)
    return n


def upsample(cond_enc, c25_chunk):
    """(T, 2048) @25Hz → (1, L, 2048) nearest,与 ConditionEncoder 末段一致。"""
    x = c25_chunk.transpose(0, 1)[None]
    x = F.interpolate(x, size=latent_len(cond_enc, c25_chunk.shape[0]), mode="nearest")
    return x.transpose(1, 2)


@torch.no_grad()
def denoise(cond_fn, num_frames, dit, scheduler, vocoder, cond_enc, steps, seed, device):
    """denoise_decode.denoise_one 的条件可替换版:cond_fn(start, end) → (1, L, 2048)。"""
    generator = torch.Generator(device).manual_seed(seed)
    chunk_starts = [0] if num_frames <= CHUNK_FRAMES else list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))
    previous_latent = previous_condition = None
    latent_chunks = []
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + CHUNK_FRAMES, num_frames)
        condition = cond_fn(chunk_start, chunk_end).to(dit.dtype)
        overlap = 0
        if previous_latent is not None:
            overlap = min(previous_latent.shape[-1], condition.shape[1])
            condition[:, :overlap] = previous_condition[:, :overlap]
        latents = randn_tensor((1, dit.config.in_channels, condition.shape[1]), generator=generator,
                               device=torch.device(device), dtype=condition.dtype)
        noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), device=device)
        condition_pair = torch.cat([condition, torch.zeros_like(condition)], dim=0)
        for t in scheduler.timesteps:
            if overlap > 0:
                tv = t.to(latents.dtype)
                latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * tv) * noise_prompt + tv * previous_latent[..., :overlap]
            noise_pred = dit(hidden_states=latents.repeat(2, 1, 1), timestep=t.expand(2).to(latents.dtype),
                             encoder_hidden_states=condition_pair, return_dict=False)[0]
            velocity = noise_pred[1:2] + GUIDANCE * (noise_pred[0:1] - noise_pred[1:2])
            latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        if overlap > 0:
            latents[..., :overlap] = previous_latent[..., :overlap]
        os_ = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT)
        oe_ = max(os_, latents.shape[-1] - OVERLAP_LATENT)
        previous_latent, previous_condition = latents[..., os_:oe_], condition[:, os_:oe_]
        latent_chunks.append(latents)
    hop = int(cond_enc.config.output_hop_length)
    wavs = []
    for i, latents in enumerate(latent_chunks):
        w = vocoder(latents.to(vocoder.dtype))
        left = 0 if i == 0 else CROP_LEFT_LATENT * hop
        right = 0 if i == len(latent_chunks) - 1 else CROP_RIGHT_LATENT * hop
        wavs.append(w[..., left: w.shape[-1] - right])
    return torch.cat(wavs, dim=-1).float().clamp(-1, 1)[0]                                   # (2, S)


def read_zip(meta_row, name):
    with zipfile.ZipFile(ZIPS / f"{meta_row['shard']}.zip") as zf:
        return zf.read(f"{meta_row['id']}/{meta_row['id']}.{name}")


def logmel_l1(a, b, sr=44100):
    n = min(a.shape[-1], b.shape[-1])
    mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=2048, hop_length=512, n_mels=128).to(a.device)
    la = torch.log(mel(a[..., :n].mean(0)) + 1e-5)
    lb = torch.log(mel(b[..., :n].mean(0)) + 1e-5)
    return float((la - lb).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--frames", type=int, default=750)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--variants", default="orig,zero,swap,noise10,noise30,davrecon")
    args = ap.parse_args()
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    ra, rb = meta[args.a], meta[args.b]

    cond_enc = MiniMaxMusic3ConditionEncoder.from_pretrained(MODEL, subfolder="condition_encoder", torch_dtype=torch.bfloat16).to(device).eval()
    dit = MiniMaxMusic3Transformer1DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to(device).eval()
    vocoder = MiniMaxMusic3Vocoder.from_pretrained(MODEL, subfolder="vocoder", torch_dtype=torch.bfloat16).to(device).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder="scheduler")
    sr = int(vocoder.config.sampling_rate)

    T = args.frames
    fh_a = torch.load(HID / f"{args.a}.pt", map_location=device)[:T]
    fh_b = torch.load(HID / f"{args.b}.pt", map_location=device)[:T]
    assert fh_a.shape[0] == T and fh_b.shape[0] == T, (fh_a.shape, fh_b.shape)
    c_a, c_b = cond25(cond_enc, fh_a).float(), cond25(cond_enc, fh_b).float()              # (T, 2048)
    # 一致性检查:c25→nearest 与 ConditionEncoder 全程输出相同
    full = cond_enc(fh_a[None, :200]).float()
    mine = upsample(cond_enc, c_a[:200].to(torch.bfloat16)).float()
    print(f"c25 path check: max|Δ|={(full - mine).abs().max():.3e}  c25 std/ch mean={c_a.std(0).mean():.4f}", flush=True)

    ref = torch.from_numpy(sf.read(io.BytesIO(read_zip(ra, "flac")), dtype="float32", always_2d=True)[0].T)
    n_out = out_latents(cond_enc, T)
    ref_len = n_out * int(cond_enc.config.output_hop_length)
    ref = ref[:, :ref_len].to(device)
    sf.write(OUT / f"{args.a}.ref.wav", ref.T.cpu().numpy(), sr)

    gen = torch.Generator("cpu").manual_seed(args.seed)
    std_a = c_a.std(0, keepdim=True)
    noise = torch.randn(c_a.shape, generator=gen).to(device) * std_a

    def mk(c):
        return lambda s, e: upsample(cond_enc, c[s:e].to(torch.bfloat16))

    variants = {
        "orig": mk(c_a),
        "zero": lambda s, e: torch.zeros(1, latent_len(cond_enc, e - s), c_a.shape[1], device=device, dtype=torch.bfloat16),
        "swap": mk(c_b),
        "noise10": mk(c_a + noise * 0.1 ** 0.5),
        "noise30": mk(c_a + noise * 0.3 ** 0.5),
    }
    results = {}
    for name in args.variants.split(","):
        t0 = time.time()
        if name == "davrecon":
            lat = np.load(io.BytesIO(read_zip(ra, "vae.npy"))).astype(np.float32)          # (L, 128)
            with torch.no_grad():
                wav = vocoder(torch.from_numpy(lat[:n_out].T[None]).to(device, vocoder.dtype)).float().clamp(-1, 1)[0]
        else:
            wav = denoise(variants[name], T, dit, scheduler, vocoder, cond_enc, args.steps, args.seed, device)
        path = OUT / f"{args.a}.{name}.wav"
        sf.write(path, wav.T.cpu().numpy(), sr)
        d = logmel_l1(wav, ref)
        results[name] = dict(logmel_l1_vs_ref=round(d, 4), sec=round(wav.shape[-1] / sr, 2), t=round(time.time() - t0, 1))
        print(f"{name:9s} logmel_L1={d:.4f} len={wav.shape[-1] / sr:.2f}s {time.time() - t0:.1f}s", flush=True)
    results["ref_vs_ref_b"] = round(logmel_l1(ref, torch.from_numpy(sf.read(io.BytesIO(read_zip(rb, "flac")), dtype="float32", always_2d=True)[0].T)[:, :ref_len].to(device)), 4)
    (OUT / f"{args.a}.expA.json").write_text(json.dumps(dict(a=args.a, b=args.b, frames=T, steps=args.steps, results=results), indent=1))
    for w in sorted(OUT.glob(f"{args.a}.*.wav")):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", "160k", str(w.with_suffix(".mp3"))], check=True)
    print("DONE", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
