"""SCION 阶段 3:嫁接解码 —— YuE2 latent → adapter → c25 → nearest ↑ → MM3 DiT → vocoder → wav。

两种输入:
  --id mf-1-xxxxx      pairs/yue2lat 里的 latent(= MM3 自生成音频过 YuE2 VAE encoder),"经嫁接的重建":
                       与原 flac 比 = 嫁接保留了多少内容;与 TF-orig 解码比 = adapter 误差造成的额外损失
  --latent path.npy    任意 (T', 64) YuE2 latent(比如 YuE2 LM 自己生成的),真正的接穗→砧木推理

用法: CUDA_VISIBLE_DEVICES=7 python graft_decode.py --run v1 --id mf-1-000415 [--frames 750]
"""
import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/cache/zhangjing/fugue/scion")
from data import PAIRS, align_index, load_meta as load_rows  # noqa: E402
from expA import MODEL, denoise, latent_len, logmel_l1, out_latents, read_zip, upsample  # noqa: E402
from train_adapter import RUNS, Adapter  # noqa: E402
from diffusers.models import MiniMaxMusic3ConditionEncoder, MiniMaxMusic3Transformer1DModel, MiniMaxMusic3Vocoder  # noqa: E402
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402

OUT = Path("/cache/zhangjing/fugue/scion/graft")


def load_adapter(run, dev):
    ck = torch.load(RUNS / run / "best.pt", map_location=dev)
    a = ck["args"]
    model = Adapter(d=a["d"], layers=a["layers"], heads=a["heads"], linear=a["linear"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    st = np.load(RUNS / run / "stats.npz")
    stats = {k: torch.from_numpy(st[k]).to(dev) for k in ("x_mean", "x_std", "y_mean", "y_std")}
    print(f"adapter {run}: step {ck['step']} eval {json.dumps({k: round(v, 4) for k, v in ck['eval'].items()})}", flush=True)
    return model, stats


@torch.no_grad()
def predict_c25(model, stats, z, dev, win=768, pad=128):
    """z (T, 64) 已对齐的 YuE2 latent → c25 (T, 2048) 原始空间。
    长于 win 时分块:每块取 [s-pad, e+pad) 上下文、只保留中间 [s, e) 的输出(训练 crop=768,分块反而更贴训练分布)。"""
    x = (torch.from_numpy(z).to(dev) - stats["x_mean"]) / stats["x_std"]
    T = x.shape[0]
    out = torch.empty(T, stats["y_mean"].shape[0], device=dev)
    for s in range(0, T, win):
        e = min(s + win, T)
        a, b = max(0, s - pad), min(T, e + pad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x[a:b][None])[0].float()
        out[s:e] = p[s - a: s - a + (e - s)]
    return out * stats["y_std"] + stats["y_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--id", default=None)
    ap.add_argument("--latent", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--frames", type=int, default=750)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    model, stats = load_adapter(args.run, dev)

    cond_enc = MiniMaxMusic3ConditionEncoder.from_pretrained(MODEL, subfolder="condition_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
    dit = MiniMaxMusic3Transformer1DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to(dev).eval()
    vocoder = MiniMaxMusic3Vocoder.from_pretrained(MODEL, subfolder="vocoder", torch_dtype=torch.bfloat16).to(dev).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder="scheduler")
    sr = int(vocoder.config.sampling_rate)

    res = {}
    if args.id:
        rows = {r["id"]: r for r in load_rows()}
        r = rows[args.id]
        tag = args.tag or f"{args.id}.{args.run}"
        z_full = np.load(PAIRS / "yue2lat" / f"{args.id}.npy").astype(np.float32)
        T_all = int(r["T"]) - 1
        T = min(args.frames, T_all)
        z = z_full[align_index(T_all, z_full.shape[0])][:T]
        c_true = torch.from_numpy(np.load(PAIRS / "c25" / f"{args.id}.npy")[:T].astype(np.float32)).to(dev)
        c_pred = predict_c25(model, stats, z, dev)
        res["r2_var_song"] = round(float(1 - ((c_true - c_pred) ** 2).sum() / ((c_true - c_true.mean(0)) ** 2).sum()), 4)
        print(f"song R²var (first {T} frames) = {res['r2_var_song']}", flush=True)
        ref = torch.from_numpy(sf.read(io.BytesIO(read_zip(r, "flac")), dtype="float32", always_2d=True)[0].T)
        ref = ref[:, : out_latents(cond_enc, T) * int(cond_enc.config.output_hop_length)].to(dev)
        sf.write(OUT / f"{tag}.ref.wav", ref.T.cpu().numpy(), sr)
    else:
        z = np.load(args.latent).astype(np.float32)
        T = min(args.frames, z.shape[0]) if args.frames > 0 else z.shape[0]
        z = z[:T]
        tag = args.tag or f"{Path(args.latent).stem}.{args.run}"
        c_pred = predict_c25(model, stats, z, dev)
        c_true = ref = None

    t0 = time.time()
    wav = denoise(lambda s, e: upsample(cond_enc, c_pred[s:e].to(torch.bfloat16)), T, dit, scheduler, vocoder, cond_enc, args.steps, args.seed, dev)
    res["decode_sec"] = round(time.time() - t0, 1)
    sf.write(OUT / f"{tag}.graft.wav", wav.T.cpu().numpy(), sr)
    if ref is not None:
        res["logmel_l1_graft_vs_ref"] = round(logmel_l1(wav, ref), 4)
        wav_o = denoise(lambda s, e: upsample(cond_enc, c_true[s:e].to(torch.bfloat16)), T, dit, scheduler, vocoder, cond_enc, args.steps, args.seed, dev)
        sf.write(OUT / f"{tag}.tforig.wav", wav_o.T.cpu().numpy(), sr)
        res["logmel_l1_tforig_vs_ref"] = round(logmel_l1(wav_o, ref), 4)
        res["logmel_l1_graft_vs_tforig"] = round(logmel_l1(wav, wav_o), 4)
    print(json.dumps(res), flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps(dict(run=args.run, id=args.id, latent=args.latent, frames=T, **res), indent=1))
    for w in OUT.glob(f"{tag}.*.wav"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", "160k", str(w.with_suffix(".mp3"))], check=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
