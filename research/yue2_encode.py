"""SCION 接穗侧:把 pairs8k 全库 flac 过 YuE2 VAE encoder(posterior mean),存 (T', 64) fp16。graphtokenizer env。

44.1k → 48k 重采样后编码,T' ≈ dur×25。与 MM3 帧的对齐(200/100 窗拼接时间轴漂移)留给训练侧 align_index()。

用法: CUDA_VISIBLE_DEVICES=0 python yue2_encode.py [--split train] [--limit N] [--shard i --nshards n]
"""
import argparse
import io
import json
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
LAT = Path("/cache/zhangjing/fugue/scion/pairs/yue2lat")
SR = 48000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    device = "cuda"
    LAT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(META)]
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    rows.sort(key=lambda r: (r["shard"], r["id"]))                     # 同 zip 相邻,少开文件
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    rows = [r for r in rows if not (LAT / f"{r['id']}.npy").exists()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} songs to encode", flush=True)

    vae = YuE2VAE.from_pretrained(V, device=device, local_files_only=True).eval()
    t0 = time.time()
    zf, zname = None, None
    fails = 0
    for i, r in enumerate(rows):
        try:
            if r["shard"] != zname:
                if zf:
                    zf.close()
                zf, zname = zipfile.ZipFile(ZIPS / f"{r['shard']}.zip"), r["shard"]
            audio, sr = sf.read(io.BytesIO(zf.read(f"{r['id']}/{r['id']}.flac")), dtype="float32", always_2d=True)
            x = torch.from_numpy(audio.T)
            if x.shape[0] == 1:
                x = x.repeat(2, 1)
            x = torchaudio.functional.resample(x, sr, SR) if sr != SR else x
            with torch.inference_mode():
                z = vae.encode(x[None].to(device))[0].T.cpu()                # (T', 64)
            np.save(LAT / f"{r['id']}.tmp.npy", z.numpy().astype(np.float16))
            (LAT / f"{r['id']}.tmp.npy").rename(LAT / f"{r['id']}.npy")      # 原子落盘,多进程并行不撞
        except Exception as e:
            fails += 1
            print(f"FAIL {r['id']}: {type(e).__name__}: {e}", flush=True)
            continue
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[{i + 1}/{len(rows)}] {r['id']} T'={z.shape[0]} T-1={r['T'] - 1} "
                  f"{el / (i + 1):.2f}s/song eta {el / (i + 1) * (len(rows) - i - 1) / 60:.0f}min", flush=True)
    print(f"DONE {len(rows) - fails} ok, {fails} fail, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
