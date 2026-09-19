"""Fugue 服务 A(graphtokenizer env):SheetSage2 转谱 + YuE2 生成(AR+NAR → latent64)+ YuE2 自家 VAE 解码(A/B 对照用)。
常驻一张卡,HTTP 接口,配合服务 B(fugue_graft_service.py,mm3 env)与编排 CLI(fugue_cover.py)。

  POST /transcribe  {audio_path}                                  → {abc, key_lab, vocal_notes, instrumental_notes, duration_seconds, sec}
  POST /generate    {style, lyrics, abc?, cot, seed, max_sem, decode_yue2}
                                                                  → {latent_path, yue2_wav_path?, semantic_tokens, truncated, latent_frames, t_ar, t_nar, t_dec}
  GET  /health

用法: CUDA_VISIBLE_DEVICES=7 python fugue_yue2_service.py --port 8650
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

Y = os.environ.get("FUGUE_YUE2", "/cache/zhangjing/models/_ms/models/m-a-p--YuE2-3B/snapshots/master")
V = os.environ.get("FUGUE_YUE2_VAE", "/cache/zhangjing/models/_ms/models/m-a-p--YuE2-Vae/snapshots/master")
SS2 = os.environ.get("FUGUE_SHEETSAGE2", "/cache/zhangjing/models/_ms/models/m-a-p--SheetSage2/snapshots/master")
MERT = os.environ.get("FUGUE_MERT", "/cache/zhangjing/models/_ms/models/m-a-p--MERT-v2-FullSong/snapshots/master")
WORK = Path(os.environ.get("FUGUE_WORK", "/cache/zhangjing/fugue/scion/svc"))
WORK.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="fugue-yue2")
LOCK = threading.Lock()
STATE = {}


class TranscribeReq(BaseModel):
    audio_path: str


class GenerateReq(BaseModel):
    style: str
    lyrics: str = "[Instrumental]"
    abc: str | None = None
    cot: str = "full"
    seed: int = 0
    max_sem: int = 9000
    decode_yue2: bool = True
    tag: str | None = None


def load_all():
    from yue2 import YuE2Pipeline
    from yue2.modeling_vae import YuE2VAE
    t0 = time.time()
    pipe = YuE2Pipeline.from_pretrained(Y, vae=V, local_files_only=True, progress=False)
    pipe._load_model()
    vae = YuE2VAE.from_pretrained(V, device="cuda", local_files_only=True).eval()
    STATE.update(pipe=pipe, vae=vae)
    print(f"[svcA] YuE2 + VAE loaded {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)


def ss2():
    if "ss2" not in STATE:
        sys.path.insert(0, SS2)
        from transformers import AutoModel
        t0 = time.time()
        STATE["ss2"] = AutoModel.from_pretrained(SS2, base_model_path=MERT, trust_remote_code=True, local_files_only=True).eval().to("cuda")
        print(f"[svcA] SheetSage2 loaded {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return STATE["ss2"]


@app.get("/health")
def health():
    return dict(ok=True, gpu_gib=round(torch.cuda.memory_allocated() / 2**30, 2), ss2_loaded="ss2" in STATE)


@app.post("/transcribe")
def transcribe(r: TranscribeReq):
    if not os.path.exists(r.audio_path):
        raise HTTPException(404, r.audio_path)
    with LOCK:
        t0 = time.time()
        tmp = tempfile.mkdtemp(prefix="ss2_", dir=WORK)
        try:
            res = ss2().transcribe(r.audio_path, output_dir=tmp)
            p = Path(tmp) / "score.abc"
            abc = p.read_text() if p.exists() else None
            kl = Path(tmp) / "key.lab"
            out = dict(abc=abc, key_lab=kl.read_text().strip()[:300] if kl.exists() else None,
                       **{k: res.get(k) for k in ("duration_seconds", "melody_notes", "vocal_notes", "instrumental_notes", "abc_measures", "abc_error")},
                       sec=round(time.time() - t0, 1))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            torch.cuda.empty_cache()
    return out


@app.post("/generate")
def generate(r: GenerateReq):
    from yue2.protocol import Sampling, SongRequest
    pipe, vae = STATE["pipe"], STATE["vae"]
    tag = r.tag or uuid.uuid4().hex[:10]
    with LOCK:
        try:
            req = SongRequest(style=r.style, lyrics=r.lyrics, cot=r.cot, abc=r.abc if r.cot != "off" else None, seed=r.seed, id=tag)
        except Exception as e:
            raise HTTPException(400, f"bad request: {e}")
        t0 = time.time()
        plan = pipe.plan(request=req)
        sem = pipe.generate_semantic(plan, sampling=Sampling(max_tokens=r.max_sem, min_tokens=min(200, r.max_sem)))
        t_ar = time.time() - t0
        lat = pipe.synthesize(sem).astype(np.float32)                                   # (T', 64)
        t_nar = time.time() - t0 - t_ar
        lat_path = WORK / f"{tag}.latent.npy"
        np.save(lat_path, lat)
        out = dict(tag=tag, latent_path=str(lat_path), abc_tokens=len(plan.abc_ids), semantic_tokens=len(sem.tokens),
                   truncated=bool(sem.truncated), latent_frames=int(lat.shape[0]), t_ar=round(t_ar, 1), t_nar=round(t_nar, 1),
                   abc=plan.abc if r.abc is None and r.cot != "off" else None)
        if r.decode_yue2:
            t1 = time.time()
            with torch.inference_mode():
                z = torch.from_numpy(lat).T[None].to("cuda")
                wav = vae.decode_tiled(z, output_device="cpu")[0].clamp(-1, 1).T.numpy()
            wp = WORK / f"{tag}.yue2vae.wav"
            sf.write(wp, wav, 48000)
            out.update(yue2_wav_path=str(wp), t_dec=round(time.time() - t1, 1))
        (WORK / f"{tag}.gen.json").write_text(json.dumps(dict(request=r.model_dump(), **out), ensure_ascii=False, indent=1))
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--preload_ss2", action="store_true")
    args = ap.parse_args()
    load_all()
    if args.preload_ss2:
        ss2()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
