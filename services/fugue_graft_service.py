"""Fugue 服务 B(mm3 env):adapter v4 + MM3 DiT + vocoder 常驻,把 YuE2 latent64 嫁接成 44.1k 立体声。

  POST /graft   {latent_path, seed, steps, tag?, max_frames?}   → {wav_path, frames, decode_sec}
  GET  /health

用法: CUDA_VISIBLE_DEVICES=7 python fugue_graft_service.py --port 8651 --run v4
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fugue import FugueAdapter, Rootstock  # noqa: E402

MODEL = os.environ.get("FUGUE_MM3", "/cache/zhangjing/models/MiniMax-Music3")
ADAPTER = os.environ.get("FUGUE_ADAPTER", "HenryZ838978/fugue-scion-v4")
WORK = Path(os.environ.get("FUGUE_WORK", "/cache/zhangjing/fugue/scion/svc"))
WORK.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="fugue-graft")
LOCK = threading.Lock()
STATE = {}


class GraftReq(BaseModel):
    latent_path: str
    seed: int = 7
    steps: int = 30
    tag: str | None = None
    max_frames: int = 0


def load_all(run):
    t0 = time.time()
    STATE.update(adapter=FugueAdapter(run), root=Rootstock(MODEL), run=run)
    print(f"[svcB] adapter {run} + DiT + vocoder loaded {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)


@app.get("/health")
def health():
    return dict(ok=True, run=STATE.get("run"), gpu_gib=round(torch.cuda.memory_allocated() / 2**30, 2))


@app.post("/graft")
def graft(r: GraftReq):
    if not os.path.exists(r.latent_path):
        raise HTTPException(404, r.latent_path)
    tag = r.tag or Path(r.latent_path).name.replace(".latent.npy", "") or uuid.uuid4().hex[:10]
    with LOCK:
        t0 = time.time()
        z = np.load(r.latent_path).astype(np.float32)
        if r.max_frames > 0:
            z = z[: r.max_frames]
        T = z.shape[0]
        wav = STATE["root"](STATE["adapter"](z), steps=r.steps, seed=r.seed)
        sr = STATE["root"].sr
        wp = WORK / f"{tag}.graft.wav"
        sf.write(wp, wav.T.cpu().numpy(), sr)
        out = dict(tag=tag, wav_path=str(wp), frames=T, sec=round(wav.shape[-1] / sr, 1), decode_sec=round(time.time() - t0, 1), run=STATE["run"])
        (WORK / f"{tag}.graft.json").write_text(json.dumps(dict(request=r.model_dump(), **out), indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8651)
    ap.add_argument("--run", default=ADAPTER, help="adapter dir or HF repo id")
    args = ap.parse_args()
    load_all(args.run)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
