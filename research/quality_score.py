"""SCION 音质评测:Audiobox Aesthetics(Meta,四轴 0–10:PQ 制作质量 / PC 制作复杂度 / CE 内容愉悦 / CU 内容有用)
+ CLAP 文本探针(大调 vs 小调、caption 相似度)。mm3 env。

对 expA / expB / graft / yue2gen 里所有 wav 打分,按"同 latent 的 YuE2-VAE 解码 vs 嫁接解码"配对输出。
CLAP 探针:major/minor 各一组文本,取 (major−minor) 余弦差作"调式倾向",用于量化 Henry 耳判的"YuE2 大调布鲁斯 vs MM3 小调硬摇滚"。

用法: CUDA_VISIBLE_DEVICES=7 python quality_score.py [--glob 'graft/*.wav' ...]
"""
import argparse
import glob
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path("/cache/zhangjing/fugue/scion")
MAJOR = ["upbeat major key music, bright and happy", "a cheerful song in a major key"]
MINOR = ["dark minor key music, sad and tense", "a melancholic song in a minor key"]
BLUES = ["blues rock with a shuffle groove"]
HARDROCK = ["traditional hard rock with distorted guitars"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="*", default=["expA/*.wav", "expB/*.wav", "graft/*.wav", "yue2gen/*.wav"])
    ap.add_argument("--out", default="quality_scores.json")
    ap.add_argument("--max_sec", type=float, default=30.0)
    args = ap.parse_args()
    from audiobox_aesthetics.infer import initialize_predictor
    from transformers import ClapModel, ClapProcessor
    ab = initialize_predictor()
    clap = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").cuda().eval()
    proc = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")

    def t_emb(texts):
        with torch.no_grad():
            i = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            e = clap.get_text_features(**i)
            e = e if torch.is_tensor(e) else e.pooler_output
        return torch.nn.functional.normalize(e, dim=-1).mean(0)

    t_maj, t_min, t_blues, t_hr = (torch.nn.functional.normalize(t_emb(x), dim=-1) for x in (MAJOR, MINOR, BLUES, HARDROCK))

    wavs = sorted({w for g in args.glob for w in glob.glob(str(ROOT / g))})
    prev = {}
    if (ROOT / args.out).exists():
        prev = {r["file"]: r for r in json.load(open(ROOT / args.out))}
    rows = []
    for w in wavs:
        rel = str(Path(w).relative_to(ROOT))
        if rel in prev:
            rows.append(prev[rel]); continue
        y, sr = sf.read(w, dtype="float32", always_2d=True)
        y = y[: int(args.max_sec * sr)]
        abx = ab.forward([{"path": torch.from_numpy(y.T.copy()), "sample_rate": sr}])[0]
        y48 = librosa.resample(y.mean(1), orig_sr=sr, target_sr=48000)
        with torch.no_grad():
            i = proc(audio=y48, sampling_rate=48000, return_tensors="pt").to("cuda")
            e = clap.get_audio_features(**i)
            e = e if torch.is_tensor(e) else e.pooler_output
            e = torch.nn.functional.normalize(e, dim=-1)[0]
        r = dict(file=rel, sec=round(len(y) / sr, 1), **{k: round(float(abx[k]), 3) for k in ("PQ", "PC", "CE", "CU")},
                 clap_major_minus_minor=round(float(e @ t_maj - e @ t_min), 4),
                 clap_blues_minus_hardrock=round(float(e @ t_blues - e @ t_hr), 4))
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open(ROOT / args.out, "w"), indent=1)
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
