"""SCION eval 打分(生成完成后跑):
  身份   Discogs-VINet:每条生成物对 210 首原曲检索(余弦),Hit@1 / MRR / mAP(单相关 = MRR);上界 = 原曲 60s 截段当 query
  旋律   SheetSage2 回转谱 DTW(Ins 声部;eval/gen 的 retrans.jsonl 由 ss2/retrans.py 产出)
  风格   CLAP 音频–文本相似(对目标 style 文本)
  音质   频谱物理量(>8k/>12k 能量占比、99% rolloff、side/mid)+ Audiobox PQ(仅参考)
按臂(fullscore-yue2vae / fullscore-graft / noscore-yue2vae / noscore-graft)汇总;配对差(同 latent 的 graft − yue2vae)给中位与胜率。

用法: CUDA_VISIBLE_DEVICES=0 python eval_score.py [--skip_retrans]
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path("/cache/zhangjing/fugue/scion")
GEN = ROOT / "eval" / "gen"
EV = ROOT / "eval"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/cache/zhangjing/fugue/ss2")


def spectral(path, sec=60):
    y, sr = sf.read(path, dtype="float32", always_2d=True); y = y[: int(sec * sr)]
    m = y.mean(1); S = np.abs(librosa.stft(m, n_fft=4096, hop_length=1024)) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=4096); tot = S.sum() + 1e-12
    roll = float(np.median(librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr, roll_percent=0.99)))
    side, mid = y[:, 0] - y[:, 1], y[:, 0] + y[:, 1]
    return dict(hi8k=float(S[fr >= 8000].sum() / tot), hi12k=float(S[fr >= 12000].sum() / tot), rolloff99=roll,
                side_mid_db=float(10 * np.log10(side.var() / (mid.var() + 1e-9) + 1e-9)))


def identity(names, embs, orig_names, orig_embs):
    """每个 query 的正确原曲 = tag 里的 id。返回 per-query rank。"""
    idx = {n: i for i, n in enumerate(orig_names)}
    sims = embs @ orig_embs.T
    ranks = {}
    for n, s in zip(names, sims):
        sid = n.split("__")[0]
        if sid not in idx:
            continue
        order = np.argsort(-s)
        ranks[n] = int(np.where(order == idx[sid])[0][0]) + 1
    return ranks


def summarize(ranks):
    r = np.array(list(ranks.values()))
    return dict(n=len(r), hit1=float((r == 1).mean()), hit10=float((r <= 10).mean()), mrr=float((1 / r).mean()), median_rank=float(np.median(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_retrans", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    recs = {Path(p).stem: json.load(open(p)) for p in glob.glob(str(GEN / "*.json"))}
    wavs = sorted(glob.glob(str(GEN / "*.wav")))
    print(f"{len(recs)} items, {len(wavs)} wavs", flush=True)

    # 1. 身份
    from vinet_embed import load_vinet, cqt_feat, embed
    orig = np.load(EV / "vinet_orig_full.npz"); o60 = np.load(EV / "vinet_orig_60s.npz")
    m = load_vinet(dev)
    cache = EV / "vinet_gen.npz"
    if cache.exists():
        z = np.load(cache); names, embs = list(z["names"]), z["emb"]
    else:
        names, embs = [], []
    done = set(names)
    for k, w in enumerate(wavs):
        n = Path(w).name
        if n in done:
            continue
        embs = list(embs); embs.append(embed(m, cqt_feat(w), dev)); names.append(n)
        if (k + 1) % 100 == 0:
            print(f"vinet {k + 1}/{len(wavs)}", flush=True)
    embs = np.stack(embs); np.savez(cache, names=np.array(names), emb=embs)
    ranks = identity(names, embs, list(orig["names"]), orig["emb"])
    upper = identity([f"{n}__orig60" for n in o60["names"]], o60["emb"], list(orig["names"]), orig["emb"])
    del m; torch.cuda.empty_cache()

    # 2. 风格(CLAP)+ 音质(物理量 + Audiobox)
    from transformers import ClapModel, ClapProcessor
    from audiobox_aesthetics.infer import initialize_predictor
    clap = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    ab = initialize_predictor()
    tcache = {}

    def temb(t):
        if t not in tcache:
            with torch.no_grad():
                i = proc(text=[t], return_tensors="pt", padding=True, truncation=True).to(dev)
                e = clap.get_text_features(**i); e = e if torch.is_tensor(e) else e.pooler_output
            tcache[t] = torch.nn.functional.normalize(e, dim=-1)[0]
        return tcache[t]

    per = {}
    pcache = EV / "per_item.json"
    if pcache.exists():
        per = json.load(open(pcache))
    for k, w in enumerate(wavs):
        n = Path(w).name
        if n in per:
            continue
        tag, dec = n.rsplit(".", 2)[0], n.rsplit(".", 2)[1]
        rec = recs.get(tag)
        if not rec:
            continue
        y, sr = sf.read(w, dtype="float32", always_2d=True); y = y[: int(60 * sr)]
        y48 = librosa.resample(y.mean(1), orig_sr=sr, target_sr=48000)
        with torch.no_grad():
            i = proc(audio=y48, sampling_rate=48000, return_tensors="pt").to(dev)
            e = clap.get_audio_features(**i); e = e if torch.is_tensor(e) else e.pooler_output
            e = torch.nn.functional.normalize(e, dim=-1)[0]
        abx = ab.forward([{"path": torch.from_numpy(y.T.copy()), "sample_rate": sr}])[0]
        per[n] = dict(tag=tag, id=rec["id"], arm=rec["arm"], dec=dec, style_id=rec["style_id"],
                      clap_style=float(e @ temb(rec["style"])), PQ=float(abx["PQ"]), CE=float(abx["CE"]), CU=float(abx["CU"]),
                      **spectral(w), vinet_rank=ranks.get(n))
        if (k + 1) % 50 == 0:
            print(f"score {k + 1}/{len(wavs)}", flush=True); json.dump(per, open(pcache, "w"))
    json.dump(per, open(pcache, "w"))

    # 3. 旋律 DTW(可选,需先跑 retrans)
    rt = GEN / "retrans.jsonl"
    if rt.exists() and not args.skip_retrans:
        from followscore import pitches, dtw_norm
        from abcutil import truncate_abc
        gen = {json.loads(l)["tag"]: json.loads(l) for l in open(rt)}
        for n, r in per.items():
            g = gen.get(n[:-4])
            if not g:
                continue
            src = (Path("/cache/zhangjing/fugue/ss2/out") / r["id"] / "score.abc").read_text()
            dur = json.load(open(Path("/cache/zhangjing/fugue/ss2/out") / r["id"] / "result.json"))["duration_seconds"]
            ref = truncate_abc(src, 60 / dur) if dur > 60 else src
            for voice in ("Ins", "Vocal"):
                pr, pg = pitches(ref, voice), pitches(g.get("abc"), voice)
                r[f"dtw_{voice}"] = dtw_norm(np.diff(pr), np.diff(pg)) if len(pr) > 1 and len(pg) > 1 else None
        json.dump(per, open(pcache, "w"))

    # 4. 汇总
    arms = defaultdict(list)
    for n, r in per.items():
        arms[f"{r['arm']}-{r['dec']}"].append(r)
    keys = ["clap_style", "PQ", "CE", "hi8k", "hi12k", "rolloff99", "side_mid_db", "dtw_Ins"]
    summary = {}
    print(f"\n{'arm':22s} {'n':>4} {'Hit@1':>6} {'Hit@10':>6} {'MRR':>6} | " + " ".join(f"{k:>10}" for k in keys))
    for arm, rs in sorted(arms.items()):
        rk = {r["tag"]: r["vinet_rank"] for r in rs if r["vinet_rank"]}
        s = summarize(rk) if rk else {}
        med = {k: float(np.nanmedian([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None for k in keys}
        summary[arm] = dict(**s, median=med)
        print(f"{arm:22s} {len(rs):>4} {s.get('hit1', 0):>6.3f} {s.get('hit10', 0):>6.3f} {s.get('mrr', 0):>6.3f} | " + " ".join(f"{(med[k] if med[k] is not None else float('nan')):>10.4f}" for k in keys))
    su = summarize(upper); summary["upper_orig60"] = su
    print(f"{'upper: orig 60s':22s} {su['n']:>4} {su['hit1']:>6.3f} {su['hit10']:>6.3f} {su['mrr']:>6.3f}")
    # 配对:同 latent graft − yue2vae
    pair = defaultdict(dict)
    for n, r in per.items():
        pair[(r["tag"])][r["dec"]] = r
    for arm in ("fullscore", "noscore"):
        ps = [p for p in pair.values() if "graft" in p and "yue2vae" in p and p["graft"]["arm"] == arm]
        if not ps:
            continue
        d = {k: [p["graft"][k] - p["yue2vae"][k] for p in ps if p["graft"].get(k) is not None and p["yue2vae"].get(k) is not None] for k in keys}
        rk = [(p["graft"]["vinet_rank"], p["yue2vae"]["vinet_rank"]) for p in ps if p["graft"].get("vinet_rank") and p["yue2vae"].get("vinet_rank")]
        summary[f"paired_{arm}"] = {k: dict(median_diff=float(np.median(v)), graft_better=float(np.mean(np.array(v) > 0)), n=len(v)) for k, v in d.items() if v}
        summary[f"paired_{arm}"]["vinet_rank"] = dict(graft_le=float(np.mean([a <= b for a, b in rk])), n=len(rk))
        print(f"\npaired {arm} (graft − yue2vae), n={len(ps)}:")
        for k, v in summary[f"paired_{arm}"].items():
            print(f"  {k:14s} {v}")
    json.dump(summary, open(EV / "summary.json", "w"), indent=1)
    print("SCORE DONE", flush=True)


if __name__ == "__main__":
    main()
