"""SCION eval 生成:ood216 真曲(210 首有 SheetSage2 谱)× 2 目标 style,三臂:
  fullscore  YuE2 供谱(cot=full)→ latent → ① YuE2-Vae 解码  ② v4 嫁接
  noscore    YuE2 无谱(cot=off,只有 style)→ 同样两种解码(只跑 style A,作"身份应归零"的对照)
每首截前 --seconds 秒(默认 60,=1500 帧)。产物 eval/gen/<id>__<style>__<arm>.{yue2vae,graft}.wav + .json。
调两个常驻服务;--svc_a/--svc_b 指定端口以便双卡并行;--shard i --nshards n 切分工作表;可重入(已有产物跳过)。

用法: python eval_gen.py --shard 0 --nshards 2 --svc_a 8650 --svc_b 8651
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import requests

ROOT = Path("/cache/zhangjing/fugue/scion")
SS2OUT = Path("/cache/zhangjing/fugue/ss2/out")
OUT = ROOT / "eval" / "gen"
STYLES = [
    "intimate solo piano ballad with soft strings, cinematic, instrumental",
    "1990s Britpop guitar rock, driving drums, instrumental",
    "smooth jazz trio, upright bass, brushed drums, Rhodes piano, instrumental",
    "synthwave, retro analog synths, punchy drum machine, instrumental",
    "acoustic folk, fingerpicked guitar, warm and gentle, instrumental",
    "orchestral film score, full strings and brass, epic, instrumental",
]


def worklist(ids, seconds):
    items = []
    for i, sid in enumerate(ids):
        sa, sb = i % 6, (i + 3) % 6
        items.append(dict(id=sid, style_id=sa, arm="fullscore"))
        items.append(dict(id=sid, style_id=sb, arm="fullscore"))
        items.append(dict(id=sid, style_id=sa, arm="noscore"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--svc_a", type=int, default=8650)
    ap.add_argument("--svc_b", type=int, default=8651)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    A, B = f"http://127.0.0.1:{args.svc_a}", f"http://127.0.0.1:{args.svc_b}"
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.load(open(ROOT / "ood_ids.json"))
    items = [it for k, it in enumerate(worklist(ids, args.seconds)) if k % args.nshards == args.shard]
    if args.limit:
        items = items[: args.limit]
    print(f"[shard {args.shard}/{args.nshards}] {len(items)} items", flush=True)
    t_all = time.time()
    done = 0
    for n, it in enumerate(items):
        tag = f"{it['id']}__S{it['style_id']}__{it['arm']}"
        if (OUT / f"{tag}.json").exists():
            continue
        style = STYLES[it["style_id"]]
        abc = (SS2OUT / it["id"] / "score.abc").read_text()
        t0 = time.time()
        try:
            g = requests.post(f"{A}/generate", json=dict(style=style, lyrics="[Instrumental]", abc=abc if it["arm"] == "fullscore" else None,
                                                         cot="full" if it["arm"] == "fullscore" else "off", seed=args.seed,
                                                         max_sem=int(args.seconds * 25), decode_yue2=True, tag=tag), timeout=3600).json()
            if "latent_path" not in g:
                raise RuntimeError(f"generate: {g}")
            gr = requests.post(f"{B}/graft", json=dict(latent_path=g["latent_path"], seed=7, steps=30, tag=tag), timeout=3600).json()
            if "wav_path" not in gr:
                raise RuntimeError(f"graft: {gr}")
            shutil.move(g["yue2_wav_path"], OUT / f"{tag}.yue2vae.wav")
            shutil.move(gr["wav_path"], OUT / f"{tag}.graft.wav")
            Path(g["latent_path"]).unlink(missing_ok=True)
            rec = dict(tag=tag, **it, style=style, seed=args.seed, seconds=args.seconds,
                       yue2={k: g.get(k) for k in ("abc_tokens", "semantic_tokens", "truncated", "latent_frames", "t_ar", "t_nar", "t_dec")},
                       graft={k: gr.get(k) for k in ("frames", "sec", "decode_sec", "run")}, item_sec=round(time.time() - t0, 1))
            (OUT / f"{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            done += 1
            el = time.time() - t_all
            print(f"[{n + 1}/{len(items)}] {tag} {rec['item_sec']}s  avg {el / done:.0f}s/item  eta {el / done * (len(items) - n - 1) / 60:.0f} min", flush=True)
        except Exception as e:
            print(f"FAIL {tag}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            (OUT / f"{tag}.fail").write_text(str(e))
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    main()
