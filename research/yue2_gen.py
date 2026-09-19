"""SCION 接穗侧推理:YuE2 按外部 ABC(+style+lyrics)生成 → semantic token → NAR ODE → latent (T',64) 存盘
→ YuE2 自家 VAE 解码存 wav(作为"未嫁接"的对照)。graphtokenizer env。

--id 取 pairs8k 某首的 abc(SheetSage2 转谱,补回 Q: 行)/caption 首句当 style/lyrics;这样 YuE2 生成的是
"同一份谱面的 YuE2 版本",graft 后与 MM3 原曲对听,看旋律是否守住 + 音质归谁。

用法: CUDA_VISIBLE_DEVICES=7 python yue2_gen.py --id mf-1-000415 --max_sem 750 [--seed 0] [--cot off]
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import soundfile as sf
from yue2 import YuE2Pipeline
from yue2.protocol import Sampling, SongRequest

Y = "/cache/zhangjing/models/_ms/models/m-a-p--YuE2-3B/snapshots/master"
V = "/cache/zhangjing/models/_ms/models/m-a-p--YuE2-Vae/snapshots/master"
META = "/cache/zhangjing/fugue/ss2/pairs8k/meta.jsonl"
OUT = Path("/cache/zhangjing/fugue/scion/yue2gen")


def style_of(cap):
    m = re.search(r"Global Metadata:\s*(.+?)(?:\.\s|\n|$)", cap)
    s = (m.group(1) if m else cap.split("\n")[0])[:180]
    return re.sub(r"\s+", " ", s).strip(" .,")


def lyr_of(l):
    return re.sub(r"\[(\w+)\]", lambda m: "[" + m.group(1).capitalize() + "]", l.strip())


def with_tempo(abc, caption):
    """pairs8k 的 abc 去掉了 Q: 行;YuE2 自己写的谱带 Q:1/4=BPM,从 caption 补回。"""
    if any(l.startswith("Q:") for l in abc.split("\n")):
        return abc
    m = re.search(r"(\d{2,3})\s*BPM", caption)
    if not m:
        return abc
    lines = abc.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("L:")), 3)
    lines.insert(i + 1, f"Q:1/4={m.group(1)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None, help="pairs8k id:取其 abc/caption/lyrics")
    ap.add_argument("--abc_file", default=None, help="外部 ABC 文件(真曲 cover 用,与 --style/--lyrics 搭配)")
    ap.add_argument("--style", default=None)
    ap.add_argument("--lyrics", default="[Instrumental]")
    ap.add_argument("--max_sem", type=int, default=750)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cot", default="full", help="full=用外部 ABC;off=不给谱(对照)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.id:
        rows = {json.loads(l)["id"]: json.loads(l) for l in open(META)}
        r = rows[args.id]
        abc_src, style, lyrics, sid = with_tempo(r["abc"], r["caption"]), style_of(r["caption"]), lyr_of(r["lyrics"]), args.id
    else:
        abc_src, style, lyrics = open(args.abc_file).read(), args.style, args.lyrics
        sid = re.sub(r"[^A-Za-z0-9_.-]", "-", Path(args.abc_file).stem)[:60]
    tag = args.tag or f"{sid}.yue2{args.cot}.s{args.seed}"

    pipe = YuE2Pipeline.from_pretrained(Y, vae=V, local_files_only=True, progress=False)
    abc = abc_src if args.cot != "off" else None
    req = SongRequest(style=style, lyrics=lyrics, cot=args.cot, abc=abc, seed=args.seed, id=sid)
    t0 = time.time()
    plan = pipe.plan(request=req)
    sem = pipe.generate_semantic(plan, sampling=Sampling(max_tokens=args.max_sem, min_tokens=min(200, args.max_sem)))
    t_ar = time.time() - t0
    lat = pipe.synthesize(sem)                                                     # (T', 64) f32
    t_nar = time.time() - t0 - t_ar
    np.save(OUT / f"{tag}.latent.npy", lat.astype(np.float32))
    wav = pipe.decode(lat)                                                         # (S, 2) 48k
    sf.write(OUT / f"{tag}.yue2vae.wav", wav, 48000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT / f"{tag}.yue2vae.wav"), "-b:a", "160k", str(OUT / f"{tag}.yue2vae.mp3")], check=True)
    # 域差缓解候选:生成的 latent 过一遍 VAE decode→encode,投影回 encoder 流形(adapter 训练时见的是 encoder 输出)
    from yue2.modeling_vae import YuE2VAE
    import torch
    vae = YuE2VAE.from_pretrained(V, device=pipe.device, local_files_only=True).eval()
    with torch.inference_mode():
        lat_re = vae.encode(torch.from_numpy(wav.T.copy())[None].to(pipe.device))[0].T.cpu().numpy()
    np.save(OUT / f"{tag}.latent_re.npy", lat_re.astype(np.float32))
    re_diff = float(np.abs(lat_re[: len(lat)] - lat[: len(lat_re)]).mean())
    info = dict(id=sid, tag=tag, cot=args.cot, seed=args.seed, style=req.style, abc_tokens=len(plan.abc_ids), semantic_tokens=len(sem.tokens),
                truncated=bool(sem.truncated), latent_frames=int(lat.shape[0]), lat_std=round(float(lat.std()), 4), reencode_mean_abs_diff=round(re_diff, 4),
                t_ar=round(t_ar, 1), t_nar=round(t_nar, 1), abc_head=(abc or "")[:200], semantic=list(map(int, sem.tokens)))
    (OUT / f"{tag}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "semantic"}, ensure_ascii=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
