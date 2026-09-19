"""Fugue cover 编排 CLI:真曲 → SheetSage2 → YuE2(换 style)→ adapter v4 → MM3 DiT/vocoder。调两个常驻服务(8650 / 8651)。

  python fugue_cover.py --audio song.flac --style "intimate solo piano ballad, 75 BPM, A minor" \
      [--lyrics lyrics.txt | --instrumental] [--cot full|melody] [--seed 0] [--seconds 0=整曲] [--out outdir] [--tag name]
  → outdir/<tag>.{abc,graft.wav,yue2vae.wav,json}

--abc 可直接给谱(跳过转谱);--no_yue2_decode 省掉 A/B 对照解码。
style 写法与 YuE2 一致:逗号分隔的自由 tag,"流派/年代, 乐器, 情绪, BPM, 调式, 拍号",英文。
lyrics 写法:段落标记 [Intro] [Verse] [Chorus] [Bridge] [Outro],每行一句;器乐曲写 [Instrumental]。
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests

A = "http://127.0.0.1:8650"
B = "http://127.0.0.1:8651"


def with_tempo(abc, bpm):
    if any(l.startswith("Q:") for l in abc.split("\n")) or not bpm:
        return abc
    lines = abc.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("L:")), 3)
    lines.insert(i + 1, f"Q:1/4={bpm}")
    return "\n".join(lines)


def strip_chords(abc):
    """cot=melody 用:去掉 "Chord" 标记。"""
    return re.sub(r'"[^"]*"', "", abc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--lyrics", default=None, help="文件路径或直接文本")
    ap.add_argument("--instrumental", action="store_true")
    ap.add_argument("--abc", default=None, help="直接给 ABC 文件,跳过转谱")
    ap.add_argument("--cot", default="full", choices=["full", "melody"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0, help="0 = 整曲;否则截前 N 秒(25 帧/秒)")
    ap.add_argument("--dit_seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="covers")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no_yue2_decode", action="store_true")
    ap.add_argument("--mp3", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = Path(args.audio)
    tag = args.tag or re.sub(r"[^A-Za-z0-9_.-]", "-", src.stem)[:50] + f".s{args.seed}"
    t_all = time.time()
    rec = dict(tag=tag, audio=str(src), style=args.style, cot=args.cot, seed=args.seed)

    # 1. 谱
    if args.abc:
        abc = Path(args.abc).read_text(); rec["abc_source"] = args.abc
    else:
        r = requests.post(f"{A}/transcribe", json=dict(audio_path=str(src.resolve())), timeout=1800).json()
        if not r.get("abc"):
            raise SystemExit(f"transcribe failed: {r}")
        abc = r["abc"]
        rec.update(transcribe={k: r.get(k) for k in ("sec", "vocal_notes", "instrumental_notes", "abc_measures", "key_lab", "duration_seconds")})
        m = re.search(r"(\d{2,3})\s*BPM", args.style)
        abc = with_tempo(abc, m.group(1) if m else None)
    if args.cot == "melody":
        abc = strip_chords(abc)
    (out / f"{tag}.abc").write_text(abc)

    # 2. 词
    if args.instrumental or not args.lyrics:
        lyrics = "[Instrumental]"
    else:
        lyrics = Path(args.lyrics).read_text() if Path(args.lyrics).exists() else args.lyrics
    rec["lyrics_head"] = lyrics[:120]

    # 3. YuE2
    max_sem = int(args.seconds * 25) if args.seconds > 0 else 9000
    g = requests.post(f"{A}/generate", json=dict(style=args.style, lyrics=lyrics, abc=abc, cot=args.cot, seed=args.seed,
                                                 max_sem=max_sem, decode_yue2=not args.no_yue2_decode, tag=tag), timeout=7200).json()
    if "latent_path" not in g:
        raise SystemExit(f"generate failed: {g}")
    rec["yue2"] = {k: g.get(k) for k in ("abc_tokens", "semantic_tokens", "truncated", "latent_frames", "t_ar", "t_nar", "t_dec")}
    if g.get("yue2_wav_path"):
        shutil.copy(g["yue2_wav_path"], out / f"{tag}.yue2vae.wav")

    # 4. graft
    gr = requests.post(f"{B}/graft", json=dict(latent_path=g["latent_path"], seed=args.dit_seed, steps=args.steps, tag=tag), timeout=7200).json()
    if "wav_path" not in gr:
        raise SystemExit(f"graft failed: {gr}")
    shutil.copy(gr["wav_path"], out / f"{tag}.graft.wav")
    rec["graft"] = {k: gr.get(k) for k in ("frames", "sec", "decode_sec", "run")}
    rec["total_sec"] = round(time.time() - t_all, 1)
    (out / f"{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    if args.mp3:
        for w in out.glob(f"{tag}.*.wav"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", "160k", str(w.with_suffix(".mp3"))], check=True)
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
