"""Fugue end-to-end cover CLI (single process, single GPU: YuE2 ~7 GB + MM3 acoustic side ~5 GB in bf16).

  python -m fugue.cover --audio song.flac --style "intimate solo piano ballad, cinematic" --out out/
  python -m fugue.cover --abc score.abc --style "1990s Britpop guitar rock" --lyrics lyrics.txt --out out/

Model locations come from env vars (or flags): FUGUE_MM3, FUGUE_YUE2, FUGUE_YUE2_VAE, FUGUE_SHEETSAGE2, FUGUE_MERT, FUGUE_ADAPTER.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import soundfile as sf

from .adapter import FugueAdapter
from .graft import Rootstock
from .scion import Scion, Transcriber, strip_chords, with_tempo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="song to cover (transcribed with SheetSage2)")
    ap.add_argument("--abc", default=None, help="or: ABC score file (skip transcription)")
    ap.add_argument("--style", required=True)
    ap.add_argument("--lyrics", default=None, help="file or text; omit for instrumental")
    ap.add_argument("--cot", default="full", choices=["full", "melody"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0, help="0 = full song")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--dit_seed", type=int, default=7)
    ap.add_argument("--out", default="fugue_out")
    ap.add_argument("--native", action="store_true", help="also write YuE2's own VAE decode for A/B")
    ap.add_argument("--mm3", default=os.environ.get("FUGUE_MM3", "MiniMaxAI/MiniMax-Music3"))
    ap.add_argument("--yue2", default=os.environ.get("FUGUE_YUE2", "m-a-p/YuE2-3B"))
    ap.add_argument("--yue2_vae", default=os.environ.get("FUGUE_YUE2_VAE", "m-a-p/YuE2-Vae"))
    ap.add_argument("--sheetsage2", default=os.environ.get("FUGUE_SHEETSAGE2", "m-a-p/SheetSage2"))
    ap.add_argument("--mert", default=os.environ.get("FUGUE_MERT", "m-a-p/MERT-v2-FullSong"))
    ap.add_argument("--adapter", default=os.environ.get("FUGUE_ADAPTER", "HenryZ838978/fugue-scion-v4"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "-", Path(args.audio or args.abc).stem)[:50] + f".s{args.seed}"
    t0 = time.time(); rec = dict(tag=tag, style=args.style, cot=args.cot, seed=args.seed)

    if args.abc:
        abc = Path(args.abc).read_text()
    else:
        tr = Transcriber(args.sheetsage2, args.mert)
        r = tr(args.audio)
        if not r["abc"]:
            raise SystemExit(f"transcription failed: {r}")
        abc = r["abc"]; rec["transcribe"] = {k: v for k, v in r.items() if k != "abc"}
        del tr
    m = re.search(r"(\d{2,3})\s*BPM", args.style)
    abc = with_tempo(abc, m.group(1) if m else None)
    if args.cot == "melody":
        abc = strip_chords(abc)
    (out / f"{tag}.abc").write_text(abc)
    lyrics = "[Instrumental]" if not args.lyrics else (Path(args.lyrics).read_text() if Path(args.lyrics).exists() else args.lyrics)

    scion = Scion(args.yue2, args.yue2_vae)
    t1 = time.time()
    lat, info = scion.generate(args.style, lyrics, abc, args.cot, args.seed, int(args.seconds * 25) if args.seconds > 0 else 9000)
    rec["yue2"] = dict(**info, sec=round(time.time() - t1, 1))
    if args.native:
        wav, sr = scion.decode_native(lat); sf.write(out / f"{tag}.yue2_native.wav", wav, sr)

    adapter = FugueAdapter(args.adapter)
    root = Rootstock(args.mm3)
    t2 = time.time()
    wav = root(adapter(lat), steps=args.steps, seed=args.dit_seed)
    sf.write(out / f"{tag}.fugue.wav", wav.T.cpu().numpy(), root.sr)
    rec["graft"] = dict(frames=int(lat.shape[0]), sec=round(time.time() - t2, 1)); rec["total_sec"] = round(time.time() - t0, 1)
    (out / f"{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
