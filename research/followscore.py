"""SCION 闭环:follow/ 目录里各变体 wav 经 SheetSage2 转谱(ss2/retrans.py → retrans.jsonl)后,
与输入 ABC(pairs8k meta 的 abc,截到生成时长)比 DTW 归一化音程距离 + 音级直方图余弦。
函数抄 ss2/followscore.py(那是脚本,top-level argparse,不可 import);Vocal / Ins 两个声部都算
(器乐曲 Vocal 全休止,只有 Ins 有意义)。

读法:graftfull 的 dtw 应接近 yue2full(嫁接没丢 YuE2 读进去的谱),且显著低于 graftoff / zero。

用法: python followscore.py [--dir follow]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/cache/zhangjing/fugue/ss2")
from abcutil import parse_blocks, truncate_abc  # noqa: E402
from abcprobe import strip_chords  # noqa: E402

META = "/cache/zhangjing/fugue/ss2/pairs8k/meta.jsonl"
BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE = re.compile(r"(?P<acc>[\^_=]*)(?P<n>[A-Ga-g])(?P<oct>[,']*)")


def pitches(abc, voice):
    if not abc:
        return np.array([], dtype=float)
    _, blocks = parse_blocks(strip_chords(abc))
    out = []
    for _cm, name, body in blocks:
        if name != voice:
            continue
        for line in body:
            for m in NOTE.finditer(line):
                n = m.group("n")
                p = BASE[n.upper()] + (72 if n.islower() else 60)
                for c in m.group("oct"):
                    p += 12 if c == "'" else -12
                a = m.group("acc")
                p += a.count("^") - a.count("_")
                out.append(p)
    return np.asarray(out, dtype=float)


def dtw_norm(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    C = np.abs(a[:, None] - b[None, :])
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Ci, Dp, Dc = C[i - 1], D[i - 1], D[i]
        for j in range(1, m + 1):
            Dc[j] = Ci[j - 1] + min(Dp[j], Dc[j - 1], Dp[j - 1])
    return float(D[n, m] / (n + m))


def pc_cos(a, b):
    ha, hb = np.zeros(12), np.zeros(12)
    for x in a:
        ha[int(x) % 12] += 1
    for x in b:
        hb[int(x) % 12] += 1
    return float(ha @ hb / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/cache/zhangjing/fugue/scion/follow")
    ap.add_argument("--abc_override", nargs="*", default=[], help="sid=path.abc:dur 用外部谱(真曲)")
    args = ap.parse_args()
    d = Path(args.dir)
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(META)}
    over = {}
    for s in args.abc_override:
        sid, rest = s.split("=", 1)
        p, dur = rest.rsplit(":", 1)
        over[sid] = (open(p).read(), float(dur))
    gen = {json.loads(l)["tag"]: json.loads(l) for l in open(d / "retrans.jsonl")}
    rows = []
    for tag in sorted(gen):
        if "__" not in tag:
            continue
        sid, var = tag.rsplit("__", 1)
        if sid in over:
            abc_full, dur = over[sid]
        elif sid in meta:
            abc_full, dur = meta[sid]["abc"], float(meta[sid]["dur"])
        else:
            continue
        info = sf.info(d / f"{tag}.wav")
        gsec = info.frames / info.samplerate
        ref_abc = truncate_abc(abc_full, gsec / dur) if dur > gsec else abc_full
        g_abc = gen[tag].get("abc")
        r = dict(id=sid, var=var, gsec=round(gsec, 1))
        for voice in ("Vocal", "Ins"):
            pr, pg = pitches(ref_abc, voice), pitches(g_abc, voice)
            r[f"{voice}_n_ref"], r[f"{voice}_n_gen"] = len(pr), len(pg)
            r[f"{voice}_dtw"] = round(dtw_norm(np.diff(pr), np.diff(pg)), 3)
            r[f"{voice}_pc"] = round(pc_cos(pr, pg), 3)
        rows.append(r)
    print(f"{'id':<14} {'var':<12} {'sec':>5} | {'Voc n_ref':>9} {'n_gen':>5} {'dtw':>6} {'pc':>6} | {'Ins n_ref':>9} {'n_gen':>5} {'dtw':>6} {'pc':>6}")
    for r in rows:
        print(f"{r['id']:<14} {r['var']:<12} {r['gsec']:>5} | {r['Vocal_n_ref']:>9} {r['Vocal_n_gen']:>5} {r['Vocal_dtw']:>6} {r['Vocal_pc']:>6} | "
              f"{r['Ins_n_ref']:>9} {r['Ins_n_gen']:>5} {r['Ins_dtw']:>6} {r['Ins_pc']:>6}")
    json.dump(rows, open(d / "followscore.json", "w"), indent=1)


if __name__ == "__main__":
    main()
