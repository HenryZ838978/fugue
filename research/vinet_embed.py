"""Discogs-VINet(ISMIR 2024,raraz15)身份嵌入:音频 → 22.05k mono → CQT(hop 512, 84 bins)→ pad 8000 → mean-downsample 20 →
max 归一 → CQTNet → 512-d L2 向量。与官方 inference.py 等价,只是不依赖 essentia(用 librosa 读音频)。
checkpoint 的键名 features.* / proj.0.weight 是旧版命名,这里映射到当前代码的 front_end.* / proj.lin.weight。

用法: python vinet_embed.py --out eval/vinet_orig.npz --list ood_ids.json   (原曲整曲)
      python vinet_embed.py --out eval/vinet_gen.npz --glob 'eval/gen/*.wav'   (生成物)
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

VINET = "/cache/zhangjing/models/discogs-vinet"
sys.path.insert(0, VINET)
from model.nets import CQTNet  # noqa: E402

AUDIO = Path("/cache/zhangjing/MusicGraph/corpus/audio")
SR, HOP, BINS, BPO, CTX, DS = 22050, 512, 84, 12, 8000, 20


def load_vinet(dev):
    m = CQTNet(ch_in=32, ch_out=512, norm="bn", pool="adaptive_max", l2_normalize=True, projection="linear")
    sd = torch.load(f"{VINET}/logs/checkpoints/Discogs-VINet/model_checkpoint.pth", map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k.replace("features.", "front_end.").replace("proj.0.weight", "proj.lin.weight"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval()


def cqt_feat(path, max_sec=None):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=max_sec)
    c = np.abs(librosa.cqt(y=y, sr=SR, hop_length=HOP, n_bins=BINS, bins_per_octave=BPO)).T.astype(np.float16).astype(np.float32)   # (T, F)
    if c.shape[0] < CTX:
        c = np.pad(c, ((0, CTX - c.shape[0]), (0, 0)))
    T = (c.shape[0] // DS) * DS
    c = c[:T].reshape(-1, DS, BINS).mean(1)
    c = np.clip(c, 0, None)
    c /= c.max() + 1e-6
    return torch.from_numpy(c.T)                                                        # (F, T')


@torch.no_grad()
def embed(m, feat, dev):
    with torch.autocast(dev, dtype=torch.float16):
        return m(feat[None, None].to(dev)).float().cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=None, help="json id 列表 → MusicGraph 原曲 .m4a")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--max_sec", type=float, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_vinet(dev)
    if args.list:
        files = [(i, str(AUDIO / f"{i}.m4a")) for i in json.load(open(args.list))]
    else:
        files = [(Path(p).name, p) for p in sorted(glob.glob(args.glob))]
    names, embs = [], []
    for k, (name, p) in enumerate(files):
        try:
            embs.append(embed(m, cqt_feat(p, args.max_sec), dev)); names.append(name)
        except Exception as e:
            print(f"FAIL {name}: {e}", flush=True)
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(files)}", flush=True)
    np.savez(args.out, names=np.array(names), emb=np.stack(embs))
    print("DONE", len(names), flush=True)


if __name__ == "__main__":
    main()
