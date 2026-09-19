"""SCION 阶段 1/2:adapter 训练 —— YuE2 latent (T,64) → MM3 c25 (T,2048),纯回归。

结构:Linear(64→d) → 2×Conv1d(k=5) 局部前端 → 深度可分 conv 相对位置(wav2vec2 式,无绝对位置,任意长度可推)
      → N 层双向 Transformer(pre-LN) → LayerNorm → Linear(d→2048)。
损失:标准化空间(每通道 z-score)masked MSE;验证报原始空间 R²var(DiT 看到的尺度)与 per-channel R²。
--linear 退化为纯线性基线(阶段 1 的"薄 adapter")。

用法: CUDA_VISIBLE_DEVICES=7 python train_adapter.py --name v1 --layers 6 --d 512 --steps 12000
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAIRS, CropDataset, compute_stats, load_meta, load_pair, load_stats

RUNS = Path("/cache/zhangjing/fugue/scion/runs")


class Adapter(nn.Module):
    def __init__(self, d_in=64, d=512, d_out=2048, layers=6, heads=8, dropout=0.1, linear=False):
        super().__init__()
        self.linear = linear
        if linear:
            self.lin = nn.Conv1d(d_in, d_out, kernel_size=3, padding=1)        # = MM3 自己 proj 的形状
            return
        self.inp = nn.Linear(d_in, d)
        self.front = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2))
        self.pos = nn.Sequential(nn.Conv1d(d, d, 63, padding=31, groups=16), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_out))

    def forward(self, x, mask=None):
        """x (B, T, 64) 标准化后;mask (B, T) 1=有效。返回 (B, T, 2048) 标准化空间。"""
        if self.linear:
            return self.lin(x.transpose(1, 2)).transpose(1, 2)
        h = self.inp(x).transpose(1, 2)
        h = h + self.front(h)
        h = h + self.pos(h)
        h = h.transpose(1, 2)
        kpm = None if mask is None else (mask < 0.5)
        h = self.blocks(h, src_key_padding_mask=kpm)
        return self.out(h)


@torch.no_grad()
def evaluate(model, rows, stats, dev, max_T=2000):
    model.eval()
    xm, xs = torch.from_numpy(stats["x_mean"]).to(dev), torch.from_numpy(stats["x_std"]).to(dev)
    ym, ys = torch.from_numpy(stats["y_mean"]).to(dev), torch.from_numpy(stats["y_std"]).to(dev)
    res = tot = 0.0
    res_ch = torch.zeros(2048, device=dev, dtype=torch.float64); tot_ch = torch.zeros_like(res_ch)
    gm = torch.from_numpy(stats["y_mean"]).to(dev).double()
    for r in rows:
        x, y = load_pair(r)
        x, y = torch.from_numpy(x[:max_T]).to(dev), torch.from_numpy(y[:max_T]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(((x - xm) / xs)[None])[0].float()
        p = p * ys + ym
        d = ((y.double() - p.double()) ** 2).sum(0); t = ((y.double() - gm) ** 2).sum(0)
        res_ch += d; tot_ch += t
    r2_ch = 1 - res_ch / tot_ch
    model.train()
    return dict(r2_var=float(1 - res_ch.sum() / tot_ch.sum()), r2_chmean=float(r2_ch.mean()),
                r2_ch_p10=float(r2_ch.quantile(0.1)), r2_ch_p90=float(r2_ch.quantile(0.9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--var_weight", type=float, default=0.0,
                    help="通道损失权重 w_c=(1-α)+α·std_c²/mean(std²):0=标准化 MSE(各通道等权),1=原始尺度 MSE(=R²var 的权重)")
    ap.add_argument("--x_noise", type=float, default=0.0,
                    help="训练时给输入 latent 加高斯噪声(标准化单位);推理时 latent 来自 YuE2 NAR 而非 VAE encoder,decode→encode 往返差 ≈0.15")
    args = ap.parse_args()
    dev = "cuda"
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=1))

    tr = load_meta("train")
    va = load_meta("val")[: args.n_val]
    print(f"train {len(tr)} songs, val {len(va)} songs", flush=True)
    stats_path = PAIRS / "stats.npz"
    stats = load_stats(stats_path) if stats_path.exists() else compute_stats(tr, n=400, path=stats_path)
    np.savez(run / "stats.npz", **stats)

    model = Adapter(d=args.d, layers=args.layers, heads=args.heads, linear=args.linear).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    ds = CropDataset(tr, stats, crop=args.crop, size=args.steps * args.bs)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, drop_last=True, persistent_workers=True)

    log = open(run / "train.log", "a")
    ys2 = torch.from_numpy(stats["y_std"] ** 2).to(dev)
    ch_w = (1 - args.var_weight) + args.var_weight * ys2 / ys2.mean()            # (2048,) 和为 2048
    best = -1e9
    t0 = time.time()
    ema_loss = None
    for step, (x, y, m) in enumerate(dl, 1):
        x, y, m = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        if args.x_noise > 0:
            x = x + args.x_noise * torch.randn_like(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x, m)
        loss = ((((p.float() - y) ** 2) * ch_w).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        if step % 100 == 0:
            rec = dict(step=step, loss=round(ema_loss, 4), gn=round(float(gn), 3), lr=f"{sched.get_last_lr()[0]:.2e}", min=round((time.time() - t0) / 60, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, va, stats, dev)
            rec = dict(step=step, eval={k: round(v, 4) for k, v in ev.items()})
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save(dict(model=model.state_dict(), args=vars(args), step=step, eval=ev), run / "last.pt")
            if ev["r2_var"] > best:
                best = ev["r2_var"]
                torch.save(dict(model=model.state_dict(), args=vars(args), step=step, eval=ev), run / "best.pt")
        if step >= args.steps:
            break
    print(f"DONE best r2_var={best:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
