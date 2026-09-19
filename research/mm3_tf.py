"""SCION 砧木侧:teacher-forcing 复原 MM3 生成时的 frame_hiddens,并算到 ConditionEncoder 的 25Hz 输出 c25。

pairs8k 的 codes 就是 MM3 采样出来的码,把它们喂回 LM(prompt + 逐帧 embedding)与 depth 解码器
(每帧 8 位置 causal 前向),得到的 hidden 与生成时的 hidden 相同(同一序列,只差 batch 归约顺序)。
布局与 ar_batch_gen 的 hidden.pt 一致:hidden 第 j 行 = [LM 消费 codes[0..j] 后的 hidden ‖ 生成 codes[j+1]
的 c1..c7 时 depth 的 7 个 hidden],对应音频第 j 帧(codes 第 0 行是未发射的 priming 行)。

c25 = proj(layer_scale · softmax(w)·hidden)  —— ConditionEncoder 去掉最后的 nearest 上采样,(T-1, 2048)。
这是 adapter 的回归目标;DiT 看到的 condition 就是 c25 每帧重复 3.445 次。

用法:
  CUDA_VISIBLE_DEVICES=6 python mm3_tf.py --ids mf-1-000000 mf-1-000001 --save_hidden   # 实验 A 用
  CUDA_VISIBLE_DEVICES=6 python mm3_tf.py --split train --limit 2000                    # 批量出 c25
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/cache/zhangjing/fugue/selfdistill/v2")
from ar_batch_gen import MODEL, _AUDIO_CODE_OFFSET, assemble_prompt, embed_audio_frame_batched  # noqa: E402
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM  # noqa: E402
from diffusers.models import MiniMaxMusic3ConditionEncoder, MiniMaxMusic3RVQDepthDecoder  # noqa: E402

META = "/cache/zhangjing/fugue/ss2/pairs8k/meta.jsonl"
CODES = "/cache/zhangjing/fugue/ss2/pairs8k/codes.i16.npy"
OUT = Path("/cache/zhangjing/fugue/scion/pairs")
DEPTH_BATCH = 1024


def load_meta():
    rows = [json.loads(l) for l in open(META)]
    return {r["id"]: r for r in rows}


def song_codes(meta_row):
    codes = np.load(CODES, mmap_mode="r")
    return np.asarray(codes[meta_row["off"]: meta_row["off"] + meta_row["T"]]).astype(np.int64)


@torch.no_grad()
def tf_frame_hiddens(lm, depth, tokenizer, caption, lyrics, codes, device):
    """codes (T, 8) 含 priming 行。返回 frame_hiddens (T-1, 8*4096) bf16。"""
    ids = assemble_prompt(tokenizer, caption, lyrics).to(device)
    codes = torch.as_tensor(codes, dtype=torch.long, device=device)
    nc = int(depth.config.num_codebooks)
    av = int(depth.config.audio_vocab_size)
    frame_emb = embed_audio_frame_batched(lm, depth, codes[:-1], nc, av).transpose(0, 1)   # (1, T-1, 4096)
    inputs = torch.cat([lm.model.embed_tokens(ids)[None], frame_emb], dim=1)
    out = lm.model(inputs_embeds=inputs, use_cache=False)
    h = out.last_hidden_state[0, ids.shape[0]:]                                             # (T-1, 4096)

    nxt = codes[1:]
    parts = []
    for s in range(0, h.shape[0], DEPTH_BATCH):
        hb, cb = h[s: s + DEPTH_BATCH], nxt[s: s + DEPTH_BATCH]
        seq = [depth.projection(hb).unsqueeze(1),
               depth.projection(lm.model.embed_tokens(cb[:, 0] + _AUDIO_CODE_OFFSET)).unsqueeze(1)]
        for k in range(1, nc - 1):
            seq.append(depth.projection(depth.audio_embeddings(cb[:, k] + (k - 1) * av)).unsqueeze(1))
        dh = depth(torch.cat(seq, dim=1))                                                   # (b, 8, 4096)
        parts.append(dh[:, 1:nc].flatten(1))                                                # 位置 1..7 预测 c1..c7
    depth_hidden = torch.cat(parts, dim=0)
    return torch.cat([h, depth_hidden], dim=-1).to(torch.bfloat16)


@torch.no_grad()
def cond25(cond_enc, frame_hiddens):
    """ConditionEncoder 去掉 interpolate:(T, 8*4096) → (T, 2048)。"""
    T = frame_hiddens.shape[0]
    nl, hd = cond_enc.config.num_condition_layers, cond_enc.config.condition_hidden_dim
    x = frame_hiddens[None].to(cond_enc.dtype).transpose(1, 2).reshape(1, nl, hd, T)
    w = torch.softmax(cond_enc.layer_weight_logits, dim=0).to(x.dtype)
    x = torch.einsum("blht,l->bht", x, w) * cond_enc.layer_scale.to(x.dtype)
    return cond_enc.proj(x)[0].transpose(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--split", default=None, help="train/val/test;与 --ids 二选一")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_T", type=int, default=0)
    ap.add_argument("--save_hidden", action="store_true", help="另存 (T-1, 32768) bf16 的 frame_hiddens .pt")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    device = "cuda"
    (args.out / "c25").mkdir(parents=True, exist_ok=True)
    if args.save_hidden:
        (args.out / "hidden").mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    if args.ids:
        rows = [meta[i] for i in args.ids]
    else:
        rows = sorted((r for r in meta.values() if r["split"] == args.split and r["T"] >= args.min_T), key=lambda r: r["id"])
    rows = [r for r in rows if not (args.out / "c25" / f"{r['id']}.npy").exists() or args.save_hidden]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} songs to do", flush=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    lm = Qwen3ForCausalLM.from_pretrained(MODEL, subfolder="language_model", torch_dtype=torch.bfloat16).to(device).eval()
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(MODEL, subfolder="rvq_depth_decoder", torch_dtype=torch.bfloat16).to(device).eval()
    cond_enc = MiniMaxMusic3ConditionEncoder.from_pretrained(MODEL, subfolder="condition_encoder", torch_dtype=torch.bfloat16).to(device).eval()
    print(f"models resident: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    t0 = time.time()
    fails = 0
    for i, r in enumerate(rows):
        try:
            codes = song_codes(r)
            fh = tf_frame_hiddens(lm, depth, tokenizer, r["caption"], r["lyrics"], codes, device)
            c = cond25(cond_enc, fh)
            np.save(args.out / "c25" / f"{r['id']}.npy", c.float().cpu().numpy().astype(np.float16))
            if args.save_hidden:
                torch.save(fh.cpu(), args.out / "hidden" / f"{r['id']}.pt")
        except Exception as e:
            fails += 1
            print(f"FAIL {r['id']}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if (i + 1) % 20 == 0 or args.ids:
            el = time.time() - t0
            print(f"[{i + 1}/{len(rows)}] {r['id']} T={r['T']} c25 std={c.float().std():.4f} "
                  f"{el / (i + 1):.2f}s/song eta {el / (i + 1) * (len(rows) - i - 1) / 60:.0f}min", flush=True)
    print(f"DONE {len(rows) - fails} ok, {fails} fail, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
