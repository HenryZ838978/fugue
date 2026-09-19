#!/bin/bash
# 抽卡方差:① 同一 YuE2 latent 换 DiT seed ×4(v4);② 同一 ABC 换 YuE2 seed ×2,各走 YuE2-VAE 与 v4 嫁接;③ 全部打质量分
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
P=/cache/zhangjing/miniconda3/envs/mm3/bin/python
G=/cache/zhangjing/miniconda3/envs/graphtokenizer/bin/python
{
for s in 11 12 13 14; do
  $P graft_decode.py --run v4 --latent yue2gen/mf-1-000415.yue2full.s0.latent.npy --seed $s --tag mf-1-000415.yue2full.s0.v4.dit$s
done
for ys in 1 2; do
  $G yue2_gen.py --id mf-1-000415 --max_sem 750 --seed $ys
  $P graft_decode.py --run v4 --latent yue2gen/mf-1-000415.yue2full.s$ys.latent.npy --tag mf-1-000415.yue2full.s$ys.v4
done
$P quality_score.py
} > seeds.log 2>&1
echo "SEEDS DONE" >> seeds.log
