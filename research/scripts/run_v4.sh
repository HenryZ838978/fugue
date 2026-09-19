#!/bin/bash
# v4 = v3 + 输入 latent 噪声增广 0.15(对齐推理时 NAR latent 的域差);GPU6,等 v3 结束
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6
while ! grep -q "^exit=" runs_v3.log 2>/dev/null; do sleep 60; done
/cache/zhangjing/miniconda3/envs/mm3/bin/python train_adapter.py --name v4 --layers 8 --d 768 --heads 12 --crop 768 --bs 24 --steps 20000 --eval_every 2000 --n_val 150 --workers 10 --var_weight 0.5 --x_noise 0.15 > runs_v4.log 2>&1
echo "exit=$?" >> runs_v4.log
