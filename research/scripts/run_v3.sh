#!/bin/bash
# v3 = v2 同结构,通道损失按方差加权 α=0.5(向 R²var / DiT 实际尺度倾斜);GPU6,与 v2 并行
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6
/cache/zhangjing/miniconda3/envs/mm3/bin/python train_adapter.py --name v3 --layers 8 --d 768 --heads 12 --crop 768 --bs 24 --steps 20000 --eval_every 2000 --n_val 150 --workers 10 --var_weight 0.5 > runs_v3.log 2>&1
echo "exit=$?" >> runs_v3.log
