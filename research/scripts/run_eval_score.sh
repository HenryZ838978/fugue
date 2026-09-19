#!/bin/bash
# 等两个 shard 生成完 → 回转谱(GPU0)→ 打分(GPU0)
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
while ! (grep -q "GEN DONE" evalgen.0.log && grep -q "GEN DONE" evalgen.1.log); do sleep 60; done
cd /cache/zhangjing/fugue/ss2 && /cache/zhangjing/miniconda3/envs/graphtokenizer/bin/python retrans.py --dir /cache/zhangjing/fugue/scion/eval/gen > /cache/zhangjing/fugue/scion/eval_retrans.log 2>&1
cd /cache/zhangjing/fugue/scion && /cache/zhangjing/miniconda3/envs/mm3/bin/python eval_score.py > eval_score.log 2>&1
echo "exit=$?" >> eval_score.log
