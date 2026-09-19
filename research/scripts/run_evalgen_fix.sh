#!/bin/bash
cd /cache/zhangjing/fugue/scion
P=/cache/zhangjing/miniconda3/envs/mm3/bin/python
$P eval_gen.py --shard 0 --nshards 2 --svc_a 8650 --svc_b 8651 > evalgen_fix.0.log 2>&1 &
$P eval_gen.py --shard 1 --nshards 2 --svc_a 8660 --svc_b 8661 > evalgen_fix.1.log 2>&1 &
wait
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
cd /cache/zhangjing/fugue/ss2 && /cache/zhangjing/miniconda3/envs/graphtokenizer/bin/python retrans.py --dir /cache/zhangjing/fugue/scion/eval/gen >> /cache/zhangjing/fugue/scion/eval_retrans.log 2>&1
cd /cache/zhangjing/fugue/scion && /cache/zhangjing/miniconda3/envs/mm3/bin/python eval_score.py > eval_score.log 2>&1
echo "exit=$?" >> eval_score.log
echo "FIX ALL DONE" >> evalgen_fix.0.log
