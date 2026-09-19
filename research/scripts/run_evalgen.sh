#!/bin/bash
cd /cache/zhangjing/fugue/scion
P=/cache/zhangjing/miniconda3/envs/mm3/bin/python
$P eval_gen.py --shard 0 --nshards 2 --svc_a 8650 --svc_b 8651 > evalgen.0.log 2>&1 &
$P eval_gen.py --shard 1 --nshards 2 --svc_a 8660 --svc_b 8661 > evalgen.1.log 2>&1 &
wait
