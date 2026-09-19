#!/bin/bash
cd /cache/zhangjing/fugue/scion
P=/cache/zhangjing/miniconda3/envs/mm3/bin/python
$P fugue_cover.py --audio /cache/zhangjing/fugue/refs_full/02_004-s-ave.flac --style "intimate solo piano ballad with soft strings, 75 BPM, A minor, cinematic, instrumental" --instrumental --tag s-ave.full.piano --mp3 > full1.log 2>&1
echo "exit=$?" >> full1.log
