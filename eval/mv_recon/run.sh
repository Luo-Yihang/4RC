#!/bin/bash
set -e

workdir='.'

model_name='4rc'
checkpoint="Luo-Yihang/4RC"

output_dir="${workdir}/eval_results/mv_recon/${model_name}/"
echo "$output_dir"

python eval/mv_recon/launch.py \
    --output_dir="$output_dir" \
    --size=518 \
    --model_name="arc" \
    --weights="$checkpoint" \
    --pi3_umeyama_align
