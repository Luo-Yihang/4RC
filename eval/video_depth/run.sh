#!/bin/bash
set -e

workdir='.'

datasets=('sintel' 'bonn' 'kitti')
model_name='4rc'
checkpoint="Luo-Yihang/4RC"

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${model_name}/${data}"
    echo "$output_dir"

    python eval/video_depth/launch.py \
    --output_dir="$output_dir" \
    --eval_dataset="$data" \
    --checkpoint="$checkpoint"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale&shift"
done