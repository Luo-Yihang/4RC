#!/bin/bash
set -e

# Dense Tracking Evaluation Script

workdir='.'
checkpoint="Luo-Yihang/4RC"
model_type="4rc"

# Dense tracking datasets
datasets=('waymo' 'kubric')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/dense_track/${data}"
    echo "Evaluating dense tracking on ${data}..."
    echo "Output dir: $output_dir"

    python eval/track/launch.py \
        --checkpoint_dir="$checkpoint" \
        --output_dir="$output_dir" \
        --data_type="$data" \
        --data_root="./data" \
        --num_frames=64 \
        --pose_eval_stride=1 \
        --size=512 \
        --patch_size=14 \
        --model_type="$model_type" \
        --track_query_idx=11 \
        --align="sim3" \
        --save_predictions \
        --seed=42

    echo "✓ Completed $data"
    echo "---"
done

echo "✅ Dense tracking evaluation completed!"
echo "Results saved in: eval_results/dense_track/"
