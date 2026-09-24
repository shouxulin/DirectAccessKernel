#!/bin/bash
# Usage: ./mem_bound.sh <gpu_name> <model_name>
#   e.g. ./mem_bound.sh GH200 facebook/opt-30b
#        ./mem_bound.sh RTX6000 facebook/opt-30b

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <gpu_name> <model_name>"
    exit 1
fi

gpu_name="${1^^}"
model_name="$2"
bsz=8
prompt_len=32

case "$gpu_name" in
    GH200)   peak_offload=0.08 ;;
    RTX6000) peak_offload=0.03 ;;
    *)
        echo "Unsupported gpu_name: $gpu_name (expected GH200 or RTX6000)"
        exit 1
        ;;
esac

# for offloading_ratio in 0 $peak_offload 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
for offloading_ratio in 0.03; do
    if [[ "$offloading_ratio" == "$peak_offload" ]]; then
        cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name} --use_config"
    else
        cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name}"
    fi
    echo $cmd
    eval $cmd
    echo "###############################"
done
