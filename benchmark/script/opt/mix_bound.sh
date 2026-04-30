bsz=512
model_name="facebook/opt-30b"
gpu_name="GH200"
prompt_len=32

# only offload model weights
host_cache_size=0
for offloading_ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt_attn.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --use_config --max_new_tokens 32 --host_cache_size $host_cache_size --attn_impl vdcores_opt"
    echo $cmd
    eval $cmd
    echo "###############################"
done

# start offloading kv cache
for offloading_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    host_cache_size=$(awk -v bsz="$bsz" -v ratio="$offloading_ratio" 'BEGIN { printf "%.0f", bsz * ratio }')
    cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt_attn.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload 1.0 --gpu ${gpu_name} --use_config --max_new_tokens 32 --host_cache_size $host_cache_size --attn_impl vdcores_opt"
    echo $cmd
    eval $cmd
    echo "###############################"
done





