
# bsz=$1
# prompt_len=$2
bsz=512
model_name="meta-llama/Llama-2-7b-hf"
gpu_name="GH200"
prompt_len=32

# # for offloading_ratio in 0 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
# for offloading_ratio in 1.0; do
#     if [[ "$offloading_ratio" == "0.08" ]]; then
#         cmd="CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_llama.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name} --use_config"
#     else
#         cmd="CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_llama.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name}"
#     fi
#     echo $cmd
#     eval $cmd
#     echo "###############################"
# done






# split attention + new kernel

host_cache_size=0
# for offloading_ratio in 0.05 0.1 0.25 0.5 0.75 1.0; do
for offloading_ratio in 1.0; do
    host_cache_size=$(awk -v bsz="$bsz" -v ratio="$offloading_ratio" 'BEGIN { printf "%.0f", bsz * ratio }')
    cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_llama_attn.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload 1.0 --gpu ${gpu_name} --use_config --max_new_tokens 32 --host_cache_size $host_cache_size --attn_impl vdcores_opt"
    echo $cmd
    eval $cmd
    echo "###############################"
done



# --host_cache_size $host_cache_size --attn_impl vdcores_opt