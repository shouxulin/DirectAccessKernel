
# bsz=$1
# prompt_len=$2
bsz=8
model_name="facebook/opt-30b"
gpu_name="GH200"


prompt_len=32
while [ $prompt_len -le 32 ]; do
    # for offloading_ratio in 0 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for offloading_ratio in 0.08; do
        if [[ "$offloading_ratio" == "0.08" ]]; then
            cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name} --use_config"
        else
            cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name}"
        fi
        echo $cmd
        eval $cmd
        echo "###############################"
    done

    prompt_len=$((prompt_len * 2))
done


# # split attention + old kernel
# prompt_len=1024
# host_cache_size=80
# while [ $prompt_len -le 1024 ]; do
#     # for offloading_ratio in 0 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#     for offloading_ratio in 1; do
#         if [[ "$offloading_ratio" == "0.08" ]]; then
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:25 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --use_config -attn_impl vdcores_opt --host_cache_size $host_cache_size"
#         else
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:25 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --attn_impl vdcores_opt --host_cache_size $host_cache_size"
#         fi
#         echo $cmd
#         eval $cmd
#         echo "###############################"
#     done

#     prompt_len=$((prompt_len * 2))
# done

# # dynamic cache
# prompt_len=256
# host_cache_size=0
# while [ $prompt_len -le 256 ]; do
#     # for offloading_ratio in 0 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#     for offloading_ratio in 0.3; do
#         if [[ "$offloading_ratio" == "0.08" ]]; then
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:25 python benchmark_opt_deynamic_cache.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --use_config"
#         else
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:25 python benchmark_opt_dynamic_cache.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name}"
#         fi
#         echo $cmd
#         eval $cmd
#         echo "###############################"
#     done

#     prompt_len=$((prompt_len * 2))
# done


# # split attention
# prompt_len=32
# host_cache_size=0
# while [ $prompt_len -le 1024 ]; do
#     # for offloading_ratio in 0 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#     for offloading_ratio in 1; do
#         if [[ "$offloading_ratio" == "0.08" ]]; then
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:25 python benchmark_opt.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --use_config --max_new_tokens 32"
#         else
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:25 python benchmark_opt.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --max_new_tokens 32"
#         fi
#         echo $cmd
#         eval $cmd
#         echo "###############################"
#     done

#     prompt_len=$((prompt_len * 2))
# done



# # split attention + new kernel
# prompt_len=32
# host_cache_size=51
# while [ $prompt_len -le 32 ]; do
#     # for offloading_ratio in 0 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#     for offloading_ratio in 0.1; do
#         if [[ "$offloading_ratio" == "0.08" ]]; then
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt_attn.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --max_new_tokens 32 --host_cache_size $host_cache_size --attn_impl vdcores_opt"
#         else
#             cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt_attn.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --use_config --max_new_tokens 32 --host_cache_size $host_cache_size --attn_impl vdcores_opt"
#         fi
#         echo $cmd
#         eval $cmd
#         echo "###############################"
#     done

#     prompt_len=$((prompt_len * 2))
# done

# --host_cache_size $host_cache_size --attn_impl vdcores_opt