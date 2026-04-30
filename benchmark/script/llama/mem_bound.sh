bsz=8
model_name="meta-llama/Llama-2-7b-hf"
gpu_name="GH200"
prompt_len=32

for offloading_ratio in 0 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    if [[ "$offloading_ratio" == "0.08" ]]; then
        cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_llama.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name} --use_config"
    else
        cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_llama.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name}"
    fi
    echo $cmd
    eval $cmd
    echo "###############################"
done

