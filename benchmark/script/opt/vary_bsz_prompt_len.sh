cd ..
make pyext arch=90a TILE_N=32
cd benchmark

gpu_name="GH200"
model_name="facebook/opt-30b"
bsz=32
prompt_len=1024
offloading_ratio=0.3
cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name}"
echo $cmd
eval $cmd

cd ..
make pyext arch=90a TILE_N=128
cd benchmark

gpu_name="GH200"
model_name="facebook/opt-30b"
bsz=128
prompt_len=256
offloading_ratio=0.3
cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt.py --model_path $model_name --max_new_tokens 32 --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --graph --gpu ${gpu_name}"
echo $cmd
eval $cmd


gpu_name="GH200"
model_name="facebook/opt-30b"
bsz=128
prompt_len=512
offloading_ratio=0
host_cache_size=80
cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt_attn_vary.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --use_config --max_new_tokens 32 --host_cache_size $host_cache_size --attn_impl vdcores_opt"
echo $cmd
eval $cmd


gpu_name="GH200"
model_name="facebook/opt-30b"
bsz=128
prompt_len=1024
offloading_ratio=0
host_cache_size=110
cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python benchmark_opt_attn_vary.py --model_path $model_name --batch_size $bsz --prompt_len $prompt_len --offload $offloading_ratio --gpu ${gpu_name} --use_config --max_new_tokens 32 --host_cache_size $host_cache_size --attn_impl vdcores_opt"
echo $cmd
eval $cmd




