bash script/opt/mem_bound.sh GH200 facebook/opt-30b ./result/fig10.csv
bash script/opt/mem_bound.sh GH200 facebook/opt-6.7b ./result/fig10.csv

bash script/opt/mix_bound.sh ./result/fig11.csv
bash script/llama/mix_bound.sh ./result/fig11.csv

bash script/opt/vary_bsz_prompt_len.sh ./result/fig12.csv

bash script/opt/mix_bound_uniform.sh ./result/fig13.csv
bash script/llama/mix_bound_uniform.sh ./result/fig13.csv
