TOT_CUDA="0,1,2,3,4,5,6,7"

BASE_MODEL=""
LORA_MODEL=""
OPAMP_MODEL=""

# CUDA_VISIBLE_DEVICES=${TOT_CUDA} python evaluate_benchmark.py \
#     --modelname llama3.1-8B-inst \
#     --dataset en_refine \
#     --plm $BASE_MODEL \
#     --temp 0.5 \
#     --noise_rate 0.9 

# CUDA_VISIBLE_DEVICES=${TOT_CUDA} python evaluate_benchmark.py \
#     --modelname qlora \
#     --dataset en_refine \
#     --plm $BASE_MODEL \
#     --lora $LORA_MODEL \
#     --temp 0.5 \
#     --noise_rate 0.9 