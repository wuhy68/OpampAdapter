# dataset : ["inconsistent", "unanswerable", "counterfactual"]

TOT_CUDA="0,1,2,3,4,5,6,7"

BASE_MODEL=""
LORA_MODEL=""
OPAMP_MODEL=""

# CUDA_VISIBLE_DEVICES=${TOT_CUDA} python evaluate_faith.py \
#     --modelname llama3.1-8B-inst \
#     --dataset counterfactual \
#     --plm $BASE_MODEL \
#     --temp 0.2 \


CUDA_VISIBLE_DEVICES=${TOT_CUDA} python evaluate_faith.py \
    --modelname qlora \
    --dataset counterfactual \
    --plm $BASE_MODEL \
    --lora $LORA_MODEL \
    --temp 0.2 \


# CUDA_VISIBLE_DEVICES=${TOT_CUDA} python evaluate_faith.py \
#     --modelname opamp_d10 \
#     --dataset counterfactual \
#     --plm $BASE_MODEL \
#     --lora $LORA_MODEL \
#     --opamp $OPAMP_MODEL \
#     --temp 0.2 \
