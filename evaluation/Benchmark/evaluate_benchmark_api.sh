API_KEY = "token-abc123"

python evaluate_benchmark_api.py \
    --modelname gpt-4o \
    --dataset en_refine \
    --temp 0.5 \
    --api_key $API_KEY