# 使用 `HuggingFace` 评测 HuggingFace 中 AutoModel 支持的模型
# 使用 `HuggingFaceCausalLM` 评测 HuggingFace 中 AutoModelForCausalLM 支持的模型
from opencompass.models.huggingface import HuggingFaceNoisyLM

models = [
    dict(
        type=HuggingFaceNoisyLM,
        # 以下参数为 `HuggingFaceCausalLM` 的初始化参数
        path='meta-llama/Llama-3.1-8B',
        peft_path="/root/autodl-tmp/peft/opamp/adapter_model",
        tokenizer_path='meta-llama/Llama-3.1-8B',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        batch_padding=False,
        # 以下参数为各类模型都有的参数，非 `HuggingFaceCausalLM` 的初始化参数
        abbr='llama-8b-base',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=16,              # 批次大小
        run_cfg=dict(num_gpus=1),   # 运行配置，用于指定资源需求
    )
]