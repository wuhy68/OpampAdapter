import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from opampllm.opampllama.configuration_opampllama import OpampLlamaConfig
from opampllm.opampllama.modeling_opampllama import LlamaForCausalLM

from opampllm.opampqwen.configuration_opampqwen import OpampQwen2Config
from opampllm.opampqwen.modeling_opampqwen import Qwen2ForCausalLM

from peft import PeftModel

import torch

from transformers_utils import (
    get_keys_to_not_convert,
    _load_pretrained_model,
)
import transformers.integrations
import transformers.modeling_utils

transformers.integrations.get_keys_to_not_convert = get_keys_to_not_convert
transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (
    _load_pretrained_model
)


def merge_lora_to_base_model():
    model_path = "./models/Qwen2.5-7B"
    peft_path="./outputs/qwen2.5-7B-na_d10_sft/checkpoint-2000/"
    opamp_path="./outputs/qwen2.5-7B-na_d10_sft/checkpoint-2000/na_model.bin"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    model_config = OpampQwen2Config.from_pretrained(model_path)
    model_config.pretraining_tp = 1  ## without tensor parallelism rank
    model_config.auto_map = {
        "AutoConfig": "configuration_opampqwen.OpampQwen2Config",
        "AutoModelForCausalLM": "modeling_opampqwen.Qwen2ForCausalLM"
    }
    
    # NA Config
    model_config.adapter_dim = 64
    model_config.adapter_rank = model_config.adapter_dim
    model_config.adapter_scaling = 1.0
    model_config.adapter_layer = 3
    model_config.adapter_lambda = 10

    model = Qwen2ForCausalLM.from_pretrained(
        model_path,        
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )

    opamp_weights = torch.load(opamp_path, map_location=torch.device("cpu"))
    weights_dict = {}
    for k, v in opamp_weights.items():
        new_k = k.replace("base_model.model.", "") if "base_model.model." in k else k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict, strict=False)

    model = PeftModel.from_pretrained(
        model,
        peft_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model = model.merge_and_unload()

    save_path = "./outputs/qwen2.5-opamp-7B/"
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

def test_loading():

    path = "./outputs/qwen2.5-opamp-7B/"
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        # attn_implementation="flash_attention_2", 
        trust_remote_code=True
    ).eval()

    bos_token = tokenizer.bos_token
    if bos_token == None:
        bos_token = ""
    eos_token = tokenizer.eos_token
    print(bos_token, eos_token)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params/(1000000000):.2f}B total parameters.')

    inputs = tokenizer(f'{bos_token}<|system|>You are a helpful assistant.<|user|>\nHow are you?{eos_token}\n<|assistant|>\n', return_tensors='pt')
    # inputs = tokenizer('### Human:\nHow are you?\n### Assistant:\n', return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

if __name__ == '__main__':
    # merge_lora_to_base_model()
    test_loading()