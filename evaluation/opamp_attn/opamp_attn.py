import json

import torch
import transformers
from peft import PeftModel

from configuration_opampllama import OpampLlamaConfig
from modeling_opampllama import LlamaForCausalLM

def load_model_and_tokenizer(lora=None, opamp=None):
    if opamp != "":
        model_config = OpampLlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B")
        model_config.pretraining_tp = 1  ## without tensor parallelism rank

        model_config.adapter_dim = 64
        model_config.adapter_rank = model_config.adapter_dim
        model_config.adapter_scaling = 1.0
        model_config.adapter_lambda = 10
        model_config.adapter_layer = 1

        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            config=model_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    if opamp != "":
        print("load opamp adapter")

        opamp_weights = torch.load(opamp, map_location=torch.device("cpu"))
        weights_dict = {}
        for k, v in opamp_weights.items():
            new_k = k.replace("base_model.model.", "") if "base_model.model." in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict, strict=False)

    if lora != "":
        print("load lora module")
        model = PeftModel.from_pretrained(
            model,
            lora,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    # model = model.merge_and_unload()

    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate(model, tokenizer, text, temperature, split_word):

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_config = transformers.GenerationConfig(
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1,
        repetition_penalty=1.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=512, # max_length=max_new_tokens+input_sequence
        min_new_tokens=1, # min_length=min_new_tokens+input_sequence
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            output_attentions=True,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
        )

    seq = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    seq = seq.split(f"{split_word}")[-1].strip().replace('�','')

    return seq, generation_output.attentions


if __name__ == "__main__":

    # 载入模型
    opamp = "/root/autodl-tmp/peft/opamp/na_model.bin"
    opamp_qlora = "/root/autodl-tmp/peft/opamp/adapter_model"
    model, tokenizer = load_model_and_tokenizer(opamp_qlora, opamp)

    # 加载数据
    dataset = json.load(open("./coqa.json"))
    doclist = dataset[0]["noisy_doc"].split("[Document]:")
    doclist = doclist[1:]
    for i in range(len(doclist)):
        doclist[i] = "[Document]:" + doclist[i]
    tokenlen = ["<|system|>\nYou are an accurate and reliable AI assistant that can answer questions with the help of retrieved documents. Retrieved documents may contain noisy irrelevant information. Please respond with the exact answer based on the retrieved documents ignoring noisy irrelevant details.\n<|user|>\n"]
    tokenlen = tokenlen + doclist
    tokenlen.append(dataset[0]["question"] + "\n<|assistant|>\n")

    tokennum = []
    for i in range(len(tokenlen)):
        if i == 0:
            tokennum.append(len(tokenizer(tokenlen[i])["input_ids"]))
        else:
            tokennum.append(len(tokenizer(tokenlen[i])["input_ids"])-1)

    # 进行推理
    model.eval()
    bos_token = tokenizer.bos_token
    if bos_token is None:
        bos_token = ""
    eos_token = tokenizer.eos_token
    prompt = dataset[0]["noisy_doc"] + dataset[0]["question"]
    inputs = f"<|system|>\nYou are an accurate and reliable AI assistant that can answer questions with the help of retrieved documents. Retrieved documents may contain noisy irrelevant information. Please respond with the exact answer based on the retrieved documents ignoring noisy irrelevant details.\n<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(inputs, return_tensors='pt')
    inputs = inputs.to(model.device)

    with torch.no_grad(): 
        output = model(**inputs, output_attentions=True)


    # 得到attention值
    lay0_attn = output.attentions[0][0][0,0,-1,:]
    lay15_attn = output.attentions[0][15][0,0,-1,:]
    lay31_attn = output.attentions[0][31][0,0,-1,:]

    lay0_attn1 = output.attentions[1][0][0,0,-1,:]
    lay15_attn1 = output.attentions[1][15][0,0,-1,:]
    lay31_attn1 = output.attentions[1][31][0,0,-1,:]

    lay0_attn2 = output.attentions[2][0][0,0,-1,:]
    lay15_attn2 = output.attentions[2][15][0,0,-1,:]
    lay31_attn2 = output.attentions[2][31][0,0,-1,:]


    # 分组保存attention值
    tokennum = [53, 251, 184, 178, 325, 404, 341, 25]

    groups0 = torch.split(lay0_attn, tokennum)
    groups15 = torch.split(lay15_attn, tokennum)
    groups31 = torch.split(lay31_attn, tokennum)

    group_sums0 = [group.sum().item() for group in groups0]
    group_sums15 = [group.sum().item() for group in groups15]
    group_sums31 = [group.sum().item() for group in groups31]

    opamp_attn = {'group_sums0':group_sums0, 'group_sums15':group_sums15, 'group_sums31':group_sums31}
    with open("./opamp_attn.json", "w") as f:
        json.dump(opamp_attn, f)

    groups0 = torch.split(lay0_attn1, tokennum)
    groups15 = torch.split(lay15_attn1, tokennum)
    groups31 = torch.split(lay31_attn1, tokennum)

    group_sums0 = [group.sum().item() for group in groups0]
    group_sums15 = [group.sum().item() for group in groups15]
    group_sums31 = [group.sum().item() for group in groups31]

    opamp_attn = {'group_sums0':group_sums0, 'group_sums15':group_sums15, 'group_sums31':group_sums31}
    with open("./opamp_attn1.json", "w") as f:
        json.dump(opamp_attn, f)

    groups0 = torch.split(lay0_attn2, tokennum)
    groups15 = torch.split(lay15_attn2, tokennum)
    groups31 = torch.split(lay31_attn2, tokennum)

    group_sums0 = [group.sum().item() for group in groups0]
    group_sums15 = [group.sum().item() for group in groups15]
    group_sums31 = [group.sum().item() for group in groups31]

    opamp_attn = {'group_sums0':group_sums0, 'group_sums15':group_sums15, 'group_sums31':group_sums31}
    with open("./opamp_attn2.json", "w") as f:
        json.dump(opamp_attn, f)