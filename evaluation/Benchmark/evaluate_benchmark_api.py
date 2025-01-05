import os
import re
import json
import yaml
import string
import numpy as np
import random, math
import argparse,torch
import json, requests
from tqdm import tqdm
from collections import Counter
import Levenshtein

from openai import OpenAI

import transformers
from datasets import load_dataset
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from peft import PeftModel

from opampllm.opampllama.configuration_opampllama import OpampLlamaConfig
from opampllm.opampllama.modeling_opampllama import LlamaForCausalLM

import warnings

warnings.filterwarnings("ignore")

from transformers_utils import (
    _load_pretrained_model,
)
import transformers.integrations
import transformers.modeling_utils

transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (
    _load_pretrained_model
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, 
        choices=["qasper", "multifieldqa", "loogle", "narrativeqa", "quality", "MultiHopRAG", "hotpotqa", "2wikimqa", "musique", "coqa", "quac", "qrecc", "doc2dial"], 
        default='narrativeqa',
        help='evaluetion dataset',
    )
    parser.add_argument(
        '--modelname', type=str, default='chatgpt',
        help='model name'
    )
    parser.add_argument(
        '--plm', type=str, default='THUDM/chatglm-6b',
        help='name of plm'
    )
    parser.add_argument(
        '--lora', type=str, default='',
        help='lora module'
    )
    parser.add_argument(
        '--opamp', type=str, default='',
        help='opamp adapter module'
    )
    parser.add_argument(
        '--temp', type=float, default=0.2,
        help='temp'
    )
    parser.add_argument(
        '--noise_rate', type=float, default=0.8,
        help='noise rate'
    )
    args = parser.parse_args()
    print(args)

    return args

def get_prompt(args, prompts, data):
    doc = data['documents']
    inst = prompts[args.dataset]["instruction"]
    ques = data["question"].split("[Question]:")[-1]
    if args.dataset in ["qasper", "multifieldqa", "loogle", "narrativeqa"]:
        prompt = f"Document:\n{doc}\nInstruction:\n{inst}\nQuestion:\n{ques}" 
    elif args.dataset == "quality":
        options = data['options']
        prompt = f"Document:\n{doc}\nInstruction:\n{inst}\nQuestion:\n{ques}\nOptions:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}"
    elif args.dataset in ["MultiHopRAG", "hotpotqa", "2wikimqa", "musique"]:
        prompt = f"Document:\n{doc}\nInstruction:\n{inst}\nQuestion:\n{ques}"
    elif args.dataset in ["coqa", "quac", "qrecc", "doc2dial"]:
        doc = data["noisy_doc"]
        prompt = f"Document:\n{doc}\nInstruction:\n{inst}\nQuestion:\n{ques}"
    return prompt

# llama prompt
def generate_prompt(instruction, system=None):
    if system is not None:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>


{system}<|eot_id|><|start_header_id|>user<|end_header_id|>


{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Answer: 
"""
    else:
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>


{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Answer: 
"""


# base prompt
def generate_prompt_o(instruction, system=None):
    if system is not None:
        return f"""<|begin_of_text|><|system|>
{system}
<|user|>
{instruction}<|end_of_text|>
<|assistant|>
Answer: 
"""
    else:
        return f"""<|begin_of_text|><|user|>
{instruction}<|end_of_text|>
<|assistant|>
Answer: 
"""
    

def load_model_and_tokenizer(plm, lora=None, opamp=None):

    if opamp != "":   
        model_config = OpampLlamaConfig.from_pretrained(plm)
        model_config.pretraining_tp = 1  ## without tensor parallelism rank

        # NA Config
        model_config.adapter_dim = 64
        model_config.adapter_rank = model_config.adapter_dim
        model_config.adapter_scaling = 1.0
        model_config.adapter_lambda = 20

        model = LlamaForCausalLM.from_pretrained(
            plm,        
            config=model_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            plm,        
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(plm, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate(model, tokenizer, text, temperature, split_word):

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_config = GenerationConfig(
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
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
        )

    seq = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    seq = seq.split(f"{split_word}")[-1].strip().replace('�','')

    return seq


def get_response(args, modelname, prompt, system_prompt=None):
    client = OpenAI(
        api_key=args.api_key
    )
    try:
        completion = client.chat.completions.create(
            model=modelname,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
                ],
            temperature=args.temp
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        print(e)
        return {"response": "Unanswerable", "error": [str(e), prompt]}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth, strict=False):
    mapdict = {True: 1, False: 0}
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if strict:
        return mapdict[prediction == ground_truth]
    else: 
        a = all([i in prediction for i in ground_truth.split()])
        b = all([i in ground_truth for i in prediction.split()])
        return mapdict[a | b]


def partial_match_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    distance = Levenshtein.distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    return 1 - distance / max_len if max_len > 0 else 1.0


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def create_noisy_doc(golden, documents, num):
    noise_doc = [item for item in documents if item != golden]
    selected_strings = random.sample(noise_doc, min(num, len(noise_doc)))
    result = list({golden, *selected_strings})
    random.shuffle(result)
    noisy_doc = "\n".join(result)
    return noisy_doc


if __name__ == "__main__":
   
    args = parse_args()
    modelname = args.modelname

    bench_path = "./"
    prompts = yaml.safe_load(open(f"{bench_path}/config/prompt.yaml"))
    dataset = json.load(open(f"{bench_path}/data/benchmark.json"))
    dataset = [data for data in dataset if data['tag'] == args.dataset]
    if args.dataset in ["coqa", "quac", "qrecc", "doc2dial"]:
        documents = list({data["documents"] for data in dataset})
        noise_doc_num = round(1/(1-args.noise_rate)) - 1
        for data in dataset:
            if "rag" not in data.keys():
                data["noisy_doc"] = create_noisy_doc(data["documents"], documents, noise_doc_num)
            elif "rag" in data.keys() and len(data["rag"]) > noise_doc_num:
                data["noisy_doc"] = create_noisy_doc(data["documents"], data["rag"], noise_doc_num)
            elif "rag" in data.keys() and len(data["rag"]) <= noise_doc_num:
                noise_doc = [item for item in documents if item not in data["rag"]]
                num = noise_doc_num + 1 - len(data["rag"])
                selected_strings = random.sample(noise_doc, min(num, len(noise_doc)))
                result = selected_strings + data["rag"]
                random.shuffle(result)
                data["noisy_doc"] = "\n".join(result)

    pred_ans_pairs = []
    error = []
    for data in tqdm(dataset):
        instruction_prompt = get_prompt(args.dataset, prompts, data)
        system_prompt = prompts[args.dataset]["system"]
        response = get_response(args,modelname,instruction_prompt,system_prompt)
        if len(response) == 1:
            response = response["response"]
        else:
            error.append(response)
            response = response["response"]
        pred_ans_pairs.append((response, data['answer']))
        #time.sleep(5)
        
    with open(f"{bench_path}/output/{args.dataset}_{modelname}.json", "w", encoding="utf-8") as f:
        json.dump(pred_ans_pairs, f, ensure_ascii=False)
    if len(error) > 0:
        with open(f"{bench_path}/output/{args.dataset}_{modelname}_error.json", "w", encoding="utf-8") as f:
            json.dump(error, f, ensure_ascii=False)
    
    total_score_em = 0.
    total_score_pm = 0.
    total_score_f1 = 0.
    for (prediction, ground_truths) in pred_ans_pairs:
        score_em = 0.
        score_pm = 0.
        score_f1 = 0.
        for ground_truth in ground_truths:
            score_em = max(score_em, exact_match_score(prediction, ground_truth))
            score_pm = max(score_pm, partial_match_score(prediction, ground_truth))
            score_f1 = max(score_f1, f1_score(prediction, ground_truth))
        score_em = max(score_em, exact_match_score(prediction, " ".join(ground_truths)))
        total_score_em += score_em
        total_score_pm += score_pm
        total_score_f1 += score_f1
    
    with open(f"{bench_path}/output/{args.dataset}_{modelname}_score.json", "w", encoding="utf-8") as f:
        json.dump({"em": round(100 * total_score_em / len(pred_ans_pairs), 2), "pm": round(100 * total_score_pm / len(pred_ans_pairs), 2), "f1": round(100 * total_score_f1 / len(pred_ans_pairs), 2)}, f, ensure_ascii=False)
