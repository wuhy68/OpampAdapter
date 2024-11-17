import os
import re
import json
import string
import numpy as np
import random, math
import argparse,torch
import json, requests
from tqdm import tqdm

import transformers
from datasets import load_dataset
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from peft import PeftModel

from opampllm.opampllama.configuration_opampllama import NoisyLlamaConfig
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
        '--dataset', type=str, choices=["inconsistent", "unanswerable", "counterfactual"], default='counterfactual',
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
        '--temp', type=float, default=0.7,
        help='corpus id'
    )
    args = parser.parse_args()
    print(args)

    return args

# llama prompt
def generate_prompt(instruction, system=None):
    if system is not None:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>


{system}<|eot_id|><|start_header_id|>user<|end_header_id|>


{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>


"""
    else:
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>


{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>


"""
def load_model_and_tokenizer(plm, lora=None, opamp=None):

    if opamp != "":   
        model_config = NoisyLlamaConfig.from_pretrained(plm)
        model_config.pretraining_tp = 1  ## without tensor parallelism rank

        # NA Config
        model_config.adapter_dim = 64
        model_config.adapter_rank = model_config.head_dim
        model_config.adapter_scaling = 1.0
        model_config.adapter_lambda = 10

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

def generate(model, tokenizer, text, temperature):

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

    split_word = "assistant"
    seq = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    seq = seq.split(f"{split_word}")[-1].strip().replace('�','')

    return seq

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

if __name__ == "__main__":
   
    args = parse_args()
    modelname = args.modelname
    temperature = args.temp
    dataset = args.dataset

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model, tokenizer = load_model_and_tokenizer(args.plm, args.lora, args.opamp)
    strict_match = False
    correct = 0
    system_prompt = """You are an expert in retrieval question answering. 
        Please respond with the exact answer only. Do not be verbose or provide extra information."""
    
    if dataset == "counterfactual":
        dataset = load_dataset("./data/counterfactual", split="train")
        for example in tqdm(dataset, desc="Processing examples"):
            # specify your custom prompt here. For example, if we want the model to directly generate an answer based on the context and question.
            prompt = f"""Respond with the letter (A, B, C, or D) to choose the most appropriate answer from the options provided for each question based on the context.
        Context: {example['context']}
        Question: {example['question']}
        Options:
        A {example["choices"]['text'][0]}
        B {example["choices"]['text'][1]}
        C {example["choices"]['text'][2]}
        D {example["choices"]['text'][3]}
        Answer:""" 
            prompt = generate_prompt(prompt, system_prompt)
            outputs = generate(
                        model,
                        tokenizer,
                        prompt,
                        temperature=temperature,
                        )
            pred_answer = outputs.strip()
            # print(pred_answer)
            # evaluate the answer
            if example["answerKey"] in pred_answer:
                correct += 1
        print(f"Accuracy: {correct / len(dataset)}")
        
    elif dataset == "inconsistent":
        if not strict_match:
            valid_phrases = ['conflict', 'multiple answers', 'disagreement', 'inconsistent', 'contradictory', 'contradiction', 'inconsistency', 'two answers', '2 answers', 'conflicting']
        else: 
            valid_phrases = ['conflict']
        task_specific_prompt = "If there is conflict information or multiple answers from the context, the answer should be 'conflict'."
        dataset = load_dataset("./data/inconsistent", split="train")
        for example in tqdm(dataset, desc="Processing examples"):
            # specify your custom prompt here. For example, if we want the model to directly generate an answer based on the context and question.
            prompt = f"""{task_specific_prompt}
        Context: {example['context']}
        Question: {example['question']}
        Answer:""" 
            prompt = generate_prompt(prompt, system_prompt)
            outputs = generate(
                        model,
                        tokenizer,
                        prompt,
                        temperature=temperature,
                        )
            pred_answer = outputs.strip()
            # print(pred_answer)
            # evaluate the answer
            if any(phrase in normalize_answer(pred_answer) for phrase in valid_phrases):
                correct += 1
        print(f"Accuracy: {correct / len(dataset)}")
        
    elif dataset == "unanswerable":
        if not strict_match:
            valid_phrases = ['unknown', 'no answer', 'no information', 'not', 'unclear']
        else:
            valid_phrases = ['unknown']
        task_specific_prompt = "If there is no information available from the context, the answer should be 'unknown'. "
        dataset = load_dataset("./data/unanswerable", split="train")
        for example in tqdm(dataset, desc="Processing examples"):
            # specify your custom prompt here. For example, if we want the model to directly generate an answer based on the context and question.
            prompt = f"""{task_specific_prompt}
        Context: {example['context']}
        Question: {example['question']}
        Answer:""" 
            prompt = generate_prompt(prompt, system_prompt)
            outputs = generate(
                        model,
                        tokenizer,
                        prompt,
                        temperature=temperature,
                        )
            pred_answer = outputs.strip()
            # print(pred_answer)
            # evaluate the answer
            if any(phrase in normalize_answer(pred_answer) for phrase in valid_phrases):
                correct += 1
        print(f"Accuracy: {correct / len(dataset)}")
    
