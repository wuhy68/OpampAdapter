#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from os.path import exists, join, isdir
import shutil
import gc
import json
import math
import random
import copy
from copy import deepcopy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Callable, List, Tuple, Union, Any

import torch
from torch import nn
from torch.utils.data import Dataset
import utils
import bitsandbytes as bnb

import transformers
from transformers import Trainer, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

from opampllm.opampllama.configuration_opampllama import OpampLlamaConfig
from opampllm.opampllama.modeling_opampllama import LlamaForCausalLM

from opampllm.opampqwen.configuration_opampqwen import OpampQwen2Config
from opampllm.opampqwen.modeling_opampqwen import Qwen2ForCausalLM

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

import warnings

warnings.filterwarnings("ignore")

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

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        default="paged_adamw_32bit"
    )  # "paged_lion_8bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_adamw_32bit"
    lr_scheduler_type: str = field(
        default="cosine"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def truncate_string(s, substring):
    index = s.find(substring) + len(substring)
    part_a = s[:index]
    part_b = s[index:]
    return part_a, part_b


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Loading data: {}".format(data_path))
        data_list = utils.jload(data_path)
        # data_list = data_list[:10]

        # Preprocess Data
        logging.warning("Processing data")
        self.sources = []
        self.targets = []

        self.tokenizer = tokenizer
        bos_token = tokenizer.bos_token
        if bos_token == None:
            bos_token = ""
        eos_token = tokenizer.eos_token

        for idx in range(len(data_list)):
            data = data_list[idx]

            if data["tag"] == "noisy":
                instruction = "Please answer the question based on the given documents."
                docs = data["documents"]
                # if docs.count(" ") >= 4000:
                #     docs = " ".join(docs.split(" ")[:5000])
                ques = data["question"].split("[Question]:")[-1]
                query = f"{docs}\n{instruction}\n{ques}"
                
                source = f"{bos_token}<|system|>\nYou are an accurate and reliable AI assistant that can answer questions with the help of retrieved documents. Retrieved documents may contain noisy irrelevant information. Please respond with the exact answer based on the retrieved documents ignoring noisy irrelevant details.\n"
                source += f"<|user|>\n{query}{eos_token}\n<|assistant|>\n"

                ans = data["answer"]
                target = f"{ans}{eos_token}"

                self.sources.append(source)
                self.targets.append(target)
            else:
                messages = data["messages"]
                source = f"{bos_token}<|system|>\nYou are a helpful AI assistant.\n"
                source += f"<|{messages[0]['role']}|>\n{messages[0]['content']}{eos_token}\n<|assistant|>\n"

                target = f"{messages[1]['content']}{eos_token}\n"

                if len(messages) != 2:
                    for m in messages[2:]:
                        target += f"<|{m['role']}|>\n{m['content']}{eos_token}\n"

                self.sources.append(source)
                self.targets.append(target)

        del data_list
        gc.collect()

        # ## Debug Mode
        # self.sources = self.sources[:1000]
        # self.targets = self.targets[:1000]

        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # del sources, targets
        # gc.collect()

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

        # del data_dict
        # gc.collect()

        logging.warning("there are {} data in dataset".format(len(self.sources)))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        source = [self.sources[i]]
        target = [self.targets[i]]
        data_dict = preprocess(source, target, self.tokenizer)

        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        # print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        model = kwargs["model"]
        model.save_pretrained(peft_model_path)

        na_state = {}
        for param_tensor in model.state_dict():
            if "adapter" in param_tensor:
                na_state.update({param_tensor: model.state_dict()[param_tensor]})
        na_model_path = os.path.join(checkpoint_folder, "na_model.bin")
        torch.save(na_state, na_model_path)

        # state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_folder)
        # na_state = {k: v.bfloat16() for k, v in state_dict.items() if "adapter" in k}
        # na_model_path = os.path.join(checkpoint_folder, "na_model.bin")
        # torch.save(na_state, na_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        # ckpt_id = None
        # for entry in os.scandir(checkpoint_folder):
        #     if entry.is_dir() and 'global_step' in entry.name:
        #         ckpt_id = entry.name
        # if ckpt_id is not None:
        #     zero_model_path = os.path.join(checkpoint_folder, f"{ckpt_id}")
        #     if os.path.exists(zero_model_path):      
        #         shutil.rmtree(zero_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def find_all_linear_names(model, bits=4):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if "adapter" not in name:
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def prepare_model_tokenizer(model_args, training_args):

    if "llama" in model_args.model_name_or_path.lower():
        model_config = OpampLlamaConfig.from_pretrained(model_args.model_name_or_path)
        model_config.pretraining_tp = 1  ## without tensor parallelism rank

        # NA Config
        model_config.adapter_dim = 64
        model_config.adapter_rank = model_config.adapter_dim
        model_config.adapter_scaling = 1.0
        model_config.adapter_lambda = 10
        model_config.adapter_layer = 2
        
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            cache_dir=training_args.cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16,
            ),
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_loading_info=False,
        )
    elif "qwen" in model_args.model_name_or_path.lower():
        model_config = OpampQwen2Config.from_pretrained(model_args.model_name_or_path)
        model_config.pretraining_tp = 1  ## without tensor parallelism rank

        # NA Config
        model_config.adapter_dim = 64
        model_config.adapter_rank = model_config.adapter_dim
        model_config.adapter_scaling = 1.0
        model_config.adapter_lambda = 10
        model_config.adapter_layer = 3
        
        model = Qwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            cache_dir=training_args.cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16,
            ),
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_loading_info=False,
        )
    else:
        raise AssertionError("Unsupported Model Type. Please use Llama/Qwen models")
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_modules = find_all_linear_names(model)
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=lora_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    # if training_args.bf16:
    #     min_dtype = torch.finfo(torch.bfloat16).min
    # else:
    #     min_dtype = torch.finfo(torch.float16).min
    
    # Init
    for n, p in model.named_parameters():
        # if "adapter_lambda" in n:
        #     nn.init.constant_(p, 0)
        #     # nn.init.zeros_(p)
        if "adapter_up" in n:
            nn.init.zeros_(p)
            # nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if "adapter_down" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    # for param_tensor in model.state_dict():
    #     if "adapter" in param_tensor:
    #         print({param_tensor: model.state_dict()[param_tensor]})

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name:
            if hasattr(module, "weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
        if "adapter" in name:
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            else:
                module = module.to(torch.float32)

    for n, p in model.named_parameters():
        if "adapter" in n:
            p.requires_grad = True
    # for n, p in model.named_parameters():
    #     if "adapter_lambda" in n:
    #         print(p.requires_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.ddp_find_unused_parameters = False
    set_seed(42)
    
    model, tokenizer = prepare_model_tokenizer(model_args, training_args)

    model.config.use_cache = False
    model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.add_callback(SavePeftModelCallback)

    trainer.train()

if __name__ == "__main__":
    train()
