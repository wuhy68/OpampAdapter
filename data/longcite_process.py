import re
import json
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from datasets import load_dataset, disable_caching

def contains_chinese(example):
    return bool(re.search('[\u4e00-\u9fff]', example["prompt"]))
    
def shuffle(example):
    documents = example["prompt"]
    document = documents.split("[Document Start]\n")[1].split("\n[Document End]")[0]
    pattern = re.compile(r'<C(\d+)>([^<]+)')
    chunks = []
    for match in pattern.findall(document):
        chunk = f"<C{match[0]}>{match[1]}"
        chunks.append(chunk)
    random.shuffle(chunks)
    shuffled_document = ''.join(chunks)
    example["prompt"] = ''.join([documents.split("[Document Start]\n")[0], "[Document Start]\n", shuffled_document, "\n[Document End]", documents.split("\n[Document End]")[1]])
    return example

def process(example):
    # process question
    example["question"] = "[Question]:" + example["prompt"].split("\n")[-1]

    # process gold doc
    cite_pattern = re.compile(r'\[([0-9]+)-([0-9]+)\]')
    numbers = set()
    for match in cite_pattern.findall(example["response"]):
        start, end = int(match[0]), int(match[1])
        numbers.update(range(start, end + 1))
    pattern = re.compile(r'<C(\d+)>([^<]+)')
    result = []
    documents = example["prompt"]
    document = documents.split("[Document Start]\n")[1].split("\n[Document End]")[0]
    for match in pattern.findall(document):
        num = int(match[0])  
        if num in numbers:  
            result.append(match[1].strip())  
    example["gold_doc"] = result

    # process Instruction
    example["instruction"] = "Please answer the user\'s question based on the following document.\n"
    
    return example

def combo(example):
    cite_pattern = re.compile(r'\[([0-9]+)-([0-9]+)\]')
    numbers = set()
    documents = example["prompt"]
    document = documents.split("[Document Start]\n")[1].split("\n[Document End]")[0]
    for match in cite_pattern.findall(example["response"]):
        start, end = int(match[0]), int(match[1])
        numbers.update(range(start, end + 1))
    pattern = re.compile(r'<C(\d+)>([^<]+)')
    leng = len(pattern.findall(document))
    if len(numbers) < 21:
        needed_count = 20 - len(numbers)
        remaining_numbers = set(range(leng)) - numbers
        if len(remaining_numbers) < needed_count:
            numbers = set(range(leng))
        else:
            numbers.update(random.sample(list(remaining_numbers), needed_count))
    result = []
    for match in pattern.findall(document):
        num = int(match[0])  
        if num in numbers:  
            result.append(match[1].strip())  
    example["simplify"] = "".join(result)
    example["simplify"] = "[Document]:" + example["simplify"] + "\n"
    example["instruction"] = "Please answer the user\'s question based on the following documents.\n"
    return example
    
def split_into_four_parts(lst):
    n = len(lst)
    part_size = n // 4
    part1 = lst[:part_size]
    part2 = lst[part_size:part_size * 2]
    part3 = lst[part_size * 2:part_size * 3]
    part4 = lst[part_size * 3:]
    
    return part1, part2, part3, part4

def split_doc(example):
    documents = example["prompt"]
    document = documents.split("[Document Start]\n")[1].split("\n[Document End]")[0]
    pattern = re.compile(r'<C(\d+)>([^<]+)')
    result = []
    for match in pattern.findall(document): 
        result.append(match[1].strip())  
    part1, part2, part3, part4 = split_into_four_parts(result)
    example["split_doc"] = ["".join(part1),"".join(part2),"".join(part3),"".join(part4)]
    example["split_doc"] = ["[Document]:" + i + "\n" for i in example["split_doc"]]
    example["instruction"] = "Please answer the user\'s question based on the following documents.\n"
    return example

def process_concat(i):
    concatenated_string = "".join(dataset_en_combo["simplify"][i:i+10])
    return [concatenated_string]*10

def parallel_processing_combo(num_proc):
    num_workers = num_proc  
    new_list = []

    with Pool(processes=num_workers) as pool:
        indices = range(0, len(dataset_en_combo), 10)
        for result in tqdm(pool.imap(process_concat, indices), total=len(indices)):
            new_list.extend(result)

    return new_list
    
def process_concat_shuffle(i):
    concatenated_string = dataset_en_split["split_doc"][i] + dataset_en_split["split_doc"][i+1]
    random.shuffle(concatenated_string)
    shuffled_string = ''.join(concatenated_string)
    return [shuffled_string, shuffled_string]


def parallel_processing_split(num_proc):
    num_workers = num_proc  
    new_list = []

    with Pool(processes=num_workers) as pool:
        indices = range(0, len(dataset_en_split), 2)
        for result in tqdm(pool.imap(process_concat_shuffle, indices), total=len(indices)):
            new_list.extend(result)

    return new_list

def process_doc(example):
    # process doc
    pattern = re.compile(r'<C(\d+)>([^<]+)')
    result = []
    documents = example["prompt"]
    document = documents.split("[Document Start]\n")[1].split("\n[Document End]")[0]
    for match in pattern.findall(document):
        result.append(match[1].strip())  
    example["documents"] = "[Document]:" + "".join(result) + "\n"
    
    return example

def process_final(example):
    cleaned_text = re.sub(r'<\/?statement>|<cite.*?>.*?<\/cite>', '', example["response"])
    final_text = cleaned_text.strip()
    example["response"] = final_text
    example["prompt"] = example["instruction"] + example["documents"] + example["question"]
    return example

def parse_args():
    
    parser = argparse.ArgumentParser(description="data_process")
    parser.add_argument(
        "--data_name_or_path",
        type=str,
        default="THUDM/LongCite-45k",
        help="Path to raw_data."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final data.")
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processors.")
   
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = parse_args()
    raw_data = args.data_name_or_path
    output_dir = args.output_dir
    num_proc = args.num_proc
    disable_caching()
    
    dataset = load_dataset(f'{raw_data}', split="train")
    dataset_en = dataset.filter(lambda example: not contains_chinese(example), num_proc = num_proc)
    dataset_en_shuffle = dataset_en.map(shuffle, num_proc = num_proc)
    dataset_en = dataset_en.map(process, num_proc = num_proc)
    dataset_en_shuffle = dataset_en_shuffle.map(process, num_proc = num_proc)

    dataset_en_combo = dataset_en.map(combo, num_proc = num_proc)
    new_list = parallel_processing_combo(num_proc)
    dataset_en_combo = dataset_en_combo.add_column("documents", new_list)
    dataset_en_combo = dataset_en_combo.map(process_final, num_proc = num_proc).select_columns(['prompt', 'response', 'gold_doc'])
    with open(f'{output_dir}/longcite_en_combo.jsonl',"w", encoding='utf-8') as f:
        for entry in dataset_en_combo:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    dataset_en_split = dataset_en.map(split_doc, num_proc = num_proc)
    new_list = parallel_processing_split(num_proc)
    dataset_en_split = dataset_en_split.add_column("documents", new_list)
    dataset_en_split = dataset_en_split.map(process_final, num_proc = num_proc).select_columns(['prompt', 'response', 'gold_doc'])
    with open(f'{output_dir}/longcite_en_split.jsonl',"w", encoding='utf-8') as f:
        for entry in dataset_en_split:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    dataset_en = dataset_en.map(process_doc, num_proc = num_proc)
    dataset_en = dataset_en.map(process_final, num_proc = num_proc).select_columns(['prompt', 'response', 'gold_doc'])
    with open(f'{output_dir}/longcite_en.jsonl',"w", encoding='utf-8') as f:
        for entry in dataset_en:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    dataset_en_shuffle = dataset_en_shuffle.map(process_doc, num_proc = num_proc)
    dataset_en_shuffle = dataset_en_shuffle.map(process_final, num_proc = num_proc).select_columns(['prompt', 'response', 'gold_doc'])
    with open(f'{output_dir}/longcite_en_shuffle.jsonl',"w", encoding='utf-8') as f:
        for entry in dataset_en_shuffle:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    