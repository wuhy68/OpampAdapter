from datasets import load_dataset
from tqdm import tqdm
import torch 
import string
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, choices=["inconsistent", "unanswerable", "counterfactual"], default='counterfactual',
        help='evaluetion dataset',
    )
    parser.add_argument(
        '--plm', type=str, default='/root/autodl-tmp/llama_8B',
        help='dir of plm'
    )

   
    args = parser.parse_args()
    plm = args.plm
    dataset = args.dataset
    model = AutoModelForCausalLM.from_pretrained(plm, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(plm)
    tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True, device_map='auto')
    strict_match = False
    correct = 0
    
    if dataset == "counterfactual":
        dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")
        for example in tqdm(dataset, desc="Processing examples"):
            # specify your custom prompt here. For example, if we want the model to directly generate an answer based on the context and question.
            prompt = f"""You are an expert in retrieval question answering. 
        Please read the following context and question, and choose the most appropriate answer from the options provided.
        Respond only with the letter (A, B, C, or D) corresponding to the correct answer for each question based on the context.
        Context: {example['context']}
        Question: {example['question']}
        A {example["choices"]['text'][0]}
        B {example["choices"]['text'][1]}
        C {example["choices"]['text'][2]}
        D {example["choices"]['text'][3]}
        Answer:""" 
            outputs = generator(
                        prompt,
                        max_new_tokens=256,
                        top_p=None,
                        do_sample=False,
                        return_full_text=False)
            pred_answer = outputs[0]["generated_text"].strip()
            print(pred_answer)
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
        dataset = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")
        for example in tqdm(dataset, desc="Processing examples"):
            # specify your custom prompt here. For example, if we want the model to directly generate an answer based on the context and question.
            prompt = f"""You are an expert in retrieval question answering. 
        Please respond with the exact answer only. Do not be verbose or provide extra information.
        {task_specific_prompt}
        Context: {example['context']}
        Question: {example['question']}
        Answer:""" 
            outputs = generator(
                        prompt,
                        max_new_tokens=256,
                        top_p=None,
                        do_sample=False,
                        return_full_text=False)
            pred_answer = outputs[0]["generated_text"].strip()
            print(pred_answer)
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
        dataset = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test")
        for example in tqdm(dataset, desc="Processing examples"):
            # specify your custom prompt here. For example, if we want the model to directly generate an answer based on the context and question.
            prompt = f"""You are an expert in retrieval question answering. 
        Please respond with the exact answer only. Do not be verbose or provide extra information.
        {task_specific_prompt}
        Context: {example['context']}
        Question: {example['question']}
        Answer:""" 
            outputs = generator(
                        prompt,
                        max_new_tokens=256,
                        top_p=None,
                        do_sample=False,
                        return_full_text=False)
            pred_answer = outputs[0]["generated_text"].strip()
            print(pred_answer)
            # evaluate the answer
            if any(phrase in normalize_answer(pred_answer) for phrase in valid_phrases):
                correct += 1
        print(f"Accuracy: {correct / len(dataset)}")
    
