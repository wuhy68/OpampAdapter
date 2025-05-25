# Benchmark

This project provides a framework for benchmarking OpAmp and other commercial LLMs on various question-answering and text-generation tasks. It supports evaluation using local models (including those fine-tuned with LoRA or OpAMP) and API-based models like OpenAI's GPT series.



## Features

* Supports multiple datasets like QASPER, NarrativeQA, HotpotQA, etc.
* Evaluates models using metrics such as Exact Match (EM), Partial Match (PM), and F1-score.
* Handles local Hugging Face transformer models, PEFT LoRA modules, and custom OpAMP adapters.
* Integrates with OpenAI API for evaluating models like GPT-4o.
* Generates detailed output reports with scores and potential errors.
* Supports noisy document injection for evaluating retrieval augmented generation (RAG) robustness for specific datasets.

---



## Configuration

### 1. Prompts (`config/prompt.yaml`)
This YAML file defines the instruction and system prompts used for each dataset. You can customize prompts here if needed.

### 2. Dataset (`data/benchmark.json`)
This JSON file contains the data for benchmarking. Each entry should include fields like `tag` (dataset name), `documents`, `question`, and `answer`. For datasets like "quality", an `options` field is also expected.

### 3. API Keys
**For OpenAI API evaluations:** Set your OpenAI API key in the `evaluate_benchmark_api.sh` script or directly pass it as an argument.

```bash
API_KEY="your-openai-api-key"
```

---



## How to Run

There are two main ways to evaluate models: using local/custom models or using the OpenAI API.

### 1. Evaluating Local Models 

This method uses `evaluate_benchmark.py` and is typically orchestrated by `evaluate_benchmark.sh`.

**Script:** `evaluate_benchmark.py`
**Key Arguments:**

* `--dataset`: The dataset to evaluate on (e.g., `narrativeqa`, `hotpotqa`).
* `--modelname`: A custom name for the model run, used for output files.
* `--plm`: Path or name of the pretrained language model (e.g., `THUDM/chatglm-6b`).
* `--lora`: Path to the LoRA module (if any).
* `--opamp`: Path to the OpAMP adapter module (if any).
* `--temp`: Temperature for generation.
* `--noise_rate`: Noise rate for datasets like `coqa`, `quac` (default 0.8).

**Using the Shell Script (`evaluate_benchmark.sh`):**

1.  Modify `evaluate_benchmark.sh` to set paths for `BASE_MODEL`, `LORA_MODEL`, `OPAMP_MODEL` if needed.
2.  Run the script:
    ```bash
    bash evaluate_benchmark.sh
    ```
    The script provides commented-out examples for different configurations (base model, LoRA, OpAMP). Uncomment and modify the section you wish to run. For example, to run a qlora model:
    ```bash
    python evaluate_benchmark.py \
        --modelname qlora \
        --dataset hotpotqa \
        --plm $BASE_MODEL \
        --lora $LORA_MODEL \
        --temp 0.5
    ```

### 2. Evaluating API-based Models (e.g., OpenAI)

This method uses `evaluate_benchmark_api.py` and `evaluate_benchmark_api.sh`.

**Script:** `evaluate_benchmark_api.py`
**Key Arguments:**
* `--dataset`: The dataset to evaluate on.
* `--modelname`: The name of the OpenAI model (e.g., `gpt-4o`, `gpt-3.5-turbo`).
* `--temp`: Temperature for generation.
* `--api_key`: Your OpenAI API key.
* `--noise_rate`: Noise rate for specific datasets (default 0.8).

**Using the Shell Script (`evaluate_benchmark_api.sh`):**
1.  Edit `evaluate_benchmark_api.sh` to set your `API_KEY`.
    ```bash
    API_KEY="your-openai-api-key"
    ```
2.  Modify other parameters like `--modelname` and `--dataset` as needed.
3.  Run the script:
    ```bash
    bash evaluate_benchmark_api.sh
    ```
    Example command within the script:
    ```bash
    python evaluate_benchmark_api.py \
        --modelname gpt-4o \
        --dataset hotpotqa \
        --temp 0.5 \
        --api_key $API_KEY
    ```

---

## Output

Evaluation results are saved in the `output/` directory. For each run, two JSON files are typically generated:
* `{dataset}_{modelname}.json`: Contains a list of prediction and ground truth answer pairs.
* `{dataset}_{modelname}_score.json`: Contains the calculated EM, PM, and F1 scores.
* `{dataset}_{modelname}_error.json` (if errors occurred with the API): Contains details about any API errors encountered.

---

This README provides a basic guide. Refer to the comments and code within the scripts for more detailed understanding.