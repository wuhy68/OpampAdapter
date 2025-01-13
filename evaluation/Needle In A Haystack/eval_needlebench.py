from mmengine.config import read_base
with read_base():
    from opencompass.configs.models.Mymodel.llama_8b_base import models as llama_8b_base
    # from opencompass.configs.models.Mymodel.llama_8b_qlora import models as llama_8b_qlora
    # from opencompass.configs.models.Mymodel.llama_8b_opamp import models as llama_8b_opamp

    # Evaluate needlebench_4k, adjust the configuration to use 8k, 32k, 128k, 200k, or 1000k if necessary.
    from opencompass.configs.datasets.needlebench.needlebench_4k.needlebench_4k import needlebench_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # only eval original "needle in a haystack test" in needlebench_4k
    # from opencompass.configs.datasets.needlebench.needlebench_4k.needlebench_single_4k import needlebench_zh_datasets, needlebench_en_datasets
    # from opencompass.configs.summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # eval Ancestral Tracing Challenge(ATC)
    # from opencompass.configs.datasets.needlebench.atc.atc_choice_50 import needlebench_datasets
    # from opencompass.configs.summarizers.needlebench import atc_summarizer_50 as summarizer

datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

for m in llama_8b_base:
    m['max_seq_len'] = 32768 # Ensure model can receive the full length of long texts, adjust for other models based on their supported maximum sequence length.
    m['max_out_len'] = 2000 # Ensure complete responses from the model in multi-needle retrieval tasks.

models = llama_8b_base

work_dir = './outputs/needlebench'
