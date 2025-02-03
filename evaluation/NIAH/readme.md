# How to run the Needle In A Haystack experiment



* **Environment**

```shell
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass # upload eval_needlebench.py
pip install -e .
pip install peft==0.10
```



* **Replacement and addition of files**

```shell
cd ./opencompass/configs/models # upload Mymodel

# In Mymodel, you should upload the path, tokenizer_path, peft_path

cd ./opencompass/models # upload modeling_opampllama.py, configuration_opampllama.py and replace huggingface.py

# In huggingface.py line 846:
# opamp_weights = torch.load(ada_path, map_location=torch.device("cpu"))
# replace the ada_path
```



* **run**

```shell
python run.py eval_needlebench.py 

# modify the experiment config in eval_needlebench.py
# modify depths_list and context_lengths in ./opencompass/configs/datasets/needlebench
```

