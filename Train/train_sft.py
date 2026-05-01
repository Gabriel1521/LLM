

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

import torch
from datasets import load_dataset

from datasets import Dataset, DatasetDict
import pandas as pd

model_dir="C:/Users/Gabriela/.cache/modelscope/hub/models/qwen/Qwen1___5-0___5B-Chat"

print("Hello")

# 1. 加载CSV到DataFrame
train_df = pd.read_csv('Train/data/gsm8k_train.csv')
test_df = pd.read_csv('Train/data/gsm8k_test.csv')  # 如果存在

# 2. 转换为Dataset格式
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print("Dataset loaded.")