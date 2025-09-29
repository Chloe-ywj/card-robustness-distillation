import os
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import config
from datasets import load_dataset

def get_data_loaders():
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("加载IMDB训练集...")
    train_dataset_raw = load_dataset("imdb", split="train")
    print("加载IMDB测试集...")
    test_dataset_raw = load_dataset("imdb", split="test")
    print(f"训练集: {len(train_dataset_raw)} 个样本")
    print(f"测试集: {len(test_dataset_raw)} 个样本")

    # 定义预处理函数
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=config.MAX_LENGTH
        )
        model_inputs["labels"] = examples["label"]
        return model_inputs

    # 应用预处理
    train_dataset_encoded = train_dataset_raw.map(
        preprocess_function, 
        batched=True, 
        remove_columns=['text']
    )
    test_dataset_encoded = test_dataset_raw.map(
        preprocess_function, 
        batched=True, 
        remove_columns=['text']
    )

    train_sample_size = len(train_dataset_encoded)
    test_sample_size = len(test_dataset_encoded)

    # 打乱并选择样本
    final_train_encoded = train_dataset_encoded.select(range(train_sample_size))
    final_test_encoded = test_dataset_encoded.select(range(test_sample_size))
    final_test_raw = test_dataset_raw.select(range(test_sample_size))

    print(f"使用 {len(final_train_encoded)} 个训练样本，{len(final_test_encoded)} 个测试样本")

    # 设置 PyTorch 格式
    available_columns = []
    for col in ['input_ids', 'attention_mask', 'labels']:
        if col in final_train_encoded.column_names:
            available_columns.append(col)
    
    final_train_encoded.set_format('torch', columns=available_columns)
    final_test_encoded.set_format('torch', columns=available_columns)

    # 创建 DataLoader
    train_loader = DataLoader(
        final_train_encoded, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(
        final_test_encoded, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False  # 测试集不需要打乱
    )

    return train_loader, test_loader, tokenizer, final_test_raw