import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import config

from peft import PeftModel, PeftConfig

def load_peft_model(model_path):
    # 获取基础模型名称
    config = PeftConfig.from_pretrained(model_path)
    base_model_name = config.base_model_name_or_path
    
    # 加载基础模型
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,  # 根据你的任务调整
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # 加载适配器
    model = PeftModel.from_pretrained(model, model_path)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer
model, _ = load_peft_model("./models/deberta-v2-xxlarge")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}") # 输出： Total parameters: 3,407,370

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

student_model = AutoModelForSequenceClassification.from_pretrained(config.STUDENT_MODEL, num_labels=2)
total_params_student = sum(p.numel() for p in student_model.parameters())
print(f"Total parameters: {total_params_student:,}")
trainable_params_student = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params_student:,}")
