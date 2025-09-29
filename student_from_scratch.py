import torch
from transformers import AutoModelForSequenceClassification,set_seed
from torch.optim import AdamW
from tqdm import tqdm
import os
import config
import data_loader
from evaluate import evaluate_model

def student_from_scratch_train():
    set_seed(config.SEED)
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training Mode: Student from Scratch (Clean Data Only)")

    # 加载数据
    train_loader, test_loader, tokenizer, raw_test_dataset = data_loader.get_data_loaders()
    num_labels = 2

    # 只加载学生模型
    student_model = AutoModelForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, 
        num_labels=num_labels
    ).to(device)
    print(f"\nTraining {config.STUDENT_MODEL} from scratch on clean data...\n")

    # 设置优化器和损失函数（移除蒸馏）
    optimizer = AdamW(student_model.parameters(), lr=config.LEARNING_RATE)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    best_clean_accuracy = 0.0
    best_robust_score = 0.0
    best_clean_epoch = -1
    best_robust_epoch = -1
    best_robust_acc=0.0
    result={}
    
    BEST_CLEAN_MODEL_PATH = "./distbert_test_results/A0.5B0.5/student_scratch_clean_model"
    BEST_ROBUST_MODEL_PATH = "./distbert_test_results/A0.5B0.5/student_scratch_robust_model"

    os.makedirs(BEST_CLEAN_MODEL_PATH, exist_ok=True)
    os.makedirs(BEST_ROBUST_MODEL_PATH, exist_ok=True)

    # 只在干净数据上训练
    for epoch in range(config.EPOCHS):
        student_model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}"):
            # 准备输入数据
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Student 模型前向传播
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 计算总损失
            loss = config.GAMMA*ce_loss_fn(student_outputs.logits, labels)

            # 反向传播和优化 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Average Training Loss: {avg_train_loss:.4f}")

        # 周期性评估
        if (epoch + 1) % 4 == 0 or (epoch + 1) == config.EPOCHS:
            print(f"\n--- Running evaluation at Epoch {epoch + 1} ---")
            clean_accuracy, robust_accuracies = evaluate_model(student_model, test_loader, device, tokenizer, raw_test_dataset)

            # 计算平均鲁棒性得分
            valid_robust_accs = [acc if isinstance(acc, (float, int)) else 0.0 for acc in robust_accuracies.values()]
            current_robust_score = sum(valid_robust_accs) / len(valid_robust_accs) if valid_robust_accs else 0.0
            print(f"Epoch {epoch + 1} | Average Robustness Score: {current_robust_score:.2f}%")

            # 检查并保存最佳干净准确率模型
            if clean_accuracy > best_clean_accuracy:
                print(f"New best clean accuracy: {clean_accuracy:.2f}% (was {best_clean_accuracy:.2f}%). Saving model to {BEST_CLEAN_MODEL_PATH}")
                best_clean_accuracy = clean_accuracy
                best_clean_epoch = epoch + 1
                student_model.save_pretrained(BEST_CLEAN_MODEL_PATH)
                tokenizer.save_pretrained(BEST_CLEAN_MODEL_PATH)

            # 检查并保存最佳鲁棒性模型
            if current_robust_score > best_robust_score:
                print(f"New best robust score: {current_robust_score:.2f}% (was {best_robust_score:.2f}%). Saving model to {BEST_ROBUST_MODEL_PATH}")
                best_robust_score = current_robust_score
                best_robust_acc = clean_accuracy
                result = robust_accuracies.copy()
                best_robust_epoch = epoch + 1
                student_model.save_pretrained(BEST_ROBUST_MODEL_PATH)
                tokenizer.save_pretrained(BEST_ROBUST_MODEL_PATH)
            print("-" * 50)

    # 7. 训练结束总结
    print("\n" + "="*60)
    print("           TRAINING COMPLETE - STUDENT FROM SCRATCH")
    print("="*60)
    print(f"Best Clean Accuracy: {best_clean_accuracy:.2f}% (achieved at Epoch {best_clean_epoch})")
    print(f"   Model saved at: {BEST_CLEAN_MODEL_PATH}")
    print("-" * 60)
    print(f"Best Average Robust Score: {best_robust_score:.2f}% (achieved at Epoch {best_robust_epoch})")
    print(f"best_robust_acc :{best_robust_acc:.2f}")
    for attack_name, acc in result.items():
        if isinstance(acc, (int, float)):
            print(f"{attack_name} 鲁棒准确率: {acc:.2f}%")
        else:
            print(f"{attack_name} 鲁棒准确率: {acc}")
    print(f"   Model saved at: {BEST_ROBUST_MODEL_PATH}")
    print("="*60)

if __name__ == '__main__':
    student_from_scratch_train()