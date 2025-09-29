import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,set_seed
from torch.optim import AdamW
from tqdm import tqdm
import os
import data_loader
import loss_functions
from evaluate import evaluate_model  # 导入评估函数
import config

TEACHER_MODEL = "dfurman/deberta-v2-xxl-imdb-v0.1"
STUDENT_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 2e-5
ALPHA = 0.5  # 知识蒸馏损失权重
TEMPERATURE = 3.0  # 蒸馏温度

def standard_kd_train():
    set_seed(config.SEED)
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training Mode: Standard Knowledge Distillation (No Adversarial Samples)")

    # 加载数据
    train_loader, test_loader, tokenizer, raw_test_dataset = data_loader.get_data_loaders()
    num_labels = 2  # IMDB是二分类

    # 加载模型
    #teacher_model = AutoModelForSequenceClassification.from_pretrained("dfurman/deberta-v2-xxl-imdb-v0.1", num_labels=num_labels).to(device)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(config.TEACHER_MODEL, num_labels=num_labels).to(device)
    student_model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL, num_labels=num_labels
    ).to(device)
    print("\nStarting Standard Knowledge Distillation...\n")

    # 冻结教师模型
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 设置优化器
    optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    best_clean_accuracy = 0.0
    best_robust_score = 0.0  # 记录最佳鲁棒性得分
    best_clean_epoch = -1
    best_robust_epoch = -1
    best_robust_acc=0.0
    result={}
    
    BEST_CLEAN_MODEL_PATH = "./distbert_test_results/A0.5B0.5/standard_kd_best_clean_model"
    BEST_ROBUST_MODEL_PATH = "./distbert_test_results/A0.5B0.5/standard_kd_best_robust_model"

    os.makedirs(BEST_CLEAN_MODEL_PATH, exist_ok=True)
    os.makedirs(BEST_ROBUST_MODEL_PATH, exist_ok=True)

    # 4. 训练循环 - 只使用干净样本
    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0
        total_kd_loss = 0
        total_ce_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            # 准备输入数据
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Teacher 前向传播
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Student 前向传播
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 计算损失 - 只有CE损失和KD损失
            loss_ce = ce_loss_fn(student_outputs.logits, labels)
            
            # 知识蒸馏损失（KL散度）
            loss_kd = loss_functions.distillation_loss(student_outputs.logits, teacher_outputs.logits)
            
            # 总损失 = CE损失 + α * KD损失
            loss = config.GAMMA*loss_ce + ALPHA * loss_kd

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_kd_loss += loss_kd.item()

        # 打印训练统计信息
        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_kd_loss = total_kd_loss / len(train_loader)
        
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Total Loss: {avg_loss:.4f} | CE Loss: {avg_ce_loss:.4f} | KD Loss: {avg_kd_loss:.4f}")
        
        # 每4轮进行评估
        if (epoch + 1) % 4 == 0 or (epoch + 1) == EPOCHS:
            print(f"\n--- Running robustness evaluation at Epoch {epoch + 1} ---")
            
            # 使用评估函数测试鲁棒性
            clean_accuracy, robust_accuracies = evaluate_model(
                student_model, test_loader, device, tokenizer, raw_test_dataset
            )
            
            # 计算平均鲁棒性得分
            valid_robust_accs = [acc if isinstance(acc, (float, int)) else 0.0 for acc in robust_accuracies.values()]
            current_robust_score = sum(valid_robust_accs) / len(valid_robust_accs) if valid_robust_accs else 0.0
            
            print(f"Clean Accuracy: {clean_accuracy:.2f}%")
            print(f"Average Robustness Score: {current_robust_score:.2f}%")
            for attack_name, acc in robust_accuracies.items():
                print(f"{attack_name}: {acc:.2f}%")

            # 保存最佳干净准确率模型
            if clean_accuracy > best_clean_accuracy:
                print(f"New best clean accuracy: {clean_accuracy:.2f}%. Saving model...")
                best_clean_accuracy = clean_accuracy
                best_clean_epoch = epoch + 1
                student_model.save_pretrained(BEST_CLEAN_MODEL_PATH)
                tokenizer.save_pretrained(BEST_CLEAN_MODEL_PATH)

            # 保存最佳鲁棒性模型
            if current_robust_score > best_robust_score:
                print(f"New best robust score: {current_robust_score:.2f}%. Saving model...")
                best_robust_score = current_robust_score
                best_robust_acc = clean_accuracy
                result=robust_accuracies.copy()
                best_robust_epoch = epoch + 1
                student_model.save_pretrained(BEST_ROBUST_MODEL_PATH)
                tokenizer.save_pretrained(BEST_ROBUST_MODEL_PATH)
            
            print("-" * 50)
        # ================================================

    # ======================= 最终总结 =======================
    print("\n" + "="*60)
    print("           STANDARD KD TRAINING COMPLETE & SUMMARY")
    print("="*60)
    print(f"Best Clean Accuracy: {best_clean_accuracy:.2f}% (Epoch {best_clean_epoch})")
    print(f"   Model saved at: {BEST_CLEAN_MODEL_PATH}")
    print("-" * 60)
    print(f"Best Robustness Score: {best_robust_score:.2f}% (Epoch {best_robust_epoch})")
    print(f"best_robust_acc :{best_robust_acc:.2f}")
    for attack_name, acc in result.items():
        if isinstance(acc, (int, float)):
            print(f"{attack_name} 鲁棒准确率: {acc:.2f}%")
        else:
            print(f"{attack_name} 鲁棒准确率: {acc}")
    print(f"   Model saved at: {BEST_ROBUST_MODEL_PATH}")
    print("="*60)

if __name__ == '__main__':
    standard_kd_train()