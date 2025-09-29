import torch
from transformers import AutoModelForSequenceClassification,set_seed
from torch.optim import AdamW
from tqdm import tqdm
import os # 导入 os 模块

import config
import data_loader
import attack
import loss_functions
from evaluate import evaluate_model

def AKD_train():
    set_seed(config.SEED)
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, test_loader, tokenizer, raw_test_dataset = data_loader.get_data_loaders()
    num_labels = 2

    # 加载模型
    teacher_model = AutoModelForSequenceClassification.from_pretrained(config.TEACHER_MODEL, num_labels=num_labels).to(device)
    #teacher_model = AutoModelForSequenceClassification.from_pretrained("dfurman/deberta-v2-xxl-imdb-v0.1", num_labels=num_labels).to(device)
    student_model = AutoModelForSequenceClassification.from_pretrained(config.STUDENT_MODEL, num_labels=num_labels, output_hidden_states=True).to(device)
    print("\nStarting distillation...\n")

    # 冻结教师模型
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 设置优化器和损失函数
    optimizer = AdamW(student_model.parameters(), lr=config.LEARNING_RATE)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    best_clean_accuracy = 0.0
    best_robust_score = 0.0  # 使用平均鲁棒性得分作为指标
    best_clean_epoch = -1
    best_robust_epoch = -1
    best_robust_acc=0.0
    result={}
    
    BEST_CLEAN_MODEL_PATH = "./distbert_test_results/A0.5B0.5/Akd_clean_model"
    BEST_ROBUST_MODEL_PATH = "./distbert_test_results/A0.5B0.5/Akd_robust_model"

    # 确保保存目录存在
    os.makedirs(BEST_CLEAN_MODEL_PATH, exist_ok=True)
    os.makedirs(BEST_ROBUST_MODEL_PATH, exist_ok=True)
    # ================================================================

    # 训练循环
    for epoch in range(config.EPOCHS):
        student_model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}"):
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            teacher_input_batch = {**inputs, "labels": labels}

            # 生成对抗样本嵌入
            perturbed_embeddings, attack_type = attack.generate_diverse_adv_examples(
                student_model, teacher_input_batch, config.FGSM_EPSILON)

            # Teacher 模型前向传播
            with torch.no_grad():
                teacher_outputs_clean = teacher_model(**teacher_input_batch, output_hidden_states=True)

            # Student 模型前向传播
            student_outputs_clean = student_model(**teacher_input_batch, output_hidden_states=True)

            student_outputs_adv = student_model(inputs_embeds=perturbed_embeddings,
                                                attention_mask=inputs['attention_mask'],
                                                output_hidden_states=True)

            # 计算总损失 
            loss_ce = ce_loss_fn(student_outputs_clean.logits, labels)
            loss_kd_clean = loss_functions.distillation_loss(student_outputs_clean.logits, teacher_outputs_clean.logits)
            loss_kd_adv = loss_functions.distillation_loss(student_outputs_adv.logits, teacher_outputs_clean.logits)
            loss = config.GAMMA*loss_ce + config.ALPHA * (loss_kd_clean + 2 * loss_kd_adv) / 2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Average Training Loss: {avg_train_loss:.4f}")

        # 每4轮进行评估
        if (epoch + 1) % 4 == 0 or (epoch + 1) == config.EPOCHS:
            print(f"\n--- Running evaluation at Epoch {epoch + 1} ---")
            clean_accuracy, robust_accuracies = evaluate_model(student_model, test_loader, device, tokenizer, raw_test_dataset)

            # 计算平均鲁棒性得分（只考虑成功的攻击评估）
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
                best_robust_acc= clean_accuracy
                result=robust_accuracies.copy()
                best_robust_epoch = epoch + 1
                student_model.save_pretrained(BEST_ROBUST_MODEL_PATH)
                tokenizer.save_pretrained(BEST_ROBUST_MODEL_PATH)
            print("-" * 50)

    print("\n" + "="*50)
    print("           TRAINING COMPLETE & SUMMARY")
    print("="*50)
    print(f"Best Clean Accuracy: {best_clean_accuracy:.2f}% (achieved at Epoch {best_clean_epoch})")
    print(f"   Model saved at: {BEST_CLEAN_MODEL_PATH}")
    print("-" * 50)
    print(f"Best Average Robust Score: {best_robust_score:.2f}% (achieved at Epoch {best_robust_epoch})")
    print(f"best_robust_acc :{best_robust_acc:.2f}")
    for attack_name, acc in result.items():
        if isinstance(acc, (int, float)):
            print(f"{attack_name} 鲁棒准确率: {acc:.2f}%")
        else:
            print(f"{attack_name} 鲁棒准确率: {acc}")
    print(f"   Model saved at: {BEST_ROBUST_MODEL_PATH}")
    print("="*50)

if __name__ == '__main__':
    AKD_train()