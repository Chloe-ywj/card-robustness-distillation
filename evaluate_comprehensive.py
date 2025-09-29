import torch
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import evaluate_model
import data_loader
import json
import datetime

def load_model_and_tokenizer(model_name, device):
    """从Hugging Face Hub加载模型和tokenizer"""
    print(f"从Hugging Face Hub加载模型: {model_name}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        print(f"✅ 模型 {model_name} 加载成功")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        raise

def evaluate_single_model(model_name, test_loader, device, raw_test_dataset):
    try:
        print(f"\n{'='*50}")
        print(f"评估模型: {model_name}")
        print(f"{'='*50}")
        
        # 加载模型和tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # 运行评估
        clean_accuracy, robust_accuracies = evaluate_model(
            model, test_loader, device, tokenizer, raw_test_dataset
        )
        
        # 返回结果
        return {
            'clean_accuracy': clean_accuracy,
            'robust_accuracies': robust_accuracies,
            'model_name': model_name,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"评估模型 {model_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e)
        }

def save_results(results, output_file="model_evaluation_results.json"):
    """保存评估结果到JSON文件"""
    # 添加时间戳
    results_with_meta = {
        'evaluation_date': datetime.datetime.now().isoformat(),
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 直接从Hugging Face Hub加载模型
    model_name = "./A0.3B0.7/best_robust_student_model"
    output_file = "model_evaluation_results.json"
    
    # 加载数据
    print("加载测试数据...")
    train_loader, test_loader, tokenizer, raw_test_dataset = data_loader.get_data_loaders()
    
    # 评估单个模型
    result = evaluate_single_model(model_name, test_loader, device, raw_test_dataset)
    
    # 保存结果
    results = {model_name: result}
    save_results(results, output_file)
    
    # 打印最终摘要
    print("\n" + "="*60)
    print("模型评估完成 - 最终摘要")
    print("="*60)
    
    if result['status'] == 'success':
        print(f"模型: {model_name}")
        print(f"干净准确率: {result['clean_accuracy']:.2f}%")
        for attack_name, acc in result['robust_accuracies'].items():
            if isinstance(acc, (int, float)):
                print(f"{attack_name}: {acc:.2f}%")
            else:
                print(f"{attack_name}: {acc}")
    else:
        print(f"模型评估失败: {result['error']}")

if __name__ == "__main__":
    main()
