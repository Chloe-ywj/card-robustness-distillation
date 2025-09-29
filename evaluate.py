import torch 
import os 
from textattack.attack_recipes import TextFoolerJin2019, PWWSRen2019, DeepWordBugGao2018 
from textattack.models.wrappers import HuggingFaceModelWrapper 
from textattack import Attacker, Attack 
from textattack.datasets import Dataset 
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification 
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification 
from textattack.search_methods import GreedyWordSwapWIR 
from textattack.transformations import WordSwapEmbedding 
import config 
import nltk 

# 配置本地文件路径
os.environ['TA_CACHE_DIR'] = r'C:\Users\vipuser\.cache\textattack'
os.environ['TFHUB_CACHE_DIR'] = r'C:\Users\vipuser\.cache\tfhub'
os.environ['TFHUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# NLTK数据下载
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('corpora/omw')
except LookupError:
    print("下载NLTK averaged_perceptron_tagger数据...")
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')

def _run_single_attack(attack_recipe, model_wrapper, attack_dataset):
    """
    运行单个攻击并返回鲁棒准确率。
    """
    attack_name = attack_recipe.__name__
    print(f"\n{'='*20}\n运行攻击: {attack_name}\n{'='*20}")
    
    try:
        if attack_name == "TextFoolerJin2019":
            attack = create_textfooler_optimized(model_wrapper)
        else:
            attack = attack_recipe.build(model_wrapper)
            
        # 配置攻击参数
        from textattack import AttackArgs
        attack_args = AttackArgs(
            num_examples=config.NUM_ATTACK_SAMPLES,
            disable_stdout=False,
            random_seed=42,
            shuffle=False
        )
        
        attacker = Attacker(attack, attack_dataset, attack_args=attack_args)
        print(f"开始对抗攻击评估...")
        attack_results = list(attacker.attack_dataset())
        
        # 使用TextAttack的统计信息
        if hasattr(attacker, 'attack_stats'):
            stats = attacker.attack_stats
            num_successful = stats.num_successful_attacks
            num_failures = stats.num_failed_attacks
            num_skipped = stats.num_skipped_attacks
            robust_accuracy = stats.accuracy_under_attack * 100
            
            print(f"\n--- {attack_name} 详细结果 ---")
            print(f"处理样本总数: {len(attack_results)}")
            print(f"成功攻击（模型被欺骗）: {num_successful}")
            print(f"失败攻击（模型抵抗）: {num_failures}")
            print(f"跳过攻击: {num_skipped}")
            print(f"鲁棒准确率（抵抗攻击的比例）: {robust_accuracy:.2f}%")
            
            print(f"\nTextAttack统计信息:")
            print(f"- 原始准确率: {stats.original_accuracy * 100:.1f}%")
            print(f"- 攻击后准确率: {stats.accuracy_under_attack * 100:.1f}%")
            print(f"- 攻击成功率: {stats.attack_success_rate * 100:.1f}%")
            
            return robust_accuracy
        else:
            # 手动解析结果对象
            num_successful = 0
            num_failures = 0
            num_skipped = 0
            
            for result in attack_results:
                # 根据结果类型判断
                result_type = type(result).__name__
                
                if 'Successful' in result_type:
                    num_successful += 1
                elif 'Failed' in result_type:
                    num_failures += 1
                elif 'Skipped' in result_type:
                    num_skipped += 1
                else:
                    # 尝试检查结果类名
                    class_name = result.__class__.__name__
                    if 'Successful' in class_name:
                        num_successful += 1
                    elif 'Failed' in class_name:
                        num_failures += 1
                    elif 'Skipped' in class_name:
                        num_skipped += 1
                    else:
                        # 最后尝试使用字符串表示
                        result_str = str(result).lower()
                        if 'success' in result_str:
                            num_successful += 1
                        elif 'fail' in result_str:
                            num_failures += 1
                        elif 'skip' in result_str:
                            num_skipped += 1
                        else:
                            print(f"无法解析的结果类型: {class_name}")
            
            total_processed = len(attack_results)
            if total_processed > 0:
                robust_accuracy = (num_failures / total_processed) * 100
            else:
                robust_accuracy = 0.0
                
            print(f"\n--- {attack_name} 详细结果 (手动解析) ---")
            print(f"处理样本总数: {total_processed}")
            print(f"成功攻击（模型被欺骗）: {num_successful}")
            print(f"失败攻击（模型抵抗）: {num_failures}")
            print(f"跳过攻击: {num_skipped}")
            print(f"鲁棒准确率（抵抗攻击的比例）: {robust_accuracy:.2f}%")
            
            return robust_accuracy
        
    except Exception as e:
        print(f"攻击 {attack_name} 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_textfooler_optimized(model_wrapper):
    """
    创建优化的TextFooler攻击 - 只使用WordEmbeddingDistance约束
    """
    print("创建优化的TextFooler攻击...")
    
    # 创建变换
    transformation = WordSwapEmbedding(max_candidates=50)

    constraints = [
        RepeatModification(),
        StopwordModification(),
    ]
    
    # 添加WordEmbeddingDistance约束
    try:
        try:
            semantic_constraint = WordEmbeddingDistance(min_cosine_sim=0.7)
            print("✓ 使用min_cosine_sim参数添加WordEmbeddingDistance约束")
        except:
            semantic_constraint = WordEmbeddingDistance(min_cos_sim=0.7)
            print("✓ 使用min_cos_sim参数添加WordEmbeddingDistance约束")
        
        constraints.append(semantic_constraint)
        print("✓ 成功添加WordEmbeddingDistance语义约束")
        
    except Exception as e:
        print(f"添加WordEmbeddingDistance约束失败: {e}")
        print("将继续使用基础约束（无语义约束）")
    
    goal_function = UntargetedClassification(model_wrapper)
    
    search_method = GreedyWordSwapWIR(wir_method="gradient")
    print("使用WIR方法: gradient")
    
    attack = Attack(goal_function, constraints, transformation, search_method)
    attack.name = "TextFoolerJin2019 (Optimized)"
    
    return attack

def check_textattack_constraints():

    print("TextAttack 0.3.10 可用语义约束:")
    
    try:
        from textattack.constraints.semantics import WordEmbeddingDistance
        print("WordEmbeddingDistance 可用")
    except ImportError as e:
        print(f"WordEmbeddingDistance 不可用: {e}")
    
    try:
        from textattack.constraints.semantics import BERTScore
        print("BERTScore 可用")
    except ImportError:
        print("BERTScore 不可用")

def evaluate_model(model, test_loader, device, tokenizer, raw_test_dataset):
    """
    评估模型在干净样本上的准确率以及在多种对抗攻击下的鲁棒性。
    """
    model.eval()
    
    # 检查可用的约束
    check_textattack_constraints()
    
    # 评估干净样本准确率 ---
    print("\n--- 评估干净准确率 ---")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    clean_accuracy = 100 * correct / total
    print(f'干净准确率: {clean_accuracy:.2f}%')
    
    # --- 2. 准备对抗攻击评估 ---
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    num_attack_samples = config.NUM_ATTACK_SAMPLES
    attack_examples = []
    
    print(f"\n准备 {num_attack_samples} 个样本进行对抗攻击...")
    for i, example in enumerate(raw_test_dataset):
        if i >= num_attack_samples:
            break
        text = example.get('text', example.get('sentence', ''))
        label = example.get('label', example.get('labels', 0))
        attack_examples.append((text, label))
    
    attack_dataset = Dataset(attack_examples, shuffle=False)
    
    attacks_to_run = [
        TextFoolerJin2019,
        DeepWordBugGao2018,
        PWWSRen2019
    ]
    
    robust_accuracies = {}
    
    for attack_recipe in attacks_to_run:
        robust_acc = _run_single_attack(attack_recipe, model_wrapper, attack_dataset)
        if robust_acc is not None:
            robust_accuracies[attack_recipe.__name__] = robust_acc
        else:
            robust_accuracies[attack_recipe.__name__] = "Failed"
    
    print("\n--- 最终评估摘要 ---")
    print(f"干净准确率: {clean_accuracy:.2f}%")
    for attack_name, acc in robust_accuracies.items():
        if isinstance(acc, (int, float)):
            print(f"{attack_name} 鲁棒准确率: {acc:.2f}%")
        else:
            print(f"{attack_name} 鲁棒准确率: {acc}")
    
    return clean_accuracy, robust_accuracies