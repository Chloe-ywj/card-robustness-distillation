import torch
import torch.nn.functional as F
import random

def fgsm_on_embedding(model, inputs, epsilon):
    """
    在词嵌入层上执行 FGSM 攻击
    """
    # 提取词嵌入矩阵
    embedding_matrix = model.get_input_embeddings()

    # 获取输入的 one-hot 编码对应的嵌入
    input_embeddings = embedding_matrix(inputs['input_ids'])
    input_embeddings.retain_grad()
    
    # 前向传播获取 logits
    outputs = model(inputs_embeds=input_embeddings, attention_mask=inputs['attention_mask'])
    logits = outputs.logits

    # 计算损失
    loss = F.cross_entropy(logits, inputs['labels'])
    model.zero_grad()
    loss.backward(retain_graph=True)

    # 获取嵌入的梯度
    grad = input_embeddings.grad.data

    # 计算扰动并添加到原始嵌入上
    perturbed_embeddings = input_embeddings + epsilon * grad.sign()

    return perturbed_embeddings

def random_perturbation(model, inputs, epsilon):
    """
    随机扰动 - 增加多样性
    """
    # 获取原始嵌入
    embedding_matrix = model.get_input_embeddings()
    input_embeddings = embedding_matrix(inputs['input_ids'])
    
    # 生成随机扰动（均匀分布）
    random_noise = torch.rand_like(input_embeddings) * 2 - 1  # [-1, 1]
    random_noise = random_noise * epsilon
    
    perturbed_embeddings = input_embeddings + random_noise
    
    return perturbed_embeddings

def pgd_on_embedding(model, inputs, epsilon, steps=3, alpha=0.01):
    """
    PGD攻击 - 多步攻击，更强的对抗样本
    """
    embedding_matrix = model.get_input_embeddings()
    original_embeddings = embedding_matrix(inputs['input_ids'])
    
    # 初始化扰动
    perturbed_embeddings = original_embeddings.clone().detach()
    
    for step in range(steps):
        perturbed_embeddings.requires_grad = True
        
        # 前向传播
        outputs = model(inputs_embeds=perturbed_embeddings, 
                       attention_mask=inputs['attention_mask'])
        loss = F.cross_entropy(outputs.logits, inputs['labels'])
        
        # 计算梯度
        model.zero_grad()
        loss.backward()
        
        # 更新扰动
        grad = perturbed_embeddings.grad.data
        perturbation = alpha * grad.sign()
        perturbed_embeddings = perturbed_embeddings + perturbation
        
        # 投影回epsilon球内
        delta = perturbed_embeddings - original_embeddings
        delta = torch.clamp(delta, -epsilon, epsilon)
        perturbed_embeddings = original_embeddings + delta
        
        perturbed_embeddings = perturbed_embeddings.detach()
    
    return perturbed_embeddings

def semantic_perturbation(model, inputs, epsilon):
    """
    语义保持的扰动 - 在梯度方向上加随机性
    """
    # 先获取FGSM梯度
    embedding_matrix = model.get_input_embeddings()
    input_embeddings = embedding_matrix(inputs['input_ids'])
    input_embeddings.retain_grad()
    
    outputs = model(inputs_embeds=input_embeddings, attention_mask=inputs['attention_mask'])
    loss = F.cross_entropy(outputs.logits, inputs['labels'])
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    grad = input_embeddings.grad.data
    
    # 在梯度方向上加随机性
    random_component = torch.randn_like(grad) * 0.3  # 30%的随机性
    combined_direction = grad.sign() + random_component
    combined_direction = combined_direction.sign()  # 重新归一化
    
    perturbed_embeddings = input_embeddings + epsilon * combined_direction
    
    return perturbed_embeddings

def generate_diverse_adv_examples(model, inputs, epsilon):
    """
    生成多种类型的对抗样本，随机选择一种
    返回: 对抗样本嵌入和攻击类型名称
    """
    attack_methods = [
        ('fgsm', lambda: fgsm_on_embedding(model, inputs, epsilon)),
        ('random', lambda: random_perturbation(model, inputs, epsilon)),
        ('pgd', lambda: pgd_on_embedding(model, inputs, epsilon, steps=2)),
        ('semantic', lambda: semantic_perturbation(model, inputs, epsilon))
    ]
    
    # 随机选择一种攻击方法（可以调整权重）
    attack_name, attack_func = random.choice(attack_methods)
    
    # 对于PGD，使用稍小的epsilon（因为是多步攻击）
    if attack_name == 'pgd':
        actual_epsilon = epsilon * 0.8
        perturbed_embeddings = pgd_on_embedding(model, inputs, actual_epsilon, steps=2)
    else:
        perturbed_embeddings = attack_func()
    
    return perturbed_embeddings, attack_name

def generate_diverse_adv_examples_weighted(model, inputs, epsilon):
    attack_methods = [
        ('fgsm', lambda: fgsm_on_embedding(model, inputs, epsilon), 0.4),
        ('random', lambda: random_perturbation(model, inputs, epsilon), 0.3),
        ('pgd', lambda: pgd_on_embedding(model, inputs, epsilon*0.8, steps=2), 0.2),
        ('semantic', lambda: semantic_perturbation(model, inputs, epsilon), 0.1)
    ]
    
    choices, weights = zip(*[(func, weight) for _, func, weight in attack_methods])
    names = [name for name, _, _ in attack_methods]
    
    selected_idx = random.choices(range(len(choices)), weights=weights, k=1)[0]
    attack_name = names[selected_idx]
    perturbed_embeddings = choices[selected_idx]()
    
    return perturbed_embeddings, attack_name
