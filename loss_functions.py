import torch
import torch.nn.functional as F
import config

def distillation_loss(student_logits, teacher_logits):
    """
    KL 散度蒸馏损失
    """
    soft_teacher_logits = F.softmax(teacher_logits / config.TEMPERATURE, dim=-1)
    soft_student_logits = F.log_softmax(student_logits / config.TEMPERATURE, dim=-1)
    return F.kl_div(soft_student_logits,soft_teacher_logits, reduction='batchmean') * (config.TEMPERATURE ** 2)

def representation_transfer_loss(student_rep_clean, student_rep_adv, temperature=0.1):
    """
    增强的对比表征传递损失
    使用InfoNCE损失，包含正负样本对比
    """
    batch_size = student_rep_clean.size(0)
    
    # 正样本对：干净样本和对应的对抗样本
    positive_pairs = F.cosine_similarity(student_rep_clean, student_rep_adv, dim=-1)
    
    # 创建负样本：同一batch中的其他样本
    negative_similarities = []
    for i in range(batch_size):
        negatives = torch.cat([student_rep_adv[:i], student_rep_adv[i+1:]])
        similarities = F.cosine_similarity(
            student_rep_clean[i].unsqueeze(0).expand_as(negatives), 
            negatives, 
            dim=-1
        )
        negative_similarities.append(similarities)
    
    negative_similarities = torch.stack(negative_similarities)
    
    # InfoNCE损失
    numerator = torch.exp(positive_pairs / temperature)
    denominator = numerator + torch.exp(negative_similarities / temperature).sum(dim=1)
    
    loss = -torch.log(numerator / denominator).mean()
    return loss