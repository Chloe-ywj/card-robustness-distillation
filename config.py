# config.py
# 模型参数
#TEACHER_MODEL = "textattack/bert-base-uncased-imdb"
TEACHER_MODEL = "dfurman/deberta-v2-xxl-imdb-v0.1"
STUDENT_MODEL = "distilbert-base-uncased"

# 数据集参数
DATASET_NAME = "./datasets/aclImdb" 
MAX_LENGTH = 256
BATCH_SIZE = 32
SEED=42

# 训练参数
EPOCHS = 20
LEARNING_RATE = 2e-5

# 损失函数权重 (gamma，alpha 和 beta)
GAMMA = 1.0
ALPHA = 0.5 # 知识蒸馏损失权重 0.4 0.3
BETA = 0.5  # 对比表征传递损失权重0.6 0.7
TEMPERATURE = 3.0 # 蒸馏温度

# 对抗攻击参数
FGSM_EPSILON = 0.12 # FGSM 扰动大小

NUM_ATTACK_SAMPLES = 20
