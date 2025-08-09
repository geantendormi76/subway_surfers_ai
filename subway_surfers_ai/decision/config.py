# C:\Users\zhz\Deepl\subway_surfers_ai\subway_surfers_ai\decision\config.py

class TrainConfig:
    """
    训练超参数配置
    """
    def __init__(self, **kwargs):
        self.device = 'cuda'
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 0.1
        self.num_workers = 8
        self.sequence_length = 30
        self.action_resample_window = 15
        self.random_interval = 3
        for k, v in kwargs.items():
            setattr(self, k, v)

class ModelConfig:
    """
    模型架构超参数配置
    """
    def __init__(self, **kwargs):
        self.state_dim = 8
        self.act_dim = 5
        self.n_embd = 128
        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.1
        
        # --- [核心修正] ---
        # block_size 现在明确指代 Transformer 能处理的最大序列长度
        # 它应该等于 训练时使用的 sequence_length * 3
        self.block_size = 30 * 3 # 默认值
        
        self.max_timestep = 15000
        
        for k, v in kwargs.items():
            setattr(self, k, v)