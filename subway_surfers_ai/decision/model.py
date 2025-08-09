# C:\Users\zhz\Deepl\subway_surfers_ai\subway_surfers_ai\decision\model.py (v4 - 索引安全版)

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class StARformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.state_patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels=config.state_dim, out_channels=config.n_embd, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.n_embd * 8 * 4, config.n_embd)
        )
        
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.action_encoder = nn.Embedding(config.act_dim, config.n_embd)
        self.rtg_encoder = nn.Linear(1, config.n_embd)
        self.timestep_encoder = nn.Embedding(config.max_timestep, config.n_embd)
        
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.action_head = nn.Linear(config.n_embd, config.act_dim)

    def forward(self, states, actions, rtgs, timesteps):
        B, T, C, H, W = states.shape

        state_embeddings_global = self.state_patch_encoder(states.view(-1, C, H, W)).view(B, T, -1)
        action_embeddings = self.action_encoder(actions)
        rtg_embeddings = self.rtg_encoder(rtgs)
        
        # --- [核心修正] ---
        # 1. 裁剪timestep索引，防止其超出Embedding层的范围，这是导致CUDA assert的直接原因。
        timesteps = timesteps.clamp(0, self.config.max_timestep - 1)
        timestep_embeddings = self.timestep_encoder(timesteps)
        # --- [修正结束] ---
        
        token_sequence = torch.stack(
            [rtg_embeddings, state_embeddings_global, action_embeddings], dim=2
        ).view(B, 3 * T, self.config.n_embd)
        
        # --- [鲁棒性修正] ---
        # 2. 确保后续操作的序列长度与token_sequence的实际长度对齐，避免因广播导致尺寸不匹配。
        current_seq_len = token_sequence.shape[1]
        
        token_sequence += self.pos_emb[:, :current_seq_len, :]
        
        time_embedding_sequence = timestep_embeddings.repeat_interleave(3, dim=1)
        token_sequence += time_embedding_sequence[:, :current_seq_len, :]
        # --- [修正结束] ---

        x = self.drop(token_sequence)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 从交错序列中只取出对应于 State 的部分来预测下一个动作
        state_preds = x[:, 1::3, :]
        action_logits = self.action_head(state_preds)
        
        return action_logits