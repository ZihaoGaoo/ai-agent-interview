# 大语言模型 (LLM) 面试题汇总

## 1. Transformer 基础

### Q1: 什么是 Transformer? 它的核心组件有哪些?

**Transformer** 是 2017 年提出的基于注意力机制的深度学习架构,完全摒弃了 RNN/CNN 结构。

**核心组件**:
- **Multi-Head Attention**: 多头注意力,并行捕捉多种关系
- **Feed-Forward Network**: 前馈神经网络
- **Positional Encoding**: 位置编码,引入序列位置信息
- **Add & Norm**: 残差连接和层归一化

```python
# 简化的 Multi-Head Attention
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(attn_output)
```

### Q2: 为什么 Transformer 需要位置编码?

**原因**: Attention 机制本身不包含位置信息,相同词汇在不同位置会有相同的输出。

**位置编码方式**:
1. **Sinusoidal PE**: 正弦周期编码
2. **Learned PE**: 可学习的参数

```python
import torch
import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    # Sinusoidal 位置编码
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return torch.FloatTensor(pe)
```

### Q3: Transformer vs RNN/LSTM 的优势?

| 方面 | Transformer | RNN/LSTM |
|------|-------------|----------|
| 并行计算 | ✅ 可并行 | ❌ 串行 |
| 长距离依赖 | ✅ O(1) | ❌ O(n) 梯度问题 |
| 训练速度 | ✅ 快 | ❌ 慢 |
| 内存 | ❌ O(n²) | ✅ O(n) |

## 2. LLM 训练与微调

### Q4: LLM 的训练流程?

```
预训练 (Pretraining) → 对齐微调 (Fine-tuning) → 人类对齐 (RLHF)
```

```python
# 简化的预训练流程
class Pretrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def compute_loss(self, input_ids, labels):
        outputs = self.model(input_ids)
        # Next token prediction loss
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), 
                                     shift_labels.view(-1))
        return loss
```

### Q5: 什么是 LoRA? 它的原理?

**LoRA (Low-Rank Adaptation)**: 通过低秩矩阵近似来实现参数高效微调。

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.rank = rank
        # 原始权重冻结
        self.weight = None
        # 新增的低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0
    
    def forward(self, x):
        if self.weight is None:
            return torch.matmul(x, self.lora_A @ self.lora_B) * self.scaling
        # 原始权重 + LoRA
        return torch.matmul(x, self.weight) + \
               torch.matmul(x, self.lora_A @ self.lora_B) * self.scaling
```

**优势**:
- 参数量减少 90%+
- 无推理延迟
- 可插拔

### Q6: 什么是 RLHF? 如何实现?

**RLHF (Reinforcement Learning from Human Feedback)**:
1. 收集人类反馈数据
2. 训练奖励模型 (Reward Model)
3. 使用 PPO 算法优化

```python
# 简化的 RLHF 流程
class RLHF:
    def __init__(self, sft_model, reward_model, ref_model):
        self.sft_model = sft_model
        self.reward_model = reward_model
        self.ref_model = ref_model
    
    def compute_ppo_loss(self, prompts, responses, old_log_probs, advantages):
        # 1. 计算新策略的 log_prob
        logits = self.sft_model(responses).logits
        new_log_probs = torch.log_softmax(logits, dim=-1)
        
        # 2. PPO 裁剪损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 0.8, 1.2) * advantages
        loss = -torch.min(ratio * advantages, clipped).mean()
        
        # 3. KL 散度惩罚
        kl = (self.ref_model(responses) - self.sft_model(responses))
        loss += 0.01 * kl
        
        return loss
```

### Q7: LLM 推理优化有哪些方法?

| 方法 | 描述 |
|------|------|
| **量化** | INT8/INT4 量化,减少显存 |
| **KV Cache** | 缓存注意力键值对 |
| **Flash Attention** | IO 优化的注意力计算 |
| **Continuous Batching** | 动态批次处理 |
| **Speculative Decoding** | 投机解码加速 |

```python
# KV Cache 示例
class KVCacheAttention(nn.Module):
    def __init__(self):
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, q, k, v, use_cache=True):
        if use_cache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=1)
                self.v_cache = torch.cat([self.v_cache, v], dim=1)
            k, v = self.k_cache, self.v_cache
        
        # 计算注意力
        attn_output = self.scaled_dot_product_attention(q, k, v)
        return attn_output
```

## 3. Prompt Engineering

### Q8: 如何设计有效的 Prompt?

**CRISP 原则**:
- **C**larity: 清晰明确
- **R**elevance: 相关上下文
- **I**nformation: 充足信息
- **S**pecificity: 具体细节
- **P**recision: 精确输出格式

```python
# Good vs Bad Prompt 对比
BAD_PROMPT = "写代码"

GOOD_PROMPT = """
你是一位Python后端工程师。请帮我写一个用户登录的API接口。

要求:
1. 使用Flask框架
2. 密码用bcrypt加密存储
3. 返回JWT token
4. 包含异常处理

请给出完整的可运行代码。
"""
```

### Q9: 什么是 Chain-of-Thought (CoT)?

思维链提示,让模型逐步推理。

```python
# Standard prompt
prompt = "What is 13 + 27?"

# CoT prompt
cot_prompt = """
Q: What is 13 + 27?
Let's think step by step.
13 + 27 = 40
So the answer is 40.

Q: What is 125 * 4?
Let's think step by step.
"""
```

## 4. 常见面试题

### Q10: Transformer 中为什么用 LayerNorm 而不是 BatchNorm?

- **BatchNorm**: 跨样本归一化,对序列长度敏感
- **LayerNorm**: 跨特征归一化,更稳定

```python
# LayerNorm 在每个样本内归一化
# 输入: [batch, seq_len, hidden_dim]
# LayerNorm 对 [hidden_dim] 归一化
```

### Q11: 注意力机制的复杂度是多少?

**O(n² × d)**:
- n: 序列长度
- d: 隐藏层维度

**优化方法**:
- Sparse Attention: O(n√n)
- Linear Attention: O(n)

### Q12: 如何处理 LLM 的幻觉问题?

1. **RAG**: 检索增强生成
2. **CoT**: 思维链推理
3. **事实核查**: 交叉验证
4. **微调**: 使用高质量数据
5. **控制温度**: 降低随机性

---

> 参考: Attention is All You Need, LLM 论文集合
