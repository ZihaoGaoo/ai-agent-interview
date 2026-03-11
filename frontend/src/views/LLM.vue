<template>
  <div class="llm-page">
    <h2>🧠 大模型 (LLM) 面试题</h2>
    
    <el-collapse v-model="activeNames">
      <el-collapse-item title="Transformer 核心组件" name="1">
        <div class="answer">
          <ul>
            <li><strong>Multi-Head Attention</strong>: 多头注意力</li>
            <li><strong>Feed-Forward Network</strong>: 前馈神经网络</li>
            <li><strong>Positional Encoding</strong>: 位置编码</li>
            <li><strong>Add & Norm</strong>: 残差连接和层归一化</li>
          </ul>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="Transformer vs RNN" name="2">
        <div class="answer">
          <table>
            <tr><th>方面</th><th>Transformer</th><th>RNN</th></tr>
            <tr><td>并行计算</td><td>✅</td><td>❌</td></tr>
            <tr><td>长距离依赖</td><td>✅ O(1)</td><td>❌ O(n)</td></tr>
            <tr><td>复杂度</td><td>O(n²×d)</td><td>O(n×d²)</td></tr>
          </table>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="LLM 训练流程" name="3">
        <div class="answer">
          <p>预训练 → 对齐微调 → RLHF</p>
          <pre><code># 1. Pretraining - Next Token Prediction
# 2. SFT - Supervised Fine-tuning  
# 3. RLHF - Reinforcement Learning from Human Feedback
#    - 训练 Reward Model
#    - PPO 优化</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="LoRA 原理" name="4">
        <div class="answer">
          <p><strong>Low-Rank Adaptation</strong>: 通过低秩矩阵近似实现参数高效微调</p>
          <pre><code>class LoRALayer:
    def __init__(self, in_features, out_features, rank=8):
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x):
        # W0 + BA
        return torch.matmul(x, self.weight) + \
               torch.matmul(x, self.lora_A @ self.lora_B) * scaling</code></pre>
          <p>优势: 参数量减少 90%+，无推理延迟</p>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="LLM 推理优化方法" name="5">
        <div class="answer">
          <ul>
            <li><strong>量化</strong>: INT8/INT4，减少显存</li>
            <li><strong>KV Cache</strong>: 缓存注意力键值对</li>
            <li><strong>Flash Attention</strong>: IO 优化的注意力计算</li>
            <li><strong>Continuous Batching</strong>: 动态批次处理</li>
            <li><strong>Speculative Decoding</strong>: 投机解码</li>
          </ul>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="为什么用 LayerNorm?" name="6">
        <div class="answer">
          <p><strong>BatchNorm</strong>: 跨样本归一化，对序列长度敏感</p>
          <p><strong>LayerNorm</strong>: 跨特征归一化，更稳定，Transformer 更适合</p>
        </div>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElCollapse, ElCollapseItem } from 'element-plus'

const activeNames = ref(['1'])
</script>

<style scoped>
.llm-page {
  padding: 20px;
}

.answer {
  padding: 10px;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}

th {
  background: #f5f5f5;
}

pre {
  background: #f5f5f5;
  padding: 15px;
  border-radius: 5px;
  overflow-x: auto;
}

ul {
  padding-left: 20px;
}
</style>
