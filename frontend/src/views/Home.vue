<template>
  <div class="home">
    <el-card class="intro-card">
      <h2>🎯 面试目标岗位</h2>
      <el-descriptions :column="2" border>
        <el-descriptions-item label="岗位">阿里云智能 - AI Agent 研发工程师</el-descriptions-item>
        <el-descriptions-item label="方向">AI Coding - 智能化代码生成</el-desielements-item>
        <el-descriptions-item label="地点">杭州</el-descriptions-item>
        <el-descriptions-item label="经验要求">2年+</el-descriptions-item>
      </el-descriptions>
    </el-card>
    
    <el-row :gutter="20" class="tech-stack">
      <el-col :span="8" v-for="item in techItems" :key="item.title">
        <el-card class="tech-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span class="icon">{{ item.icon }}</span>
              <span>{{ item.title }}</span>
            </div>
          </template>
          <ul>
            <li v-for="tech in item.techs" :key="tech">{{ tech }}</li>
          </ul>
        </el-card>
      </el-col>
    </el-row>
    
    <el-card class="roadmap-card">
      <template #header>
        <h3>📚 复习路线</h3>
      </template>
      <el-timeline>
        <el-timeline-item 
          v-for="(item, index) in roadmap" 
          :key="index"
          :timestamp="item.timestamp"
          placement="top"
        >
          <el-card>
            <h4>{{ item.title }}</h4>
            <p>{{ item.desc }}</p>
          </el-card>
        </el-timeline-item>
      </el-timeline>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElCard, ElRow, ElCol, ElDescriptions, ElDescriptionsItem, ElTimeline, ElTimelineItem } from 'element-plus'

const techItems = ref([
  {
    icon: '🐍',
    title: '编程语言',
    techs: ['Python', 'Go', 'Java', 'TypeScript']
  },
  {
    icon: '🧠',
    title: '大模型技术',
    techs: ['Transformer', 'LLM 微调', '推理优化', 'LangChain']
  },
  {
    icon: '🤖',
    title: 'AI Agent',
    techs: ['RAG', 'Agent Framework', 'MCP 协议', '多智能体']
  }
])

const roadmap = ref([
  { timestamp: '第1周', title: 'Python 基础', desc: '巩固 Python 核心知识，包括语法、装饰器、异步编程等' },
  { timestamp: '第2周', title: '大模型基础', desc: 'Transformer、LLM 训练、微调、RLHF 等' },
  { timestamp: '第3周', title: 'AI Agent', desc: 'LangChain、ReAct、Agent 框架、工具调用' },
  { timestamp: '第4周', title: 'RAG 技术', desc: '向量检索、分块策略、Reranking、向量数据库' },
  { timestamp: '第5周', title: '系统设计', desc: '分布式架构、高并发、CI/CD' },
  { timestamp: '第6周', title: '算法准备', desc: '手写代码、算法题练习' }
])
</script>

<style scoped>
.home {
  padding: 20px;
}

.intro-card {
  margin-bottom: 20px;
}

.tech-stack {
  margin: 20px 0;
}

.tech-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
}

.icon {
  font-size: 24px;
}

ul {
  list-style: none;
  padding: 0;
}

li {
  padding: 8px 0;
  border-bottom: 1px solid #eee;
}

li:last-child {
  border-bottom: none;
}

.roadmap-card h3 {
  margin: 0;
}
</style>
