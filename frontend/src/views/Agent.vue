<template>
  <div class="agent-page">
    <h2>🤖 AI Agent 面试题</h2>
    
    <el-collapse v-model="activeNames">
      <el-collapse-item title="什么是 AI Agent?" name="1">
        <div class="answer">
          <p><strong>AI Agent</strong> = 大模型 + 工具 + 记忆 + 规划能力</p>
          <table>
            <tr><th>组件</th><th>作用</th></tr>
            <tr><td>LLM</td><td>推理引擎</td></tr>
            <tr><td>Planner</td><td>任务分解</td></tr>
            <tr><td>Memory</td><td>短期/长期记忆</td></tr>
            <tr><td>Tools</td><td>搜索、API、计算器等</td></tr>
          </table>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="LangChain 核心组件" name="2">
        <div class="answer">
          <pre><code>from langchain.agents import AgentExecutor
from langchain.llms import OpenAI

# 定义工具
tools = [Tool(name="Search", func=search.run)]

# 创建 Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 执行
result = agent_executor.run("query")</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="ReAct 模式" name="3">
        <div class="answer">
          <p><strong>Reasoning + Acting</strong>: 思考 → 行动 → 观察 → 反思</p>
          <pre><code>while not done:
    thought = llm.think(task)    # 思考
    action = llm.act(thought)    # 行动
    obs = execute(action)        # 观察
    if evaluate(obs):            # 评估
        done = True</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="多 Agent 协作模式" name="4">
        <div class="answer">
          <ul>
            <li><strong>Sequential</strong>: 串行执行</li>
            <li><strong>Parallel</strong>: 并行执行</li>
            <li><strong>Supervisor</strong>: 主管分配任务</li>
            <li><strong>Router</strong>: 路由选择</li>
            <li><strong>Swarm</strong>: 蜂群协作</li>
          </ul>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="MCP 协议" name="5">
        <div class="answer">
          <p><strong>Model Context Protocol</strong>: Anthropic 提出的 Agent 通信协议</p>
          <p>用于标准化 Agent 之间的通信、工具调用等</p>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="Agent 常见失败模式" name="6">
        <div class="answer">
          <ul>
            <li><strong>循环</strong>: 重复执行相同动作</li>
            <li><strong>幻觉</strong>: 虚构不存在的工具</li>
            <li><strong>过度推理</strong>: 想太多但不做</li>
            <li><strong>工具滥用</strong>: 频繁调用工具</li>
            <li><strong>上下文丢失</strong>: 长任务中丢失信息</li>
          </ul>
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
.agent-page {
  padding: 20px;
}

.answer {
  padding: 10px;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
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
