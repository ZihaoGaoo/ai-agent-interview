# AI Agent 面试题汇总

## 1. Agent 基础概念

### Q1: 什么是 AI Agent? 它和传统程序有什么区别?

**AI Agent** = 大模型 + 工具 + 记忆 + 规划能力

| 特性 | 传统程序 | AI Agent |
|------|----------|----------|
| 逻辑 | 预设规则 | LLM 推理 |
| 输入 | 固定格式 | 自然语言 |
| 行为 | 确定性 | 概率性 |
| 能力 | 单一 | 多任务 |

```python
# 简化的 Agent 架构
class AIAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm          # 大模型
        self.tools = tools      # 工具集
        self.memory = memory    # 记忆
    
    def run(self, task):
        # 1. 理解任务
        plan = self.llm.plan(task)
        
        # 2. 执行计划
        for step in plan:
            # 3. 使用工具
            result = self.execute_tool(step)
            
            # 4. 观察环境
            self.memory.add(result)
            
            # 5. 反思调整
            if not self.is_progress(result):
                plan = self.replan(task, result)
        
        return self.memory.get_result()
```

### Q2: Agent 的核心组件有哪些?

```
┌─────────────────────────────────┐
│           Agent                 │
├─────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐     │
│  │   LLM   │  │  Planner │     │
│  └─────────┘  └──────────┘     │
│  ┌─────────┐  ┌──────────┐     │
│  │ Memory  │  │  Tools   │     │
│  └─────────┘  └──────────┘     │
│  ┌─────────┐  ┌──────────┐     │
│  │   Env   │  │ Evaluator│     │
│  └─────────┘  └──────────┘     │
└─────────────────────────────────┘
```

- **LLM**: 推理引擎
- **Planner**: 任务分解
- **Memory**: 短期/长期记忆
- **Tools**: 搜索引擎、API、计算器等
- **Environment**: 交互环境
- **Evaluator**: 结果评估

## 2. Agent 框架

### Q3: LangChain 的核心组件?

```python
from langchain.agents import AgentExecutor, Tool
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper

# 1. 定义工具
search = SerpAPIWrapper()
tools = [
    Tool(name="Search", func=search.run, description="搜索信息")
]

# 2. 创建 Agent
llm = OpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

# 3. 执行
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.run("2024年奥运会金牌榜")
```

### Q4: 什么是 ReAct (Reasoning + Acting)?

ReAct = 推理 + 行动,让 Agent 结合思考和工具使用。

```python
# ReAct 的核心循环
thoughts = []
actions = []
observations = []

while not done:
    # 1. 思考
    thought = llm.think(task, thoughts, actions, observations)
    thoughts.append(thought)
    
    # 2. 行动
    action = llm.select_action(thought, tools)
    actions.append(action)
    
    # 3. 观察
    observation = execute(action)
    observations.append(observation)
    
    # 4. 评估
    if evaluate(observation):
        done = True
```

### Q5: 什么是 Agentic Workflow?

代理工作流,通过多步骤协作完成任务。

```python
# 代理工作流示例
class AgenticWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
    
    def run(self, topic):
        # 并行研究
        research_results = self.researcher.concurrent_search([
            f"{topic} 技术原理",
            f"{topic} 应用场景", 
            f"{topic} 发展趋势"
        ])
        
        # 串行写作
        draft = self.writer.write(research_results)
        
        # 迭代审查
        for _ in range(3):
            feedback = self.reviewer.review(draft)
            if feedback.is_good:
                break
            draft = self.writer.revise(draft, feedback)
        
        return draft
```

## 3. 多 Agent 系统

### Q6: 多 Agent 协作模式有哪些?

| 模式 | 描述 | 场景 |
|------|------|------|
| **Sequential** | 串行执行 | 流程任务 |
| **Parallel** | 并行执行 | 独立任务 |
| **Supervisor** | 主管分配 | 任务分发 |
| **Router** | 路由选择 | 分类任务 |
| **Swarm** | 蜂群协作 | 复杂问题 |

```python
# Supervisor 模式
class SupervisorAgent:
    def __init__(self):
        self.agents = {
            "research": ResearchAgent(),
            "code": CodeAgent(),
            "write": WriterAgent()
        }
        self.llm = OpenAI()
    
    def route(self, task):
        # LLM 决定使用哪个 Agent
        decision = self.llm.decide(f"任务: {task}, 可用Agent: {list(self.agents.keys())}")
        agent_name = decision["agent"]
        return self.agents[agent_name].run(task)
```

### Q7: 什么是 MCP (Model Context Protocol)?

MCP 是 Anthropic 提出的 Agent 通信协议。

```python
# MCP 协议示例
class MCPMessage:
    def __init__(self, role, content, tool_calls=None):
        self.role = role          # user/assistant/system
        self.content = content    # 文本内容
        self.tool_calls = tool_calls or []  # 工具调用
    
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls]
        }
```

## 4. Agent 评测

### Q8: 如何评测 Agent 能力?

| 维度 | 指标 |
|------|------|
| **准确性** | 任务完成率、答案正确率 |
| **效率** | 步数、时间、成本 |
| **稳定性** | 一致性、方差 |
| **安全性** | 拒绝率、越狱率 |
| **可解释性** | 可追踪性、可调试性 |

```python
# Agent 评测框架
class AgentBenchmark:
    def __init__(self, tasks, metrics):
        self.tasks = tasks
        self.metrics = metrics
    
    def evaluate(self, agent):
        results = []
        for task in self.tasks:
            result = agent.run(task)
            scores = {}
            for metric in self.metrics:
                scores[metric] = metric.measure(task, result)
            results.append(scores)
        
        return self.aggregate(results)
```

## 5. 实际应用

### Q9: 如何设计一个 AI Coding Agent?

```python
class AICodingAgent:
    def __init__(self):
        self.llm = CodeGenLLM()
        self.tools = [
            SearchTool(),       # 搜索文档
            GrepTool(),        # 代码搜索
            BashTool(),        # 执行命令
            EditTool(),        # 编辑文件
            ReadTool(),        # 读取文件
        ]
        self.memory = ContextWindow()
    
    def implement_feature(self, feature_request):
        # 1. 理解需求
        spec = self.llm.understand(feature_request)
        
        # 2. 分析现有代码
        relevant_code = self.search_relevant_code(spec)
        
        # 3. 生成代码
        plan = self.plan_implementation(spec, relevant_code)
        
        # 4. 执行并验证
        for step in plan:
            code = self.llm.generate(step, relevant_code)
            self.apply_and_test(code)
        
        return self.get_summary()
```

### Q10: Agent 常见的失败模式?

1. **循环**: 重复执行相同动作
2. **幻觉**: 虚构不存在的工具或结果
3. **过度推理**: 想太多但不做
4. **工具滥用**: 频繁调用工具导致效率低
5. **上下文丢失**: 长任务中丢失重要信息

---

> 参考: LangChain, AutoGPT, ReAct, Claude Agent 论文
