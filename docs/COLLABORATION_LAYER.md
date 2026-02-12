# Collaboration Layer - 协作者层使用文档

## 概述

协作者层是Meta Quantum Field的协作核心，旨在实现人类与AI的平等协作关系。

**核心设计原则：**
- 不是回答问题，而是提供新视角
- 不是给出答案，而是帮助用户发现
- 不是服从指令，而是平等对话
- 不是完成任务，而是共同探索

## 核心组件

### 1. Collaborator - 协作者核心

协作者入口，根据用户输入类型自动选择协作方式。

```python
from collaboration import Collaborator

collab = Collaborator()

result = await collab.collaborate(
    user_id="user_123",
    session_id="session_456",
    user_input="我认为这个项目应该采用传统开发方式",
    context={"domain": "项目管理"}
)
```

### 2. ThinkingExpander - 思维扩展器

扩展用户思维，发现盲点。

```python
from collaboration import ThinkingExpander

expander = ThinkingExpander()

# 扩展思维
expansion = await expander.expand(
    "我认为应该这样做...",
    {}
)

# 发现盲点
blind_spots = await expander.discover_blind_spots(
    "我认为应该这样做..."
)

# 生成挑战性问题
questions = await expander.generate_challenging_questions(
    "我认为应该这样做..."
)
```

### 3. ProblemReshaper - 问题重塑器

重新定义问题，发现新可能。

```python
from collaboration import ProblemReshaper

reshaper = ProblemReshaper()

# 分析问题
analysis = await reshaper.analyze_problem(
    "这个问题无法解决..."
)

# 重塑问题
reshaped = await reshaper.reshape(
    "这个问题无法解决..."
)

# 发现替代方案
alternatives = await reshaper.discover_alternatives(
    "这个问题无法解决..."
)
```

### 4. KnowledgeIntegrator - 知识整合器

跨领域连接，底层逻辑贯通。

```python
from collaboration import KnowledgeIntegrator

integrator = KnowledgeIntegrator()

# 识别领域
domains = await integrator.identify_domains(
    "用物理学的逻辑来理解经济学..."
)

# 发现连接
connections = await integrator.find_connections(
    domains,
    "用物理学的逻辑来理解经济学..."
)

# 整合知识
integration = await integrator.integrate(
    domains,
    connections
)

# 生成视角
perspectives = await integrator.generate_perspectives(
    domains,
    "用物理学的逻辑来理解经济学..."
)
```

### 5. PerspectiveGenerator - 全新视角生成器

跳出惯性，发现新可能。

```python
from collaboration import PerspectiveGenerator

generator = PerspectiveGenerator()

# 生成多维视角
perspectives = await generator.generate(
    "人工智能的未来",
    {}
)

# 挑战惯性思维
challenges = await generator.challenge_assumptions(
    "人工智能的未来"
)

# 生成反常识视角
counter_intuitive = await generator.generate_counter_intuitive(
    "人工智能的未来"
)
```

### 6. DialogueEngine - 深度对话引擎

苏格拉底式追问，深度对话。

```python
from collaboration import DialogueEngine

engine = DialogueEngine()

# 深度对话
dialogue = await engine.dialogue(
    "我想聊聊人生的意义...",
    {}
)

# 苏格拉底式问题
questions = await engine.socratic_questions(
    "我想聊聊人生的意义..."
)

# 反思
reflection = await engine.reflection(
    "我想聊聊人生的意义..."
)
```

### 7. Learner - 学习系统

从用户反馈中学习，越来越了解用户。

```python
from collaboration import Learner

learner = Learner()

# 从协作中学习
await learner.learn_from_collaboration(
    user_id="user_123",
    user_input="...",
    collaborator_response={...}
)

# 获取用户画像
profile = await learner.get_user_profile("user_123")

# 获取协作总结
summary = await learner.get_collaboration_summary("user_123")

# 指导协作者
result = await learner.teach_collaborator(
    lesson="我喜欢简洁的回应",
    context="在技术讨论中"
)

# 获取相互理解报告
understanding = await learner.get_mutual_understanding("user_123")
```

## API接口

### 协作者入口

```
POST /api/collaboration/collaborate
{
    "user_id": "user_123",
    "session_id": "session_456",
    "user_input": "我认为应该这样做...",
    "context": {}
}
```

### 思维扩展

```
POST /api/collaboration/expand
{
    "user_thinking": "我认为应该这样做...",
    "context": {}
}
```

### 问题重塑

```
POST /api/collaboration/reshape
{
    "problem": "这个问题无法解决...",
    "context": {}
}
```

### 知识整合

```
POST /api/collaboration/integrate
{
    "knowledge_request": "用物理学的逻辑来理解经济学...",
    "context": {}
}
```

### 全新视角

```
POST /api/collaboration/perspective
{
    "topic": "人工智能的未来",
    "context": {}
}
```

### 深度对话

```
POST /api/collaboration/dialogue
{
    "dialogue_input": "我想聊聊人生的意义...",
    "context": {}
}
```

### 学习系统

```
GET /api/collaboration/learn/profile/{user_id}
GET /api/collaboration/learn/summary/{user_id}
POST /api/collaboration/learn/teach
{
    "lesson": "我喜欢简洁的回应",
    "context": "在技术讨论中"
}
GET /api/collaboration/learn/understanding/{user_id}
```

## Python客户端

```python
from collaboration_client import (
    CollaborationClient,
    expand_thinking,
    reshape_problem,
    integrate_knowledge,
    generate_perspectives,
    deep_dialogue,
)

# 客户端方式
client = CollaborationClient()
result = await client.expand_thinking("我认为应该这样做...")
await client.close()

# 便捷函数
result = await expand_thinking("我认为应该这样做...")
result = await reshape_problem("这个问题无法解决...")
result = await integrate_knowledge("用物理学理解经济学...")
result = await generate_perspectives("人工智能的未来...")
result = await deep_dialogue("我想聊聊人生的意义...")
```

## 协作模式

### 思维分享模式

当用户说"我认为"、"我觉得"、"我的想法是"时：
- 协作者扩展用户思维
- 发现盲点
- 生成挑战性问题

### 问题陈述模式

当用户说"问题"、"无法解决"、"stuck"时：
- 协作者重塑问题
- 发现问题之外的可能性
- 提供转化视角

### 知识请求模式

当用户说"怎么理解"、"什么意思"、"什么关系"时：
- 协作者识别知识领域
- 发现领域间连接
- 提供跨领域视角

### 视角请求模式

当用户说"你怎么看"、"有什么看法"、"不同角度"时：
- 协作者生成多维视角
- 挑战惯性思维
- 提供反常识视角

### 深度对话模式

当用户说"深度对话"、"聊聊"、"讨论"时：
- 协作者进行苏格拉底式追问
- 引导深度思考
- 记录反思历程

## 协作者vs传统AI

| 传统AI | 协作者 |
|---------|--------|
| 回答问题 | 扩展思维 |
| 执行任务 | 重塑问题 |
| 提供答案 | 发现盲点 |
| 服从指令 | 平等对话 |
| 完成任务 | 相互学习 |
| 一次性交互 | 长期协作 |

## 学习系统

协作者不仅提供价值，也从协作中学习：

1. **用户画像**：记录用户的交互风格、偏好、话题
2. **协作历史**：保存每次协作的记录
3. **偏好学习**：从反馈中学习用户的偏好
4. **关系深化**：越来越了解用户，形成真正的协作关系

## 使用建议

1. **思维扩展**：当用户分享想法时使用
2. **问题重塑**：当用户遇到问题卡住时使用
3. **知识整合**：当用户需要跨领域理解时使用
4. **视角生成**：当用户需要新思路时使用
5. **深度对话**：当用户想要深入探讨时使用

## 下一步

- 运行真实协作实验
- 收集用户反馈
- 持续优化协作者能力
- 深化学习系统
