# Quantum Field Agent 完全指南
## 量子场架构智能体系统

---

## 一、核心理念（为什么这样做）

### 1.1 传统架构 vs 量子场架构

| 维度 | 传统Agent架构 | 量子场架构 |
|------|-------------|-----------|
| **结构** | 分层（感知→推理→行动） | 场域（共振→干涉→坍缩） |
| **组件关系** | 硬性连接（if/else） | 软连接（概率权重） |
| **记忆** | 外部检索（RAG） | 场内驻留（上下文） |
| **工具调用** | 离散选择（MCP/Function Call） | 连续叠加（多技能共振） |
| **可观测性** | 过程日志（每一步记录） | 边界快照（仅I/O） |
| **哲学** | 机械论（A→B→C） | 量子论（叠加态+坍缩） |

### 1.2 关键洞察

**"过程即幻觉，I/O即实相"**
- 人类需要过程可观测（满足控制欲）
- AI实际只有输入（Intent）和输出（Result）是真实的
- 中间路径是概率云，观测即坍缩

**LLM作为场介质**
- 不是"被调用的工具"，而是"能力干涉发生的场所"
- 上下文窗口 = 量子场容器
- Attention机制 = 技能共振

---

## 二、系统架构（三要素）

### 2.1 核心公式
智能 = LLM(∑(技能向量 × 记忆向量 × 意图向量))
输出 = Collapse(智能, 约束条件)


### 2.2 三要素详解

#### 要素1：LLM（场介质/大脑）
- **作用**：提供语义理解、推理、生成能力
- **配置**：支持OpenAI/Claude/DeepSeek/本地模型
- **关键参数**：上下文窗口越大，场容量越大

#### 要素2：Skills（技能库/手脚）
- **本质**：函数集合（Python函数）
- **注册**：通过描述让LLM知道"我能做什么"
- **激活**：LLM根据意图自动选择（可多选并行）
- **示例**：
  - `search_weather`：查天气
  - `calculate`：数学计算
  - `send_email`：发送邮件
  - `save_memory`：保存记忆

#### 要素3：Memory（记忆库/经验）
- **存储**：SQLite数据库（轻量，文件级）
- **内容**：对话历史、用户偏好、长期事实
- **使用**：自动注入LLM上下文，影响共振结果
- **特点**：无需外部检索，直接驻留场内

---

## 三、代码结构说明

### 3.1 文件组织

quantum-field-agent/
├── backend/
│   ├── main.py              # 主程序：FastAPI + LLM集成
│   ├── requirements.txt     # Python依赖
│   ├── .env                 # 环境变量（API密钥）
│   └── quantum_memory.db    # SQLite数据库（自动创建）
├── frontend/
│   └── index.html           # Web界面：量子场可视化
└── README.md                # 本文档


### 3.2 关键代码逻辑

#### 共振阶段（技能选择）
```python
# LLM自动决定激活哪些技能
response = client.chat.completions.create(
    messages=messages,
    tools=available_skills,  # 告诉LLM有哪些工具
    tool_choice="auto"       # 自动选择
)
坍缩阶段（结果生成）

Python：

# 如果有工具调用，执行后再次生成自然语言
if response.tool_calls:
    results = execute_tools(response.tool_calls)
    final_response = client.chat.completions.create(
        messages=messages + results
    )
    
记忆驻留

Python：

# 自动保存到SQLite
save_memory(user_id, role, content)
# 自动读取最近N条作为上下文
memory = get_memory(user_id, limit=10)

四、部署步骤（手把手）

步骤1：环境准备

# 1. 安装Python 3.9+
python --version

# 2. 创建项目文件夹
mkdir quantum-field-agent
cd quantum-field-agent
mkdir backend frontend

# 3. 保存代码文件
# 将本文档中的代码分别保存到对应位置

步骤2：后端配置

cd backend

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env，填入你的OpenAI API Key

.env文件示例：

OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

步骤3：启动系统

# 启动后端（在backend目录）
python main.py
# 看到：Uvicorn running on http://0.0.0.0:8000

# 启动前端（直接双击或用简易服务器）
cd ../frontend
python -m http.server 8080
# 或直接双击index.html用浏览器打开

步骤4：验证测试

打开浏览器访问 http://localhost:8000（后端）或打开前端HTML
输入："查北京天气"
观察：界面显示"共振中"→技能节点高亮→返回结果
输入："记住我喜欢喝咖啡"
输入："我喜欢喝什么"（测试记忆）

五、使用指南

5.1 基础用法

直接输入意图（自然语言）：
"查上海明天天气"
"计算 25*4+100"
"发邮件给test@example.com说你好"
"记住我的工位是A-123"
系统会自动：
理解意图（共振）
选择技能（激活场节点）
执行操作（坍缩）
保存记忆（场状态更新）
5.2 领域切换（垂直/全能）

点击界面上的"生活"/"办公"/"计算"按钮：
通用场：加载所有技能（全能模式）
生活场：仅加载天气等生活技能（垂直模式）
办公场：仅加载邮件等办公技能（垂直模式）
原理：通过domain_focus参数过滤技能列表，改变场密度。

5.3 记忆管理

自动记忆：
所有对话自动保存
LLM自动引用历史（如用户之前说过的话）
手动记忆：
"记住我是VIP客户"
"记得下周三是会议"
查看记忆：
访问 http://localhost:8000/memory/user_001
清空记忆：
访问 http://localhost:8000/memory/user_001（DELETE请求）
六、扩展开发

6.1 添加新技能

在main.py中找到register_default_skills，添加：

{
    "name": "your_skill_name",
    "description": "技能描述，LLM靠这个理解何时使用",
    "domain": "your_domain",  # 分类：life/office/math等
    "func": lambda param: "返回结果"
}

示例：添加翻译技能

{
    "name": "translate",
    "description": "翻译文本，参数：text（原文）、target_lang（目标语言）",
    "domain": "office",
    "func": lambda text, target_lang: f"已翻译{text}到{target_lang}"
}

6.2 接入真实API

修改技能函数，接入外部服务：

import requests

def real_weather(city: str) -> str:
    """接入和风天气真实API"""
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.weather.com/v1/current?city={city}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    return f"{city}当前温度：{data['temp']}°C，天气：{data['text']}"
    
6.3 更换LLM模型

修改.env：

# 使用DeepSeek（便宜，中文好）
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-deepseek-key
MODEL_NAME=deepseek-chat

# 或使用Claude（强推理）
OPENAI_BASE_URL=https://api.anthropic.com/v1
# ...配置相应key

七、故障排除

问题1：CORS错误（前端无法连接后端）

症状：浏览器控制台显示"CORS policy"
解决：后端已配置allow_origins=["*"]，确保前端访问http://localhost:8000而非https
问题2：API Key无效

症状：报错"Authentication Error"
解决：检查.env文件中的OPENAI_API_KEY是否以sk-开头
问题3：技能不触发

症状：LLM直接回复，不调用工具
解决：
检查技能description是否清晰描述使用场景
确保tools参数正确传入
尝试更明确的用户输入（如"使用天气工具查北京"）

问题4：记忆不生效

症状：AI忘记之前说过的话
解决：
检查get_memory是否返回数据
确认messages构造时包含记忆
增加limit参数（默认只读最近5条）
八、哲学总结

8.1 为什么这比传统架构好？

极简：3个文件 vs 传统6层架构的20+个服务
统一：没有"Agent vs Tool vs Workflow"的区分，只有"场"
涌现：能力组合是自动的，无需预设工作流
人化：像与人对话一样自然，无需学习"如何下指令"
8.2 边界与限制

当前实现：
适合个人助手、小型企业应用
技能数量<50时效果最佳
单用户响应延迟1-3秒（依赖LLM）
不适合：
超大规模并发（需加Redis缓存）
强审计要求场景（过程不可观测）
多Agent复杂协作（需扩展A2A协议）

九、下一步（进化路径）

语音接入：接入浏览器Web Speech API
多设备：通过WebSocket同步场状态到手机/音箱
视觉能力：添加OCR技能（识别图片文字）
自动化：添加定时触发器（定时任务）
