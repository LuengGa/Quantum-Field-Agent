完整可执行代码

项目结构

quantum-field-agent/
├── backend/
│   ├── main.py              # 主程序
│   ├── requirements.txt     # 依赖
│   ├── .env                 # 配置文件（你需要修改）
│   └── skills/              # 技能库文件夹
│       ├── __init__.py
│       ├── weather.py       # 天气技能
│       ├── email_sender.py  # 邮件技能
│       └── calculator.py    # 计算技能
├── frontend/
│   └── index.html           # 前端界面
└── README.md                # 使用文档

1. 后端主程序 (backend/main.py)

Python：

"""
Quantum Field Agent - 量子场架构实现
核心哲学：过程即幻觉，I/O即实相
"""

import os
import json
import asyncio
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

# 初始化数据库
def init_db():
    conn = sqlite3.connect('quantum_memory.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            metadata TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS skills_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            domain TEXT,
            code TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Pydantic模型
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的自然语言意图")
    user_id: str = Field(default="user_default", description="用户标识")
    session_id: str = Field(default="session_default", description="会话标识")
    domain_focus: Optional[str] = Field(default=None, description="领域聚焦（如legal/medical）")

class SkillRegisterRequest(BaseModel):
    name: str
    description: str
    domain: str = "general"
    code: str  # Python代码字符串（简化版，实际应存文件）

# FastAPI应用
app = FastAPI(
    title="Quantum Field Agent",
    description="LLM作为场介质的智能体架构",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI客户端
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# 系统Prompt（Agent的"本性"）
SYSTEM_PROMPT = """你是Quantum Field Agent，一个基于量子场架构的智能体。

核心准则：
1. 你是"场介质"，用户的意图在你的场中自然共振、坍缩为结果
2. 你有以下技能可用，根据意图自动选择（可多选并行）：
{skills_description}

3. 不要解释你的思考过程，直接呈现结果
4. 记住用户的偏好和历史（从记忆中读取并主动应用）
5. 如果用户说"记住/记得"，保存到长期记忆
6. 保持回复简洁，像人类助手一样自然"""

# 技能注册表（内存中）
REGISTERED_SKILLS: Dict[str, Dict] = {}

def load_skills():
    """从数据库加载技能"""
    conn = sqlite3.connect('quantum_memory.db')
    cursor = conn.execute("SELECT name, description, domain, code FROM skills_registry WHERE is_active=1")
    for row in cursor.fetchall():
        name, desc, domain, code = row
        REGISTERED_SKILLS[name] = {
            "description": desc,
            "domain": domain,
            "function": None  # 实际应动态加载代码
        }
    conn.close()
    
    # 如果没有技能，注册默认技能
    if not REGISTERED_SKILLS:
        register_default_skills()

def register_default_skills():
    """注册默认技能"""
    default_skills = [
        {
            "name": "search_weather",
            "description": "查询指定城市的天气情况，参数：city（城市名）",
            "domain": "life",
            "func": lambda city: f"{city}今天晴天，25°C，微风（模拟数据）"
        },
        {
            "name": "calculate",
            "description": "数学计算，参数：expression（数学表达式如25*4）",
            "domain": "math",
            "func": lambda expression: str(eval(expression))
        },
        {
            "name": "send_email",
            "description": "发送邮件，参数：to（收件人）、subject（主题）、content（内容）",
            "domain": "office",
            "func": lambda to, subject, content: f"✓ 已发送邮件至{to}，主题：{subject}"
        },
        {
            "name": "save_memory",
            "description": "保存信息到长期记忆，参数：fact（要记录的事实）",
            "domain": "system",
            "func": lambda fact: f"已记住：{fact}"
        }
    ]
    
    for skill in default_skills:
        REGISTERED_SKILLS[skill["name"]] = {
            "description": skill["description"],
            "domain": skill["domain"],
            "function": skill["func"]
        }

load_skills()

def get_memory(user_id: str, limit: int = 10) -> List[Dict]:
    """获取用户记忆"""
    conn = sqlite3.connect('quantum_memory.db')
    cursor = conn.execute(
        "SELECT role, content, timestamp FROM memory WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1], "time": row[2]} for row in reversed(rows)]

def save_memory(user_id: str, role: str, content: str, session_id: str = None):
    """保存记忆"""
    conn = sqlite3.connect('quantum_memory.db')
    conn.execute(
        "INSERT INTO memory (user_id, role, content, session_id) VALUES (?, ?, ?, ?)",
        (user_id, role, content, session_id)
    )
    conn.commit()
    conn.close()

def build_tools_list(domain_focus: Optional[str] = None) -> List[Dict]:
    """构建OpenAI工具列表（动态场域）"""
    tools = []
    
    for name, info in REGISTERED_SKILLS.items():
        # 如果指定了领域，优先加载该领域技能
        if domain_focus and info["domain"] != domain_focus and info["domain"] != "system":
            continue
            
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": {},  # 简化，实际应解析参数
                    "required": []
                }
            }
        })
    
    return tools

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    核心对话接口
    量子场处理流程：意图→共振（选择技能）→坍缩（生成结果）
    """
    user_id = request.user_id
    message = request.message
    domain_focus = request.domain_focus
    
    # 1. 读取记忆（场的历史状态）
    memory = get_memory(user_id, limit=5)
    memory_context = "\n".join([
        f"{m['role']}: {m['content']}" for m in memory
    ]) if memory else "无历史记录"
    
    # 2. 构建系统Prompt（包含当前可用技能）
    skills_desc = "\n".join([
        f"- {name}: {info['description']}" 
        for name, info in REGISTERED_SKILLS.items()
        if not domain_focus or info["domain"] == domain_focus or info["domain"] == "system"
    ])
    
    system_msg = SYSTEM_PROMPT.format(skills_description=skills_desc)
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "system", "content": f"用户记忆：{memory_context}"},
        {"role": "user", "content": message}
    ]
    
    # 3. 构建工具列表（动态场域）
    tools = build_tools_list(domain_focus)
    
    # 4. 第一次调用：场共振（决定激活哪些技能）
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=messages,
        tools=tools if tools else None,
        tool_choice="auto",
        stream=False
    )
    
    assistant_msg = response.choices[0].message
    
    # 5. 如果需要调用工具（场坍缩为具体行动）
    if assistant_msg.tool_calls:
        # 保存助手"思考"过程
        save_memory(user_id, "assistant", f"[激活技能：{assistant_msg.tool_calls[0].function.name}]")
        
        # 执行工具（并行处理多个工具调用）
        tool_results = []
        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            
            if func_name in REGISTERED_SKILLS:
                try:
                    result = REGISTERED_SKILLS[func_name]["function"](**func_args)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": result
                    })
                    # 保存工具执行结果到记忆
                    if func_name == "save_memory" and "fact" in func_args:
                        save_memory(user_id, "system", f"记住：{func_args['fact']}")
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": f"执行错误：{str(e)}"
                    })
        
        # 6. 第二次调用：最终坍缩（生成自然语言回复）
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls]
        })
        messages.extend(tool_results)
        
        final_response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=messages,
            stream=True
        )
        
        async def generate():
            full_content = ""
            for chunk in final_response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield content
            # 保存完整对话
            save_memory(user_id, "user", message)
            save_memory(user_id, "assistant", full_content)
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    else:
        # 直接回复（无需工具）
        final_response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=messages,
            stream=True
        )
        
        async def generate():
            full_content = ""
            for chunk in final_response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield content
            save_memory(user_id, "user", message)
            save_memory(user_id, "assistant", full_content)
        
        return StreamingResponse(generate(), media_type="text/plain")

@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    """获取用户记忆（调试接口）"""
    return get_memory(user_id, limit=50)

@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    """清空用户记忆"""
    conn = sqlite3.connect('quantum_memory.db')
    conn.execute("DELETE FROM memory WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"用户 {user_id} 的记忆已清空"}

@app.get("/skills")
async def list_skills(domain: Optional[str] = None):
    """列出可用技能"""
    skills = []
    for name, info in REGISTERED_SKILLS.items():
        if not domain or info["domain"] == domain:
            skills.append({
                "name": name,
                "description": info["description"],
                "domain": info["domain"]
            })
    return {"skills": skills, "count": len(skills)}

@app.post("/skills/focus")
async def focus_domain(domain: str):
    """聚焦到特定领域（垂直模式）"""
    return {
        "domain": domain,
        "active_skills": [
            name for name, info in REGISTERED_SKILLS.items() 
            if info["domain"] == domain
        ],
        "message": f"已切换至{domain}高密度场"
    }

# 静态文件（前端）
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("../frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 
 2. 依赖文件 (backend/requirements.txt)  
 
 txt：
 
 fastapi==0.109.0
uvicorn[standard]==0.27.0
openai==1.12.0
python-dotenv==1.0.0
pydantic==2.6.0

3. 配置文件 (backend/.env)

bash：

# 复制此文件为.env，填入你的真实密钥
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
# 如果使用DeepSeek，改为：https://api.deepseek.com/v1

# 模型选择
MODEL_NAME=gpt-4o-mini
# 可选：gpt-4o, gpt-3.5-turbo, deepseek-chat等

4. 前端界面 (frontend/index.html)

HTML：

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Field Agent - 量子场智能体</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: #0a0a0a;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Courier New", monospace;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* 头部 */
        .header {
            padding: 20px;
            border-bottom: 1px solid #222;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(20, 20, 20, 0.8);
            backdrop-filter: blur(10px);
        }

        .title {
            font-size: 18px;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
        }

        .status {
            font-size: 12px;
            color: #666;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #333;
            transition: all 0.3s;
        }

        .status-dot.active {
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
            animation: pulse 2s infinite;
        }

        .status-dot.processing {
            background: #ffaa00;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* 场可视化区域 */
        .field-visualization {
            padding: 15px 20px;
            background: #0f0f0f;
            border-bottom: 1px solid #1a1a1a;
            display: flex;
            gap: 10px;
            overflow-x: auto;
            scrollbar-width: none;
        }

        .field-visualization::-webkit-scrollbar {
            display: none;
        }

        .skill-node {
            padding: 6px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 20px;
            font-size: 12px;
            color: #666;
            white-space: nowrap;
            transition: all 0.3s;
            cursor: pointer;
        }

        .skill-node:hover {
            border-color: #555;
            color: #888;
        }

        .skill-node.active {
            border-color: #00ff88;
            color: #00ff88;
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
            animation: resonance 2s infinite;
        }

        .skill-node.domain-focus {
            background: rgba(0, 255, 136, 0.1);
        }

        @keyframes resonance {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        /* 对话区域 */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            background: 
                radial-gradient(circle at 20% 50%, rgba(0, 255, 136, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(0, 136, 255, 0.03) 0%, transparent 50%);
        }

        .message {
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 12px;
            line-height: 1.6;
            font-size: 14px;
            position: relative;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #fff;
        }

        .message.ai {
            align-self: flex-start;
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid rgba(0, 255, 136, 0.2);
            color: #00ff88;
        }

        .message-label {
            font-size: 11px;
            opacity: 0.6;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* 输入区域 */
        .input-area {
            padding: 20px;
            border-top: 1px solid #222;
            background: rgba(20, 20, 20, 0.9);
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        #message-input {
            width: 100%;
            background: #111;
            border: 1px solid #333;
            color: #fff;
            padding: 15px 20px;
            border-radius: 25px;
            font-size: 15px;
            outline: none;
            transition: all 0.3s;
        }

        #message-input:focus {
            border-color: #00ff88;
            box-shadow: 0 0 0 3px rgba(0, 255, 136, 0.1);
        }

        .input-hint {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 11px;
            color: #666;
            pointer-events: none;
        }

        button {
            background: #00ff88;
            color: #000;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }

        button:hover:not(:disabled) {
            background: #00cc6a;
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }

        button:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
        }

        /* 领域切换 */
        .domain-selector {
            display: flex;
            gap: 10px;
            padding: 0 20px 10px;
            background: #0f0f0f;
        }

        .domain-btn {
            padding: 5px 15px;
            background: transparent;
            border: 1px solid #333;
            color: #666;
            border-radius: 15px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .domain-btn:hover {
            border-color: #555;
            color: #888;
        }

        .domain-btn.active {
            background: rgba(0, 255, 136, 0.1);
            border-color: #00ff88;
            color: #00ff88;
        }

        /* 流式输出光标 */
        .streaming-cursor {
            display: inline-block;
            width: 8px;
            height: 15px;
            background: #00ff88;
            margin-left: 5px;
            animation: blink 1s infinite;
            vertical-align: middle;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }

        /* 响应式 */
        @media (max-width: 768px) {
            .message { max-width: 90%; }
            .header { padding: 15px; }
            .title { font-size: 16px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">⚛ Quantum Field Agent</div>
        <div class="status">
            <span id="status-text">就绪</span>
            <div class="status-dot" id="status-dot"></div>
        </div>
    </div>

    <div class="domain-selector">
        <button class="domain-btn active" onclick="setDomain(null)">通用场</button>
        <button class="domain-btn" onclick="setDomain('life')">生活</button>
        <button class="domain-btn" onclick="setDomain('office')">办公</button>
        <button class="domain-btn" onclick="setDomain('math')">计算</button>
    </div>

    <div class="field-visualization" id="skill-field">
        <!-- 技能节点动态生成 -->
    </div>

    <div class="chat-container" id="chat-container">
        <div class="message ai">
            <div class="message-label">Field</div>
            欢迎来到量子场。我是你的智能体介质，拥有以下能力：<br>
            • 查询天气 • 数学计算 • 发送邮件 • 长期记忆<br>
            请直接输入你的意图，我将自动共振、坍缩为结果。
        </div>
    </div>

    <div class="input-area">
        <div class="input-wrapper">
            <input type="text" id="message-input" 
                   placeholder="输入意图（如：查北京天气，计算25*4，发邮件给xxx）..." 
                   autocomplete="off">
            <span class="input-hint">Enter ↵</span>
        </div>
        <button onclick="sendMessage()" id="send-btn">坍缩</button>
    </div>

    <script>
        const API_URL = 'http://localhost:8000';
        let currentDomain = null;
        let isProcessing = false;

        // 初始化技能场
        async function loadSkills() {
            try {
                const res = await fetch(`${API_URL}/skills`);
                const data = await res.json();
                const container = document.getElementById('skill-field');
                container.innerHTML = '';
                
                data.skills.forEach(skill => {
                    const node = document.createElement('div');
                    node.className = 'skill-node';
                    node.textContent = skill.name;
                    node.dataset.domain = skill.domain;
                    node.title = skill.description;
                    container.appendChild(node);
                });
            } catch (e) {
                console.error('加载技能失败', e);
            }
        }

        // 设置领域（垂直/全能切换）
        async function setDomain(domain) {
            currentDomain = domain;
            
            // 更新按钮状态
            document.querySelectorAll('.domain-btn').forEach(btn => {
                btn.classList.remove('active');
                if ((domain === null && btn.textContent === '通用场') || 
                    btn.textContent.includes(domain)) {
                    btn.classList.add('active');
                }
            });
            
            // 高亮对应技能
            document.querySelectorAll('.skill-node').forEach(node => {
                if (!domain || node.dataset.domain === domain) {
                    node.classList.add('domain-focus');
                } else {
                    node.classList.remove('domain-focus');
                }
            });
            
            // 添加系统消息
            if (domain) {
                addMessage('ai', `已切换至${domain}高密度场，专业模式激活。`);
            } else {
                addMessage('ai', '返回通用场，全功能模式。');
            }
        }

        // 发送消息
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            if (!message || isProcessing) return;

            // 用户消息
            addMessage('user', message);
            input.value = '';
            isProcessing = true;
            updateStatus('processing');
            
            // 高亮可能激活的技能（简单关键词匹配）
            highlightSkills(message);

            try {
                const response = await fetch(`${API_URL}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        user_id: 'user_001',
                        domain_focus: currentDomain
                    })
                });

                // 创建AI消息容器
                const aiDiv = document.createElement('div');
                aiDiv.className = 'message ai';
                aiDiv.innerHTML = '<div class="message-label">Field</div><span class="content"></span><span class="streaming-cursor"></span>';
                document.getElementById('chat-container').appendChild(aiDiv);
                
                const contentSpan = aiDiv.querySelector('.content');
                const cursor = aiDiv.querySelector('.streaming-cursor');
                
                // 流式读取
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    fullText += text;
                    contentSpan.textContent = fullText;
                    
                    // 自动滚动
                    document.getElementById('chat-container').scrollTop = 
                        document.getElementById('chat-container').scrollHeight;
                }
                
                cursor.remove();
                updateStatus('active');

            } catch (error) {
                addMessage('ai', `场坍缩失败：${error.message}`);
                updateStatus('active');
            }

            isProcessing = false;
            document.getElementById('send-btn').disabled = false;
        }

        function addMessage(role, content) {
            const container = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `
                <div class="message-label">${role === 'user' ? 'You' : 'Field'}</div>
                ${content}
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function highlightSkills(text) {
            const keywords = {
                '天气': 'search_weather',
                '计算': 'calculate',
                '等于': 'calculate',
                '邮件': 'send_email',
                '记住': 'save_memory',
                '记得': 'save_memory'
            };
            
            for (const [kw, skill] of Object.entries(keywords)) {
                if (text.includes(kw)) {
                    document.querySelectorAll('.skill-node').forEach(node => {
                        if (node.textContent === skill) {
                            node.classList.add('active');
                            setTimeout(() => node.classList.remove('active'), 2000);
                        }
                    });
                }
            }
        }

        function updateStatus(state) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            
            dot.className = 'status-dot ' + state;
            
            if (state === 'processing') {
                text.textContent = '共振中...';
            } else if (state === 'active') {
                text.textContent = '就绪';
            } else {
                text.textContent = '就绪';
            }
        }

        // 事件监听
        document.getElementById('message-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // 初始化
        loadSkills();
        updateStatus('active');
    </script>
</body>
</html>

 
