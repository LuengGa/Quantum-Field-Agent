QUANTUM_FIELD_GUIDEv2.5

项目结构（V2.5）

quantum-field-v2.5/
├── docker-compose.yml
├── backend/
│   ├── main.py                    # API入口（集成纠缠网络）
│   ├── audit_core.py              # V2.0审计（复用）
│   ├── entanglement_network.py    # 纠缠网络核心 ⭐
│   ├── agent_node.py              # Agent节点定义
│   ├── collaborative_task.py      # 协作任务管理
│   ├── shared_memory.py           # 共享记忆池
│   ├── field_manager.py           # 集成版场管理器
│   └── requirements.txt
└── frontend/
    └── entanglement_dashboard.html # 网络可视化 ⭐
    
1. 纠缠网络核心（backend/entanglement_network.py）

"""
Quantum Field V2.5 - 纠缠网络核心
实现A2A（Agent-to-Agent）协议：状态共享、协作坍缩、干涉融合
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Set, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import redis.asyncio as redis
from datetime import datetime

class EntanglementStrength(Enum):
    """纠缠强度等级"""
    WEAK = 0.3      # 信息同步（松散协作）
    MEDIUM = 0.6    # 状态共享（紧密协作）
    STRONG = 0.9    # 联合坍缩（深度绑定）
    MAXIMAL = 1.0   # 完全纠缠（量子隐形传态模拟）

@dataclass
class EntanglementLink:
    """纠缠链接（无向边）"""
    agent_a: str
    agent_b: str
    strength: EntanglementStrength
    created_at: float
    shared_memory_key: str          # Redis中共享数据的key
    last_sync: float = 0.0
    entanglement_vector: List[float] = field(default_factory=list)  # 纠缠态向量

class EntanglementNetwork:
    """
    量子纠缠网络管理器
    管理Agent间的纠缠关系、共享状态、协作调度
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.links: Dict[Tuple[str, str], EntanglementLink] = {}  # 纠缠链接表
        self.agents: Dict[str, 'AgentNode'] = {}                  # Agent注册表
        self.task_queue: asyncio.Queue = asyncio.Queue()          # 协作任务队列
        self.running = True
        
        # 启动后台同步任务
        asyncio.create_task(self._sync_loop())
    
    async def register_agent(self, agent: 'AgentNode'):
        """注册Agent到网络"""
        self.agents[agent.agent_id] = agent
        # 发布Agent上线公告（用于发现）
        await self.redis.publish("agent:discovery", json.dumps({
            "event": "online",
            "agent_id": agent.agent_id,
            "capabilities": agent.capabilities,
            "timestamp": time.time()
        }))
        print(f"[纠缠网络] Agent {agent.agent_id} 已注册")
    
    async def entangle(self, agent_a_id: str, agent_b_id: str, 
                      strength: EntanglementStrength = EntanglementStrength.MEDIUM):
        """
        建立Agent间纠缠（核心操作）
        类似于量子纠缠的制备过程
        """
        if agent_a_id not in self.agents or agent_b_id not in self.agents:
            raise ValueError("Agent未注册")
        
        # 规范化键（无向）
        link_key = tuple(sorted([agent_a_id, agent_b_id]))
        
        # 生成共享记忆池key
        shared_key = f"entangled:{link_key[0]}:{link_key[1]}"
        
        link = EntanglementLink(
            agent_a=agent_a_id,
            agent_b=agent_b_id,
            strength=strength,
            created_at=time.time(),
            shared_memory_key=shared_key,
            entanglement_vector=np.random.randn(1536).tolist()  # 随机初始化纠缠态
        )
        
        self.links[link_key] = link
        
        # 在Redis中建立共享空间
        await self.redis.hset(shared_key, mapping={
            "strength": strength.value,
            "created": link.created_at,
            "agent_a": agent_a_id,
            "agent_b": agent_b_id,
            "sync_count": 0
        })
        
        # 双向通知
        await self._notify_entanglement(link)
        
        return link
    
    async def disentangle(self, agent_a_id: str, agent_b_id: str):
        """解除纠缠（退相干）"""
        link_key = tuple(sorted([agent_a_id, agent_b_id]))
        if link_key in self.links:
            link = self.links.pop(link_key)
            # 清理共享空间
            await self.redis.delete(link.shared_memory_key)
            # 通知双方
            await self._notify_disentanglement(link)
            print(f"[纠缠网络] {agent_a_id} <-> {agent_b_id} 纠缠已解除")
    
    async def discover_agents(self, capability: Optional[str] = None, 
                             exclude: Optional[str] = None) -> List[Dict]:
        """
        Agent发现协议（基于能力匹配）
        类似DNS，但基于语义能力而非地址
        """
        matches = []
        for agent_id, agent in self.agents.items():
            if exclude and agent_id == exclude:
                continue
            if capability is None or capability in agent.capabilities:
                matches.append({
                    "agent_id": agent_id,
                    "capabilities": agent.capabilities,
                    "current_load": agent.current_load,
                    "entangled_with": self._get_entangled_partners(agent_id)
                })
        return matches
    
    def _get_entangled_partners(self, agent_id: str) -> List[str]:
        """获取与指定Agent纠缠的所有伙伴"""
        partners = []
        for (a, b), link in self.links.items():
            if agent_id == a:
                partners.append(b)
            elif agent_id == b:
                partners.append(a)
        return partners
    
    async def collaborative_collapse(self, initiator_id: str, task: str,
                                   participants: List[str],
                                   mode: str = "interference") -> AsyncGenerator[str, None]:
        """
        协作坍缩（多Agent联合处理）
        
        mode参数:
        - interference: 干涉融合（默认，类似量子干涉）
        - consensus: 共识机制（投票）
        - cascade: 级联（A→B→C链式）
        """
        if mode == "interference":
            async for token in self._interference_collapse(initiator_id, task, participants):
                yield token
        elif mode == "consensus":
            async for token in self._consensus_collapse(initiator_id, task, participants):
                yield token
    
    async def _interference_collapse(self, initiator_id: str, task: str, 
                                    participants: List[str]) -> AsyncGenerator[str, None]:
        """
        干涉融合坍缩（核心算法）
        所有Agent同时坍缩，结果干涉叠加产生最终输出
        """
        print(f"[协作坍缩] 发起者: {initiator_id}, 参与者: {participants}, 任务: {task[:50]}...")
        
        # 1. 并行激发所有参与Agent（叠加态准备）
        collapse_tasks = []
        for agent_id in participants:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                # 注入共享上下文（纠缠效应）
                shared_context = await self._get_shared_context(initiator_id, agent_id)
                task_future = self._collapse_single_agent(agent, task, shared_context)
                collapse_tasks.append((agent_id, task_future))
        
        if not collapse_tasks:
            yield "[错误] 无可用Agent参与协作"
            return
        
        # 2. 等待所有Agent部分坍缩（并行）
        partial_results = {}
        for agent_id, future in collapse_tasks:
            try:
                result = await future
                partial_results[agent_id] = result
                print(f"[协作坍缩] {agent_id} 贡献: {result[:30]}...")
            except Exception as e:
                partial_results[agent_id] = f"[错误: {str(e)}]"
        
        # 3. 干涉融合（使用Meta-LLM或加权融合）
        async for token in self._interference_fusion(task, participants, partial_results):
            yield token
        
        # 4. 更新共享记忆（纠缠态更新）
        await self._update_shared_memory(initiator_id, participants, task, 
                                        json.dumps(partial_results))
    
    async def _collapse_single_agent(self, agent: 'AgentNode', task: str, 
                                    shared_context: Dict) -> str:
        """单个Agent的坍缩（带共享上下文）"""
        # 构造增强提示（包含纠缠信息）
        enhanced_prompt = f"""
[纠缠上下文]
协作伙伴: {shared_context.get('partners', [])}
共享记忆: {shared_context.get('memory', '无')}
纠缠强度: {shared_context.get('strength', 0)}

[任务]
{task}

请基于以上上下文生成你的专业贡献。
"""
        # 调用Agent的场进行坍缩
        return await agent.collapse(enhanced_prompt)
    
    async def _interference_fusion(self, task: str, agents: List[str], 
                                  results: Dict[str, str]) -> AsyncGenerator[str, None]:
        """
        干涉融合算法（类似量子干涉仪）
        构建性干涉（共识）增强，破坏性干涉（冲突）消解
        """
        # 构建干涉提示
        interference_prompt = f"""任务：{task}

多Agent坍缩结果（待干涉融合）：
"""
        for agent_id, result in results.items():
            interference_prompt += f"\n[{agent_id}]: {result}\n"
        
        interference_prompt += """
请作为量子干涉仪，融合以上坍缩结果：
1. 识别建设性干涉（共识部分）→ 保留并增强
2. 识别破坏性干涉（冲突部分）→ 分析原因并调和
3. 产生相位一致的联合坍缩结果（高质量综合）

融合原则：
- 保留专业细节但消除冗余
- 解决逻辑冲突（标记*）
- 输出统一、连贯、可执行的结论
"""
        
        # 使用更强的模型进行融合（Meta-LLM）
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await client.chat.completions.create(
            model="gpt-4o",  # 使用强模型进行融合
            messages=[{"role": "user", "content": interference_prompt}],
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _consensus_collapse(self, initiator_id: str, task: str, 
                                 participants: List[str]) -> AsyncGenerator[str, None]:
        """共识机制坍缩（投票式）"""
        # 收集各Agent的离散决策（是/否/选项）
        votes = {}
        for agent_id in participants:
            agent = self.agents[agent_id]
            vote = await agent.collapse(f"[共识任务] {task}\n请只回答：同意/反对/弃权")
            votes[agent_id] = vote
        
        # 统计共识
        agree = sum(1 for v in votes.values() if "同意" in v)
        disagree = sum(1 for v in votes.values() if "反对" in v)
        
        result = f"共识结果: {agree}票同意, {disagree}票反对, {len(votes)-agree-disagree}票弃权"
        yield result
    
    async def _get_shared_context(self, agent_a: str, agent_b: str) -> Dict:
        """获取两个Agent间的共享上下文（纠缠态）"""
        link_key = tuple(sorted([agent_a, agent_b]))
        if link_key not in self.links:
            return {"partners": [], "memory": "", "strength": 0}
        
        link = self.links[link_key]
        memory_data = await self.redis.hget(link.shared_memory_key, "last_memory")
        
        return {
            "partners": [agent_b] if agent_a == link.agent_a else [agent_a],
            "memory": memory_data or "",
            "strength": link.strength.value,
            "entanglement_vector": link.entanglement_vector
        }
    
    async def _update_shared_memory(self, initiator: str, participants: List[str], 
                                   task: str, result_summary: str):
        """更新所有参与者的共享记忆池"""
        timestamp = time.time()
        for agent_id in participants:
            if agent_id == initiator:
                continue
            
            link_key = tuple(sorted([initiator, agent_id]))
            if link_key in self.links:
                link = self.links[link_key]
                # 追加到共享记忆
                memory_entry = f"[{datetime.now().isoformat()}] 协作任务: {task[:30]}... 结果: {result_summary[:50]}..."
                await self.redis.hset(link.shared_memory_key, "last_memory", memory_entry)
                await self.redis.hincrby(link.shared_memory_key, "sync_count", 1)
    
    async def _notify_entanglement(self, link: EntanglementLink):
        """通知双方Agent纠缠建立"""
        for agent_id in [link.agent_a, link.agent_b]:
            await self.redis.publish(f"agent:{agent_id}:entangle", json.dumps({
                "event": "entangled",
                "partner": link.agent_b if agent_id == link.agent_a else link.agent_a,
                "strength": link.strength.value,
                "shared_key": link.shared_memory_key
            }))
    
    async def _notify_disentanglement(self, link: EntanglementLink):
        """通知退相干"""
        for agent_id in [link.agent_a, link.agent_b]:
            await self.redis.publish(f"agent:{agent_id}:disentangle", json.dumps({
                "event": "disentangled",
                "partner": link.agent_b if agent_id == link.agent_a else link.agent_a
            }))
    
    async def _sync_loop(self):
        """后台同步循环：维持纠缠态、同步场状态"""
        while self.running:
            try:
                # 定期同步强纠缠Agent的状态（模拟量子纠缠的非局域性）
                for (a, b), link in self.links.items():
                    if link.strength == EntanglementStrength.STRONG:
                        # 强纠缠：定期状态同步
                        await self._sync_agent_states(a, b)
                
                await asyncio.sleep(5)  # 每5秒检查一次
            except Exception as e:
                print(f"[纠缠同步错误] {e}")
    
    async def _sync_agent_states(self, agent_a: str, agent_b: str):
        """同步两个Agent的场状态（强纠缠效应）"""
        # 实际应用中，这里可以同步内存向量、偏好设置等
        pass
    
    async def get_network_topology(self) -> Dict:
        """获取网络拓扑（用于可视化）"""
        nodes = []
        for agent_id, agent in self.agents.items():
            nodes.append({
                "id": agent_id,
                "capabilities": agent.capabilities,
                "load": agent.current_load,
                "type": agent.agent_type
            })
        
        edges = []
        for (a, b), link in self.links.items():
            edges.append({
                "source": a,
                "target": b,
                "strength": link.strength.value,
                "age": time.time() - link.created_at
            })
        
        return {"nodes": nodes, "edges": edges, "agent_count": len(nodes), "link_count": len(edges)}

import os

2. Agent节点定义（backend/agent_node.py）

"""
Agent节点：纠缠网络中的基本单元
每个Agent是一个独立的量子场，但可与其他Agent纠缠
"""

from typing import List, Dict, Optional, AsyncGenerator
import asyncio

class AgentNode:
    """
    网络中的Agent节点
    包装原有的QuantumField，添加网络属性
    """
    
    def __init__(self, agent_id: str, capabilities: List[str], 
                 agent_type: str = "general"):
        self.agent_id = agent_id
        self.capabilities = capabilities  # 能力标签（如["legal", "contract"]）
        self.agent_type = agent_type      # 类型：general/specialist/hybrid
        self.current_load = 0             # 当前负载（用于调度）
        self.entangled_with: List[str] = []  # 纠缠伙伴列表
        self.field_state: Optional[Dict] = None  # 场状态引用
        self.message_queue: asyncio.Queue = asyncio.Queue()  # 消息队列
        
        # 统计
        self.collapse_count = 0
        self.total_tokens_generated = 0
    
    async def collapse(self, prompt: str) -> str:
        """
        Agent的坍缩操作（实际调用底层Field）
        这里简化实现，实际应接入V2.0的AuditableFieldManager
        """
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.current_load += 1
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content
            self.collapse_count += 1
            self.total_tokens_generated += len(result)
            return result
        finally:
            self.current_load -= 1
    
    async def receive_entanglement_signal(self, signal: Dict):
        """接收纠缠信号（来自网络的异步通知）"""
        await self.message_queue.put(signal)
    
    def to_dict(self) -> Dict:
        """序列化"""
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "type": self.agent_type,
            "load": self.current_load,
            "entangled_count": len(self.entangled_with),
            "stats": {
                "collapses": self.collapse_count,
                "tokens": self.total_tokens_generated
            }
        }
        
3. 协作任务管理（backend/collaborative_task.py）

"""
协作任务管理：跟踪多Agent协作的生命周期
"""

from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import time
import uuid

class TaskStatus(Enum):
    PENDING = "pending"         # 等待中
    COLLAPSING = "collapsing"   # 正在坍缩（处理中）
    FUSING = "fusing"           # 干涉融合中
    COMPLETED = "completed"     # 完成
    FAILED = "failed"          # 失败

@dataclass
class CollaborativeTask:
    """协作任务实例"""
    task_id: str
    initiator: str
    participants: List[str]
    original_intent: str
    status: TaskStatus
    created_at: float
    completed_at: Optional[float] = None
    partial_results: Dict[str, str] = None
    final_result: Optional[str] = None
    audit_trail: List[Dict] = None  # 审计日志（V2.0集成）
    
    def __post_init__(self):
        if self.partial_results is None:
            self.partial_results = {}
        if self.audit_trail is None:
            self.audit_trail = []
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "initiator": self.initiator,
            "participants": self.participants,
            "intent": self.original_intent[:100] + "...",
            "status": self.status.value,
            "duration": (self.completed_at or time.time()) - self.created_at,
            "progress": f"{len(self.partial_results)}/{len(self.participants)}"
        }

class TaskManager:
    """协作任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, CollaborativeTask] = {}
    
    def create_task(self, initiator: str, participants: List[str], 
                   intent: str) -> CollaborativeTask:
        """创建新任务"""
        task = CollaborativeTask(
            task_id=str(uuid.uuid4())[:8],
            initiator=initiator,
            participants=participants,
            original_intent=intent,
            status=TaskStatus.PENDING,
            created_at=time.time()
        )
        self.tasks[task.task_id] = task
        return task
    
    def update_status(self, task_id: str, status: TaskStatus):
        """更新任务状态"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if status == TaskStatus.COMPLETED:
                self.tasks[task_id].completed_at = time.time()
    
    def record_partial(self, task_id: str, agent_id: str, result: str):
        """记录部分结果"""
        if task_id in self.tasks:
            self.tasks[task_id].partial_results[agent_id] = result
    
    def get_task(self, task_id: str) -> Optional[CollaborativeTask]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def list_active_tasks(self) -> List[Dict]:
        """列出活跃任务"""
        return [t.to_dict() for t in self.tasks.values() 
                if t.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]]
                
4. API入口集成（backend/main.py 更新）

"""
V2.5 API入口 - 集成纠缠网络
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# 导入V2.5模块
from entanglement_network import EntanglementNetwork, EntanglementStrength
from agent_node import AgentNode
from collaborative_task import TaskManager
from field_manager import AuditableFieldManager  # V2.0复用

app = FastAPI(title="Quantum Field V2.5 - Entanglement Network")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化全局组件
network = EntanglementNetwork()
task_manager = TaskManager()

# 预注册示例Agent（实际应从数据库/配置加载）
@app.on_event("startup")
async def setup_agents():
    # 法律Agent
    legal = AgentNode("legal_expert", ["legal", "contract", "compliance"], "specialist")
    await network.register_agent(legal)
    
    # 财务Agent
    finance = AgentNode("finance_expert", ["finance", "accounting", "tax"], "specialist")
    await network.register_agent(finance)
    
    # 技术Agent
    tech = AgentNode("tech_expert", ["coding", "architecture", "debug"], "specialist")
    await network.register_agent(tech)
    
    # 通用助手
    general = AgentNode("general_assistant", ["general", " coordination"], "general")
    await network.register_agent(general)
    
    print("[系统] 4个示例Agent已注册到纠缠网络")

# 数据模型
class EntangleRequest(BaseModel):
    agent_a: str
    agent_b: str
    strength: float = 0.6  # 0.0-1.0

class CollaborativeRequest(BaseModel):
    initiator: str
    task: str
    participants: List[str]
    mode: str = "interference"  # interference/consensus

class DiscoverRequest(BaseModel):
    capability: Optional[str] = None
    exclude: Optional[str] = None

# ===== 纠缠网络API =====

@app.post("/network/entangle")
async def create_entanglement(req: EntangleRequest):
    """建立Agent间纠缠"""
    try:
        strength = EntanglementStrength(min(1.0, max(0.0, req.strength)))
        link = await network.entangle(req.agent_a, req.agent_b, strength)
        return {
            "status": "entangled",
            "agents": [link.agent_a, link.agent_b],
            "strength": link.strength.value,
            "shared_key": link.shared_memory_key
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/network/disentangle")
async def remove_entanglement(agent_a: str, agent_b: str):
    """解除纠缠"""
    await network.disentangle(agent_a, agent_b)
    return {"status": "disentangled"}

@app.get("/network/discover")
async def discover_agents(capability: Optional[str] = None, 
                         exclude: Optional[str] = None):
    """发现Agent（基于能力）"""
    agents = await network.discover_agents(capability, exclude)
    return {"agents": agents, "count": len(agents)}

@app.get("/network/topology")
async def get_topology():
    """获取网络拓扑（可视化用）"""
    return await network.get_network_topology()

@app.post("/network/collaborate")
async def collaborative_collapse(req: CollaborativeRequest):
    """
    发起协作坍缩（多Agent联合处理）
    流式返回融合结果
    """
    # 验证所有参与者存在
    for agent_id in req.participants:
        if agent_id not in network.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} 未找到")
    
    # 创建任务
    task = task_manager.create_task(req.initiator, req.participants, req.task)
    
    async def generate():
        try:
            task_manager.update_status(task.task_id, TaskStatus.COLLAPSING)
            
            async for token in network.collaborative_collapse(
                req.initiator, req.task, req.participants, req.mode
            ):
                yield token
            
            task_manager.update_status(task.task_id, TaskStatus.COMPLETED)
            
        except Exception as e:
            task_manager.update_status(task.task_id, TaskStatus.FAILED)
            yield f"\n[协作错误: {str(e)}]"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/network/tasks")
async def list_tasks():
    """列出活跃协作任务"""
    return task_manager.list_active_tasks()

@app.get("/network/agents/{agent_id}")
async def get_agent_info(agent_id: str):
    """获取Agent详细信息"""
    if agent_id not in network.agents:
        raise HTTPException(status_code=404, detail="Agent未找到")
    return network.agents[agent_id].to_dict()

# ===== 兼容V2.0的单Agent接口 =====

@app.post("/chat")
async def single_agent_chat(message: str, user_id: str = "default"):
    """
    单Agent模式（向后兼容V2.0）
    自动路由到general_assistant或创建临时Agent
    """
    if "general_assistant" in network.agents:
        agent = network.agents["general_assistant"]
        result = await agent.collapse(message)
        return {"response": result}
    else:
        raise HTTPException(status_code=503, detail="无可用Agent")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
5. 网络可视化前端（frontend/entanglement_dashboard.html）

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Field V2.5 - 纠缠网络可视化</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            background: #0a0a0a;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        .sidebar {
            width: 300px;
            background: #111;
            border-right: 1px solid #333;
            padding: 20px;
            overflow-y: auto;
        }
        
        .main {
            flex: 1;
            position: relative;
        }
        
        #network-viz {
            width: 100%;
            height: 100%;
        }
        
        .agent-card {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .agent-card:hover {
            border-color: #00ff88;
            background: #222;
        }
        
        .agent-card.selected {
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0,255,136,0.3);
        }
        
        .entanglement-controls {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #333;
        }
        
        button {
            background: #00ff88;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            width: 100%;
            margin-top: 10px;
        }
        
        button:hover {
            opacity: 0.8;
        }
        
        button:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
        }
        
        .collaboration-panel {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            border: 1px solid #00ff88;
            padding: 20px;
            border-radius: 10px;
            width: 600px;
            display: none;
        }
        
        .node circle {
            fill: #1a1a1a;
            stroke: #00ff88;
            stroke-width: 2px;
            cursor: pointer;
        }
        
        .node text {
            fill: #fff;
            font-size: 12px;
            pointer-events: none;
        }
        
        .link {
            stroke: #333;
            stroke-width: 2px;
        }
        
        .link.strong {
            stroke: #00ff88;
            stroke-width: 4px;
        }
        
        .link.weak {
            stroke: #555;
            stroke-dasharray: 5,5;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>⚛ 纠缠网络</h2>
            <div id="agent-list"></div>
            
            <div class="entanglement-controls">
                <h3>操作</h3>
                <button onclick="refreshTopology()">刷新拓扑</button>
                <button onclick="autoEntangle()">自动纠缠（推荐）</button>
                <div id="selected-actions" style="margin-top:20px;display:none;">
                    <p>已选择: <span id="selected-name"></span></p>
                    <button onclick="startCollaboration()">发起协作</button>
                </div>
            </div>
            
            <div style="margin-top:30px;font-size:12px;color:#666;">
                <p>提示：点击节点选择Agent，拖动查看连接。</p>
                <p>绿色粗线=强纠缠，虚线=弱纠缠</p>
            </div>
        </div>
        
        <div class="main">
            <svg id="network-viz"></svg>
            
            <div class="collaboration-panel" id="collab-panel">
                <h3>协作坍缩</h3>
                <textarea id="collab-task" rows="3" style="width:100%;background:#111;color:#fff;border:1px solid #333;padding:10px;" placeholder="输入协作任务（如：审查这份合同的法律和财务风险）..."></textarea>
                <div style="margin-top:10px;">
                    <label>模式：</label>
                    <select id="collab-mode" style="background:#111;color:#fff;border:1px solid #333;">
                        <option value="interference">干涉融合（默认）</option>
                        <option value="consensus">共识机制</option>
                    </select>
                </div>
                <button onclick="executeCollaboration()" style="margin-top:10px;">执行协作坍缩</button>
                <button onclick="closeCollab()" style="background:#333;color:#fff;">取消</button>
                <div id="collab-result" style="margin-top:15px;max-height:200px;overflow-y:auto;background:#0a0a0a;padding:10px;border-radius:4px;"></div>
            </div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:8000';
        let selectedAgents = new Set();
        let simulation;
        
        async function loadAgents() {
            const res = await fetch(`${API_URL}/network/topology`);
            const data = await res.json();
            
            // 渲染侧边栏
            const list = document.getElementById('agent-list');
            list.innerHTML = '';
            data.nodes.forEach(agent => {
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.dataset.id = agent.id;
                card.innerHTML = `
                    <strong>${agent.id}</strong>
                    <div style="font-size:11px;color:#888;margin-top:5px;">
                        ${agent.capabilities.join(', ')}
                    </div>
                    <div style="font-size:10px;color:#00ff88;margin-top:5px;">
                        负载: ${agent.load} | 纠缠: ${agent.entangled_count}
                    </div>
                `;
                card.onclick = () => toggleAgent(agent.id);
                list.appendChild(card);
            });
            
            // 渲染D3网络图
            renderNetwork(data);
        }
        
        function renderNetwork(data) {
            const svg = d3.select("#network-viz");
            svg.selectAll("*").remove();
            
            const width = svg.node().parentElement.clientWidth;
            const height = svg.node().parentElement.clientHeight;
            svg.attr("width", width).attr("height", height);
            
            // 力学仿真
            simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.edges).id(d => d.id).distance(150))
                .force("charge", d3.forceManyBody().strength(-500))
                .force("center", d3.forceCenter(width / 2, height / 2));
            
            // 绘制连线（纠缠）
            const link = svg.append("g")
                .selectAll("line")
                .data(data.edges)
                .enter().append("line")
                .attr("class", d => `link ${d.strength > 0.7 ? 'strong' : d.strength < 0.4 ? 'weak' : ''}`)
                .attr("stroke-width", d => d.strength * 5);
            
            // 绘制节点
            const node = svg.append("g")
                .selectAll("g")
                .data(data.nodes)
                .enter().append("g")
                .attr("class", "node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended))
                .on("click", (event, d) => toggleAgent(d.id));
            
            node.append("circle")
                .attr("r", d => 20 + d.capabilities.length * 3)
                .attr("fill", d => selectedAgents.has(d.id) ? "#00ff88" : "#1a1a1a");
            
            node.append("text")
                .attr("dx", 25)
                .attr("dy", 5)
                .text(d => d.id);
            
            // 更新位置
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node.attr("transform", d => `translate(${d.x},${d.y})`);
            });
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }
        
        function toggleAgent(agentId) {
            if (selectedAgents.has(agentId)) {
                selectedAgents.delete(agentId);
                document.querySelector(`.agent-card[data-id="${agentId}"]`)?.classList.remove('selected');
            } else {
                selectedAgents.add(agentId);
                document.querySelector(`.agent-card[data-id="${agentId}"]`)?.classList.add('selected');
            }
            
            // 更新UI
            const actions = document.getElementById('selected-actions');
            if (selectedAgents.size > 0) {
                actions.style.display = 'block';
                document.getElementById('selected-name').textContent = Array.from(selectedAgents).join(', ');
            } else {
                actions.style.display = 'none';
            }
            
            // 更新D3节点颜色
            d3.selectAll(".node circle")
                .attr("fill", d => selectedAgents.has(d.id) ? "#00ff88" : "#1a1a1a");
        }
        
        async function autoEntangle() {
            // 自动建立推荐纠缠（法律-财务等）
            const pairs = [
                ['legal_expert', 'finance_expert'],
                ['tech_expert', 'general_assistant']
            ];
            
            for (const [a, b] of pairs) {
                await fetch(`${API_URL}/network/entangle`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({agent_a: a, agent_b: b, strength: 0.8})
                });
            }
            
            alert('已自动建立推荐纠缠关系');
            loadAgents();
        }
        
        function startCollaboration() {
            if (selectedAgents.size < 2) {
                alert('请至少选择2个Agent进行协作');
                return;
            }
            document.getElementById('collab-panel').style.display = 'block';
        }
        
        function closeCollab() {
            document.getElementById('collab-panel').style.display = 'none';
            document.getElementById('collab-result').innerHTML = '';
        }
        
        async function executeCollaboration() {
            const task = document.getElementById('collab-task').value;
            const mode = document.getElementById('collab-mode').value;
            const initiator = Array.from(selectedAgents)[0];
            const participants = Array.from(selectedAgents);
            
            const resultDiv = document.getElementById('collab-result');
            resultDiv.innerHTML = '<div style="color:#888;">协作坍缩中...</div>';
            
            try {
                const response = await fetch(`${API_URL}/network/collaborate`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        initiator: initiator,
                        task: task,
                        participants: participants,
                        mode: mode
                    })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let result = '';
                
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    result += decoder.decode(value);
                    resultDiv.innerHTML = '<div style="white-space:pre-wrap;">' + result + '</div>';
                }
                
            } catch (e) {
                resultDiv.innerHTML = `<div style="color:#ff0000;">错误: ${e.message}</div>`;
            }
        }
        
        function refreshTopology() {
            loadAgents();
        }
        
        // 初始化
        loadAgents();
        setInterval(loadAgents, 5000); // 自动刷新
    </script>
</body>
</html>

6. 部署与测试

# 1. 启动V2.5（包含Redis）
docker-compose up -d

# 2. 查看Agent列表
curl http://localhost:8000/network/discover

# 3. 建立纠缠（法律-财务）
curl -X POST http://localhost:8000/network/entangle \
  -d '{"agent_a": "legal_expert", "agent_b": "finance_expert", "strength": 0.8}'

# 4. 发起协作坍缩（流式）
curl -X POST http://localhost:8000/network/collaborate \
  -d '{
    "initiator": "general_assistant",
    "task": "审查这份合同：甲方需支付100万，但乙方违约责任不明确，请从法律和财务角度分析风险",
    "participants": ["legal_expert", "finance_expert"],
    "mode": "interference"
  }'

# 5. 打开可视化界面
open http://localhost:8000/network/topology

V2.5核心特性

A2A协议实现：Agent间通过Redis Pub/Sub共享状态（纠缠）
协作坍缩：多Agent并行处理，Meta-LLM干涉融合
能力发现：基于标签的Agent发现（无需硬编码地址）
可视化网络：D3.js实时显示纠缠拓扑和强度
V2.0兼容：保留审计链，记录纠缠事件
