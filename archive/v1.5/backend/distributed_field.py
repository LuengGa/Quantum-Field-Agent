"""
分布式量子场核心（V1.5）
实现：场分片、状态序列化、流式Pub/Sub
"""

import os
import json
import asyncio
import hashlib
import pickle
import time
from typing import Dict, Optional, AsyncGenerator, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from datetime import datetime

import redis.asyncio as redis
from redis.asyncio.client import PubSub


@dataclass
class FieldState:
    """
    量子场状态（可序列化）
    包含：记忆向量、用户偏好、当前上下文、技能激活历史
    """

    user_id: str
    memory_vector: list  # 嵌入向量（而非原始文本，节省空间）
    preference_vector: list
    activated_skills: list  # 最近激活的技能ID
    entropy: float  # 场熵（混乱度，用于判断是否需要offload）
    last_update: float
    session_context: dict  # 当前会话上下文

    def serialize(self) -> bytes:
        """压缩序列化（使用pickle+压缩）"""
        import zlib

        return zlib.compress(pickle.dumps(asdict(self)))

    @classmethod
    def deserialize(cls, data: bytes) -> "FieldState":
        import zlib

        return cls(**pickle.loads(zlib.decompress(data)))


class DistributedQuantumField:
    """
    分布式量子场管理器
    职责：场定位、状态恢复、流式结果路由
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.local_cache: Dict[str, Any] = {}  # L1缓存（进程内）
        self.locks: Dict[str, asyncio.Lock] = {}  # 用户级锁（防止并发修改场状态）

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        """获取用户级锁（保证场状态修改原子性）"""
        if user_id not in self.locks:
            self.locks[user_id] = asyncio.Lock()
        return self.locks[user_id]

    async def locate_field(self, user_id: str) -> Optional[FieldState]:
        """
        定位场（类似量子力学的'测量'）
        策略：先查本地缓存 -> 再查Redis -> 最后新建基态
        """
        # 1. L1缓存（最近活跃）
        if user_id in self.local_cache:
            return self.local_cache[user_id]

        # 2. Redis（L2缓存）
        field_key = f"qf:field:{user_id}"
        data = await self.redis.get(field_key)

        if data:
            state = FieldState.deserialize(data)
            # 放入L1缓存（5分钟TTL）
            self.local_cache[user_id] = state
            asyncio.create_task(self._expire_local_cache(user_id, 300))
            return state

        # 3. 新建基态场
        return self._create_base_field(user_id)

    def _create_base_field(self, user_id: str) -> FieldState:
        """创建基态场（空状态，高势能）"""
        return FieldState(
            user_id=user_id,
            memory_vector=[0.0] * 1536,  # 零向量
            preference_vector=[0.0] * 1536,
            activated_skills=[],
            entropy=0.1,  # 低熵（有序）
            last_update=time.time(),
            session_context={},
        )

    async def save_field(self, state: FieldState, ttl: int = 3600):
        """
        保存场状态到Redis（坍缩后持久化）
        使用Hash Tag确保同一用户的场在同一个Redis分片（ locality）
        """
        field_key = f"qf:field:{state.user_id}"
        serialized = state.serialize()

        # 使用SETEX（原子操作）
        await self.redis.setex(field_key, ttl, serialized)

        # 同时更新L1缓存
        self.local_cache[state.user_id] = state

    async def process_intent(
        self, user_id: str, intent: str, session_id: str
    ) -> AsyncGenerator[str, None]:
        """
        核心处理流程：共振 -> 可能的分布式坍缩 -> 流式返回
        """
        async with self._get_lock(user_id):
            # 1. 恢复或创建场
            field_state = await self.locate_field(user_id)

            # 2. 判断场熵（是否需要高算力offload）
            if field_state.entropy > 0.8:
                # 高熵场：分发到计算集群（GPU节点）
                async for token in self._distributed_collapse(
                    user_id, intent, field_state, session_id
                ):
                    yield token
            else:
                # 低熵场：本地快速坍缩
                async for token in self._local_collapse(
                    user_id, intent, field_state, session_id
                ):
                    yield token

    async def _local_collapse(
        self, user_id: str, intent: str, state: FieldState, session_id: str
    ) -> AsyncGenerator[str, None]:
        """
        本地坍缩（单节点处理）
        保持与V1.0相同的逻辑，但增加了状态更新
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        # 更新场状态（增加熵）
        state.entropy = min(1.0, state.entropy + 0.1)
        state.last_update = time.time()

        # 构建提示（注入场状态）
        messages = self._build_messages(intent, state)

        # 流式生成（坍缩）
        full_response = ""
        try:
            response = await client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                messages=messages,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token

            # 坍缩完成后，更新记忆向量（在场中驻留）
            state.memory_vector = await self._update_memory_vector(
                state.memory_vector, intent, full_response
            )
            state.activated_skills.append("chat")  # 记录激活

            # 保存场状态（异步，不阻塞返回）
            asyncio.create_task(self.save_field(state))

        except Exception as e:
            yield f"[场坍缩异常: {str(e)}]"
            # 回滚熵增加
            state.entropy = max(0.1, state.entropy - 0.1)

    async def _distributed_collapse(
        self, user_id: str, intent: str, state: FieldState, session_id: str
    ) -> AsyncGenerator[str, None]:
        """
        分布式坍缩：将任务发布到计算集群，通过Redis Pub/Sub流式返回结果
        这是V1.5的核心创新
        """
        task_id = hashlib.sha256(
            f"{user_id}:{intent}:{time.time()}".encode()
        ).hexdigest()[:16]

        # 1. 发布任务到队列（Stream，支持消费者组）
        task_data = {
            "user_id": user_id,
            "intent": intent,
            "field_state": state.serialize().hex(),  # 序列化场状态传递
            "task_id": task_id,
            "session_id": session_id,
            "timestamp": time.time(),
        }

        await self.redis.xadd(
            "qf:compute_queue",  # 计算队列
            task_data,
            maxlen=10000,  # 保留最近10000个任务
        )

        # 2. 订阅结果频道（Pub/Sub）
        result_channel = f"qf:result:{task_id}"
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(result_channel)

        try:
            # 3. 等待并流式返回结果（最多30秒超时）
            timeout = 30
            start_time = time.time()
            full_result = ""

            while time.time() - start_time < timeout:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )

                if message:
                    data = json.loads(message["data"])

                    if data["type"] == "token":
                        yield data["content"]
                        full_result += data["content"]
                    elif data["type"] == "end":
                        # 坍缩完成，更新本地场状态
                        if "new_state" in data:
                            new_state = FieldState.deserialize(
                                bytes.fromhex(data["new_state"])
                            )
                            await self.save_field(new_state)
                        break
                    elif data["type"] == "error":
                        yield f"[计算场错误: {data['message']}]"
                        break

        finally:
            await pubsub.unsubscribe(result_channel)
            await pubsub.close()

    def _build_messages(self, intent: str, state: FieldState) -> list:
        """构建LLM消息（注入场状态）"""
        # 从向量恢复最近记忆（简化版，实际应有向量检索）
        recent_context = "你是Quantum Field Agent。当前场熵：{:.2f}。保持简洁。".format(
            state.entropy
        )

        return [
            {"role": "system", "content": recent_context},
            {"role": "user", "content": intent},
        ]

    async def _update_memory_vector(
        self, current_vector: list, intent: str, response: str
    ) -> list:
        """更新记忆向量（滑动平均）"""
        # 简化的向量更新：实际应使用嵌入模型
        import random

        # 模拟向量变化（实际应调用text-embedding-3-large）
        new_vector = [v * 0.9 + random.uniform(-0.1, 0.1) for v in current_vector]
        return new_vector

    async def _expire_local_cache(self, user_id: str, delay: int):
        """过期本地缓存"""
        await asyncio.sleep(delay)
        if user_id in self.local_cache:
            del self.local_cache[user_id]


# 计算节点（Worker）实现
class ComputeFieldWorker:
    """
    计算场工作节点（GPU/高算力节点）
    消费Redis队列，执行高熵坍缩，Pub/Sub返回结果
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.running = True

    async def run(self):
        """主循环：消费任务"""
        print("[计算场节点] 启动，等待高熵任务...")

        while self.running:
            try:
                # 阻塞读取队列（超时1秒，便于检查running标志）
                messages = await self.redis.xread(
                    {"qf:compute_queue": "$"},  # 从最新消息开始
                    block=1000,
                    count=1,
                )

                if not messages:
                    continue

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        await self._process_task(fields)

            except Exception as e:
                print(f"[Worker错误] {e}")
                await asyncio.sleep(1)

    async def _process_task(self, fields: dict):
        """处理单个任务"""
        task_id = fields[b"task_id"].decode()
        user_id = fields[b"user_id"].decode()
        intent = fields[b"intent"].decode()
        state_data = bytes.fromhex(fields[b"field_state"].decode())

        # 恢复场状态
        state = FieldState.deserialize(state_data)
        result_channel = f"qf:result:{task_id}"

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )

            # 高算力坍缩（可接入GPT-4/Claude-Opus等强模型）
            messages = [
                {
                    "role": "system",
                    "content": "你是高算力计算场节点，处理复杂任务。当前场熵较高，请仔细推理。",
                },
                {"role": "user", "content": intent},
            ]

            response = await client.chat.completions.create(
                model=os.getenv("WORKER_MODEL_NAME", "gpt-4o"),  # 强模型
                messages=messages,
                stream=False,  # Worker内部不流式，直接生成后分段推送
            )

            full_text = response.choices[0].message.content

            # 模拟流式推送（分段发送，保持用户体验）
            chunk_size = 5  # 每5个字符发一次
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i : i + chunk_size]
                await self.redis.publish(
                    result_channel, json.dumps({"type": "token", "content": chunk})
                )
                await asyncio.sleep(0.05)  # 模拟打字延迟

            # 更新场状态（降低熵，因为已坍缩）
            state.entropy = 0.3
            state.last_update = time.time()

            # 发送完成信号
            await self.redis.publish(
                result_channel,
                json.dumps({"type": "end", "new_state": state.serialize().hex()}),
            )

        except Exception as e:
            await self.redis.publish(
                result_channel, json.dumps({"type": "error", "message": str(e)})
            )


# 一致性哈希负载均衡（多Redis实例时使用）
class ConsistentHashRouter:
    """
    当需要多个Redis实例时，使用一致性哈希定位场
    """

    def __init__(self, redis_nodes: list):
        self.nodes = redis_nodes
        self.virtual_nodes = 150  # 虚拟节点数
        self.ring = {}
        self._build_ring()

    def _build_ring(self):
        import hashlib

        for node in self.nodes:
            for i in range(self.virtual_nodes):
                key = hashlib.md5(f"{node}:{i}".encode()).hexdigest()
                self.ring[int(key, 16)] = node
        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, user_id: str) -> str:
        import hashlib

        h = hashlib.md5(user_id.encode()).hexdigest()
        hash_int = int(h, 16)

        # 顺时针找到第一个节点
        for key in self.sorted_keys:
            if hash_int <= key:
                return self.ring[key]
        return self.ring[self.sorted_keys[0]]


# 全局实例（单例）
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
field_manager = DistributedQuantumField(redis_url)
