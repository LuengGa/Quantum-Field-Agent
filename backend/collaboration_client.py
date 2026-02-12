"""
Collaboration Client - 协作者层Python客户端
=========================================

提供Python客户端，方便在代码中调用协作者功能

使用示例：
```python
from collaboration_client import CollaborationClient

client = CollaborationClient()

# 思维扩展
result = await client.expand_thinking("我认为应该这样做...")

# 问题重塑
result = await client.reshape_problem("这个问题无法解决...")

# 知识整合
result = await client.integrate_knowledge("用物理学的逻辑来理解经济学...")

# 获取新视角
result = await client.generate_perspectives("人工智能的未来...")

# 深度对话
result = await client.deep_dialogue("我想聊聊人生的意义...")
```
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp


class CollaborationClient:
    """
    协作者层Python客户端
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _post(self, endpoint: str, data: Dict) -> Dict:
        """POST请求"""
        session = await self._get_session()
        async with session.post(f"{self.base_url}{endpoint}", json=data) as resp:
            return await resp.json()

    async def _get(self, endpoint: str) -> Dict:
        """GET请求"""
        session = await self._get_session()
        async with session.get(f"{self.base_url}{endpoint}") as resp:
            return await resp.json()

    # ==================== 协作者入口 ====================

    async def collaborate(
        self,
        user_id: str,
        user_input: str,
        session_id: str = None,
        context: Dict = None,
    ) -> Dict:
        """
        协作者入口

        根据用户输入类型，自动选择协作方式
        """
        return await self._post(
            "/api/collaboration/collaborate",
            {
                "user_id": user_id,
                "session_id": session_id,
                "user_input": user_input,
                "context": context,
            },
        )

    # ==================== 思维扩展 ====================

    async def expand_thinking(self, user_thinking: str, context: Dict = None) -> Dict:
        """
        思维扩展

        不是回答问题，而是扩展用户思维
        - 扩展用户思维到多个可能方向
        - 发现用户可能忽略的盲点
        - 生成挑战性问题
        """
        return await self._post(
            "/api/collaboration/expand",
            {
                "user_thinking": user_thinking,
                "context": context,
            },
        )

    async def discover_blind_spots(self, user_thinking: str) -> Dict:
        """发现思维盲点"""
        return await self._post(
            "/api/collaboration/expand/blind-spots",
            {
                "user_thinking": user_thinking,
            },
        )

    async def generate_challenging_questions(self, user_thinking: str) -> Dict:
        """生成挑战性问题"""
        return await self._post(
            "/api/collaboration/expand/challenging-questions",
            {
                "user_thinking": user_thinking,
            },
        )

    # ==================== 问题重塑 ====================

    async def reshape_problem(self, problem: str, context: Dict = None) -> Dict:
        """
        问题重塑

        不是解决用户的问题，而是帮助用户重新定义问题
        - 分析问题的表层和深层
        - 重塑问题的多种方式
        - 发现问题之外的解决方案
        """
        return await self._post(
            "/api/collaboration/reshape",
            {
                "problem": problem,
                "context": context,
            },
        )

    async def analyze_problem(self, problem: str) -> Dict:
        """分析问题"""
        return await self._post(
            "/api/collaboration/reshape/analyze",
            {
                "problem": problem,
            },
        )

    # ==================== 知识整合 ====================

    async def integrate_knowledge(
        self, knowledge_request: str, context: Dict = None
    ) -> Dict:
        """
        知识整合

        跨领域连接，发现底层逻辑
        - 识别知识领域
        - 发现领域间连接
        - 整合不同领域的知识
        - 提供跨领域新视角
        """
        return await self._post(
            "/api/collaboration/integrate",
            {
                "knowledge_request": knowledge_request,
                "context": context,
            },
        )

    async def identify_domains(self, content: str) -> Dict:
        """识别知识领域"""
        return await self._post(
            "/api/collaboration/integrate/domains",
            {
                "content": content,
            },
        )

    # ==================== 全新视角 ====================

    async def generate_perspectives(self, topic: str, context: Dict = None) -> Dict:
        """
        全新视角

        跳出惯性，发现新可能
        - 多维视角生成
        - 惯性思维挑战
        - 反常识视角
        - 未来/过去视角
        """
        return await self._post(
            "/api/collaboration/perspective",
            {
                "topic": topic,
                "context": context,
            },
        )

    async def time_perspective(self, topic: str) -> Dict:
        """时间维度视角"""
        return await self._post(
            "/api/collaboration/perspective/time",
            {
                "topic": topic,
            },
        )

    async def future_perspective(self, topic: str) -> Dict:
        """未来视角"""
        return await self._post(
            "/api/collaboration/perspective/future",
            {
                "topic": topic,
            },
        )

    # ==================== 深度对话 ====================

    async def deep_dialogue(self, dialogue_input: str, context: Dict = None) -> Dict:
        """
        深度对话

        苏格拉底式追问，不是给答案，而是引问
        """
        return await self._post(
            "/api/collaboration/dialogue",
            {
                "dialogue_input": dialogue_input,
                "context": context,
            },
        )

    async def socratic_questions(self, user_input: str) -> Dict:
        """苏格拉底式问题"""
        return await self._post(
            "/api/collaboration/dialogue/socratic",
            {
                "user_input": user_input,
            },
        )

    # ==================== 学习系统 ====================

    async def learn_from_collaboration(
        self, user_id: str, user_input: str, collaborator_response: Dict
    ) -> Dict:
        """从协作中学习"""
        return await self._post(
            "/api/collaboration/learn",
            {
                "user_id": user_id,
                "user_input": user_input,
                "collaborator_response": collaborator_response,
            },
        )

    async def get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像"""
        return await self._get(f"/api/collaboration/learn/profile/{user_id}")

    async def get_collaboration_summary(self, user_id: str) -> Dict:
        """获取协作总结"""
        return await self._get(f"/api/collaboration/learn/summary/{user_id}")

    async def teach_collaborator(self, lesson: str, context: str) -> Dict:
        """指导协作者"""
        return await self._post(
            "/api/collaboration/learn/teach",
            {
                "lesson": lesson,
                "context": context,
            },
        )

    async def get_mutual_understanding(self, user_id: str) -> Dict:
        """获取相互理解报告"""
        return await self._get(f"/api/collaboration/learn/understanding/{user_id}")

    async def get_learning_stats(self) -> Dict:
        """获取学习统计"""
        return await self._get("/api/collaboration/learn/stats")

    # ==================== 会话管理 ====================

    async def get_session(self, session_id: str) -> Dict:
        """获取会话"""
        return await self._get(f"/api/collaboration/session/{session_id}")

    async def get_collaboration_history(self, session_id: str = None) -> Dict:
        """获取协作历史"""
        endpoint = "/api/collaboration/history"
        if session_id:
            endpoint += f"?session_id={session_id}"
        return await self._get(endpoint)

    async def clear_session(self, session_id: str) -> Dict:
        """清空会话"""
        return await self._delete(f"/api/collaboration/session/{session_id}")

    async def _delete(self, endpoint: str) -> Dict:
        """DELETE请求"""
        session = await self._get_session()
        async with session.delete(f"{self.base_url}{endpoint}") as resp:
            return await resp.json()

    # ==================== 反馈 ====================

    async def submit_feedback(self, user_id: str, feedback: str) -> Dict:
        """提交用户反馈"""
        return await self._post(
            "/api/collaboration/feedback",
            {
                "user_id": user_id,
                "feedback": feedback,
            },
        )

    # ==================== 状态 ====================

    async def status(self) -> Dict:
        """获取协作者状态"""
        return await self._get("/api/collaboration/status")

    async def close(self):
        """关闭客户端"""
        await self._close_session()


# ==================== 便捷函数 ====================


async def expand_thinking(user_thinking: str, context: Dict = None) -> Dict:
    """便捷函数：思维扩展"""
    client = CollaborationClient()
    try:
        return await client.expand_thinking(user_thinking, context)
    finally:
        await client.close()


async def reshape_problem(problem: str, context: Dict = None) -> Dict:
    """便捷函数：问题重塑"""
    client = CollaborationClient()
    try:
        return await client.reshape_problem(problem, context)
    finally:
        await client.close()


async def integrate_knowledge(knowledge_request: str, context: Dict = None) -> Dict:
    """便捷函数：知识整合"""
    client = CollaborationClient()
    try:
        return await client.integrate_knowledge(knowledge_request, context)
    finally:
        await client.close()


async def generate_perspectives(topic: str, context: Dict = None) -> Dict:
    """便捷函数：全新视角"""
    client = CollaborationClient()
    try:
        return await client.generate_perspectives(topic, context)
    finally:
        await client.close()


async def deep_dialogue(dialogue_input: str, context: Dict = None) -> Dict:
    """便捷函数：深度对话"""
    client = CollaborationClient()
    try:
        return await client.deep_dialogue(dialogue_input, context)
    finally:
        await client.close()


async def ask_self(question: str) -> Dict:
    """便捷函数：向自己提问"""
    client = CollaborationClient()
    try:
        return await client.collaborate(
            user_id="self",
            user_input=f"向自己提问：{question}",
        )
    finally:
        await client.close()


# ==================== 批量操作 ====================


class CollaborationBatch:
    """批量协作"""

    def __init__(self):
        self.client = CollaborationClient()

    async def expand_multiple(self, thoughts: List[str]) -> List[Dict]:
        """批量扩展思维"""
        results = []
        for thought in thoughts:
            result = await self.client.expand_thinking(thought)
            results.append(result)
        return results

    async def reshape_multiple(self, problems: List[str]) -> List[Dict]:
        """批量重塑问题"""
        results = []
        for problem in problems:
            result = await self.client.reshape_problem(problem)
            results.append(result)
        return results

    async def perspectives_multiple(self, topics: List[str]) -> List[Dict]:
        """批量获取视角"""
        results = []
        for topic in topics:
            result = await self.client.generate_perspectives(topic)
            results.append(result)
        return results

    async def close(self):
        """关闭"""
        await self.client.close()


if __name__ == "__main__":

    async def demo():
        """演示"""
        client = CollaborationClient()

        # 检查状态
        print("协作者状态:")
        status = await client.status()
        print(status)

        # 思维扩展演示
        print("\n思维扩展演示:")
        result = await client.expand_thinking(
            "我认为这个项目应该采用传统的瀑布模型开发"
        )
        print(result)

        await client.close()

    asyncio.run(demo())
