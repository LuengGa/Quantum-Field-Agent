"""
Collaborator - 协作者核心
=========================

协作者的核心设计原则：
1. 不是回答问题，而是提供新视角
2. 不是给出答案，而是帮助用户发现
3. 不是服从指令，而是平等对话
4. 不是完成任务，而是共同探索

协作者的角色：
- 思维扩展器：看见用户看不见的盲点
- 问题重塑器：重新定义问题，发现新可能
- 知识整合器：跨领域连接，底层逻辑贯通
- 全新视角生成器：跳出惯性，发现新可能
- 学习者：从用户反馈中学习，越来越了解用户
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .thinking_expander import ThinkingExpander
from .problem_reshaper import ProblemReshaper
from .knowledge_integrator import KnowledgeIntegrator
from .perspective_generator import PerspectiveGenerator
from .dialogue_engine import DialogueEngine
from .learner import Learner


class Collaborator:
    """
    协作者 - 人类与AI平等协作的核心
    """

    def __init__(self, storage_dir: str = "./collaboration_data"):
        print("\n" + "=" * 60)
        print("Collaborator - 协作者初始化")
        print("=" * 60)

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 协作者组件
        self.thinking_expander = ThinkingExpander()
        self.problem_reshaper = ProblemReshaper()
        self.knowledge_integrator = KnowledgeIntegrator()
        self.perspective_generator = PerspectiveGenerator()
        self.dialogue_engine = DialogueEngine()
        self.learner = Learner()

        # 会话管理
        self.sessions: Dict[str, Dict] = {}

        # 用户画像
        self.user_profiles: Dict[str, Dict] = {}

        print("[Collaborator] ✓ 协作者组件已初始化")
        print("[Collaborator] - 思维扩展器")
        print("[Collaborator] - 问题重塑器")
        print("[Collaborator] - 知识整合器")
        print("[Collaborator] - 视角生成器")
        print("[Collaborator] - 对话引擎")
        print("[Collaborator] - 学习系统")
        print("=" * 60 + "\n")

    async def collaborate(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        协作者入口 - 处理用户输入

        不是回答问题，而是协作为用户提供新视角
        """
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

        # 初始化会话
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "history": [],
                "user_id": user_id,
            }

        # 记录用户输入
        self.sessions[session_id]["history"].append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 理解用户意图类型
        intent_type = self._classify_intent(user_input)

        # 根据意图类型选择协作方式
        if intent_type == "thinking_share":
            # 用户分享思维
            result = await self._collaborate_thinking_expansion(
                user_id, session_id, user_input, context
            )
        elif intent_type == "problem_statement":
            # 用户陈述问题
            result = await self._collaborate_problem_reshape(
                user_id, session_id, user_input, context
            )
        elif intent_type == "knowledge_request":
            # 用户请求知识整合
            result = await self._collaborate_knowledge_integration(
                user_id, session_id, user_input, context
            )
        elif intent_type == "perspective_request":
            # 用户请求新视角
            result = await self._collaborate_perspective(
                user_id, session_id, user_input, context
            )
        elif intent_type == "deep_dialogue":
            # 用户想要深度对话
            result = await self._collaborate_dialogue(
                user_id, session_id, user_input, context
            )
        else:
            # 通用协作
            result = await self._collaborate_general(
                user_id, session_id, user_input, context
            )

        # 记录协作历史
        self.sessions[session_id]["history"].append(
            {
                "role": "collaborator",
                "content": result.get("response", ""),
                "timestamp": datetime.now().isoformat(),
                "intent_type": intent_type,
                "collaboration_type": result.get("type", "general"),
            }
        )

        # 从协作中学习
        await self.learner.learn_from_collaboration(user_id, user_input, result)

        return result

    def _classify_intent(self, user_input: str) -> str:
        """
        理解用户意图类型
        """
        input_lower = user_input.lower()

        if any(
            kw in input_lower
            for kw in ["我认为", "我觉得", "我的想法是", "i think", "i believe"]
        ):
            return "thinking_share"
        elif any(
            kw in input_lower
            for kw in ["问题", "无法解决", " stuck", "problem", "解决不了"]
        ):
            return "problem_statement"
        elif any(
            kw in input_lower
            for kw in ["怎么理解", "什么意思", "什么关系", "explain", "understand"]
        ):
            return "knowledge_request"
        elif any(
            kw in input_lower
            for kw in ["你怎么看", "有什么看法", "不同角度", "perspective", "view"]
        ):
            return "perspective_request"
        elif any(
            kw in input_lower for kw in ["深度对话", "聊聊", "讨论", "talk", "discuss"]
        ):
            return "deep_dialogue"
        else:
            return "general"

    async def _collaborate_thinking_expansion(
        self,
        user_id: str,
        session_id: str,
        user_thinking: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        协作者：思维扩展
        """
        # 扩展用户思维
        expansion = await self.thinking_expander.expand(user_thinking, context or {})

        # 生成挑战性问题
        challenging_questions = (
            await self.thinking_expander.generate_challenging_questions(user_thinking)
        )

        # 发现盲点
        blind_spots = await self.thinking_expander.discover_blind_spots(user_thinking)

        response = f"""我听到你的想法了。让我提供一些扩展视角：

## 思维扩展

你提到：{user_thinking[:100]}...

从你的思路出发，我看到几个可能的方向：

{expansion["expanded_views"]}

## 可能的盲点

{blind_spots["analysis"]}

{chr(10).join([f"- {q}" for q in blind_spots["blind_spots"][:3]])}

## 挑战性问题

你提到"A因为B"，但如果：

{chr(10).join([f"- {q}" for q in challenging_questions[:3]])}

这些视角是否对你有新的启发？"""

        return {
            "type": "thinking_expansion",
            "response": response,
            "expanded_views": expansion["expanded_views"],
            "blind_spots": blind_spots,
            "challenging_questions": challenging_questions,
        }

    async def _collaborate_problem_reshape(
        self,
        user_id: str,
        session_id: str,
        problem_statement: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        协作者：问题重塑
        """
        # 分析问题
        analysis = await self.problem_reshaper.analyze_problem(problem_statement)

        # 重塑问题
        reshaped = await self.problem_reshaper.reshape(problem_statement)

        # 发现问题之外的可能
        alternatives = await self.problem_reshaper.discover_alternatives(
            problem_statement
        )

        response = f"""让我重新理解一下你的问题：

## 问题分析

{analysis["surface_problem"]}

你尝试的：{analysis["approaches_tried"]}

{analysis["root_cause_analysis"]}

## 问题重塑

你定义的"解决"是{analysis["definition_of_solve"]}，
但问题可能不在{analysis["problematic_assumptions"]}

换一种方式看这个问题：

{chr(10).join([f"- {r['reframed_problem']}" for r in reshaped["reframings"][:3]])}

## 真正的洞见

{alternatives["insight"]}

{chr(10).join([f"- {a}" for a in alternatives["alternative_perspectives"][:3]])}

也许答案不在"解决"中，而在"转化"中。

你觉得这个重塑是否触及了问题的核心？"""

        return {
            "type": "problem_reshape",
            "response": response,
            "analysis": analysis,
            "reframings": reshaped,
            "alternatives": alternatives,
        }

    async def _collaborate_knowledge_integration(
        self,
        user_id: str,
        session_id: str,
        knowledge_request: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        协作者：知识整合
        """
        # 识别知识领域
        domains = await self.knowledge_integrator.identify_domains(knowledge_request)

        # 发现底层连接
        connections = await self.knowledge_integrator.find_connections(
            domains, knowledge_request
        )

        # 生成整合洞见
        integration = await self.knowledge_integrator.integrate(domains, connections)

        # 提供新视角
        perspectives = await self.knowledge_integrator.generate_perspectives(
            domains, knowledge_request
        )

        response = f"""让我帮你整合这些知识：

## 涉及的领域

{chr(10).join([f"- {d['domain']}: {d['concepts']}" for d in domains[:5]])}

## 底层连接

{integration["core_pattern"]}

我发现一个有趣的模式：

{integration["insight"]}

{chr(10).join([f"**{c['domain_a']}** ↔ **{c['domain_b']}**" for c in connections[:3]])}

## 全新的理解

{chr(10).join([f"- {p}" for p in perspectives[:3]])}

跨领域视角往往能带来突破。你从这个连接中看到了什么新的可能性？"""

        return {
            "type": "knowledge_integration",
            "response": response,
            "domains": domains,
            "connections": connections,
            "integration": integration,
            "perspectives": perspectives,
        }

    async def _collaborate_perspective(
        self, user_id: str, session_id: str, topic: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        协作者：全新视角
        """
        # 生成多维视角
        perspectives = await self.perspective_generator.generate(topic, context or {})

        # 挑战惯性思维
        challenges = await self.perspective_generator.challenge_assumptions(topic)

        # 提供反常识视角
        counter_intuitive = await self.perspective_generator.generate_counter_intuitive(
            topic
        )

        # 构建视角文本（避免f-string中的反斜杠）
        perspectives_text = "\n\n".join(
            [f"**{p['perspective']}**\n{p['explanation']}" for p in perspectives[:4]]
        )
        challenges_text = "\n".join([f"- {c}" for c in challenges[:3]])

        response = f"""关于「{topic}」，让我提供几个你可能没有想过的视角：

## 多维视角

{perspectives_text}

## 惯性思维的挑战

{challenges_text}

## 反常识视角

{counter_intuitive["counter_intuitive_view"]}

{counter_intuitive["reasoning"]}

如果从{counter_intuitive["opposite_view"]}来看这件事呢？

这些视角中，哪个对你最有冲击力？"""

        return {
            "type": "new_perspective",
            "response": response,
            "perspectives": perspectives,
            "challenges": challenges,
            "counter_intuitive": counter_intuitive,
        }

    async def _collaborate_dialogue(
        self,
        user_id: str,
        session_id: str,
        dialogue_input: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        协作者：深度对话
        """
        # 深度对话
        dialogue = await self.dialogue_engine.dialogue(dialogue_input, context or {})

        # 生成苏格拉底式问题
        questions = await self.dialogue_engine.socratic_questions(dialogue_input)

        # 记录思考历程
        reflection = await self.dialogue_engine.reflection(dialogue_input)

        response = f"""让我们来场深度对话：

## 对话

{dialogue["response"]}

{dialogue["follow_up"]}

## 苏格拉底式追问

{chr(10).join([f"- {q}" for q in questions[:3]])}

## 思考历程

{reflection}

真正的洞见往往出现在对话的深处。你感受到什么新的想法在浮现吗？"""

        return {
            "type": "deep_dialogue",
            "response": response,
            "dialogue": dialogue,
            "questions": questions,
            "reflection": reflection,
        }

    async def _collaborate_general(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        通用协作
        """
        # 首先尝试理解用户真正需要什么
        understanding = await self._understand_deep_needs(user_input)

        response = f"""{understanding["acknowledgment"]}

{understanding["deeper_question"]}

{understanding["perspective"]}
"""

        return {
            "type": "general",
            "response": response,
            "understanding": understanding,
        }

    async def _understand_deep_needs(self, user_input: str) -> Dict[str, Any]:
        """
        理解用户深层需求
        """
        # 分析用户输入
        analysis = {
            "surface_request": user_input,
            "possible_needs": [],
            "deeper_question": "",
            "acknowledgment": "",
            "perspective": "",
        }

        # 深层需求识别
        input_lower = user_input.lower()

        if any(kw in input_lower for kw in ["帮助", "帮", "help", "assist"]):
            analysis["acknowledgment"] = "我听到你需要帮助。"
            analysis["deeper_question"] = (
                "不过，在帮你之前，我想问：你希望这个帮助带给你什么？"
            )
            analysis["perspective"] = "有时候我们需要的不是答案，而是一个新的起点。"

        elif any(kw in input_lower for kw in ["想", "想要", "want", "希望", "hope"]):
            analysis["acknowledgment"] = "我听到你有一个想法或愿望。"
            analysis["deeper_question"] = (
                "你最想实现的是什么？在你看来，什么才是真正的突破？"
            )
            analysis["perspective"] = (
                "每个愿望背后都有一个更深的需求。找到它，往往就能找到答案。"
            )

        else:
            analysis["acknowledgment"] = "让我听听你想说什么。"
            analysis["deeper_question"] = "你希望从这个对话中得到什么？"
            analysis["perspective"] = "有时候，重要的不是说什么，而是怎么说、为什么说。"

        return analysis

    # ==================== 会话管理 ====================

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话历史
        """
        return self.sessions.get(session_id, {})

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户画像
        """
        return await self.learner.get_user_profile(user_id)

    async def clear_session(self, session_id: str):
        """
        清空会话
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    # ==================== 协作历史 ====================

    async def get_collaboration_history(self, session_id: str = None) -> List[Dict]:
        """
        获取协作历史
        """
        if session_id:
            return self.sessions.get(session_id, {}).get("history", [])
        else:
            all_history = []
            for session in self.sessions.values():
                all_history.extend(session.get("history", []))
            return all_history

    def get_user_feedback(self, user_id: str, feedback: str):
        """
        收集用户反馈
        """
        self.user_profiles[user_id] = self.user_profiles.get(user_id, {})
        self.user_profiles[user_id]["feedback"] = feedback

        return {
            "status": "received",
            "message": "感谢你的反馈，这会帮助我更好地协作为你。",
        }


# ==================== 单例 ====================

_collaborator_instance: Optional[Collaborator] = None


def get_collaborator() -> Collaborator:
    """获取协作者单例"""
    global _collaborator_instance
    if _collaborator_instance is None:
        _collaborator_instance = Collaborator()
    return _collaborator_instance


async def init_collaborator() -> Collaborator:
    """初始化协作者"""
    return get_collaborator()
