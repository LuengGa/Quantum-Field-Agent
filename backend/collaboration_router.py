"""
Collaboration Router - 协作者层API路由
====================================

提供协作者层的API接口：
- /api/collaboration/collaborate - 协作者入口
- /api/collaboration/expand - 思维扩展
- /api/collaboration/reshape - 问题重塑
- /api/collaboration/integrate - 知识整合
- /api/collaboration/perspective - 全新视角
- /api/collaboration/dialogue - 深度对话
- /api/collaboration/learner - 学习系统
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from .collaboration import (
    Collaborator,
    ThinkingExpander,
    ProblemReshaper,
    KnowledgeIntegrator,
    PerspectiveGenerator,
    DialogueEngine,
    Learner,
)

# 创建路由器
collaboration_router = APIRouter(prefix="/api/collaboration", tags=["Collaboration"])

# 全局协作者实例
_collaborator: Optional[Collaborator] = None


def get_collaborator() -> Collaborator:
    """获取协作者实例"""
    global _collaborator
    if _collaborator is None:
        _collaborator = Collaborator()
    return _collaborator


# ==================== 请求模型 ====================


class CollaborateRequest(BaseModel):
    user_id: str = Field(default="user_default", description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    user_input: str = Field(..., description="用户输入")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文")


class ExpandRequest(BaseModel):
    user_thinking: str = Field(..., description="用户思维")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文")


class ReshapeRequest(BaseModel):
    problem: str = Field(..., description="问题陈述")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文")


class IntegrateRequest(BaseModel):
    knowledge_request: str = Field(..., description="知识请求")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文")


class PerspectiveRequest(BaseModel):
    topic: str = Field(..., description="话题")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文")


class DialogueRequest(BaseModel):
    dialogue_input: str = Field(..., description="对话输入")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文")


class TeachRequest(BaseModel):
    lesson: str = Field(..., description="教导内容")
    context: str = Field(..., description="教导场景")


class AskSelfRequest(BaseModel):
    question: str = Field(..., description="向自己提出的问题")


# ==================== 协作者入口 ====================


@collaboration_router.post("/collaborate")
async def collaborate(request: CollaborateRequest):
    """
    协作者入口

    根据用户输入类型，自动选择协作方式：
    - 分享思维 → 思维扩展
    - 陈述问题 → 问题重塑
    - 请求知识 → 知识整合
    - 请求视角 → 全新视角
    - 深度对话 → 对话引擎
    """
    collab = get_collaborator()
    result = await collab.collaborate(
        user_id=request.user_id,
        session_id=request.session_id,
        user_input=request.user_input,
        context=request.context,
    )
    return result


# ==================== 思维扩展 ====================


@collaboration_router.post("/expand")
async def expand_thinking(request: ExpandRequest):
    """
    思维扩展器

    不是回答问题，而是扩展用户思维
    - 扩展用户思维到多个可能方向
    - 发现用户可能忽略的盲点
    - 生成挑战性问题
    """
    expander = ThinkingExpander()

    # 扩展思维
    expansion = await expander.expand(request.user_thinking, request.context)

    # 发现盲点
    blind_spots = await expander.discover_blind_spots(request.user_thinking)

    # 生成挑战性问题
    questions = await expander.generate_challenging_questions(request.user_thinking)

    return {
        "type": "thinking_expansion",
        "original_thinking": request.user_thinking,
        "expansion": expansion,
        "blind_spots": blind_spots,
        "challenging_questions": questions,
    }


@collaboration_router.post("/expand/blind-spots")
async def discover_blind_spots(user_thinking: str):
    """
    发现思维盲点

    检测用户思维中可能存在的认知偏差和盲点
    """
    expander = ThinkingExpander()
    blind_spots = await expander.discover_blind_spots(user_thinking)
    return blind_spots


@collaboration_router.post("/expand/challenging-questions")
async def generate_challenging_questions(user_thinking: str):
    """
    生成挑战性问题

    生成挑战用户假设的问题
    """
    expander = ThinkingExpander()
    questions = await expander.generate_challenging_questions(user_thinking)
    return {"challenging_questions": questions}


# ==================== 问题重塑 ====================


@collaboration_router.post("/reshape")
async def reshape_problem(request: ReshapeRequest):
    """
    问题重塑器

    不是解决用户的问题，而是帮助用户重新定义问题
    - 分析问题的表层和深层
    - 重塑问题的多种方式
    - 发现问题之外的解决方案
    """
    reshaper = ProblemReshaper()

    # 分析问题
    analysis = await reshaper.analyze_problem(request.problem)

    # 重塑问题
    reshaped = await reshaper.reshape(request.problem)

    # 发现替代方案
    alternatives = await reshaper.discover_alternatives(request.problem)

    return {
        "type": "problem_reshape",
        "original_problem": request.problem,
        "analysis": analysis,
        "reshaped": reshaped,
        "alternatives": alternatives,
    }


@collaboration_router.post("/reshape/analyze")
async def analyze_problem(problem: str):
    """
    分析问题

    分析问题的表层、深层、类型、根因
    """
    reshaper = ProblemReshaper()
    analysis = await reshaper.analyze_problem(problem)
    return analysis


# ==================== 知识整合 ====================


@collaboration_router.post("/integrate")
async def integrate_knowledge(request: IntegrateRequest):
    """
    知识整合器

    跨领域连接，发现底层逻辑
    - 识别知识领域
    - 发现领域间连接
    - 整合不同领域的知识
    - 提供跨领域新视角
    """
    integrator = KnowledgeIntegrator()

    # 识别领域
    domains = await integrator.identify_domains(request.knowledge_request)

    # 发现连接
    connections = await integrator.find_connections(domains, request.knowledge_request)

    # 整合知识
    integration = await integrator.integrate(domains, connections)

    # 生成视角
    perspectives = await integrator.generate_perspectives(
        domains, request.knowledge_request
    )

    return {
        "type": "knowledge_integration",
        "request": request.knowledge_request,
        "domains": domains,
        "connections": connections,
        "integration": integration,
        "perspectives": perspectives,
    }


@collaboration_router.post("/integrate/domains")
async def identify_domains(content: str):
    """
    识别知识领域

    分析内容涉及的知识领域
    """
    integrator = KnowledgeIntegrator()
    domains = await integrator.identify_domains(content)
    return {"domains": domains}


# ==================== 全新视角 ====================


@collaboration_router.post("/perspective")
async def generate_perspectives(request: PerspectiveRequest):
    """
    全新视角生成器

    跳出惯性，发现新可能
    - 多维视角生成
    - 惯性思维挑战
    - 反常识视角
    - 未来/过去视角
    """
    generator = PerspectiveGenerator()

    # 生成多维视角
    perspectives = await generator.generate(request.topic, request.context)

    # 挑战惯性思维
    challenges = await generator.challenge_assumptions(request.topic)

    # 生成反常识视角
    counter_intuitive = await generator.generate_counter_intuitive(request.topic)

    return {
        "type": "new_perspective",
        "topic": request.topic,
        "perspectives": perspectives,
        "challenges": challenges,
        "counter_intuitive": counter_intuitive,
    }


@collaboration_router.post("/perspective/time")
async def time_perspective(topic: str):
    """
    时间维度视角

    从过去、现在、未来的角度看问题
    """
    generator = PerspectiveGenerator()
    perspective = await generator._time_perspective(topic)
    return perspective


@collallenge_router.post("/perspective/future")
async def future_perspective(topic: str):
    """
    未来视角

    从未来的角度回看现在
    """
    generator = PerspectiveGenerator()
    perspective = await generator._future_perspective(topic)
    return perspective


# ==================== 深度对话 ====================


@collaboration_router.post("/dialogue")
async def deep_dialogue(request: DialogueRequest):
    """
    深度对话引擎

    苏格拉底式追问，不是给答案，而是引问
    - 分析用户输入
    - 生成回应
    - 生成追问
    """
    engine = DialogueEngine()

    # 对话
    dialogue = await engine.dialogue(request.dialogue_input, request.context)

    # 苏格拉底式问题
    questions = await engine.socratic_questions(request.dialogue_input)

    # 反思
    reflection = await engine.reflection(request.dialogue_input)

    return {
        "type": "deep_dialogue",
        "input": request.dialogue_input,
        "dialogue": dialogue,
        "socratic_questions": questions,
        "reflection": reflection,
    }


@collaboration_router.post("/dialogue/socratic")
async def socratic_questions(user_input: str):
    """
    苏格拉底式问题

    生成引导深度思考的问题
    """
    engine = DialogueEngine()
    questions = await engine.socratic_questions(user_input)
    return {"socratic_questions": questions}


# ==================== 学习系统 ====================


@collaboration_router.post("/learn")
async def learn_from_collaboration(
    user_id: str, user_input: str, collaborator_response: Dict[str, Any]
):
    """
    从协作中学习

    协作者从每次协作中学习，越来越了解用户
    """
    collab = get_collaborator()
    await collab.learner.learn_from_collaboration(
        user_id, user_input, collaborator_response
    )
    return {"status": "learned", "message": "已从这次协作中学习"}


@collaboration_router.get("/learn/profile/{user_id}")
async def get_user_profile(user_id: str):
    """
    获取用户画像

    返回协作者对用户的了解程度
    """
    collab = get_collaborator()
    profile = await collab.learner.get_user_profile(user_id)
    return profile


@collaboration_router.get("/learn/summary/{user_id}")
async def get_collaboration_summary(user_id: str):
    """
    获取协作总结

    返回用户与协作者的协作历史
    """
    collab = get_collaborator()
    summary = await collab.learner.get_collaboration_summary(user_id)
    return summary


@collaboration_router.post("/learn/teach")
async def teach_collaborator(request: TeachRequest):
    """
    指导协作者

    用户可以直接告诉协作者如何更好地协作为自己
    """
    collab = get_collaborator()
    result = await collab.learner.teach_collaborator(
        lesson=request.lesson,
        context=request.context,
    )
    return result


@collaboration_router.get("/learn/understanding/{user_id}")
async def get_mutual_understanding(user_id: str):
    """
    获取相互理解报告

    协作者向用户汇报：我理解你什么
    """
    collab = get_collaborator()
    understanding = await collab.learner.get_mutual_understanding(user_id)
    return understanding


@collaboration_router.get("/learn/stats")
async def get_learning_stats():
    """
    获取学习统计

    返回协作者的学习统计信息
    """
    collab = get_collaborator()
    stats = collab.learner.get_learning_stats()
    return stats


# ==================== 会话管理 ====================


@collaboration_router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    获取会话历史
    """
    collab = get_collaborator()
    session = await collab.get_session(session_id)
    return session


@collaboration_router.get("/history")
async def get_collaboration_history(session_id: Optional[str] = None):
    """
    获取协作历史
    """
    collab = get_collaborator()
    history = await collab.get_collaboration_history(session_id)
    return {"history": history}


@collaboration_router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    清空会话
    """
    collab = get_collaborator()
    await collab.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ==================== 反馈 ====================


@collaboration_router.post("/feedback")
async def submit_feedback(user_id: str, feedback: str):
    """
    提交用户反馈

    用户可以反馈协作者的表现，帮助改进
    """
    collab = get_collaborator()
    result = collab.get_user_feedback(user_id, feedback)
    return result


# ==================== 协作者状态 ====================


@collaboration_router.get("/status")
async def collaboration_status():
    """
    获取协作者状态
    """
    collab = get_collaborator()
    return {
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "collaborator": True,
            "thinking_expander": True,
            "problem_reshaper": True,
            "knowledge_integrator": True,
            "perspective_generator": True,
            "dialogue_engine": True,
            "learner": True,
        },
        "stats": collab.learner.get_learning_stats(),
    }
