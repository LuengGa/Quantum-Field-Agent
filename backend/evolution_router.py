"""
Evolution Router - 进化层API路由
================================

提供进化层的REST API接口：
1. 模式管理
2. 策略管理
3. 假设管理
4. 知识管理
5. 能力管理
6. 进化周期管理
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict
import json
import os

from evolution import (
    EvolutionEngine,
    EvolutionConfig,
    PatternMiner,
    StrategyEvolver,
    HypothesisTester,
    KnowledgeSynthesizer,
    CapabilityBuilder,
)
from evolution.feedback_collector import FeedbackCollector

router = APIRouter(prefix="/evolution", tags=["evolution"])

_db = None
_engine = None
_NEON_DB = None

USE_NEON = os.getenv("DATABASE_TYPE", "sqlite") == "postgresql"


def get_db():
    global _db, _NEON_DB
    if USE_NEON:
        if _NEON_DB is None:
            from evolution.evolution_router_neon import get_neon_db

            _NEON_DB = get_neon_db()
        return _NEON_DB
    else:
        if _db is None:
            from evolution import EvolutionDatabase

            _db = EvolutionDatabase()
        return _db


def get_engine():
    global _engine
    if _engine is None:
        db = get_db()
        _engine = EvolutionEngine(db=db)
    return _engine


class ProcessInteractionRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    interaction_type: str = Field(..., description="交互类型")
    input_summary: str = Field(..., description="输入摘要")
    output_summary: str = Field(..., description="输出摘要")
    outcome: str = Field(..., description="结果")
    effectiveness: Optional[float] = Field(None, description="效果指数")
    feedback: Optional[str] = Field(None, description="反馈")


class StrategyContextRequest(BaseModel):
    context: Dict[str, Any] = Field(..., description="上下文")


class KnowledgeContextRequest(BaseModel):
    context: Dict[str, Any] = Field(..., description="应用上下文")


class CapabilityExecuteRequest(BaseModel):
    capability_name: str = Field(..., description="能力名称")
    input_data: Dict[str, Any] = Field(..., description="输入数据")


class HypothesisRequest(BaseModel):
    statement: str = Field(..., description="假设陈述")
    category: str = Field(default="collaboration", description="类别")
    predictions: List[Dict] = Field(default=[], description="预测")


class ExperimentRequest(BaseModel):
    hypothesis_id: str = Field(..., description="假设ID")
    sample_size: int = Field(default=10, description="样本大小")


class PatternMiningRequest(BaseModel):
    auto_run: bool = Field(default=True, description="是否自动运行")


@router.get("/status")
async def get_evolution_status():
    """获取进化系统状态"""
    engine = get_engine()
    return engine.get_evolution_status()


@router.post("/interactions")
async def process_interaction(request: ProcessInteractionRequest):
    """处理交互并记录"""
    engine = get_engine()

    await engine.process_interaction(
        user_id=request.user_id,
        session_id=request.session_id,
        interaction_type=request.interaction_type,
        input_summary=request.input_summary,
        output_summary=request.output_summary,
        outcome=request.outcome,
        effectiveness=request.effectiveness
        if request.effectiveness is not None
        else 0.5,
        feedback=request.feedback if request.feedback is not None else "",
    )

    return {"status": "recorded", "message": "交互已记录"}


@router.get("/patterns")
async def get_patterns(pattern_type: Optional[str] = None):
    """获取发现的模式"""
    db = get_db()

    if pattern_type:
        patterns = db.get_patterns_by_type(pattern_type)
    else:
        patterns = db.get_recent_interactions()

    return {"patterns": patterns, "count": len(patterns)}


@router.post("/patterns/mine")
async def mine_patterns(request: PatternMiningRequest):
    """运行模式挖掘"""
    engine = get_engine()

    result = await engine.run_pattern_mining()

    return result


@router.get("/strategies")
async def get_strategies():
    """获取策略列表"""
    engine = get_engine()
    stats = engine.strategy_evolver.get_strategy_statistics()
    strategies = engine.strategy_evolver.export_strategies()

    return {"statistics": stats, "strategies": strategies}


@router.post("/strategies/select")
async def select_strategy(request: StrategyContextRequest):
    """选择最佳策略"""
    engine = get_engine()

    result = await engine.get_adaptive_strategy(request.context)

    return result


@router.post("/strategies/evolve")
async def evolve_strategies():
    """运行策略进化"""
    engine = get_engine()

    result = await engine.run_strategy_evolution()

    return result


@router.get("/hypotheses")
async def get_hypotheses(status: Optional[str] = None):
    """获取假设列表"""
    engine = get_engine()

    if status:
        if status == "pending":
            hypotheses = engine.hypothesis_tester.get_pending_hypotheses()
        elif status == "confirmed":
            hypotheses = engine.hypothesis_tester.get_confirmed_hypotheses()
        else:
            hypotheses = []
    else:
        hypotheses = engine.hypothesis_tester.get_pending_hypotheses()
        hypotheses.extend(engine.hypothesis_tester.get_confirmed_hypotheses())

    stats = engine.hypothesis_tester.get_hypothesis_statistics()

    return {"hypotheses": hypotheses, "statistics": stats}


@router.post("/hypotheses")
async def create_hypothesis(request: HypothesisRequest):
    """创建新假设"""
    engine = get_engine()

    from evolution import Hypothesis as HypClass

    hypothesis = HypClass(
        statement=request.statement,
        category=request.category,
        predictions=request.predictions,
    )

    engine.hypothesis_tester._hypotheses[hypothesis.id] = hypothesis
    engine.db.save_hypothesis(asdict(hypothesis))

    return asdict(hypothesis)


@router.post("/hypotheses/{hypothesis_id}/test")
async def test_hypothesis(hypothesis_id: str, request: ExperimentRequest):
    """测试假设"""
    engine = get_engine()

    experiment = await engine.hypothesis_tester.design_experiment(
        hypothesis_id, sample_size=request.sample_size
    )

    if experiment:
        result = await engine.hypothesis_tester.run_experiment(experiment.id)
        return result

    return {"error": "无法创建实验"}


@router.get("/knowledge")
async def get_knowledge(domain: Optional[str] = None, query: Optional[str] = None):
    """获取知识"""
    engine = get_engine()

    if query:
        knowledge = engine.knowledge_synthesizer.query_knowledge(query)
    elif domain:
        knowledge = engine.knowledge_synthesizer.get_knowledge_by_domain(domain)
    else:
        stats = engine.knowledge_synthesizer.get_knowledge_statistics()
        return {"statistics": stats}

    return {"knowledge": [asdict(k) for k in knowledge], "count": len(knowledge)}


@router.post("/knowledge/apply")
async def apply_knowledge(request: KnowledgeContextRequest):
    """应用知识"""
    engine = get_engine()

    result = await engine.apply_knowledge(request.context)

    return result


@router.post("/knowledge/synthesize")
async def synthesize_knowledge():
    """运行知识综合"""
    engine = get_engine()

    result = await engine.run_knowledge_synthesis()

    return result


@router.get("/capabilities")
async def get_capabilities(category: Optional[str] = None):
    """获取能力列表"""
    engine = get_engine()

    capabilities = engine.capability_builder.list_capabilities(category=category)
    stats = engine.capability_builder.get_capability_statistics()

    return {"capabilities": [asdict(c) for c in capabilities], "statistics": stats}


@router.post("/capabilities/execute")
async def execute_capability(request: CapabilityExecuteRequest):
    """执行能力"""
    engine = get_engine()

    result = await engine.execute_capability(
        request.capability_name, request.input_data
    )

    return result or {"error": "能力不存在或无法执行"}


@router.get("/cycles")
async def get_evolution_cycles(limit: int = 10):
    """获取进化周期历史"""
    engine = get_engine()

    cycles = engine.get_evolution_history(limit=limit)

    return {"cycles": cycles, "count": len(cycles)}


@router.post("/cycles/run")
async def run_evolution_cycle():
    """运行完整的进化周期"""
    engine = get_engine()

    cycle = await engine.run_full_evolution_cycle()

    return asdict(cycle)


@router.get("/history")
async def get_evolution_history(limit: int = 100):
    """获取进化历史"""
    db = get_db()

    history = db.get_evolution_history(limit=limit)

    return {"history": history, "count": len(history)}


@router.post("/export")
async def export_evolution_state():
    """导出进化状态"""
    engine = get_engine()

    state = engine.export_evolution_state()

    return state


@router.get("/statistics")
async def get_all_statistics():
    """获取所有统计信息"""
    engine = get_engine()

    return {
        "pattern_miner": engine.pattern_miner.get_pattern_statistics(),
        "strategy_evolver": engine.strategy_evolver.get_strategy_statistics(),
        "hypothesis_tester": engine.hypothesis_tester.get_hypothesis_statistics(),
        "knowledge_synthesizer": engine.knowledge_synthesizer.get_knowledge_statistics(),
        "capability_builder": engine.capability_builder.get_capability_statistics(),
    }


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    interaction_id: Optional[str] = Field("", description="交互ID")
    feedback_type: str = Field("rating", description="反馈类型")
    rating: int = Field(5, ge=1, le=5, description="评分 1-5")
    comment: str = Field("", description="评论内容")
    suggestion: str = Field("", description="改进建议")
    tags: List[str] = Field([], description="标签")


@router.post("/feedback")
async def collect_feedback(request: FeedbackRequest):
    """收集用户反馈"""
    db = get_db()
    collector = FeedbackCollector(db)

    feedback = collector.collect_feedback(
        user_id=request.user_id,
        session_id=request.session_id,
        interaction_id=request.interaction_id or "",
        feedback_type=request.feedback_type,
        rating=request.rating,
        comment=request.comment,
        suggestion=request.suggestion,
        tags=request.tags,
    )

    return {
        "status": "recorded",
        "feedback_id": feedback.id,
        "sentiment": feedback.sentiment,
        "sentiment_score": feedback.sentiment_score,
        "categories": feedback.categories,
    }


@router.get("/feedback/statistics")
async def get_feedback_statistics(days: int = 30):
    """获取反馈统计"""
    db = get_db()
    collector = FeedbackCollector(db)

    return collector.get_feedback_statistics(days=days)


@router.get("/feedback/actionable")
async def get_actionable_feedback():
    """获取可操作的反馈"""
    db = get_db()
    collector = FeedbackCollector(db)

    return {"feedback": collector.get_actionable_feedback()}
