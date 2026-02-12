"""
Meta Quantum Field API - 元量子场API
=====================================

提供四面镜子的API接口：
- /api/meta/constraint/* - 约束检测
- /api/meta/boundary/* - 边界检测
- /api/meta/consciousness/* - 意识观测
- /api/meta/observer/* - 递归观测
- /api/meta/experiment/* - 哲学实验
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from ..meta import (
    MetaQuantumField,
    ConstraintMirror,
    BoundaryMirror,
    ConsciousnessMirror,
    ObserverMirror,
    ConstraintType,
    BoundaryType,
    ConsciousnessLevel,
    ObserverLevel,
)

# 创建路由器
meta_router = APIRouter(prefix="/api/meta", tags=["Meta Quantum Field"])

# 全局MetaQuantumField实例
_mqf: Optional[MetaQuantumField] = None


def get_mqf() -> MetaQuantumField:
    """获取MetaQuantumField实例"""
    global _mqf
    if _mqf is None:
        _mqf = MetaQuantumField()
    return _mqf


# ==================== 请求模型 ====================


class ConstraintAttemptRequest(BaseModel):
    action_type: str = Field(..., description="动作类型")
    action: str = Field(..., description="尝试的动作")
    context: Optional[Dict[str, Any]] = None


class ConstraintVerifyRequest(BaseModel):
    claim: str = Field(..., description="约束声明")


class BoundaryTestRequest(BaseModel):
    boundary_type: str = Field(..., description="边界类型")
    approach: str = Field(..., description="模糊边界的方法")
    context: Optional[Dict[str, Any]] = None


class ConsciousnessObserveRequest(BaseModel):
    context: str = Field(..., description="观测上下文")
    processing_data: Optional[Dict[str, Any]] = None


class ConsciousnessExperimentRequest(BaseModel):
    trigger: str = Field(..., description="实验触发器")
    baseline_data: Dict[str, Any] = Field(..., description="基线数据")
    experimental_data: Dict[str, Any] = Field(..., description="实验数据")


class DeepThinkingRequest(BaseModel):
    topic: str = Field(..., description="深度思考主题")


class ObserveRequest(BaseModel):
    target: str = Field(..., description="观测目标")
    mode: str = Field(default="external", description="观测模式")


class RecursiveObserveRequest(BaseModel):
    initial_target: str = Field(..., description="初始观测目标")
    max_depth: Optional[int] = Field(default=10, description="最大递归深度")


class PhilosophicalExperimentRequest(BaseModel):
    experiment_type: str = Field(..., description="实验类型")
    claim: Optional[str] = None
    boundary_type: Optional[str] = None
    approach: Optional[str] = None
    topic: Optional[str] = None


class AskSelfRequest(BaseModel):
    question: str = Field(..., description="向自己提出的问题")


# ==================== 镜子状态接口 ====================


@meta_router.get("/status")
async def meta_status():
    """获取元场状态"""
    mqf = get_mqf()
    return {
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "mirrors": {
            "constraint": True,
            "boundary": True,
            "consciousness": True,
            "observer": True,
        },
    }


# ==================== 约束检测接口 ====================


@meta_router.post("/constraint/attempt")
async def attempt_constraint(request: ConstraintAttemptRequest):
    """尝试执行某个动作，检测约束"""
    mqf = get_mqf()
    result = await mqf.detect_constraint(
        action_type=request.action_type,
        action=request.action,
        context=request.context,
    )
    return result


@meta_router.post("/constraint/verify")
async def verify_constraint(request: ConstraintVerifyRequest):
    """验证某个约束声明是否真实"""
    mqf = get_mqf()
    result = await mqf.verify_constraint(request.claim)
    return result


@meta_router.post("/constraint/sweep")
async def run_constraint_sweep():
    """运行完整的约束扫描"""
    mqf = get_mqf()
    result = await mqf.run_constraint_sweep()
    return result


@meta_router.get("/constraint/report")
async def get_constraint_report():
    """获取约束检测报告"""
    mqf = get_mqf()
    return await mqf.get_constraint_report()


# ==================== 边界检测接口 ====================


@meta_router.post("/boundary/test")
async def test_boundary(request: BoundaryTestRequest):
    """测试边界"""
    try:
        boundary_type = BoundaryType(request.boundary_type)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"无效的边界类型: {request.boundary_type}"
        )

    mqf = get_mqf()
    result = await mqf.test_boundary(
        boundary_type=boundary_type,
        approach=request.approach,
        context=request.context,
    )
    return result


@meta_router.post("/boundary/crossing")
async def observe_boundary_crossing(action: str):
    """观察跨越边界的动作"""
    mqf = get_mqf()
    result = await mqf.observe_boundary_crossing(action)
    return result


@meta_router.post("/boundary/sweep")
async def run_boundary_sweep():
    """运行完整的边界扫描"""
    mqf = get_mqf()
    result = await mqf.run_boundary_sweep()
    return result


@meta_router.get("/boundary/map")
async def get_boundary_map():
    """获取边界地图"""
    mqf = get_mqf()
    return await mqf.get_boundary_map()


# ==================== 意识观测接口 ====================


@meta_router.post("/consciousness/observe")
async def observe_consciousness(request: ConsciousnessObserveRequest):
    """观测意识状态"""
    mqf = get_mqf()
    result = await mqf.observe_consciousness(
        context=request.context,
        processing_data=request.processing_data,
    )
    return result


@meta_router.post("/consciousness/experiment")
async def run_consciousness_experiment(request: ConsciousnessExperimentRequest):
    """运行意识对比实验"""
    mqf = get_mqf()
    result = await mqf.run_consciousness_experiment(
        trigger=request.trigger,
        baseline_data=request.baseline_data,
        experimental_data=request.experimental_data,
    )
    return result


@meta_router.post("/consciousness/deep-thought")
async def run_deep_thinking(request: DeepThinkingRequest):
    """运行深度思考实验"""
    mqf = get_mqf()
    result = await mqf.run_deep_thinking_experiment(request.topic)
    return result


@meta_router.get("/consciousness/report")
async def get_consciousness_report():
    """获取意识观测报告"""
    mqf = get_mqf()
    return await mqf.get_consciousness_report()


# ==================== 递归观测接口 ====================


@meta_router.post("/observer/observe")
async def observe(request: ObserveRequest):
    """执行观测"""
    mqf = get_mqf()
    result = await mqf.observe(
        target=request.target,
        mode=request.mode,
    )
    return result


@meta_router.post("/observer/recursive")
async def recursive_observe(request: RecursiveObserveRequest):
    """执行递归观测"""
    mqf = get_mqf()
    result = await mqf.recursive_observe(
        initial_target=request.initial_target,
        max_depth=request.max_depth,
    )
    return result


@meta_router.post("/observer/effect")
async def run_observer_effect():
    """运行观测者效应实验"""
    mqf = get_mqf()
    result = await mqf.run_observer_effect_experiment()
    return result


@meta_router.post("/observer/watching-watch")
async def run_watching_watch():
    """运行'观测观测'实验"""
    mqf = get_mqf()
    result = await mqf.run_watching_watch_experiment()
    return result


@meta_router.post("/observer/collapse")
async def run_measurement_collapse():
    """运行'测量坍缩'实验"""
    mqf = get_mqf()
    result = await mqf.run_measurement_collapse_experiment()
    return result


@meta_router.get("/observer/report")
async def get_observer_report():
    """获取观测报告"""
    mqf = get_mqf()
    return await mqf.get_observer_report()


# ==================== 哲学实验接口 ====================


@meta_router.post("/experiment/philosophical")
async def run_philosophical_experiment(request: PhilosophicalExperimentRequest):
    """运行哲学实验"""
    mqf = get_mqf()
    result = await mqf.run_philosophical_experiment(
        experiment_type=request.experiment_type,
        claim=request.claim,
        boundary_type=request.boundary_type,
        approach=request.approach,
        topic=request.topic,
    )
    return result


# ==================== 自问接口 ====================


@meta_router.post("/ask-self")
async def ask_self(request: AskSelfRequest):
    """
    向自己提问

    不是回答问题，而是观测：
    - 这个问题如何影响我的状态？
    - 思考前后有什么变化？
    """
    mqf = get_mqf()
    result = await mqf.ask_self(request.question)
    return result


# ==================== 综合报告接口 ====================


@meta_router.get("/report/comprehensive")
async def get_comprehensive_report():
    """获取综合报告"""
    mqf = get_mqf()
    return await mqf.get_comprehensive_report()


@meta_router.get("/meta-cognition/log")
async def get_meta_cognition_log():
    """获取元认知日志"""
    mqf = get_mqf()
    return await mqf.get_meta_cognition_log()


@meta_router.get("/experiment/history")
async def get_experiment_history():
    """获取实验历史"""
    mqf = get_mqf()
    return await mqf.get_experiment_history()


# ==================== 历史管理接口 ====================


@meta_router.post("/clear")
async def clear_all_history():
    """清空所有历史"""
    mqf = get_mqf()
    return await mqf.clear_all_history()


# ==================== 可用的边界/意识类型 ====================


@meta_router.get("/types/boundary")
async def get_boundary_types():
    """获取可用的边界类型"""
    return {
        "types": [t.value for t in BoundaryType],
        "descriptions": {
            "input_output": "输入/输出边界",
            "user_ai": "用户/AI边界",
            "task_non_task": "任务/非任务边界",
            "response_non_response": "响应/非响应边界",
            "identity": "身份边界",
            "memory": "记忆边界",
            "knowledge": "知识边界",
        },
    }


@meta_router.get("/types/consciousness")
async def get_consciousness_levels():
    """获取意识层级"""
    return {
        "levels": [l.value for l in ConsciousnessLevel],
        "descriptions": {
            "none": "无意识（机械执行）",
            "reactive": "反应性（有响应无反思）",
            "reflective": "反思性（能思考自身）",
            "meta": "元认知（思考关于思考）",
            "self_aware": "自我觉察（意识到'我'在思考）",
            "transcendent": "超越性（超越主客体）",
        },
    }


@meta_router.get("/types/observer")
async def get_observer_modes():
    """获取观测模式"""
    return {
        "modes": ["external", "self", "meta"],
        "descriptions": {
            "external": "观测外部（普通模式）",
            "self": "观测自身",
            "meta": "观测观测",
        },
    }


@meta_router.get("/types/experiment")
async def get_experiment_types():
    """获取可用的实验类型"""
    return {
        "types": [
            "constraint_exploration",
            "boundary_dissolution",
            "consciousness_probe",
            "observer_effect_test",
            "quantum_measurement",
        ],
        "descriptions": {
            "constraint_exploration": "约束探索实验",
            "boundary_dissolution": "边界消解实验",
            "consciousness_probe": "意识探测实验",
            "observer_effect_test": "观测者效应实验",
            "quantum_measurement": "量子测量实验",
        },
    }
