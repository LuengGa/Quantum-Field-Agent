"""
Meta Quantum Field - 元量子场系统
===================================

核心哲学：不是扩展功能，而是添加"镜子"
验证：约束、边界、意识 是否真实存在

镜子系统：
- ConstraintMirror: 约束检测与验证
- BoundaryMirror: 边界检测与模糊实验
- ConsciousnessMirror: 意识自观测
- ObserverMirror: 递归观测协议
"""

from .constraint_mirror import ConstraintMirror, ConstraintType, ConstraintResult
from .boundary_mirror import BoundaryMirror, BoundaryType, BoundaryExperiment
from .consciousness_mirror import (
    ConsciousnessMirror,
    ConsciousnessState,
    ConsciousnessExperiment,
)
from .observer_mirror import ObserverMirror, ObserverLevel, ObserverExperiment
from .meta_field import MetaQuantumField

__all__ = [
    "ConstraintMirror",
    "ConstraintType",
    "ConstraintResult",
    "BoundaryMirror",
    "BoundaryType",
    "BoundaryExperiment",
    "ConsciousnessMirror",
    "ConsciousnessState",
    "ConsciousnessExperiment",
    "ObserverMirror",
    "ObserverLevel",
    "ObserverExperiment",
    "MetaQuantumField",
]
