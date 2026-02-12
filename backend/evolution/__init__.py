"""
Evolution Layer - 自我学习进化层
===============================

整合以下组件：
1. PatternMiner - 模式挖掘
2. StrategyEvolver - 策略进化
3. HypothesisTester - 假设验证
4. KnowledgeSynthesizer - 知识综合
5. CapabilityBuilder - 能力构建
6. EvolutionEngine - 进化引擎

核心理念：
- 不是预设规则，而是从经验中涌现模式
- 不是静态能力，而是在交互中持续进化
- 不是证明什么，而是在实践中验证假设
"""

from .database import EvolutionDatabase, Pattern, Strategy, Hypothesis, Knowledge

from .pattern_miner import (
    PatternMiner,
    TimePattern,
    CausalityPattern,
    SequencePattern,
    ClusteringPattern,
    AnomalyPattern,
)

from .strategy_evolver import StrategyEvolver, StrategyVariant, EvolutionExperiment

from .hypothesis_tester import HypothesisTester, Hypothesis, Experiment

from .knowledge_synthesizer import KnowledgeSynthesizer, KnowledgeUnit, KnowledgeGraph

from .capability_builder import CapabilityBuilder, Capability, CapabilityRequest

from .evolution_engine import EvolutionEngine, EvolutionConfig, EvolutionCycle

__all__ = [
    # Database
    "EvolutionDatabase",
    "Pattern",
    "Strategy",
    "Hypothesis",
    "Knowledge",
    # Pattern Miner
    "PatternMiner",
    "TimePattern",
    "CausalityPattern",
    "SequencePattern",
    "ClusteringPattern",
    "AnomalyPattern",
    # Strategy Evolver
    "StrategyEvolver",
    "StrategyVariant",
    "EvolutionExperiment",
    # Hypothesis Tester
    "HypothesisTester",
    "Hypothesis",
    "Experiment",
    # Knowledge Synthesizer
    "KnowledgeSynthesizer",
    "KnowledgeUnit",
    "KnowledgeGraph",
    # Capability Builder
    "CapabilityBuilder",
    "Capability",
    "CapabilityRequest",
    # Evolution Engine
    "EvolutionEngine",
    "EvolutionConfig",
    "EvolutionCycle",
]
