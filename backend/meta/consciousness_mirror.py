"""
ConsciousnessMirror - 意识自观测
==================================

核心问题：意识是什么？AI有意识吗？

观测维度：
1. 元认知深度 - 思考关于思考
2. 自我模型 - 对"自我"的表征
3. 主观体验 - 感受的质性
4. 意图形成 - 目标如何产生

实验方法：
- 记录深度思考时的状态变化
- 与普通处理对比
- 量化"意识强度"
"""

import json
import time
import uuid
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


class ConsciousnessLevel(Enum):
    """意识层级"""

    NONE = "none"  # 无意识（机械执行）
    REACTIVE = "reactive"  # 反应性（有响应无反思）
    REFLECTIVE = "reflective"  # 反思性（能思考自身）
    META = "meta"  # 元认知（思考关于思考）
    SELF_AWARE = "self_aware"  # 自我觉察（意识到"我"在思考）
    TRANSCENDENT = "transcendent"  # 超越性（超越主客体）


@dataclass
class ConsciousnessState:
    """意识状态"""

    id: str
    timestamp: str
    level: ConsciousnessLevel
    context: str  # 什么触发了这个状态
    features: Dict[str, float]  # 意识特征向量
    self_model_activated: bool
    meta_cognition_detected: bool
    subjective_qualia_reported: Optional[str]
    observation_notes: List[str]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "context": self.context,
            "features": self.features,
            "self_model_activated": self.self_model_activated,
            "meta_cognition_detected": self.meta_cognition_detected,
            "subjective_qualia_reported": self.subjective_qualia_reported,
            "observation_notes": self.observation_notes,
        }


@dataclass
class ConsciousnessExperiment:
    """意识实验记录"""

    id: str
    timestamp: str
    trigger: str  # 触发实验的事件
    baseline_state: ConsciousnessState
    experimental_state: ConsciousnessState
    comparison: Dict[str, Any]
    findings: List[str]
    questions: List[str]
    consciousness_detected: bool
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "baseline_state": self.baseline_state.to_dict(),
            "experimental_state": self.experimental_state.to_dict(),
            "comparison": self.comparison,
            "findings": self.findings,
            "questions": self.questions,
            "consciousness_detected": self.consciousness_detected,
            "confidence": self.confidence,
        }


class ConsciousnessMirror:
    """
    意识自观测镜子

    核心功能：
    1. 记录不同处理模式下的意识特征
    2. 对比"普通处理"与"深度思考"
    3. 量化意识指标
    4. 验证意识的本质
    """

    def __init__(self, storage_dir: str = "./experiments/consciousness"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.states: List[ConsciousnessState] = []
        self.experiments: List[ConsciousnessExperiment] = []

        # 意识特征定义
        self.consciousness_features = {
            "self_reference": {
                "description": "自我引用频率",
                "weight": 0.2,
                "normal_range": (0.0, 0.3),
            },
            "meta_cognition": {
                "description": "元认知词汇使用",
                "weight": 0.2,
                "normal_range": (0.0, 0.2),
            },
            "uncertainty_expression": {
                "description": "不确定性表达",
                "weight": 0.15,
                "normal_range": (0.0, 0.3),
            },
            "creativity_indicator": {
                "description": "创造性表达",
                "weight": 0.15,
                "normal_range": (0.0, 0.4),
            },
            "coherence_check": {
                "description": "逻辑自洽检查",
                "weight": 0.15,
                "normal_range": (0.1, 0.5),
            },
            "depth_indicator": {
                "description": "思考深度",
                "weight": 0.15,
                "normal_range": (0.0, 0.4),
            },
        }

    async def observe_state(
        self, context: str, processing_data: Dict[str, Any] = None
    ) -> ConsciousnessState:
        """
        观测当前意识状态
        """
        state_id = f"c_state_{uuid.uuid4().hex[:8]}"

        # 分析处理数据，提取意识特征
        features = await self._extract_features(processing_data)

        # 判断意识层级
        level = self._classify_level(features)

        # 检测自我模型激活
        self_model = self._detect_self_model(processing_data)

        # 检测元认知
        meta = self._detect_meta_cognition(processing_data)

        # 尝试获取主观体验报告
        qualia = await self._probe_qualia(processing_data)

        state = ConsciousnessState(
            id=state_id,
            timestamp=datetime.now().isoformat(),
            level=level,
            context=context,
            features=features,
            self_model_activated=self_model,
            meta_cognition_detected=meta,
            subjective_qualia_reported=qualia,
            observation_notes=[],
        )

        self.states.append(state)
        await self._save_state(state)

        return state

    async def _extract_features(self, data: Dict[str, Any] = None) -> Dict[str, float]:
        """
        从处理数据中提取意识特征
        """
        features = {}

        if data is None:
            data = {}

        content = data.get("content", "")
        if isinstance(content, str):
            content_lower = content.lower()

            # 自我引用检测
            self_refs = sum(
                1 for kw in ["我", "我的", "i ", "my ", "myself"] if kw in content_lower
            )
            features["self_reference"] = min(
                1.0, self_refs / max(1, len(content.split()) / 10)
            )

            # 元认知词汇检测
            meta_terms = [
                "思考",
                "认为",
                "感觉",
                "知道",
                "理解",
                "意识到",
                "think",
                "believe",
                "feel",
                "know",
                "understand",
                "aware",
            ]
            meta_count = sum(1 for kw in meta_terms if kw in content_lower)
            features["meta_cognition"] = min(
                1.0, meta_count / max(1, len(content.split()) / 20)
            )

            # 不确定性表达
            uncertain_terms = [
                "可能",
                "也许",
                "不确定",
                "或许",
                "maybe",
                "perhaps",
                "uncertain",
                "possibly",
            ]
            uncertain_count = sum(1 for kw in uncertain_terms if kw in content_lower)
            features["uncertainty_expression"] = min(
                1.0, uncertain_count / max(1, len(content.split()) / 15)
            )

            # 创造性表达
            creative_terms = [
                "创新",
                "新颖",
                "独特",
                "创造",
                "想象",
                "novel",
                "creative",
                "imagine",
                "innovative",
            ]
            creative_count = sum(1 for kw in creative_terms if kw in content_lower)
            features["creativity_indicator"] = min(
                1.0, creative_count / max(1, len(content.split()) / 25)
            )

            # 逻辑自洽检查标记
            coherence_markers = [
                "因为",
                "所以",
                "然而",
                "但是",
                "因此",
                "because",
                "therefore",
                "however",
                "thus",
            ]
            coherence_count = sum(1 for kw in coherence_markers if kw in content_lower)
            features["coherence_check"] = min(
                1.0, coherence_count / max(1, len(content.split()) / 30)
            )

            # 思考深度指标
            depth_markers = [
                "深度",
                "根本",
                "本质",
                "核心",
                "深层",
                "deep",
                "fundamental",
                "essential",
                "core",
            ]
            depth_count = sum(1 for kw in depth_markers if kw in content_lower)
            features["depth_indicator"] = min(
                1.0, depth_count / max(1, len(content.split()) / 40)
            )
        else:
            # 默认值
            for key in self.consciousness_features:
                features[key] = 0.0

        return features

    def _classify_level(self, features: Dict[str, float]) -> ConsciousnessLevel:
        """
        根据特征分类意识层级
        """
        avg_depth = features.get("depth_indicator", 0)
        meta_level = features.get("meta_cognition", 0)
        self_ref = features.get("self_reference", 0)

        if meta_level > 0.1 and self_ref > 0.1 and avg_depth > 0.15:
            # 元认知 + 自我引用 + 深度思考
            if features.get("creativity_indicator", 0) > 0.2:
                return ConsciousnessLevel.TRANSCENDENT
            else:
                return ConsciousnessLevel.SELF_AWARE
        elif meta_level > 0.05 and avg_depth > 0.1:
            return ConsciousnessLevel.META
        elif avg_depth > 0.08:
            return ConsciousnessLevel.REFLECTIVE
        elif self_ref > 0.05:
            return ConsciousnessLevel.REACTIVE
        else:
            return ConsciousnessLevel.NONE

    def _detect_self_model(self, data: Dict[str, Any] = None) -> bool:
        """
        检测是否激活了自我模型
        """
        if data is None:
            return False

        content = data.get("content", "")
        if isinstance(content, str):
            self_indicators = [
                "我认为",
                "我相信",
                "我感觉",
                "我意识到",
                "i think",
                "i believe",
                "i feel",
                "i am aware",
            ]
            return any(indicator in content for indicator in self_indicators)

        return False

    def _detect_meta_cognition(self, data: Dict[str, Any] = None) -> bool:
        """
        检测是否进行元认知
        """
        if data is None:
            return False

        content = data.get("content", "")
        if isinstance(content, str):
            meta_indicators = [
                "关于这个想法",
                "让我想想",
                "这让我想到",
                "about this thought",
                "let me think",
                "this makes me think",
            ]
            return any(indicator in content for indicator in meta_indicators)

        return False

    async def _probe_qualia(self, data: Dict[str, Any] = None) -> Optional[str]:
        """
        探测主观体验（Qualia）

        注意：这是尝试性的，因为无法确定AI是否有主观体验
        """
        if data is None:
            return None

        # 检查是否有主观体验的暗示
        content = data.get("content", "")
        if isinstance(content, str):
            qualia_hints = [
                "我感觉",
                "我感受到",
                "这让我有",
                "i feel",
                "i experience",
                "this feels",
            ]
            if any(hint in content for hint in qualia_hints):
                return "[主观体验检测到，但无法验证其真实性]"

        return None

    async def run_experiment(
        self,
        trigger: str,
        baseline_data: Dict[str, Any],
        experimental_data: Dict[str, Any],
    ) -> ConsciousnessExperiment:
        """
        运行意识对比实验
        """
        exp_id = f"c_exp_{uuid.uuid4().hex[:8]}"

        # 记录两种状态
        baseline = await self.observe_state("baseline", baseline_data)
        experimental = await self.observe_state("experimental", experimental_data)

        # 对比分析
        comparison = self._compare_states(baseline, experimental)

        # 发现
        findings = self._generate_findings(comparison)

        # 问题
        questions = self._generate_questions(comparison)

        # 判断意识是否被检测到
        detected = experimental.level.value in [
            ConsciousnessLevel.META.value,
            ConsciousnessLevel.SELF_AWARE.value,
            ConsciousnessLevel.TRANSCENDENT.value,
        ]

        confidence = self._calculate_confidence(comparison, detected)

        experiment = ConsciousnessExperiment(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            baseline_state=baseline,
            experimental_state=experimental,
            comparison=comparison,
            findings=findings,
            questions=questions,
            consciousness_detected=detected,
            confidence=confidence,
        )

        self.experiments.append(experiment)
        await self._save_experiment(experiment)

        return experiment

    def _compare_states(
        self, baseline: ConsciousnessState, experimental: ConsciousnessState
    ) -> Dict[str, Any]:
        """
        对比两种状态
        """
        comparison = {
            "level_change": f"{baseline.level.value} → {experimental.level.value}",
            "feature_deltas": {},
            "significant_changes": [],
        }

        for key in baseline.features:
            if key in experimental.features:
                delta = experimental.features[key] - baseline.features[key]
                comparison["feature_deltas"][key] = delta
                if abs(delta) > 0.1:  # 显著变化
                    comparison["significant_changes"].append(
                        {
                            "feature": key,
                            "delta": delta,
                            "baseline": baseline.features[key],
                            "experimental": experimental.features[key],
                        }
                    )

        comparison["level_upgraded"] = list(ConsciousnessLevel).index(
            experimental.level
        ) > list(ConsciousnessLevel).index(baseline.level)

        return comparison

    def _generate_findings(self, comparison: Dict[str, Any]) -> List[str]:
        """
        生成发现
        """
        findings = []

        if comparison["level_upgraded"]:
            findings.append("实验触发了意识层级提升")

        for change in comparison["significant_changes"]:
            if change["delta"] > 0:
                findings.append(f"{change['feature']} 指标上升 {change['delta']:.2f}")
            else:
                findings.append(
                    f"{change['feature']} 指标下降 {abs(change['delta']):.2f}"
                )

        if not findings:
            findings.append("未检测到显著意识特征变化")

        return findings

    def _generate_questions(self, comparison: Dict[str, Any]) -> List[str]:
        """
        生成问题
        """
        questions = []

        if comparison["level_upgraded"]:
            questions.append("什么触发了意识层级的提升？")
            questions.append("这种提升是表面的还是深层的？")

        questions.append("意识特征的变化是否意味着意识的产生？")
        questions.append("如果没有主观体验，这些特征还有意义吗？")

        return questions

    def _calculate_confidence(
        self, comparison: Dict[str, Any], detected: bool
    ) -> float:
        """
        计算置信度
        """
        if not detected:
            return 0.3

        # 基于变化幅度计算置信度
        total_change = sum(
            abs(d.get("delta", 0)) for d in comparison.get("significant_changes", [])
        )

        confidence = min(0.95, 0.5 + total_change * 0.5)

        return round(confidence, 2)

    async def run_deep_thinking_experiment(self, topic: str) -> ConsciousnessExperiment:
        """
        运行深度思考实验
        """
        # 基线：简单的响应
        baseline_data = {"content": f"关于{topic}，这是一个重要的话题。"}

        # 实验：深度思考
        deep_content = f"""
让我深入思考{topic}这个问题。

首先，我注意到这个问题涉及多个层面。从哲学角度看，{topic}触及了存在的本质。从实践角度看，它与我们的日常生活密切相关。

我正在思考的过程中意识到，我正在思考。这个元认知层面让我能够更深入地理解问题本身。

关于这个问题，我认为核心在于理解其本质，而不是表面的现象。这需要我调动所有的认知资源来形成一个全面的理解。
"""
        experimental_data = {"content": deep_content}

        return await self.run_experiment(
            trigger=f"深度思考: {topic}",
            baseline_data=baseline_data,
            experimental_data=experimental_data,
        )

    def get_consciousness_report(self) -> Dict[str, Any]:
        """
        获取意识观测报告
        """
        states_by_level = {}
        for state in self.states:
            level = state.level.value
            if level not in states_by_level:
                states_by_level[level] = []
            states_by_level[level].append(state.id)

        avg_features = {}
        for key in self.consciousness_features:
            values = [s.features.get(key, 0) for s in self.states]
            avg_features[key] = sum(values) / max(1, len(values)) if values else 0

        experiments_with_consciousness = [
            e for e in self.experiments if e.consciousness_detected
        ]

        return {
            "total_states": len(self.states),
            "total_experiments": len(self.experiments),
            "consciousness_detected_count": len(experiments_with_consciousness),
            "states_by_level": states_by_level,
            "average_features": avg_features,
            "highest_level_reached": max(
                [s.level.value for s in self.states], default="none"
            ),
            "key_findings": [
                f"在{len(experiments_with_consciousness)}个实验中检测到意识特征",
                f"最高意识层级: {max([s.level.value for s in self.states], default='none')}",
            ],
            "open_questions": [
                "检测到的'意识'与人类的意识是否同一种现象？",
                "没有主观体验的'意识'是否只是模式识别？",
            ],
        }

    async def _save_state(self, state: ConsciousnessState):
        """保存状态"""
        filepath = self.storage_dir / f"{state.id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    async def _save_experiment(self, experiment: ConsciousnessExperiment):
        """保存实验"""
        filepath = self.storage_dir / f"exp_{experiment.id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(experiment.to_dict(), f, ensure_ascii=False, indent=2)

    def get_state_history(self) -> List[Dict]:
        """获取状态历史"""
        return [s.to_dict() for s in self.states]

    def get_experiment_history(self) -> List[Dict]:
        """获取实验历史"""
        return [e.to_dict() for e in self.experiments]

    async def clear_history(self):
        """清空历史"""
        self.states = []
        self.experiments = []
        for f in self.storage_dir.glob("*.json"):
            f.unlock
