"""
Hypothesis Generator - 自动假设生成器
===================================

从发现的模式中自动生成可验证的假设：
1. 模式分析 - 从模式中提取潜在规律
2. 假设构建 - 将规律转化为假设
3. 预测设计 - 为每个假设设计可验证的预测
4. 验证建议 - 提供验证方法建议

核心理念：
- 假设是连接观察和理论的桥梁
- 好的假设是可证伪的
- 假设驱动学习和发现
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import uuid


@dataclass
class AutoHypothesis:
    """自动生成的假设"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    source_patterns: List[str] = field(default_factory=list)
    category: str = "collaboration"
    predictions: List[Dict] = field(default_factory=list)
    verification_method: str = ""
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class HypothesisGenerator:
    """
    自动假设生成器

    从模式中自动生成假设：
    - 时间模式假设
    - 因果模式假设
    - 序列模式假设
    - 聚类模式假设
    - 异常模式假设
    """

    def __init__(self, db=None):
        self.db = db

    def generate_from_patterns(self, patterns: List[Dict]) -> List[AutoHypothesis]:
        """
        从模式列表生成假设

        Args:
            patterns: 发现的模式列表

        Returns:
            生成的假设列表
        """
        hypotheses = []

        patterns_by_type = defaultdict(list)
        for pattern in patterns:
            p_type = pattern.get("pattern_type", "unknown")
            patterns_by_type[p_type].append(pattern)

        for p_type, type_patterns in patterns_by_type.items():
            if p_type in ["time", "time_recurrence", "hourly_recurrence"]:
                hypotheses.extend(self._generate_time_hypotheses(type_patterns))
            elif p_type in ["causation", "action_outcome"]:
                hypotheses.extend(self._generate_causation_hypotheses(type_patterns))
            elif p_type in ["sequence", "transition"]:
                hypotheses.extend(self._generate_sequence_hypotheses(type_patterns))
            elif p_type in ["clustering", "feature_cluster"]:
                hypotheses.extend(self._generate_clustering_hypotheses(type_patterns))
            elif p_type in ["anomaly", "effectiveness_deviation"]:
                hypotheses.extend(self._generate_anomaly_hypotheses(type_patterns))

        return hypotheses

    def _generate_time_hypotheses(self, patterns: List[Dict]) -> List[AutoHypothesis]:
        """从时间模式生成假设"""
        hypotheses = []

        for pattern in patterns:
            time_range = pattern.get("time_range", {})
            occurrences = pattern.get("occurrences", 0)
            success_rate = pattern.get("success_rate", 0.5)

            if occurrences < 3:
                continue

            statement = f"在特定时间段进行交互可以提高{success_rate * 100:.0f}%的成功率"

            prediction = {
                "description": "在未来相同时间段的交互将保持相似的成功率",
                "condition": "时间段匹配",
                "expected_outcome": f"成功率接近{success_rate:.1%}",
                "probability": success_rate,
            }

            hypothesis = AutoHypothesis(
                statement=statement,
                source_patterns=[pattern.get("id", "")],
                category="timing",
                predictions=[prediction],
                verification_method="在未来一周内收集相同时间段的交互数据",
                confidence=min(occurrences / 10, 0.9),
            )

            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_causation_hypotheses(
        self, patterns: List[Dict]
    ) -> List[AutoHypothesis]:
        """从因果模式生成假设"""
        hypotheses = []

        for pattern in patterns:
            cause = pattern.get("cause", "")
            effect = pattern.get("effect", "")
            correlation = pattern.get("correlation", pattern.get("success_rate", 0.5))
            occurrences = pattern.get("occurrences", 0)

            if occurrences < 2:
                continue

            statement = f"'{cause}' 的行为会导致 '{effect}' 的结果"

            prediction1 = {
                "description": f"当{cause}存在时，{effect}更可能发生",
                "condition": f"{cause}存在",
                "expected_outcome": effect,
                "probability": correlation,
            }

            prediction2 = {
                "description": f"当{cause}不存在时，{effect}不太可能发生",
                "condition": f"{cause}不存在",
                "expected_outcome": "其他结果",
                "probability": 1 - correlation,
            }

            hypothesis = AutoHypothesis(
                statement=statement,
                source_patterns=[pattern.get("id", "")],
                category="causation",
                predictions=[prediction1, prediction2],
                verification_method="设计对照实验，比较有/无该原因的情况",
                confidence=min(occurrences / 15, 0.85),
            )

            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_sequence_hypotheses(
        self, patterns: List[Dict]
    ) -> List[AutoHypothesis]:
        """从序列模式生成假设"""
        hypotheses = []

        for pattern in patterns:
            events = pattern.get("events", [])
            following = pattern.get("following_events", [])
            support = pattern.get("min_support", 0.1)

            if len(events) < 1 or len(following) < 1:
                continue

            statement = f"'{events[0]}' 类型的交互后倾向于发生 '{following[0]}'"

            prediction = {
                "description": f"在{events[0]}后，更高概率出现{following[0]}",
                "condition": f"发生了{events[0]}",
                "expected_outcome": following[0],
                "probability": min(support * 2, 0.9),
            }

            hypothesis = AutoHypothesis(
                statement=statement,
                source_patterns=[pattern.get("id", "")],
                category="sequence",
                predictions=[prediction],
                verification_method="收集交互序列数据，统计条件概率",
                confidence=min(support * 5, 0.8),
            )

            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_clustering_hypotheses(
        self, patterns: List[Dict]
    ) -> List[AutoHypothesis]:
        """从聚类模式生成假设"""
        hypotheses = []

        for pattern in patterns:
            features = pattern.get("features", [])
            members = pattern.get("members", [])

            if len(features) < 2 or len(members) < 3:
                continue

            statement = f"具有 {', '.join(features[:3])} 特征的事件倾向于同时发生"

            prediction = {
                "description": "新事件如果具备这些特征，可能属于同一聚类",
                "condition": "具备全部或大部分特征",
                "expected_outcome": "被归入同一类别",
                "probability": 0.7,
            }

            hypothesis = AutoHypothesis(
                statement=statement,
                source_patterns=[pattern.get("id", "")],
                category="clustering",
                predictions=[prediction],
                verification_method="使用聚类算法对新数据进行分类验证",
                confidence=min(len(members) / 20, 0.85),
            )

            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_anomaly_hypotheses(
        self, patterns: List[Dict]
    ) -> List[AutoHypothesis]:
        """从异常模式生成假设"""
        hypotheses = []

        for pattern in patterns:
            normal = pattern.get("normal_pattern", {})
            anomaly = pattern.get("anomaly_signature", {})
            severity = pattern.get("severity", "low")

            statement = f"存在偏离正常范围的异常情况（严重程度: {severity}）"

            prediction = {
                "description": "异常情况会降低交互效果",
                "condition": "检测到异常模式",
                "expected_outcome": "效果指数下降",
                "probability": 0.8,
            }

            hypothesis = AutoHypothesis(
                statement=statement,
                source_patterns=[pattern.get("id", "")],
                category="anomaly",
                predictions=[prediction],
                verification_method="持续监控效果指标，及时检测异常",
                confidence=0.6,
            )

            hypotheses.append(hypothesis)

        return hypotheses

    def save_hypotheses(self, hypotheses: List[AutoHypothesis]):
        """保存假设到数据库"""
        if not self.db:
            return

        for hypothesis in hypotheses:
            data = {
                "statement": hypothesis.statement,
                "category": hypothesis.category,
                "predictions": json.dumps(hypothesis.predictions),
                "test_results": "[]",
                "status": "pending",
                "test_count": 0,
                "confidence": hypothesis.confidence,
                "evidence_count": len(hypothesis.source_patterns),
                "created_at": hypothesis.created_at,
                "last_tested": hypothesis.created_at,
            }

            self.db.save_hypothesis(data)

    def get_statistics(self, hypotheses: List[AutoHypothesis]) -> Dict:
        """获取假设统计"""
        return {
            "total": len(hypotheses),
            "by_category": defaultdict(int),
            "avg_confidence": sum(h.confidence for h in hypotheses) / len(hypotheses)
            if hypotheses
            else 0,
        }


def integrate_with_evolution_engine(engine) -> int:
    """
    将假设生成器集成到进化引擎

    在每次进化周期结束后自动生成假设
    """
    generator = HypothesisGenerator(engine.db)

    patterns = engine._get_all_patterns()

    hypotheses = generator.generate_from_patterns(patterns)

    generator.save_hypotheses(hypotheses)

    stats = generator.get_statistics(hypotheses)

    print(f"\n自动生成了 {stats['total']} 个假设")
    print(f"平均置信度: {stats['avg_confidence']:.2f}")

    return len(hypotheses)
