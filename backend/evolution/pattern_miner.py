"""
Pattern Miner - 模式挖掘器
==========================

从交互历史中发现隐藏模式：
1. 时间模式 - 何时发生什么
2. 因果模式 - 什么导致什么
3. 序列模式 - 什么跟随什么
4. 聚类模式 - 什么总是同时出现
5. 异常模式 - 什么不符合预期

核心理念：
- 模式不是预设的，而是从数据中涌现的
- 模式需要验证，不是所有相关性都是因果
- 模式会演化，需要持续更新
"""

import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import mean, stdev
from dataclasses import dataclass, field, asdict
import uuid


@dataclass
class TimePattern:
    """时间模式"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    time_range: Tuple[str, str] = ("00:00", "23:59")
    day_of_week: List[int] = field(default_factory=list)
    pattern_type: str = "recurrence"
    occurrences: int = 0
    success_rate: float = 0.5
    confidence: float = 0.0
    description: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class CausalityPattern:
    """因果模式"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cause: str = ""
    effect: str = ""
    conditions: Dict = field(default_factory=dict)
    pattern_type: str = "causation"
    correlation: float = 0.0
    lag_hours: float = 0.0
    occurrences: int = 0
    success_rate: float = 0.5
    confidence: float = 0.0
    description: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class SequencePattern:
    """序列模式"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    events: List[str] = field(default_factory=list)
    following_events: List[str] = field(default_factory=list)
    min_support: float = 0.1
    pattern_type: str = "sequence"
    occurrences: int = 0
    success_rate: float = 0.5
    confidence: float = 0.0
    description: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class ClusteringPattern:
    """聚类模式"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    cluster_id: str = ""
    features: List[str] = field(default_factory=list)
    members: List[str] = field(default_factory=list)
    centroid: Dict = field(default_factory=dict)
    pattern_type: str = "clustering"
    occurrences: int = 0
    success_rate: float = 0.5
    confidence: float = 0.0
    description: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class AnomalyPattern:
    """异常模式"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    normal_pattern: Dict = field(default_factory=dict)
    anomaly_signature: Dict = field(default_factory=dict)
    severity: str = "low"
    pattern_type: str = "anomaly"
    occurrences: int = 0
    detection_count: int = 0
    false_positive_rate: float = 0.0
    confidence: float = 0.0
    description: str = ""
    metadata: Dict = field(default_factory=dict)


class PatternMiner:
    """
    模式挖掘器

    从交互历史中自动发现和验证模式：
    - 时间模式：什么时间发生什么
    - 因果模式：什么导致什么
    - 序列模式：事件序列
    - 聚类模式：相似事件聚类
    - 异常模式：偏离正常的事件
    """

    def __init__(self, db):
        self.db = db
        self.min_occurrences = 3
        self.min_confidence = 0.6
        self.time_window_days = 30

        self._time_patterns: List[TimePattern] = []
        self._causality_patterns: List[CausalityPattern] = []
        self._sequence_patterns: List[SequencePattern] = []
        self._clustering_patterns: List[ClusteringPattern] = []
        self._anomaly_patterns: List[AnomalyPattern] = []

    async def mine_patterns(self, interactions: List[Dict] = None) -> Dict[str, List]:
        """
        挖掘所有类型的模式

        Args:
            interactions: 交互历史，如果为None则从数据库获取

        Returns:
            发现的模式字典
        """
        if interactions is None:
            interactions = self.db.get_recent_interactions(days=self.time_window_days)

        if len(interactions) < self.min_occurrences:
            return {"message": "交互数据不足", "count": len(interactions)}

        result = {
            "time_patterns": [],
            "causality_patterns": [],
            "sequence_patterns": [],
            "clustering_patterns": [],
            "anomaly_patterns": [],
        }

        time_patterns = await self._mine_time_patterns(interactions)
        causality_patterns = await self._mine_causality_patterns(interactions)
        sequence_patterns = await self._mine_sequence_patterns(interactions)
        clustering_patterns = await self._mine_clustering_patterns(interactions)
        anomaly_patterns = await self._mine_anomaly_patterns(interactions)

        for pattern in time_patterns:
            pattern_dict = self._save_time_pattern(pattern)
            result["time_patterns"].append(pattern_dict)

        for pattern in causality_patterns:
            pattern_dict = self._save_causality_pattern(pattern)
            result["causality_patterns"].append(pattern_dict)

        for pattern in sequence_patterns:
            pattern_dict = self._save_sequence_pattern(pattern)
            result["sequence_patterns"].append(pattern_dict)

        for pattern in clustering_patterns:
            pattern_dict = self._save_clustering_pattern(pattern)
            result["clustering_patterns"].append(pattern_dict)

        for pattern in anomaly_patterns:
            pattern_dict = self._save_anomaly_pattern(pattern)
            result["anomaly_patterns"].append(pattern_dict)

        return result

    async def _mine_time_patterns(self, interactions: List[Dict]) -> List[TimePattern]:
        """挖掘时间模式"""
        patterns = []

        hour_counts = Counter()
        day_counts = Counter()
        success_by_hour = defaultdict(list)
        success_by_day = defaultdict(list)

        for interaction in interactions:
            timestamp = interaction.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                hour = dt.hour
                day = dt.weekday()

                hour_counts[hour] += 1
                day_counts[day] += 1

                effectiveness = interaction.get("effectiveness")
                if effectiveness is not None:
                    success_by_hour[hour].append(effectiveness)
                    success_by_day[day].append(effectiveness)
            except:
                pass

        for hour, count in hour_counts.items():
            if count >= self.min_occurrences:
                success_rates = success_by_hour.get(hour, [])
                avg_success = mean(success_rates) if success_rates else 0.5

                pattern = TimePattern(
                    name=f"Hour_{hour}_pattern",
                    time_range=(f"{hour:02d}:00", f"{hour:02d}:59"),
                    pattern_type="hourly_recurrence",
                    occurrences=count,
                    success_rate=avg_success,
                    confidence=min(count / 20, 1.0),
                    description=f"在{hour}点时段活跃的模式",
                )
                patterns.append(pattern)

        return patterns

    async def _mine_causality_patterns(
        self, interactions: List[Dict]
    ) -> List[CausalityPattern]:
        """挖掘因果模式"""
        patterns = []

        action_outcome_map = defaultdict(lambda: {"success": 0, "total": 0})

        for interaction in interactions:
            input_summary = interaction.get("input_summary", "")
            outcome = interaction.get("outcome", "")
            effectiveness = interaction.get("effectiveness")

            actions = self._extract_actions(input_summary)

            for action in actions:
                action_outcome_map[(action, outcome)]["total"] += 1
                if effectiveness and effectiveness > 0.7:
                    action_outcome_map[(action, outcome)]["success"] += 1

        for (action, outcome), counts in action_outcome_map.items():
            if counts["total"] >= self.min_occurrences:
                correlation = counts["success"] / counts["total"]

                if correlation > 0.6 or correlation < 0.4:
                    pattern = CausalityPattern(
                        cause=action,
                        effect=outcome,
                        pattern_type="action_outcome",
                        correlation=correlation,
                        occurrences=counts["total"],
                        success_rate=correlation,
                        confidence=min(counts["total"] / 15, 1.0),
                        description=f"'{action}' 导致 '{outcome}' 的模式",
                    )
                    patterns.append(pattern)

        return patterns

    async def _mine_sequence_patterns(
        self, interactions: List[Dict]
    ) -> List[SequencePattern]:
        """挖掘序列模式"""
        patterns = []

        session_sequences = defaultdict(list)

        for interaction in interactions:
            session_id = interaction.get("session_id", "default")
            interaction_type = interaction.get("interaction_type", "unknown")
            outcome = interaction.get("outcome", "")

            session_sequences[session_id].append(
                {"type": interaction_type, "outcome": outcome}
            )

        transition_counts = defaultdict(lambda: {"count": 0, "successes": 0})

        for session_id, sequence in session_sequences.items():
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_event = sequence[i + 1]

                key = (current["type"], current["outcome"], next_event["type"])
                transition_counts[key]["count"] += 1
                if next_event.get("effectiveness", 0) > 0.7:
                    transition_counts[key]["successes"] += 1

        for (prev_type, prev_outcome, next_type), counts in transition_counts.items():
            if counts["count"] >= self.min_occurrences:
                pattern = SequencePattern(
                    name=f"Sequence_{prev_type}_to_{next_type}",
                    events=[prev_type],
                    following_events=[next_type],
                    pattern_type="transition",
                    occurrences=counts["count"],
                    success_rate=counts["successes"] / counts["count"],
                    confidence=min(counts["count"] / 15, 1.0),
                    description=f"'{prev_type}' 类型交互后倾向于 '{next_type}' 类型",
                )
                patterns.append(pattern)

        return patterns

    async def _mine_clustering_patterns(
        self, interactions: List[Dict]
    ) -> List[ClusteringPattern]:
        """挖掘聚类模式"""
        patterns = []

        feature_groups = defaultdict(list)

        for interaction in interactions:
            input_summary = interaction.get("input_summary", "")
            interaction_type = interaction.get("interaction_type", "unknown")

            keywords = self._extract_keywords(input_summary)
            features = [interaction_type] + keywords[:3]
            feature_key = tuple(sorted(features))

            feature_groups[feature_key].append(interaction)

        for features, group in feature_groups.items():
            if len(group) >= self.min_occurrences:
                success_count = sum(1 for i in group if i.get("effectiveness", 0) > 0.7)

                pattern = ClusteringPattern(
                    name=f"Cluster_{'_'.join(features[:2])}",
                    cluster_id=hashlib.md5(str(features).encode()).hexdigest()[:8],
                    features=list(features),
                    members=[i["id"] for i in group],
                    pattern_type="feature_cluster",
                    occurrences=len(group),
                    success_rate=success_count / len(group),
                    confidence=min(len(group) / 20, 1.0),
                    description=f"具有 {features} 特征的事件聚类",
                )
                patterns.append(pattern)

        return patterns

    async def _mine_anomaly_patterns(
        self, interactions: List[Dict]
    ) -> List[AnomalyPattern]:
        """挖掘异常模式"""
        patterns = []

        normal_effectiveness = []
        normal_outcomes = []

        for interaction in interactions:
            effectiveness = interaction.get("effectiveness")
            outcome = interaction.get("outcome", "")

            if effectiveness is not None:
                normal_effectiveness.append(effectiveness)
            normal_outcomes.append(outcome)

        if not normal_effectiveness:
            return patterns

        avg_effectiveness = mean(normal_effectiveness)
        std_effectiveness = (
            stdev(normal_effectiveness) if len(normal_effectiveness) > 1 else 0
        )

        anomaly_count = 0
        anomaly_signatures = []

        for interaction in interactions:
            effectiveness = interaction.get("effectiveness")

            if effectiveness is not None:
                if abs(effectiveness - avg_effectiveness) > 2 * std_effectiveness:
                    if effectiveness < avg_effectiveness - std_effectiveness:
                        anomaly_count += 1
                        anomaly_signatures.append(
                            {
                                "type": "low_effectiveness",
                                "value": effectiveness,
                                "expected": avg_effectiveness,
                            }
                        )

        if anomaly_count > 0:
            pattern = AnomalyPattern(
                name="Effectiveness_anomaly",
                normal_pattern={
                    "avg_effectiveness": avg_effectiveness,
                    "std_effectiveness": std_effectiveness,
                },
                anomaly_signature={
                    "count": anomaly_count,
                    "examples": anomaly_signatures[:5],
                },
                severity="medium" if anomaly_count < 10 else "high",
                pattern_type="effectiveness_deviation",
                detection_count=anomaly_count,
                confidence=min(anomaly_count / 20, 1.0),
                description=f"效果指标偏离正常范围 {anomaly_count} 次",
            )
            patterns.append(pattern)

        return patterns

    def _extract_actions(self, text: str) -> List[str]:
        """从文本中提取动作"""
        actions = []

        action_patterns = [
            r"询问|ask|问",
            r"请求|request|要",
            r"学习|learn|学",
            r"分析|analyze|分析",
            r"创建|create|建",
            r"搜索|search|搜",
            r"解释|explain|解释",
            r"比较|compare|比较",
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend(matches)

        return list(set(actions)) if actions else ["general"]

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """提取关键词"""
        if not text:
            return []

        words = re.findall(r"\b\w+\b", text.lower())

        stopwords = {
            "的",
            "是",
            "在",
            "了",
            "和",
            "与",
            "或",
            "我",
            "你",
            "他",
            "她",
            "它",
            "这",
            "那",
            "什么",
            "如何",
            "怎么",
            "可以",
        }

        keywords = [w for w in words if len(w) > 2 and w not in stopwords]

        return keywords[:max_keywords]

    def _save_time_pattern(self, pattern: TimePattern) -> Dict:
        """保存时间模式"""
        pattern_dict = asdict(pattern)
        self._time_patterns.append(pattern)
        self.db.save_pattern(pattern_dict)
        return pattern_dict

    def _save_causality_pattern(self, pattern: CausalityPattern) -> Dict:
        """保存因果模式"""
        pattern_dict = asdict(pattern)
        self._causality_patterns.append(pattern)
        self.db.save_pattern(pattern_dict)
        return pattern_dict

    def _save_sequence_pattern(self, pattern: SequencePattern) -> Dict:
        """保存序列模式"""
        pattern_dict = asdict(pattern)
        self._sequence_patterns.append(pattern)
        self.db.save_pattern(pattern_dict)
        return pattern_dict

    def _save_clustering_pattern(self, pattern: ClusteringPattern) -> Dict:
        """保存聚类模式"""
        pattern_dict = asdict(pattern)
        self._clustering_patterns.append(pattern)
        self.db.save_pattern(pattern_dict)
        return pattern_dict

    def _save_anomaly_pattern(self, pattern: AnomalyPattern) -> Dict:
        """保存异常模式"""
        pattern_dict = asdict(pattern)
        self._anomaly_patterns.append(pattern)
        self.db.save_pattern(pattern_dict)
        return pattern_dict

    async def match_patterns(self, context: Dict) -> List[Dict]:
        """
        根据当前上下文匹配适用的模式

        Args:
            context: 当前上下文，包含时间、输入类型、用户信息等

        Returns:
            匹配到的模式列表
        """
        matched_patterns = []

        for pattern in self._time_patterns:
            if self._match_time_pattern(pattern, context):
                matched_patterns.append(asdict(pattern))

        for pattern in self._causality_patterns:
            if self._match_causality_pattern(pattern, context):
                matched_patterns.append(asdict(pattern))

        for pattern in self._sequence_patterns:
            if self._match_sequence_pattern(pattern, context):
                matched_patterns.append(asdict(pattern))

        for pattern in self._anomaly_patterns:
            if self._match_anomaly_pattern(pattern, context):
                matched_patterns.append(asdict(pattern))

        return matched_patterns

    def _match_time_pattern(self, pattern: TimePattern, context: Dict) -> bool:
        """匹配时间模式"""
        current_time = context.get("time")
        if not current_time:
            return False

        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(
                    current_time.replace("Z", "+00:00")
                )
            except:
                return False

        time_range = pattern.time_range
        start_hour = int(time_range[0].split(":")[0])
        end_hour = int(time_range[1].split(":")[0])

        return start_hour <= current_time.hour <= end_hour

    def _match_causality_pattern(
        self, pattern: CausalityPattern, context: Dict
    ) -> bool:
        """匹配因果模式"""
        input_text = context.get("input", "")
        return pattern.cause.lower() in input_text.lower()

    def _match_sequence_pattern(self, pattern: SequencePattern, context: Dict) -> bool:
        """匹配序列模式"""
        recent_types = context.get("recent_types", [])
        return len(recent_types) > 0 and recent_types[-1] in pattern.events

    def _match_anomaly_pattern(self, pattern: AnomalyPattern, context: Dict) -> bool:
        """匹配异常模式"""
        effectiveness = context.get("effectiveness")
        if effectiveness is None:
            return False

        normal = pattern.normal_pattern
        if "avg_effectiveness" not in normal:
            return False

        threshold = normal.get("std_effectiveness", 0) * 2
        return abs(effectiveness - normal["avg_effectiveness"]) > threshold

    async def update_pattern_confidence(self, pattern_id: str, success: bool):
        """更新模式置信度"""
        all_patterns = (
            self._time_patterns
            + self._causality_patterns
            + self._sequence_patterns
            + self._clustering_patterns
            + self._anomaly_patterns
        )

        for pattern in all_patterns:
            if pattern.id == pattern_id:
                pattern.occurrences += 1
                if success:
                    pattern.success_rate = (
                        pattern.success_rate * (pattern.occurrences - 1) + 1
                    ) / pattern.occurrences
                else:
                    pattern.success_rate = (
                        pattern.success_rate
                        * (pattern.occurrences - 1)
                        / pattern.occurrences
                    )

                pattern.confidence = min(pattern.occurrences / 20, 1.0)
                pattern.last_observed = datetime.now().isoformat()

                self.db.save_pattern(asdict(pattern))
                break

    def get_pattern_statistics(self) -> Dict:
        """获取模式统计信息"""
        return {
            "time_patterns": {
                "count": len(self._time_patterns),
                "avg_confidence": mean([p.confidence for p in self._time_patterns])
                if self._time_patterns
                else 0,
            },
            "causality_patterns": {
                "count": len(self._causality_patterns),
                "avg_confidence": mean([p.confidence for p in self._causality_patterns])
                if self._causality_patterns
                else 0,
            },
            "sequence_patterns": {
                "count": len(self._sequence_patterns),
                "avg_confidence": mean([p.confidence for p in self._sequence_patterns])
                if self._sequence_patterns
                else 0,
            },
            "clustering_patterns": {
                "count": len(self._clustering_patterns),
                "avg_confidence": mean(
                    [p.confidence for p in self._clustering_patterns]
                )
                if self._clustering_patterns
                else 0,
            },
            "anomaly_patterns": {
                "count": len(self._anomaly_patterns),
                "avg_confidence": mean([p.confidence for p in self._anomaly_patterns])
                if self._anomaly_patterns
                else 0,
            },
        }
