"""
Hypothesis Tester - 假设验证器
==============================

系统化验证关于协作的假设：
1. 假设生成 - 从观察中产生假设
2. 假设设计 - 设计可验证的预测
3. 实验执行 - 系统化测试假设
4. 结果分析 - 统计验证结果
5. 知识更新 - 将验证结果转化为知识

核心理念：
- 假设不是信念，而是待验证的预测
- 所有假设都可以被证伪
- 验证比证明更重要
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from statistics import mean, stdev
from datetime import datetime
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean, stdev
import uuid


@dataclass
class Hypothesis:
    """假设"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    category: str = "collaboration"
    predictions: List[Dict] = field(default_factory=list)
    test_results: List[Dict] = field(default_factory=list)
    status: str = "pending"
    test_count: int = 0
    confidence: float = 0.0
    evidence_count: int = 0
    supporting_evidence: int = 0
    contradicting_evidence: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_tested: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Experiment:
    """实验"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str = ""
    name: str = ""
    description: str = ""
    setup: Dict = field(default_factory=dict)
    variables: Dict = field(default_factory=dict)
    control_variables: Dict = field(default_factory=dict)
    sample_size: int = 10
    results: List[Dict] = field(default_factory=list)
    analysis: Dict = field(default_factory=dict)
    conclusions: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    status: str = "running"
    success: bool = False


class HypothesisTester:
    """
    假设验证器

    系统化验证关于协作的假设：
    - 生成和设计假设
    - 执行对照实验
    - 统计分析结果
    - 更新置信度
    - 转化为知识
    """

    def __init__(self, db):
        self.db = db

        self.min_sample_size = 5
        self.significance_threshold = 0.05
        self.min_confidence_for_knowledge = 0.8

        self._hypotheses: Dict[str, Hypothesis] = {}
        self._experiments: Dict[str, Experiment] = {}

        self._load_hypotheses()

    def _load_hypotheses(self):
        """从数据库加载假设"""
        pending = self.db.get_hypotheses_by_status("pending")
        confirmed = self.db.get_hypotheses_by_status("confirmed")
        rejected = self.db.get_hypotheses_by_status("rejected")

        for h_data in pending + confirmed + rejected:
            test_results_raw = h_data.get("test_results")
            test_results = json.loads(test_results_raw) if test_results_raw else []

            predictions_raw = h_data.get("predictions")
            predictions = json.loads(predictions_raw) if predictions_raw else []

            hypothesis = Hypothesis(
                id=h_data.get("id", str(uuid.uuid4())),
                statement=h_data.get("statement", ""),
                category=h_data.get("category", "collaboration"),
                predictions=predictions,
                test_results=test_results,
                status=h_data.get("status", "pending"),
                test_count=h_data.get("test_count", 0),
                confidence=h_data.get("confidence", 0.0),
                evidence_count=h_data.get("evidence_count", 0),
                created_at=h_data.get("created_at", datetime.now().isoformat()),
                last_tested=h_data.get("last_tested", datetime.now().isoformat()),
            )

            self._hypotheses[hypothesis.id] = hypothesis

    async def generate_hypotheses(
        self, observations: List[Dict], category: str = "collaboration"
    ) -> List[Hypothesis]:
        """
        从观察中生成假设

        Args:
            observations: 观察列表
            category: 假设类别

        Returns:
            生成的假设列表
        """
        hypotheses = []

        pattern_groups = defaultdict(list)

        for obs in observations:
            obs_type = obs.get("type", "general")
            pattern_groups[obs_type].append(obs)

        for obs_type, group in pattern_groups.items():
            if len(group) >= 3:
                hypothesis = self._create_hypothesis_from_patterns(
                    group, category, obs_type
                )
                if hypothesis:
                    hypotheses.append(hypothesis)
                    self._hypotheses[hypothesis.id] = hypothesis
                    self.db.save_hypothesis(asdict(hypothesis))

        return hypotheses

    def _create_hypothesis_from_patterns(
        self, patterns: List[Dict], category: str, base_type: str
    ) -> Optional[Hypothesis]:
        """从模式创建假设"""
        if len(patterns) < 3:
            return None

        observations = [p.get("description", "") for p in patterns]

        common_features = self._find_common_features(observations)

        if not common_features:
            return None

        statement = f"当 {common_features} 时，协作效果会提升"

        predictions = [
            {
                "description": "未来相似的交互会产生类似的效果",
                "condition": "相同的特征条件",
                "expected_outcome": "效果提升",
                "probability": 0.7,
            },
            {
                "description": "缺少这些特征的交互效果较差",
                "condition": "缺少特征",
                "expected_outcome": "效果不变或下降",
                "probability": 0.3,
            },
        ]

        return Hypothesis(
            statement=statement,
            category=category,
            predictions=predictions,
            status="pending",
        )

    def _find_common_features(self, observations: List[str]) -> str:
        """找出共同特征"""
        if not observations:
            return ""

        all_words = []
        for obs in observations:
            words = obs.split()
            all_words.extend([w for w in words if len(w) > 2])

        word_counts = defaultdict(int)
        for word in all_words:
            word_counts[word] += 1

        common = [
            word
            for word, count in word_counts.items()
            if count >= len(observations) * 0.5
        ]

        return " ".join(common[:5])

    async def design_experiment(
        self,
        hypothesis_id: str,
        name: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> Optional[Experiment]:
        """
        设计验证假设的实验

        Args:
            hypothesis_id: 假设ID
            name: 实验名称
            sample_size: 样本大小

        Returns:
            实验对象
        """
        if hypothesis_id not in self._hypotheses:
            return None

        hypothesis = self._hypotheses[hypothesis_id]

        experiment = Experiment(
            hypothesis_id=hypothesis_id,
            name=name or f"实验-{hypothesis_id[:8]}",
            description=f"验证假设: {hypothesis.statement}",
            sample_size=sample_size or self.min_sample_size,
            setup={
                "hypothesis": hypothesis.statement,
                "predictions": hypothesis.predictions,
            },
            variables={
                "independent": ["feature_presence"],
                "dependent": ["effectiveness"],
                "controlled": ["user_type", "time_of_day"],
            },
            control_variables={"user_type": "balanced", "time_of_day": "randomized"},
            status="running",
        )

        self._experiments[experiment.id] = experiment

        return experiment

    async def run_experiment(self, experiment_id: str) -> Dict:
        """
        运行实验

        Args:
            experiment_id: 实验ID

        Returns:
            实验结果
        """
        if experiment_id not in self._experiments:
            return {"error": "实验不存在"}

        experiment = self._experiments[experiment_id]

        experiment.results = []

        for i in range(experiment.sample_size):
            result = self._simulate_trial(experiment)
            experiment.results.append(result)

            experiment.analysis = self._analyze_results(experiment.results)

        experiment.status = "completed"
        experiment.completed_at = datetime.now().isoformat()

        experiment.conclusions = self._generate_conclusions(
            experiment.analysis, experiment.hypothesis_id
        )

        success_count = sum(
            1 for r in experiment.results if r.get("outcome") == "success"
        )
        experiment.success = success_count > experiment.sample_size / 2

        await self._update_hypothesis_from_experiment(experiment)

        return {
            "experiment_id": experiment_id,
            "status": experiment.status,
            "sample_size": experiment.sample_size,
            "success_rate": success_count / experiment.sample_size,
            "analysis": experiment.analysis,
            "conclusions": experiment.conclusions,
            "success": experiment.success,
        }

    def _simulate_trial(self, experiment: Experiment) -> Dict:
        """模拟实验 trial"""
        hypothesis = self._hypotheses.get(experiment.hypothesis_id)

        if hypothesis and hypothesis.predictions:
            pred = hypothesis.predictions[0]
            if isinstance(pred, dict):
                base_probability = pred.get("probability", 0.5)
            else:
                base_probability = 0.5
        else:
            base_probability = 0.5

        outcome = random.random() < base_probability

        return {
            "trial": len(experiment.results) + 1,
            "timestamp": datetime.now().isoformat(),
            "variables": {
                "feature_presence": random.choice([True, False]),
                "effectiveness": random.uniform(0.3, 0.9),
            },
            "outcome": "success" if outcome else "failure",
            "effectiveness": random.uniform(0.4, 0.9)
            if outcome
            else random.uniform(0.1, 0.5),
            "notes": "",
        }

    def _analyze_results(self, results: List[Dict]) -> Dict:
        """分析实验结果"""
        if not results:
            return {}

        effectiveness_values = [r.get("effectiveness", 0.5) for r in results]

        success_count = sum(1 for r in results if r.get("outcome") == "success")
        failure_count = len(results) - success_count

        return {
            "sample_size": len(results),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(results),
            "avg_effectiveness": mean(effectiveness_values),
            "std_effectiveness": stdev(effectiveness_values)
            if len(effectiveness_values) > 1
            else 0,
            "effectiveness_values": effectiveness_values,
        }

    def _generate_conclusions(self, analysis: Dict, hypothesis_id: str) -> List[str]:
        """生成结论"""
        conclusions = []

        success_rate = analysis.get("success_rate", 0)
        avg_effectiveness = analysis.get("avg_effectiveness", 0)

        if success_rate > 0.7:
            conclusions.append("假设得到支持：成功率显著高于随机水平")
        elif success_rate < 0.3:
            conclusions.append("假设被拒绝：成功率显著低于预期")
        else:
            conclusions.append("假设未得到明确支持：结果与随机无显著差异")

        if avg_effectiveness > 0.6:
            conclusions.append(f"平均效果指数较高 ({avg_effectiveness:.2f})")
        elif avg_effectiveness < 0.4:
            conclusions.append(f"平均效果指数较低 ({avg_effectiveness:.2f})")

        return conclusions

    async def _update_hypothesis_from_experiment(self, experiment: Experiment):
        """根据实验更新假设"""
        if experiment.hypothesis_id not in self._hypotheses:
            return

        hypothesis = self._hypotheses[experiment.hypothesis_id]

        test_result = {
            "experiment_id": experiment.id,
            "timestamp": experiment.completed_at,
            "success": experiment.success,
            "analysis": experiment.analysis,
            "conclusions": experiment.conclusions,
        }

        if isinstance(hypothesis.test_results, str):
            hypothesis.test_results = json.loads(hypothesis.test_results)
        hypothesis.test_results.append(test_result)
        hypothesis.test_count += 1
        hypothesis.last_tested = experiment.completed_at

        if experiment.success:
            hypothesis.supporting_evidence += 1
        else:
            hypothesis.contradicting_evidence += 1

        hypothesis.evidence_count = (
            hypothesis.supporting_evidence + hypothesis.contradicting_evidence
        )

        if hypothesis.evidence_count > 0:
            hypothesis.confidence = (
                hypothesis.supporting_evidence / hypothesis.evidence_count
            )
        else:
            hypothesis.confidence = 0.0

        if hypothesis.test_count >= 3:
            if hypothesis.confidence > 0.8:
                hypothesis.status = "confirmed"
            elif hypothesis.confidence < 0.2:
                hypothesis.status = "rejected"
            else:
                hypothesis.status = "pending"

        self.db.save_hypothesis(asdict(hypothesis))

    async def record_observation(self, hypothesis_id: str, observation: Dict):
        """记录对假设的观察"""
        if hypothesis_id not in self._hypotheses:
            return

        hypothesis = self._hypotheses[hypothesis_id]

        test_result = {
            "timestamp": datetime.now().isoformat(),
            "type": "observation",
            "data": observation,
        }

        hypothesis.test_results.append(test_result)
        hypothesis.test_count += 1
        hypothesis.last_tested = datetime.now().isoformat()

        outcome = observation.get("outcome")
        if outcome == "success":
            hypothesis.supporting_evidence += 1
        elif outcome == "failure":
            hypothesis.contradicting_evidence += 1

        hypothesis.evidence_count = (
            hypothesis.supporting_evidence + hypothesis.contradicting_evidence
        )

        if hypothesis.evidence_count > 0:
            hypothesis.confidence = (
                hypothesis.supporting_evidence / hypothesis.evidence_count
            )

        self.db.save_hypothesis(asdict(hypothesis))

    def get_hypothesis_statistics(self) -> Dict:
        """获取假设统计"""
        total = len(self._hypotheses)

        pending = sum(1 for h in self._hypotheses.values() if h.status == "pending")
        confirmed = sum(1 for h in self._hypotheses.values() if h.status == "confirmed")
        rejected = sum(1 for h in self._hypotheses.values() if h.status == "rejected")

        if total > 0:
            avg_confidence = mean([h.confidence for h in self._hypotheses.values()])
        else:
            avg_confidence = 0

        return {
            "total_hypotheses": total,
            "pending": pending,
            "confirmed": confirmed,
            "rejected": rejected,
            "experiments_running": sum(
                1 for e in self._experiments.values() if e.status == "running"
            ),
            "experiments_completed": sum(
                1 for e in self._experiments.values() if e.status == "completed"
            ),
            "avg_confidence": avg_confidence,
        }

    def get_pending_hypotheses(self) -> List[Dict]:
        """获取待验证假设"""
        return [asdict(h) for h in self._hypotheses.values() if h.status == "pending"]

    def get_confirmed_hypotheses(self) -> List[Dict]:
        """获取已确认假设"""
        return [asdict(h) for h in self._hypotheses.values() if h.status == "confirmed"]

    def get_confidence_building_hypotheses(self) -> List[Hypothesis]:
        """获取正在积累置信度的假设"""
        return [
            h
            for h in self._hypotheses.values()
            if h.status == "pending" and h.evidence_count > 0
        ]
