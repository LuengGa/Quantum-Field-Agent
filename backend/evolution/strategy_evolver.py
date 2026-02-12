"""
Strategy Evolver - 策略进化器
==============================

根据效果自动调整协作策略：
1. 效果追踪 - 记录每个策略的效果
2. A/B测试 - 比较不同策略的效果
3. 策略变异 - 基于效果生成新策略
4. 自然选择 - 选择效果好的策略
5. 策略融合 - 组合有效策略

核心理念：
- 策略不是固定的，而是在实践中演化的
- 好的策略来自大量实验和选择
- 进化不是优化，而是适应
"""

import random
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict, dataclass, field
from collections import defaultdict
from statistics import mean
import uuid


@dataclass
class StrategyVariant:
    """策略变体"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = ""
    name: str = ""
    strategy_type: str = "adaptive"
    conditions: Dict = field(default_factory=dict)
    actions: List[Dict] = field(default_factory=list)
    parameters: Dict = field(default_factory=dict)
    total_uses: int = 0
    success_count: int = 0
    total_effectiveness: float = 0.0
    avg_effectiveness: float = 0.5
    confidence: float = 0.0
    generation: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class EvolutionExperiment:
    """进化实验"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    strategies: List[str] = field(default_factory=list)
    control_group: List[str] = field(default_factory=list)
    experiment_type: str = "ab_test"
    variables: Dict = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    status: str = "running"
    results: Dict = field(default_factory=dict)
    winner: str = ""


class StrategyEvolver:
    """
    策略进化器

    自动调整和优化协作策略：
    - 追踪策略效果
    - 生成策略变体
    - 执行A/B测试
    - 选择最优策略
    - 融合有效策略
    """

    def __init__(self, db):
        self.db = db

        self.min_uses_for_selection = 5
        self.min_effectiveness_threshold = 0.6
        self.max_generation = 10
        self.experiment_size = 10

        self._active_variants: Dict[str, StrategyVariant] = {}
        self._experiments: List[EvolutionExperiment] = []
        self._strategy_pool: Dict[str, List[str]] = defaultdict(list)

        self._load_strategies()

    def _load_strategies(self):
        """从数据库加载策略"""
        strategies = self.db.get_active_strategies()

        for strategy_data in strategies:
            variant = StrategyVariant(
                id=strategy_data.get("id", str(uuid.uuid4())),
                name=strategy_data.get("name", ""),
                strategy_type=strategy_data.get("strategy_type", "adaptive"),
                conditions=json.loads(strategy_data.get("conditions", "{}")),
                actions=json.loads(strategy_data.get("actions", "[]")),
                parameters=strategy_data.get("parameters", {}),
                total_uses=strategy_data.get("total_uses", 0),
                success_count=int(
                    strategy_data.get("total_uses", 0)
                    * (strategy_data.get("success_rate") or 0.5)
                ),
                avg_effectiveness=strategy_data.get("avg_effectiveness", 0.5),
                confidence=strategy_data.get("avg_effectiveness", 0.5),
                created_at=strategy_data.get("created_at", datetime.now().isoformat()),
                last_used=strategy_data.get("last_used", datetime.now().isoformat()),
                is_active=bool(strategy_data.get("is_active", True)),
                tags=[],
            )

            self._active_variants[variant.id] = variant
            self._strategy_pool[variant.strategy_type].append(variant.id)

    async def select_strategy(
        self, context: Dict, available_strategies: Optional[List[str]] = None
    ) -> Optional[StrategyVariant]:
        """
        根据上下文选择最佳策略

        Args:
            context: 当前上下文
            available_strategies: 可用策略列表，如果为None则自动选择

        Returns:
            选中的策略变体
        """
        if available_strategies is None:
            candidates = list(self._active_variants.values())
        else:
            candidates = [
                self._active_variants[sid]
                for sid in available_strategies
                if sid in self._active_variants
            ]

        if not candidates:
            return None

        scored_candidates = []

        for variant in candidates:
            if not variant.is_active:
                continue

            score = self._calculate_strategy_score(variant, context)
            scored_candidates.append((score, variant))

        if not scored_candidates:
            return None

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        best_score, best_variant = scored_candidates[0]

        if best_variant.total_uses < self.min_uses_for_selection:
            exploration_rate = 0.3
            if random.random() < exploration_rate:
                return random.choice(scored_candidates[:3])[1]

        return best_variant

    def _calculate_strategy_score(
        self, variant: StrategyVariant, context: Dict
    ) -> float:
        """计算策略得分"""
        base_score = variant.avg_effectiveness

        recency_weight = 0.3
        recency_score = self._calculate_recency_score(variant)

        context_match_score = self._calculate_context_match(variant, context)

        confidence_bonus = min(variant.confidence * 0.2, 0.2)

        total_score = (
            base_score * 0.4
            + recency_score * recency_weight
            + context_match_score * 0.3
            + confidence_bonus
        )

        return total_score

    def _calculate_recency_score(self, variant: StrategyVariant) -> float:
        """计算时间衰减分数"""
        if variant.total_uses == 0:
            return 0.5

        last_used = variant.last_used
        try:
            last_dt = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
            hours_ago = (datetime.now() - last_dt).total_seconds() / 3600

            decay = max(0, 1 - hours_ago / 168)
            return decay
        except:
            return 0.5

    def _calculate_context_match(
        self, variant: StrategyVariant, context: Dict
    ) -> float:
        """计算上下文匹配度"""
        conditions = variant.conditions

        if not conditions:
            return 0.5

        matches = 0
        total = 0

        for key, expected in conditions.items():
            total += 1
            actual = context.get(key)

            if isinstance(expected, list):
                if actual in expected:
                    matches += 1
            elif isinstance(expected, dict):
                if actual == expected.get("eq"):
                    matches += 1
                elif actual and actual > expected.get("gt", 0):
                    matches += 1
            elif actual == expected:
                matches += 1

        if total == 0:
            return 0.5

        return matches / total

    async def record_strategy_use(
        self,
        strategy_id: str,
        success: bool,
        effectiveness: float,
        context: Dict = None,
    ):
        """记录策略使用结果"""
        if strategy_id not in self._active_variants:
            return

        variant = self._active_variants[strategy_id]

        variant.total_uses += 1
        variant.last_used = datetime.now().isoformat()

        if success:
            variant.success_count += 1

        variant.avg_effectiveness = (
            variant.avg_effectiveness * (variant.total_uses - 1) + effectiveness
        ) / variant.total_uses

        variant.confidence = min(variant.total_uses / 20, 1.0)

        self.db.update_strategy_metrics(strategy_id, success, effectiveness)

        if context:
            self._update_strategy_evolution(variant, context)

    def _update_strategy_evolution(self, variant: StrategyVariant, context: Dict):
        """更新策略进化状态"""
        if variant.total_uses < self.min_uses_for_selection:
            return

        if variant.avg_effectiveness > self.min_effectiveness_threshold:
            if variant.generation < self.max_generation:
                self._evolve_strategy(variant, context)
        else:
            if variant.total_uses > 20:
                variant.is_active = False

    async def _evolve_strategy(self, parent: StrategyVariant, context: Dict):
        """进化策略：生成变体"""
        child = StrategyVariant(
            parent_id=parent.id,
            name=f"{parent.name}_gen{parent.generation + 1}",
            strategy_type=parent.strategy_type,
            conditions=self._mutate_conditions(parent.conditions, context),
            actions=self._mutate_actions(parent.actions),
            parameters=self._mutate_parameters(parent.parameters),
            generation=parent.generation + 1,
            tags=parent.tags.copy(),
        )

        self._active_variants[child.id] = child
        self._strategy_pool[child.strategy_type].append(child.id)

        parent.generation += 1

        self.db.save_strategy(asdict(child))

        self.db.log_evolution_event(
            event_type="strategy_evolution",
            description=f"策略 {parent.name} 进化为 {child.name}",
            changes={
                "parent_id": parent.id,
                "child_id": child.id,
                "parent_generation": parent.generation,
                "child_generation": child.generation,
            },
            impact=0.3,
        )

    def _mutate_conditions(self, conditions: Dict, context: Dict) -> Dict:
        """变异条件"""
        if not conditions:
            return {}

        mutated = conditions.copy()

        mutation_rate = 0.2

        if random.random() < mutation_rate:
            for key in list(mutated.keys()):
                if random.random() < 0.3:
                    new_value = context.get(key, mutated[key])
                    if isinstance(new_value, list):
                        mutated[key] = random.choice(new_value)
                    else:
                        mutated[key] = new_value

        return mutated

    def _mutate_actions(self, actions: List[Dict]) -> List[Dict]:
        """变异动作序列"""
        if not actions:
            return []

        mutated = [action.copy() for action in actions]

        mutation_rate = 0.2

        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutated) - 1)
            if "parameters" in mutated[idx]:
                params = mutated[idx]["parameters"]
                for key in list(params.keys()):
                    if random.random() < 0.3:
                        params[key] = params[key] * random.uniform(0.8, 1.2)
                mutated[idx]["parameters"] = params

        return mutated

    def _mutate_parameters(self, parameters: Dict) -> Dict:
        """变异参数"""
        if not parameters:
            return {}

        mutated = {}

        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                mutated[key] = value * random.uniform(0.8, 1.2)
            elif isinstance(value, str):
                if random.random() < 0.1:
                    mutated[key] = value + "_mutated"
                else:
                    mutated[key] = value
            else:
                mutated[key] = value

        return mutated

    async def run_ab_experiment(
        self, name: str, strategy_a: str, strategy_b: str, variables: Dict = None
    ) -> EvolutionExperiment:
        """运行A/B测试实验"""
        experiment = EvolutionExperiment(
            name=name,
            description=f"A/B测试: {strategy_a} vs {strategy_b}",
            strategies=[strategy_a, strategy_b],
            control_group=[],
            experiment_type="ab_test",
            variables=variables or {},
        )

        self._experiments.append(experiment)

        return experiment

    async def analyze_experiment(self, experiment_id: str) -> Dict:
        """分析实验结果"""
        experiment = None

        for exp in self._experiments:
            if exp.id == experiment_id:
                experiment = exp
                break

        if not experiment:
            return {"error": "实验不存在"}

        results = {"experiment_id": experiment_id}

        for strategy_id in experiment.strategies:
            if strategy_id in self._active_variants:
                variant = self._active_variants[strategy_id]
                results[strategy_id] = {
                    "name": variant.name,
                    "total_uses": variant.total_uses,
                    "success_rate": variant.success_count / max(variant.total_uses, 1),
                    "avg_effectiveness": variant.avg_effectiveness,
                }

        if experiment.variables:
            results["variables"] = experiment.variables

        if experiment.status == "completed":
            a_effectiveness = results.get(experiment.strategies[0], {}).get(
                "avg_effectiveness", 0
            )
            b_effectiveness = results.get(experiment.strategies[1], {}).get(
                "avg_effectiveness", 0
            )

            if b_effectiveness > a_effectiveness:
                experiment.winner = experiment.strategies[1]
                results["winner"] = experiment.strategies[1]
            else:
                experiment.winner = experiment.strategies[0]
                results["winner"] = experiment.strategies[0]

        return results

    async def fuse_strategies(
        self, strategy_ids: List[str], name: Optional[str] = None
    ) -> StrategyVariant:
        """融合多个策略"""
        parents = [
            self._active_variants[sid]
            for sid in strategy_ids
            if sid in self._active_variants
        ]

        if len(parents) < 2:
            return None

        fused_conditions = {}
        fused_actions = []
        fused_parameters = {}

        for parent in parents:
            for key, value in parent.conditions.items():
                if key not in fused_conditions:
                    fused_conditions[key] = value

            fused_actions.extend(parent.actions)
            fused_parameters.update(parent.parameters)

        if len(parents) > 0:
            avg_effectiveness = mean([p.avg_effectiveness for p in parents])
        else:
            avg_effectiveness = 0.5

        fused = StrategyVariant(
            name=name or f"Fused_{'_'.join([p.name for p in parents[:2]])}",
            strategy_type="fused",
            conditions=fused_conditions,
            actions=fused_actions[:10],
            parameters=fused_parameters,
            avg_effectiveness=avg_effectiveness,
            confidence=mean([p.confidence for p in parents]),
            tags=["fused"],
        )

        self._active_variants[fused.id] = fused

        self.db.save_strategy(asdict(fused))

        self.db.log_evolution_event(
            event_type="strategy_fusion",
            description=f"融合策略: {', '.join([p.name for p in parents])}",
            changes={"parent_ids": [p.id for p in parents], "child_id": fused.id},
            impact=0.5,
        )

        return fused

    def get_strategy_statistics(self) -> Dict:
        """获取策略统计"""
        active = [v for v in self._active_variants.values() if v.is_active]
        inactive = [v for v in self._active_variants.values() if not v.is_active]

        if active:
            avg_effectiveness = mean([v.avg_effectiveness for v in active])
            total_uses = sum([v.total_uses for v in active])
        else:
            avg_effectiveness = 0
            total_uses = 0

        return {
            "total_strategies": len(self._active_variants),
            "active_strategies": len(active),
            "inactive_strategies": len(inactive),
            "experiments": len(self._experiments),
            "avg_effectiveness": avg_effectiveness,
            "total_uses": total_uses,
            "by_type": {
                stype: len(sids) for stype, sids in self._strategy_pool.items()
            },
        }

    def export_strategies(self) -> List[Dict]:
        """导出所有策略"""
        return [asdict(v) for v in self._active_variants.values()]

    def import_strategies(self, strategies: List[Dict]):
        """导入策略"""
        for strategy_data in strategies:
            variant = StrategyVariant(
                id=strategy_data.get("id", str(uuid.uuid4())),
                name=strategy_data.get("name", ""),
                strategy_type=strategy_data.get("strategy_type", "adaptive"),
                conditions=strategy_data.get("conditions", {}),
                actions=strategy_data.get("actions", []),
                parameters=strategy_data.get("parameters", {}),
                total_uses=strategy_data.get("total_uses", 0),
                avg_effectiveness=strategy_data.get("avg_effectiveness", 0.5),
                confidence=strategy_data.get("confidence", 0.5),
                is_active=strategy_data.get("is_active", True),
            )

            self._active_variants[variant.id] = variant
