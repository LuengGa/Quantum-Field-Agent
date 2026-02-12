"""
Evolution Engine - 进化引擎
==========================

整合所有进化组件的中央引擎：
1. 协调模式挖掘、策略进化、假设验证、知识综合、能力构建
2. 管理进化循环和调度
3. 提供统一的进化接口
4. 追踪进化历史和影响

核心理念：
- 进化不是单一的组件，而是整个系统的涌现属性
- 各组件相互协作，共同进化
- 进化是持续的过程，不是终点
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

from .database import EvolutionDatabase
from .pattern_miner import PatternMiner
from .strategy_evolver import StrategyEvolver
from .hypothesis_tester import HypothesisTester
from .knowledge_synthesizer import KnowledgeSynthesizer
from .capability_builder import CapabilityBuilder
from .hypothesis_generator import HypothesisGenerator, integrate_with_evolution_engine


@dataclass
class EvolutionConfig:
    """进化配置"""

    auto_mine_patterns: bool = True
    auto_evolve_strategies: bool = True
    auto_test_hypotheses: bool = True
    auto_synthesize_knowledge: bool = True

    pattern_mining_interval_hours: int = 24
    strategy_evolution_interval_hours: int = 48
    hypothesis_testing_interval_hours: int = 72
    knowledge_synthesis_interval_hours: int = 168

    min_interactions_for_mining: int = 10
    min_uses_for_evolution: int = 5
    min_evidence_for_knowledge: int = 3

    enabled: bool = True
    log_level: str = "INFO"


@dataclass
class EvolutionCycle:
    """进化周期"""

    id: str = field(
        default_factory=lambda: str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    status: str = "running"

    pattern_mining_result: Dict = field(default_factory=dict)
    strategy_evolution_result: Dict = field(default_factory=dict)
    hypothesis_testing_result: Dict = field(default_factory=dict)
    hypothesis_generation_result: Dict = field(default_factory=dict)
    knowledge_synthesis_result: Dict = field(default_factory=dict)
    capability_building_result: Dict = field(default_factory=dict)

    interactions_processed: int = 0
    patterns_discovered: int = 0
    strategies_evolved: int = 0
    hypotheses_tested: int = 0
    hypotheses_generated: int = 0
    knowledge_synthesized: int = 0
    capabilities_built: int = 0

    overall_score: float = 0.0
    impact_assessment: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class EvolutionEngine:
    """
    进化引擎

    整合所有进化组件，提供统一的进化接口：
    - PatternMiner: 模式挖掘
    - StrategyEvolver: 策略进化
    - HypothesisTester: 假设验证
    - KnowledgeSynthesizer: 知识综合
    - CapabilityBuilder: 能力构建
    """

    def __init__(
        self,
        db: Optional[EvolutionDatabase] = None,
        config: Optional[EvolutionConfig] = None,
    ):
        self.db = db or EvolutionDatabase()
        self.config = config or EvolutionConfig()
        self.current_cycle: Optional[EvolutionCycle] = None

        self.pattern_miner = PatternMiner(self.db)
        self.strategy_evolver = StrategyEvolver(self.db)
        self.hypothesis_tester = HypothesisTester(self.db)
        self.knowledge_synthesizer = KnowledgeSynthesizer(self.db)
        self.capability_builder = CapabilityBuilder(self.db)

        self.current_cycle: EvolutionCycle = None
        self.cycle_history: List[EvolutionCycle] = []

        self._tasks: List[asyncio.Task] = []
        self._running = False

        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("EvolutionEngine")

    async def start(self):
        """启动进化引擎"""
        if self._running:
            self.logger.warning("进化引擎已经在运行")
            return

        self._running = True
        self.logger.info("进化引擎启动")

        if self.config.enabled:
            await self._schedule_evolution_tasks()

    async def stop(self):
        """停止进化引擎"""
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self.current_cycle:
            self.current_cycle.status = "interrupted"
            self.current_cycle.end_time = datetime.now().isoformat()
            self.cycle_history.append(self.current_cycle)

        self.logger.info("进化引擎停止")

    async def _schedule_evolution_tasks(self):
        """调度进化任务"""
        if self.config.auto_mine_patterns:
            self._tasks.append(asyncio.create_task(self._periodic_pattern_mining()))

        if self.config.auto_evolve_strategies:
            self._tasks.append(asyncio.create_task(self._periodic_strategy_evolution()))

        if self.config.auto_test_hypotheses:
            self._tasks.append(asyncio.create_task(self._periodic_hypothesis_testing()))

        if self.config.auto_synthesize_knowledge:
            self._tasks.append(
                asyncio.create_task(self._periodic_knowledge_synthesis())
            )

    async def _periodic_pattern_mining(self):
        """定期模式挖掘"""
        while self._running:
            try:
                await asyncio.sleep(self.config.pattern_mining_interval_hours * 3600)

                if self._running:
                    await self.run_pattern_mining()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"模式挖掘任务错误: {e}")

    async def _periodic_strategy_evolution(self):
        """定期策略进化"""
        while self._running:
            try:
                await asyncio.sleep(
                    self.config.strategy_evolution_interval_hours * 3600
                )

                if self._running:
                    await self.run_strategy_evolution()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"策略进化任务错误: {e}")

    async def _periodic_hypothesis_testing(self):
        """定期假设验证"""
        while self._running:
            try:
                await asyncio.sleep(
                    self.config.hypothesis_testing_interval_hours * 3600
                )

                if self._running:
                    await self.run_hypothesis_testing()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"假设验证任务错误: {e}")

    async def _periodic_knowledge_synthesis(self):
        """定期知识综合"""
        while self._running:
            try:
                await asyncio.sleep(
                    self.config.knowledge_synthesis_interval_hours * 3600
                )

                if self._running:
                    await self.run_knowledge_synthesis()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"知识综合任务错误: {e}")

    async def run_full_evolution_cycle(self) -> EvolutionCycle:
        """
        运行完整的进化周期

        依次执行：
        1. 模式挖掘
        2. 策略进化
        3. 假设验证
        4. 知识综合
        5. 能力构建
        """
        self.current_cycle = EvolutionCycle()
        self.logger.info(f"开始进化周期 {self.current_cycle.id}")

        try:
            self.current_cycle.pattern_mining_result = await self.run_pattern_mining()
            self.current_cycle.patterns_discovered = (
                self.current_cycle.pattern_mining_result.get("total_found", 0)
            )

            self.current_cycle.strategy_evolution_result = (
                await self.run_strategy_evolution()
            )
            self.current_cycle.strategies_evolved = (
                self.current_cycle.strategy_evolution_result.get("evolved_count", 0)
            )

            self.current_cycle.hypothesis_testing_result = (
                await self.run_hypothesis_testing()
            )
            self.current_cycle.hypotheses_tested = (
                self.current_cycle.hypothesis_testing_result.get("tested_count", 0)
            )

            self.current_cycle.knowledge_synthesis_result = (
                await self.run_knowledge_synthesis()
            )
            self.current_cycle.knowledge_synthesized = (
                self.current_cycle.knowledge_synthesis_result.get(
                    "synthesized_count", 0
                )
            )

            self.current_cycle.hypothesis_generation_result = (
                self.run_hypothesis_generation()
            )
            self.current_cycle.hypotheses_generated = (
                self.current_cycle.hypothesis_generation_result.get(
                    "generated_count", 0
                )
            )

            self.current_cycle.capability_building_result = (
                await self.run_capability_building()
            )
            self.current_cycle.capabilities_built = (
                self.current_cycle.capability_building_result.get("built_count", 0)
            )

            self.current_cycle.status = "completed"
            self.current_cycle.end_time = datetime.now().isoformat()

            self._calculate_overall_score()

            self._log_evolution_cycle()

            self.cycle_history.append(self.current_cycle)

            return self.current_cycle

        except Exception as e:
            self.logger.error(f"进化周期错误: {e}")
            self.current_cycle.status = "error"
            self.current_cycle.end_time = datetime.now().isoformat()
            raise

    async def run_pattern_mining(self) -> Dict:
        """运行模式挖掘"""
        interactions = self.db.get_recent_interactions()

        if len(interactions) < self.config.min_interactions_for_mining:
            return {
                "status": "skipped",
                "reason": "交互数据不足",
                "count": len(interactions),
                "required": self.config.min_interactions_for_mining,
            }

        result = await self.pattern_miner.mine_patterns(interactions)

        total_found = sum(len(patterns) for patterns in result.values())

        self.db.log_evolution_event(
            event_type="pattern_mining",
            description=f"挖掘到 {total_found} 个新模式",
            changes=result,
            impact=0.3,
        )

        return {
            "status": "completed",
            "total_found": total_found,
            "time_patterns": len(result.get("time_patterns", [])),
            "causality_patterns": len(result.get("causality_patterns", [])),
            "sequence_patterns": len(result.get("sequence_patterns", [])),
            "clustering_patterns": len(result.get("clustering_patterns", [])),
            "anomaly_patterns": len(result.get("anomaly_patterns", [])),
        }

    async def run_strategy_evolution(self) -> Dict:
        """运行策略进化"""
        strategies = self.strategy_evolver.get_strategy_statistics()

        evolved_count = 0

        for strategy_id in list(self.strategy_evolver._active_variants.keys()):
            variant = self.strategy_evolver._active_variants[strategy_id]

            if (
                variant.total_uses >= self.config.min_uses_for_evolution
                and variant.avg_effectiveness > 0.6
            ):
                context = {
                    "avg_strategy_effectiveness": strategies.get(
                        "avg_effectiveness", 0.5
                    )
                }

                await self.strategy_evolver._evolve_strategy(variant, context)
                evolved_count += 1

        self.db.log_evolution_event(
            event_type="strategy_evolution",
            description=f"进化了 {evolved_count} 个策略",
            changes={"evolved_count": evolved_count},
            impact=0.4,
        )

        return {
            "status": "completed",
            "evolved_count": evolved_count,
            "total_strategies": strategies.get("total_strategies", 0),
        }

    async def run_hypothesis_testing(self) -> Dict:
        """运行假设验证"""
        pending_hypotheses = self.hypothesis_tester.get_pending_hypotheses()

        tested_count = 0

        for hypothesis_data in pending_hypotheses[:5]:
            hypothesis_id = hypothesis_data.get("id")

            experiment = await self.hypothesis_tester.design_experiment(
                hypothesis_id, sample_size=10
            )

            if experiment:
                result = await self.hypothesis_tester.run_experiment(experiment.id)

                if result.get("status") == "completed":
                    tested_count += 1

        stats = self.hypothesis_tester.get_hypothesis_statistics()

        self.db.log_evolution_event(
            event_type="hypothesis_testing",
            description=f"验证了 {tested_count} 个假设",
            changes={
                "tested_count": tested_count,
                "total_pending": stats.get("pending", 0),
            },
            impact=0.3,
        )

        return {
            "status": "completed",
            "tested_count": tested_count,
            "pending_hypotheses": stats.get("pending", 0),
        }

    async def run_knowledge_synthesis(self) -> Dict:
        """运行知识综合"""
        patterns = self._get_all_patterns()

        if len(patterns) < self.config.min_evidence_for_knowledge:
            return {
                "status": "skipped",
                "reason": "模式数据不足",
                "count": len(patterns),
            }

        knowledge = await self.knowledge_synthesizer.synthesize_from_patterns(
            patterns, domain="collaboration"
        )

        synthesized_count = len(knowledge)

        self.db.log_evolution_event(
            event_type="knowledge_synthesis",
            description=f"综合了 {synthesized_count} 个知识单元",
            changes={"synthesized_count": synthesized_count},
            impact=0.5,
        )

        stats = self.knowledge_synthesizer.get_knowledge_statistics()

        return {
            "status": "completed",
            "synthesized_count": synthesized_count,
            "total_knowledge": stats.get("total_knowledge", 0),
        }

    async def run_capability_building(self) -> Dict:
        """运行能力构建"""
        built_count = 0

        requests = await self.capability_builder.analyze_capability_gap(
            {"required_capabilities": ["text_generation", "data_analysis"]}
        )

        for request in requests[:3]:
            capability = await self.capability_builder.build_capability(request)

            if capability:
                await self.capability_builder.register_capability(capability.id)
                built_count += 1

        stats = self.capability_builder.get_capability_statistics()

        self.db.log_evolution_event(
            event_type="capability_building",
            description=f"构建了 {built_count} 个新能力",
            changes={"built_count": built_count},
            impact=0.6,
        )

        return {
            "status": "completed",
            "built_count": built_count,
            "total_capabilities": stats.get("total_capabilities", 0),
        }

    def run_hypothesis_generation(self) -> Dict:
        """运行假设生成"""
        try:
            patterns = self._get_all_patterns()

            generator = HypothesisGenerator(self.db)
            hypotheses = generator.generate_from_patterns(patterns)

            generator.save_hypotheses(hypotheses)

            self.db.log_evolution_event(
                event_type="hypothesis_generation",
                description=f"从 {len(patterns)} 个模式生成了 {len(hypotheses)} 个假设",
                changes={
                    "patterns_analyzed": len(patterns),
                    "hypotheses_generated": len(hypotheses),
                },
                impact=0.3,
            )

            stats = generator.get_statistics(hypotheses)

            return {
                "status": "completed",
                "generated_count": len(hypotheses),
                "by_category": dict(stats.get("by_category", {})),
                "avg_confidence": stats.get("avg_confidence", 0),
            }
        except Exception as e:
            self.logger.error(f"假设生成错误: {e}")
            return {"status": "error", "error": str(e), "generated_count": 0}

    def _get_all_patterns(self) -> List[Dict]:
        """获取所有模式"""
        from dataclasses import asdict

        patterns = []

        patterns.extend(self.pattern_miner._time_patterns)
        patterns.extend(self.pattern_miner._causality_patterns)
        patterns.extend(self.pattern_miner._sequence_patterns)
        patterns.extend(self.pattern_miner._clustering_patterns)
        patterns.extend(self.pattern_miner._anomaly_patterns)

        return [asdict(p) for p in patterns]

    def _calculate_overall_score(self):
        """计算整体进化得分"""
        weights = {
            "pattern_mining": 0.2,
            "strategy_evolution": 0.25,
            "hypothesis_testing": 0.2,
            "knowledge_synthesis": 0.2,
            "capability_building": 0.15,
        }

        scores = {
            "pattern_mining": min(self.current_cycle.patterns_discovered / 10, 1.0),
            "strategy_evolution": min(self.current_cycle.strategies_evolved / 5, 1.0),
            "hypothesis_testing": min(self.current_cycle.hypotheses_tested / 5, 1.0),
            "knowledge_synthesis": min(
                self.current_cycle.knowledge_synthesized / 3, 1.0
            ),
            "capability_building": min(self.current_cycle.capabilities_built / 2, 1.0),
        }

        self.current_cycle.overall_score = sum(scores[k] * weights[k] for k in weights)

        self.current_cycle.impact_assessment = {
            "pattern_impact": self._assess_pattern_impact(),
            "strategy_impact": self._assess_strategy_impact(),
            "knowledge_impact": self._assess_knowledge_impact(),
        }

    def _assess_pattern_impact(self) -> Dict:
        """评估模式影响"""
        return {
            "new_patterns": self.current_cycle.patterns_discovered,
            "confidence": self.pattern_miner.get_pattern_statistics(),
        }

    def _assess_strategy_impact(self) -> Dict:
        """评估策略影响"""
        return {
            "evolved_strategies": self.current_cycle.strategies_evolved,
            "effectiveness": self.strategy_evolver.get_strategy_statistics(),
        }

    def _assess_knowledge_impact(self) -> Dict:
        """评估知识影响"""
        return {
            "new_knowledge": self.current_cycle.knowledge_synthesized,
            "graph_size": self.knowledge_synthesizer.get_knowledge_statistics(),
        }

    def _log_evolution_cycle(self):
        """记录进化周期"""
        self.db.log_evolution_event(
            event_type="evolution_cycle",
            description=f"进化周期 {self.current_cycle.id} 完成",
            changes={
                "patterns_discovered": self.current_cycle.patterns_discovered,
                "strategies_evolved": self.current_cycle.strategies_evolved,
                "hypotheses_tested": self.current_cycle.hypotheses_tested,
                "knowledge_synthesized": self.current_cycle.knowledge_synthesized,
                "capabilities_built": self.current_cycle.capabilities_built,
                "overall_score": self.current_cycle.overall_score,
            },
            impact=self.current_cycle.overall_score,
        )

    async def process_interaction(
        self,
        user_id: str,
        session_id: str,
        interaction_type: str,
        input_summary: str,
        output_summary: str,
        outcome: str,
        effectiveness: Optional[float] = None,
        feedback: Optional[str] = None,
    ):
        """
        次交互

                       处理单 记录交互，并触发相关的进化过程
        """
        self.db.log_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_type=interaction_type,
            input_summary=input_summary,
            output_summary=output_summary,
            outcome=outcome,
            effectiveness=effectiveness,
            feedback=feedback,
        )

        if self.current_cycle:
            self.current_cycle.interactions_processed += 1

        self.logger.debug(f"处理交互: {interaction_type}")

    async def get_adaptive_strategy(self, context: Dict) -> Dict:
        """获取自适应策略"""
        strategy = await self.strategy_evolver.select_strategy(context)

        if strategy:
            return {
                "strategy_id": strategy.id,
                "name": strategy.name,
                "actions": strategy.actions,
                "confidence": strategy.confidence,
            }

        return {
            "strategy_id": "default",
            "name": "默认策略",
            "actions": [],
            "confidence": 0.5,
        }

    async def apply_knowledge(self, context: Dict) -> Dict:
        """应用知识"""
        applicable = await self.knowledge_synthesizer.find_applicable_knowledge(context)

        if applicable:
            knowledge = applicable[0]

            await self.knowledge_synthesizer.validate_knowledge(
                knowledge.id, True, context
            )

            return {
                "knowledge_id": knowledge.id,
                "title": knowledge.title,
                "content": knowledge.content,
                "confidence": knowledge.confidence,
            }

        return {"status": "no_applicable_knowledge"}

    async def execute_capability(self, capability_name: str, input_data: Dict) -> Dict:
        """执行能力"""
        return await self.capability_builder.execute_capability(
            capability_name, input_data
        )

    def get_evolution_status(self) -> Dict:
        """获取进化状态"""
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "current_cycle": asdict(self.current_cycle) if self.current_cycle else None,
            "cycle_history_count": len(self.cycle_history),
            "components": {
                "pattern_miner": self.pattern_miner.get_pattern_statistics(),
                "strategy_evolver": self.strategy_evolver.get_strategy_statistics(),
                "hypothesis_tester": self.hypothesis_tester.get_hypothesis_statistics(),
                "knowledge_synthesizer": self.knowledge_synthesizer.get_knowledge_statistics(),
                "capability_builder": self.capability_builder.get_capability_statistics(),
            },
        }

    def get_evolution_history(self, limit: int = 10) -> List[Dict]:
        """获取进化历史"""
        return [asdict(cycle) for cycle in self.cycle_history[-limit:]]

    def export_evolution_state(self) -> Dict:
        """导出进化状态"""
        return {
            "exported_at": datetime.now().isoformat(),
            "config": {
                "auto_mine_patterns": self.config.auto_mine_patterns,
                "auto_evolve_strategies": self.config.auto_evolve_strategies,
                "auto_test_hypotheses": self.config.auto_test_hypotheses,
                "auto_synthesize_knowledge": self.config.auto_synthesize_knowledge,
            },
            "status": self.get_evolution_status(),
            "pattern_statistics": self.pattern_miner.get_pattern_statistics(),
            "strategy_statistics": self.strategy_evolver.get_strategy_statistics(),
            "hypothesis_statistics": self.hypothesis_tester.get_hypothesis_statistics(),
            "knowledge_statistics": self.knowledge_synthesizer.get_knowledge_statistics(),
            "capability_statistics": self.capability_builder.get_capability_statistics(),
            "evolution_history": self.get_evolution_history(100),
        }
