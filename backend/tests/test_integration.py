"""
Integration Tests - 集成测试
=========================

测试完整系统工作流程
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

from evolution.database import EvolutionDatabase
from evolution.evolution_engine import EvolutionEngine, EvolutionConfig
from evolution.pattern_miner import PatternMiner
from evolution.strategy_evolver import StrategyEvolver
from evolution.hypothesis_tester import HypothesisTester
from evolution.knowledge_synthesizer import KnowledgeSynthesizer, KnowledgeUnit
from evolution.capability_builder import (
    CapabilityBuilder,
    Capability,
    CapabilityRequest,
)
from evolution.data_collector import ContinuousDataCollector, DataPoint
from evolution.strategy_tracker import StrategyTracker


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def test_db(temp_dir):
    db_path = temp_dir / "test.db"
    db = EvolutionDatabase(db_path=str(db_path))
    yield db


@pytest.fixture
def test_engine(test_db):
    config = EvolutionConfig(
        auto_mine_patterns=False,
        auto_evolve_strategies=False,
        auto_test_hypotheses=False,
        auto_synthesize_knowledge=False,
        min_interactions_for_mining=5,
        min_uses_for_evolution=3,
        min_evidence_for_knowledge=1,
    )
    engine = EvolutionEngine(db=test_db, config=config)
    yield engine


class TestDatabaseIntegration:
    """数据库集成测试"""

    def test_pattern_crud(self, test_db):
        """测试模式CRUD"""
        pattern = {
            "id": "pat_001",
            "name": "测试模式",
            "type": "time_pattern",
            "confidence": 0.85,
        }
        test_db.save_pattern(pattern)

        retrieved = test_db.get_pattern("pat_001")
        assert retrieved is not None
        assert retrieved["name"] == "测试模式"

    def test_strategy_crud(self, test_db):
        """测试策略CRUD"""
        strategy = {
            "id": "str_001",
            "name": "测试策略",
            "type": "adaptive",
            "success_rate": 0.75,
        }
        test_db.save_strategy(strategy)

        retrieved = test_db.get_strategy("str_001")
        assert retrieved is not None
        assert retrieved["name"] == "测试策略"

    def test_hypothesis_crud(self, test_db):
        """测试假设CRUD"""
        hypothesis = {
            "id": "hyp_001",
            "statement": "测试假设",
            "category": "test",
            "confidence": 0.6,
        }
        test_db.save_hypothesis(hypothesis)

        retrieved = test_db.get_hypothesis("hyp_001")
        assert retrieved is not None
        assert retrieved["statement"] == "测试假设"


class TestPatternMiningIntegration:
    """模式挖掘集成测试"""

    def test_mine_patterns_empty(self, test_engine):
        """测试空数据模式挖掘"""
        patterns = asyncio.run(test_engine.run_pattern_mining())
        assert patterns is not None

    def test_multiple_interactions(self, test_engine):
        """测试多次交互"""
        for i in range(10):
            asyncio.run(
                test_engine.process_interaction(
                    user_id=f"user_{i % 3}",
                    session_id=f"session_{i % 2}",
                    interaction_type="question_answer",
                    input_summary=f"问题 {i}",
                    output_summary=f"回答 {i}",
                    outcome="positive",
                    effectiveness=0.7 + (i % 5) * 0.05,
                )
            )

        patterns = asyncio.run(test_engine.run_pattern_mining())
        assert patterns is not None


class TestStrategyEvolutionIntegration:
    """策略进化集成测试"""

    def test_strategy_statistics(self, test_engine):
        """测试策略统计"""
        stats = test_engine.strategy_evolver.get_strategy_statistics()
        assert "total_strategies" in stats
        assert "active_strategies" in stats

    def test_strategy_export(self, test_engine):
        """测试策略导出"""
        strategies = test_engine.strategy_evolver.export_strategies()
        assert strategies is not None


class TestKnowledgeIntegration:
    """知识集成测试"""

    def test_knowledge_synthesis(self):
        """测试知识综合"""
        db = EvolutionDatabase()
        synthesizer = KnowledgeSynthesizer(db)

        patterns = [
            {"pattern": "test1", "type": "time", "confidence": 0.85},
            {"pattern": "test2", "type": "sequence", "confidence": 0.78},
        ]

        result = asyncio.run(
            synthesizer.synthesize_from_patterns(patterns, "test_domain")
        )
        assert result is not None
        assert isinstance(result, list)

    def test_knowledge_unit(self):
        """测试知识单元"""
        unit = KnowledgeUnit(
            id="test_001",
            title="测试知识",
            domain="test",
            content="测试内容",
            confidence=0.8,
        )
        assert unit.id == "test_001"
        assert unit.confidence == 0.8


class TestCapabilityIntegration:
    """能力集成测试"""

    def test_capability_creation(self):
        """测试能力创建"""
        capability = Capability(
            id="cap_001",
            name="测试能力",
            category="analysis",
            implementation_code="def analyze(): pass",
        )
        assert capability.name == "测试能力"

    def test_capability_request(self):
        """测试能力请求"""
        request = CapabilityRequest(
            name="新能力",
            category="test",
            description="测试请求",
        )
        assert request.name == "新能力"


class TestDataCollectionIntegration:
    """数据收集集成测试"""

    def test_collect_data_points(self, test_db):
        """测试收集数据点"""
        collector = ContinuousDataCollector(test_db)

        for i in range(10):
            point = collector.collect_realtime(
                source="interaction",
                data_type="test",
                payload={"data": f"test_{i}"},
                user_id=f"user_{i}",
            )
            assert point is not None

        quality = collector.get_quality_report()
        assert quality is not None

    def test_coverage_report(self, test_db):
        """测试覆盖报告"""
        collector = ContinuousDataCollector(test_db)
        coverage = collector.get_coverage_report()
        assert coverage is not None


class TestStrategyTrackingIntegration:
    """策略追踪集成测试"""

    def test_record_effectiveness(self, test_db):
        """测试记录效果"""
        tracker = StrategyTracker(test_db)

        for i in range(10):
            tracker.record_effectiveness(
                strategy_id=f"str_{i % 3}",
                strategy_name=f"策略{i % 3}",
                effectiveness=0.6 + (i % 5) * 0.05,
                success=i % 4 != 3,
                context_type="test",
            )

        metrics = tracker.get_all_metrics()
        assert len(metrics) >= 1

    def test_effectiveness_report(self, test_db):
        """测试效果报告"""
        tracker = StrategyTracker(test_db)

        for i in range(5):
            tracker.record_effectiveness(
                strategy_id="test_str",
                strategy_name="测试策略",
                effectiveness=0.75,
                success=True,
                context_type="test",
            )

        report = tracker.get_effectiveness_report()
        assert report is not None


class TestHypothesisIntegration:
    """假设集成测试"""

    def test_hypothesis_statistics(self, test_db):
        """测试假设统计"""
        tester = HypothesisTester(test_db)
        stats = tester.get_hypothesis_statistics()
        assert "total_hypotheses" in stats


class TestFullWorkflow:
    """完整工作流程测试"""

    @pytest.mark.asyncio
    async def test_complete_interaction_flow(self, test_db):
        """测试完整交互流程"""
        config = EvolutionConfig(
            auto_mine_patterns=False,
            min_interactions_for_mining=5,
        )
        engine = EvolutionEngine(db=test_db, config=config)

        for i in range(10):
            await engine.process_interaction(
                user_id=f"user_{i % 3}",
                session_id=f"session_{i % 2}",
                interaction_type="question_answer",
                input_summary=f"问题 {i}",
                output_summary=f"回答 {i}",
                outcome="positive",
                effectiveness=0.7 + (i % 5) * 0.05,
            )

        patterns = await engine.run_pattern_mining()
        assert patterns is not None

    def test_system_components(self, test_db):
        """测试系统组件"""
        miner = PatternMiner(test_db)
        evolver = StrategyEvolver(test_db)
        tester = HypothesisTester(test_db)
        synthesizer = KnowledgeSynthesizer(test_db)
        builder = CapabilityBuilder(test_db)
        collector = ContinuousDataCollector(test_db)
        tracker = StrategyTracker(test_db)

        assert miner is not None
        assert evolver is not None
        assert tester is not None
        assert synthesizer is not None
        assert builder is not None
        assert collector is not None
        assert tracker is not None


class TestErrorHandling:
    """错误处理测试"""

    def test_get_nonexistent_pattern(self, test_db):
        """测试获取不存在的模式"""
        result = test_db.get_pattern("non_existent")
        assert result is None

    def test_get_nonexistent_strategy(self, test_db):
        """测试获取不存在的策略"""
        result = test_db.get_strategy("non_existent")
        assert result is None

    def test_get_nonexistent_hypothesis(self, test_db):
        """测试获取不存在的假设"""
        result = test_db.get_hypothesis("non_existent")
        assert result is None


class TestPerformance:
    """性能测试"""

    def test_bulk_interactions(self, test_engine):
        """测试批量交互"""
        import time

        start = time.time()

        for i in range(50):
            asyncio.run(
                test_engine.process_interaction(
                    user_id=f"user_{i % 10}",
                    session_id=f"session_{i // 10}",
                    interaction_type="question_answer",
                    input_summary=f"问题 {i}",
                    output_summary=f"回答 {i}",
                    outcome="positive",
                    effectiveness=0.8,
                )
            )

        elapsed = time.time() - start
        assert elapsed < 30

    def test_pattern_mining_performance(self, test_engine):
        """测试模式挖掘性能"""
        for i in range(20):
            asyncio.run(
                test_engine.process_interaction(
                    user_id=f"user_{i}",
                    session_id=f"session_{i}",
                    interaction_type="question_answer",
                    input_summary=f"复杂问题 {i}",
                    output_summary=f"详细回答 {i}",
                    outcome="positive",
                    effectiveness=0.85,
                )
            )

        import time

        start = time.time()

        patterns = asyncio.run(test_engine.run_pattern_mining())

        elapsed = time.time() - start
        assert elapsed < 10
        assert patterns is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
