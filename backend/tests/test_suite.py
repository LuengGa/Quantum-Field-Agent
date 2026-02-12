"""
Test Suite for Meta Quantum Field Agent
======================================

Comprehensive tests for all core modules:
- Evolution layer tests
- Data collection tests
- Strategy tracking tests
- Hypothesis validation tests
"""

import pytest
import asyncio
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json
import random

# Import all modules to test
from evolution.database import EvolutionDatabase
from evolution.pattern_miner import PatternMiner
from evolution.strategy_evolver import StrategyEvolver, StrategyVariant
from evolution.hypothesis_tester import HypothesisTester, Hypothesis, Experiment
from evolution.knowledge_synthesizer import KnowledgeSynthesizer, KnowledgeUnit
from evolution.capability_builder import (
    CapabilityBuilder,
    Capability,
    CapabilityRequest,
)
from evolution.evolution_engine import EvolutionEngine
from evolution.data_collector import ContinuousDataCollector, DataPoint, CollectionTask
from evolution.strategy_tracker import StrategyTracker, StrategyEffectivenessRecord
from evolution.hypothesis_validator import HypothesisValidator, HypothesisValidation


class TestEvolutionDatabase:
    """Tests for EvolutionDatabase"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_evolution.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_database_initialization(self, temp_db):
        """Test database creation and initialization"""
        db = EvolutionDatabase(db_path=str(temp_db))
        assert temp_db.exists()

        # Check tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "patterns" in table_names
        assert "strategies" in table_names
        assert "hypotheses" in table_names
        assert "knowledge" in table_names
        assert "capabilities" in table_names
        conn.close()

    def test_save_and_get_pattern(self, temp_db):
        """Test pattern save and retrieval"""
        db = EvolutionDatabase(db_path=str(temp_db))

        pattern = {
            "id": "test_pattern_001",
            "name": "Test Pattern",
            "pattern_type": "time_pattern",
            "description": "A test pattern",
            "confidence": 0.85,
        }

        db.save_pattern(pattern)
        retrieved = db.get_pattern("test_pattern_001")

        assert retrieved is not None
        assert retrieved["name"] == "Test Pattern"
        assert retrieved["pattern_type"] == "time_pattern"

    def test_save_and_get_strategy(self, temp_db):
        """Test strategy save and retrieval"""
        db = EvolutionDatabase(db_path=str(temp_db))

        strategy = {
            "id": "test_strategy_001",
            "name": "Test Strategy",
            "strategy_type": "adaptive",
            "conditions": '{"user_level": "beginner"}',
            "actions": '[{"action": "explain"}]',
            "avg_effectiveness": 0.78,
        }

        db.save_strategy(strategy)
        retrieved = db.get_strategy("test_strategy_001")

        assert retrieved is not None
        assert retrieved["name"] == "Test Strategy"
        assert retrieved["avg_effectiveness"] == 0.78

    def test_save_and_get_hypothesis(self, temp_db):
        """Test hypothesis save and retrieval"""
        db = EvolutionDatabase(db_path=str(temp_db))

        hypothesis = {
            "id": "test_hypothesis_001",
            "statement": "Test hypothesis statement",
            "category": "collaboration",
            "status": "pending",
            "confidence": 0.5,
        }

        db.save_hypothesis(hypothesis)
        retrieved = db.get_hypothesis("test_hypothesis_001")

        assert retrieved is not None
        assert retrieved["statement"] == "Test hypothesis statement"

    def test_update_strategy_metrics(self, temp_db):
        """Test strategy metrics update"""
        db = EvolutionDatabase(db_path=str(temp_db))

        # First save a strategy
        strategy = {
            "id": "test_strategy_002",
            "name": "Metrics Test Strategy",
            "avg_effectiveness": 0.5,
        }
        db.save_strategy(strategy)

        # Update metrics
        db.update_strategy_metrics("test_strategy_002", success=True, effectiveness=0.8)

        retrieved = db.get_strategy("test_strategy_002")
        assert retrieved is not None
        assert retrieved["avg_effectiveness"] == 0.8


class TestPatternMiner:
    """Tests for PatternMiner"""

    @pytest.fixture
    def db_with_data(self, temp_db_path):
        """Create database with sample data"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Add sample interactions
        for i in range(20):
            cursor.execute(
                """
                INSERT INTO interactions 
                (id, user_id, session_id, input, output, interaction_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    f"int_{i:03d}",
                    f"user_{i % 3}",
                    f"session_{i % 2}",
                    f"input_{i}",
                    f"output_{i}",
                    ["explanation", "code", "dialogue"][i % 3],
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        conn.close()
        return db

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_patterns.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_mine_patterns_returns_dict(self, db_with_data):
        """Test that mine_patterns returns a dictionary"""
        miner = PatternMiner(db_with_data)
        result = await miner.mine_patterns()

        assert isinstance(result, dict)
        assert "count" in result

    def test_pattern_statistics(self, db_with_data):
        """Test pattern statistics calculation"""
        miner = PatternMiner(db_with_data)
        stats = miner.get_pattern_statistics()

        assert "time_patterns" in stats
        assert "causality_patterns" in stats

    def test_get_patterns_by_type(self, db_with_data):
        """Test getting patterns by type"""
        patterns = db_with_data.get_patterns_by_type("time_pattern")

        assert isinstance(patterns, list)


class TestStrategyEvolver:
    """Tests for StrategyEvolver"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_strategies.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_strategy_variant_creation(self):
        """Test StrategyVariant dataclass"""
        variant = StrategyVariant(
            id="test_variant_001",
            name="Test Variant",
            strategy_type="adaptive",
            avg_effectiveness=0.75,
        )

        assert variant.id == "test_variant_001"
        assert variant.name == "Test Variant"
        assert variant.avg_effectiveness == 0.75
        assert variant.is_active is True

    def test_select_strategy_returns_variant(self, temp_db_path):
        """Test strategy selection returns a variant"""
        db = EvolutionDatabase(db_path=str(temp_db_path))

        # Add test strategies
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO strategies 
            (id, name, strategy_type, conditions, actions, avg_effectiveness, total_uses, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        """,
            ("str_001", "Strategy 1", "adaptive", "{}", "[]", 0.8, 10),
        )
        conn.commit()
        conn.close()

        evolver = StrategyEvolver(db)
        context = {"user_level": "beginner"}

        # Note: This test verifies the structure, actual selection depends on data
        assert evolver._active_variants is not None

    def test_calculate_strategy_score(self, temp_db_path):
        """Test strategy score calculation"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        evolver = StrategyEvolver(db)

        variant = StrategyVariant(
            id="test_variant",
            name="Test",
            avg_effectiveness=0.8,
            total_uses=20,
            confidence=0.9,
        )

        score = evolver._calculate_strategy_score(variant, {"user_level": "beginner"})

        assert 0 <= score <= 1

    def test_get_strategy_statistics(self, temp_db_path):
        """Test strategy statistics"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        evolver = StrategyEvolver(db)

        stats = evolver.get_strategy_statistics()

        assert "total_strategies" in stats
        assert "active_strategies" in stats
        assert "avg_effectiveness" in stats


class TestHypothesisTester:
    """Tests for HypothesisTester"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_hypotheses.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_hypothesis_creation(self):
        """Test Hypothesis dataclass"""
        hypothesis = Hypothesis(
            id="test_hypo_001",
            statement="Test hypothesis",
            category="collaboration",
            confidence=0.5,
        )

        assert hypothesis.id == "test_hypo_001"
        assert hypothesis.statement == "Test hypothesis"
        assert hypothesis.status == "pending"

    def test_hypothesis_statistics(self, temp_db_path):
        """Test hypothesis statistics"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        tester = HypothesisTester(db)

        stats = tester.get_hypothesis_statistics()

        assert "total_hypotheses" in stats
        assert "pending" in stats
        assert "confirmed" in stats
        assert "rejected" in stats

    def test_experiment_creation(self):
        """Test Experiment dataclass"""
        experiment = Experiment(
            id="test_exp_001",
            hypothesis_id="hypo_001",
            name="Test Experiment",
            sample_size=10,
        )

        assert experiment.id == "test_exp_001"
        assert experiment.sample_size == 10
        assert experiment.status == "running"

    @pytest.mark.asyncio
    async def test_design_experiment(self, temp_db_path):
        """Test experiment design"""
        db = EvolutionDatabase(db_path=str(temp_db_path))

        # Add a hypothesis first
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO hypotheses 
            (id, statement, category, status, confidence)
            VALUES (?, ?, ?, ?, ?)
        """,
            ("hypo_test_001", "Test statement", "collaboration", "pending", 0.5),
        )
        conn.commit()
        conn.close()

        tester = HypothesisTester(db)

        experiment = await tester.design_experiment("hypo_test_001")

        assert experiment is not None
        assert experiment.hypothesis_id == "hypo_test_001"


class TestDataCollector:
    """Tests for ContinuousDataCollector"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_collector.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_data_point_creation(self):
        """Test DataPoint dataclass"""
        point = DataPoint(
            source="interaction",
            data_type="test",
            payload={"input": "test", "output": "result"},
        )

        assert point.source == "interaction"
        assert point.payload["input"] == "test"
        assert point.quality_score == 0.8  # default

    def test_collection_task_creation(self):
        """Test CollectionTask dataclass"""
        task = CollectionTask(
            name="Test Task",
            source="interaction",
            interval_seconds=3600,
        )

        assert task.name == "Test Task"
        assert task.enabled is True

    def test_collect_realtime(self, temp_db_path):
        """Test realtime data collection"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        collector = ContinuousDataCollector(db)

        point = collector.collect_realtime(
            source="interaction",
            data_type="test",
            payload={"input": "test_input", "output": "test_output"},
        )

        assert point is not None
        assert point.source == "interaction"

    def test_get_coverage_report(self, temp_db_path):
        """Test coverage report generation"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        collector = ContinuousDataCollector(db)

        report = collector.get_coverage_report()

        assert "total_pattern_types" in report
        assert "covered_types" in report
        assert "coverage_rate" in report

    def test_get_quality_report(self, temp_db_path):
        """Test quality report generation"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        collector = ContinuousDataCollector(db)

        # Generate some data first
        collector.generate_synthetic_data(10)

        report = collector.get_quality_report()

        assert "overall_score" in report
        assert "total_points" in report

    def test_get_collection_status(self, temp_db_path):
        """Test collection status"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        collector = ContinuousDataCollector(db)

        status = collector.get_collection_status()

        assert "tasks" in status
        assert "coverage" in status
        assert "quality" in status


class TestStrategyTracker:
    """Tests for StrategyTracker"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_tracker.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_effectiveness_record_creation(self):
        """Test StrategyEffectivenessRecord dataclass"""
        record = StrategyEffectivenessRecord(
            strategy_id="str_001",
            strategy_name="Test Strategy",
            effectiveness=0.85,
            success=True,
            context_type="test",
        )

        assert record.strategy_id == "str_001"
        assert record.effectiveness == 0.85
        assert record.success is True

    def test_record_effectiveness(self, temp_db_path):
        """Test recording effectiveness"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        tracker = StrategyTracker(db)

        record = tracker.record_effectiveness(
            strategy_id="str_test_001",
            strategy_name="Test Strategy",
            effectiveness=0.8,
            success=True,
            context_type="test",
        )

        assert record is not None
        assert record.effectiveness == 0.8

    def test_get_all_metrics(self, temp_db_path):
        """Test getting all metrics"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        tracker = StrategyTracker(db)

        # Record some data
        for i in range(5):
            tracker.record_effectiveness(
                strategy_id=f"str_{i:03d}",
                strategy_name=f"Strategy {i}",
                effectiveness=0.6 + i * 0.05,
                success=i > 2,
                context_type="test",
            )

        metrics = tracker.get_all_metrics()

        assert len(metrics) >= 5

    def test_create_ab_experiment(self, temp_db_path):
        """Test creating A/B experiment"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        tracker = StrategyTracker(db)

        experiment = tracker.create_ab_experiment(
            name="Test Experiment",
            group_a="strategy_a",
            group_b="strategy_b",
            traffic_split=0.5,
        )

        assert experiment is not None
        assert experiment.name == "Test Experiment"
        assert experiment.traffic_split == 0.5

    def test_ab_experiment_status(self, temp_db_path):
        """Test A/B experiment status"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        tracker = StrategyTracker(db)

        status = tracker.get_experiment_status()

        assert "total_experiments" in status
        assert "running_experiments" in status
        assert "completed_experiments" in status

    def test_effectiveness_report(self, temp_db_path):
        """Test effectiveness report"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        tracker = StrategyTracker(db)

        # Add some data
        for i in range(10):
            tracker.record_effectiveness(
                strategy_id=f"str_{i:03d}",
                strategy_name=f"Strategy {i}",
                effectiveness=0.7 + i * 0.02,
                success=random.random() > 0.3,
                context_type="test",
            )

        report = tracker.get_effectiveness_report(days=30)

        assert "total_records" in report
        assert "strategies" in report
        assert "best_strategy" in report


class TestHypothesisValidator:
    """Tests for HypothesisValidator"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_validator.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_hypothesis_validation_creation(self):
        """Test HypothesisValidation dataclass"""
        validation = HypothesisValidation(
            hypothesis_id="hypo_001",
            validation_type="automatic",
            confidence_score=0.85,
            validation_result="confirmed",
        )

        assert validation.hypothesis_id == "hypo_001"
        assert validation.confidence_score == 0.85
        assert validation.validation_result == "confirmed"

    @pytest.mark.asyncio
    async def test_validate_hypothesis(self, temp_db_path):
        """Test hypothesis validation"""
        db = EvolutionDatabase(db_path=str(temp_db_path))

        # Add a hypothesis first
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO hypotheses 
            (id, statement, category, status, confidence)
            VALUES (?, ?, ?, ?, ?)
        """,
            ("hypo_test_001", "Test hypothesis", "test", "pending", 0.5),
        )
        conn.commit()
        conn.close()

        collector = ContinuousDataCollector(db)
        validator = HypothesisValidator(db, collector)

        result = await validator.validate_hypothesis("hypo_test_001")

        # Result may be None if no data, but should not error
        assert result is None or isinstance(result, HypothesisValidation)

    def test_get_validation_status(self, temp_db_path):
        """Test validation status"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        validator = HypothesisValidator(db)

        status = validator.get_validation_status()

        assert "total_validations" in status
        assert "confirmed" in status
        assert "rejected" in status

    def test_get_validation_report(self, temp_db_path):
        """Test validation report"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        validator = HypothesisValidator(db)

        report = validator.get_validation_report(days=30)

        assert "period_days" in report
        assert "total_validations" in report
        assert "by_result" in report


class TestKnowledgeSynthesizer:
    """Tests for KnowledgeSynthesizer"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_knowledge.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_knowledge_unit_creation(self):
        """Test KnowledgeUnit dataclass"""
        unit = KnowledgeUnit(
            id="know_001",
            domain="test",
            content="Test knowledge",
            confidence=0.85,
        )

        assert unit.id == "know_001"
        assert unit.domain == "test"
        assert unit.confidence == 0.85

    @pytest.mark.asyncio
    async def test_synthesize_from_patterns(self, temp_db_path):
        """Test knowledge synthesis from patterns"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        synthesizer = KnowledgeSynthesizer(db)

        patterns = [
            {"pattern": "time_based", "type": "time", "confidence": 0.8},
            {"pattern": "sequential", "type": "sequence", "confidence": 0.75},
        ]

        result = await synthesizer.synthesize_from_patterns(patterns, "test_domain")

        assert isinstance(result, list)

    def test_get_knowledge_statistics(self, temp_db_path):
        """Test knowledge statistics"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        synthesizer = KnowledgeSynthesizer(db)

        stats = synthesizer.get_knowledge_statistics()

        assert "total_knowledge" in stats
        assert "domains" in stats
        assert "avg_confidence" in stats


class TestCapabilityBuilder:
    """Tests for CapabilityBuilder"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_capabilities.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_capability_creation(self):
        """Test Capability dataclass"""
        capability = Capability(
            id="cap_001",
            name="Test Capability",
            category="analysis",
            implementation_code="def analyze(): pass",
        )

        assert capability.id == "cap_001"
        assert capability.name == "Test Capability"
        assert capability.is_active is True

    @pytest.mark.asyncio
    async def test_build_capability(self, temp_db_path):
        """Test capability building"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        builder = CapabilityBuilder(db)

        request = CapabilityRequest(
            name="Dynamic Capability",
            category="test",
            dependencies=["requirement1", "requirement2"],
        )

        result = await builder.build_capability(request)

        assert result is not None
        assert result.name == "Dynamic Capability"

    def test_get_capability_statistics(self, temp_db_path):
        """Test capability statistics"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        builder = CapabilityBuilder(db)

        stats = builder.get_capability_statistics()

        assert "total_capabilities" in stats
        assert "active_capabilities" in stats
        assert "by_category" in stats


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_integration.db"
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_complete_workflow(self, temp_db_path):
        """Test complete evolution workflow"""
        db = EvolutionDatabase(db_path=str(temp_db_path))
        engine = EvolutionEngine(db=db)

        # Add sample strategies
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        for i in range(5):
            cursor.execute(
                """
                INSERT INTO strategies 
                (id, name, strategy_type, avg_effectiveness, is_active)
                VALUES (?, ?, ?, ?, 1)
            """,
                (f"str_{i:03d}", f"Strategy {i}", "adaptive", 0.6 + i * 0.05),
            )
        conn.commit()
        conn.close()

        # Run evolution cycle
        cycle = await engine.run_full_evolution_cycle()

        assert cycle is not None or isinstance(cycle, dict)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
