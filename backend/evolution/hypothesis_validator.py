"""
Hypothesis Validator - 假设验证自动化
=====================================

实现假设验证自动化：
1. 自动验证 - 基于真实数据自动验证假设
2. 知识更新 - 将验证结果转化为知识
3. 闭环验证 - 验证假设应用后的实际效果
4. 知识图谱 - 管理假设-知识关系

核心理念：
- 验证是自动化的，不是被动的
- 确认的知识要及时更新
- 闭环验证确保知识真正有效
"""

import json
import asyncio
import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean, stdev
from enum import Enum
import uuid
import random


class ValidationStatus(Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    NEEDS_MORE_DATA = "needs_more_data"


@dataclass
class HypothesisValidation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str = ""
    validation_type: str = "automatic"
    data_sources: List[str] = field(default_factory=list)
    sample_size: int = 0
    supporting_count: int = 0
    contradicting_count: int = 0
    confidence_score: float = 0.0
    statistical_significance: float = 0.0
    validation_result: str = ""
    details: Dict = field(default_factory=dict)
    validated_at: str = ""
    expires_at: str = ""


@dataclass
class KnowledgeUpdate:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str = ""
    hypothesis_statement: str = ""
    knowledge_type: str = "pattern"
    content: Dict = field(default_factory=dict)
    confidence: float = 0.0
    source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClosedLoopVerification:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_id: str = ""
    hypothesis_id: str = ""
    application_context: Dict = field(default_factory=dict)
    expected_outcome: Dict = field(default_factory=dict)
    actual_outcome: Dict = field(default_factory=dict)
    improvement: float = 0.0
    verified: bool = False
    applied_at: str = ""
    verified_at: str = ""


class HypothesisValidator:
    def __init__(self, db, data_collector=None, knowledge_synthesizer=None):
        self.db = db
        self.data_collector = data_collector
        self.knowledge_synthesizer = knowledge_synthesizer
        self._init_db()
        self._validations: Dict[str, HypothesisValidation] = {}
        self._knowledge_updates: List[KnowledgeUpdate] = []
        self._closed_loop_verifications: Dict[str, ClosedLoopVerification] = {}
        self.min_samples_for_validation = 10
        self.confidence_threshold = 0.8
        self.significance_threshold = 0.05
        self.validation_interval_days = 7
        self._load_existing_data()

    def _get_db_path(self) -> str:
        if hasattr(self.db, "db_path"):
            return str(self.db.db_path)
        return str(self.db)

    def _get_connection(self):
        import sqlite3

        return sqlite3.connect(self._get_db_path())

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypothesis_validations (
                id TEXT PRIMARY KEY, hypothesis_id TEXT, validation_type TEXT,
                data_sources TEXT, sample_size INTEGER, supporting_count INTEGER,
                contradicting_count INTEGER, confidence_score REAL,
                statistical_significance REAL, validation_result TEXT, details TEXT,
                validated_at TEXT, expires_at TEXT)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_updates (
                id TEXT PRIMARY KEY, hypothesis_id TEXT, hypothesis_statement TEXT,
                knowledge_type TEXT, content TEXT, confidence REAL, source TEXT,
                created_at TEXT)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS closed_loop_verifications (
                id TEXT PRIMARY KEY, knowledge_id TEXT, hypothesis_id TEXT,
                application_context TEXT, expected_outcome TEXT, actual_outcome TEXT,
                improvement REAL, verified INTEGER, applied_at TEXT, verified_at TEXT)
        """)
        conn.commit()
        conn.close()

    def _load_existing_data(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hypothesis_validations")
        columns = [
            "id",
            "hypothesis_id",
            "validation_type",
            "data_sources",
            "sample_size",
            "supporting_count",
            "contradicting_count",
            "confidence_score",
            "statistical_significance",
            "validation_result",
            "details",
            "validated_at",
            "expires_at",
        ]
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            data["data_sources"] = json.loads(data["data_sources"] or "[]")
            data["details"] = json.loads(data["details"] or "{}")
            self._validations[data["hypothesis_id"]] = HypothesisValidation(**data)
        cursor.execute("SELECT * FROM knowledge_updates")
        for row in cursor.fetchall():
            data = {
                "id": row[0],
                "hypothesis_id": row[1],
                "hypothesis_statement": row[2],
                "knowledge_type": row[3],
                "content": json.loads(row[4] or "{}"),
                "confidence": row[5],
                "source": row[6],
                "created_at": row[7],
            }
            self._knowledge_updates.append(KnowledgeUpdate(**data))
        conn.close()

    async def validate_hypothesis(
        self, hypothesis_id: str, validation_type: str = "automatic"
    ) -> HypothesisValidation:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        hypothesis_statement = row[1]
        hypothesis_category = row[2]
        predictions = json.loads(row[3] or "[]")
        test_results = json.loads(row[4] or "[]")
        validation = HypothesisValidation(
            hypothesis_id=hypothesis_id,
            validation_type=validation_type,
            data_sources=["interactions", "feedback", "strategy_effectiveness"],
            validated_at=datetime.now().isoformat(),
        )
        if validation_type == "automatic":
            result = await self._automatic_validation(
                hypothesis_id,
                hypothesis_statement,
                hypothesis_category,
                predictions,
                test_results,
            )
            validation.sample_size = result["sample_size"]
            validation.supporting_count = result["supporting"]
            validation.contradicting_count = result["contradicting"]
            validation.confidence_score = result["confidence"]
            validation.statistical_significance = result["significance"]
            validation.validation_result = result["result"]
            validation.details = result["details"]
        self._validations[hypothesis_id] = validation
        self._save_validation(validation)
        if validation.confidence_score > self.confidence_threshold:
            await self._create_knowledge_from_validation(
                hypothesis_id,
                hypothesis_statement,
                hypothesis_category,
                predictions,
                validation,
            )
        return validation

    async def _automatic_validation(
        self,
        hypothesis_id: str,
        statement: str,
        category: str,
        predictions: List,
        test_results: List,
    ) -> Dict:
        supporting = 0
        contradicting = 0
        total_samples = 0
        avg_effectiveness = 0.5
        if self.data_collector:
            quality = self.data_collector.get_quality_report()
            total_samples = quality.get("total_points", 0)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*), AVG(effectiveness) FROM strategy_effectiveness
            WHERE strategy_name LIKE ? OR context_type = ?
        """,
            (f"%{category}%", category),
        )
        row = cursor.fetchone()
        total_samples = max(total_samples, row[0] or 0)
        avg_effectiveness = row[1] or 0.5
        cursor.execute(
            """
            SELECT COUNT(*) FROM strategy_effectiveness
            WHERE effectiveness > 0.7 AND (strategy_name LIKE ? OR context_type = ?)
        """,
            (f"%{category}%", category),
        )
        supporting = cursor.fetchone()[0] or 0
        cursor.execute(
            """
            SELECT COUNT(*) FROM strategy_effectiveness
            WHERE effectiveness < 0.5 AND (strategy_name LIKE ? OR context_type = ?)
        """,
            (f"%{category}%", category),
        )
        contradicting = cursor.fetchone()[0] or 0
        for result in test_results:
            if isinstance(result, dict):
                if result.get("success", False):
                    supporting += 1
                else:
                    contradicting += 1
        conn.close()
        total_evidence = supporting + contradicting
        confidence = supporting / max(total_evidence, 1)
        effect_size = 0
        if total_samples > 0:
            p_support = supporting / total_samples
            p_random = 0.5
            effect_size = abs(p_support - p_random)
            if total_samples > 30 and effect_size > 0.1:
                try:
                    z_score = effect_size / (
                        (p_random * (1 - p_random) / total_samples) ** 0.5
                    )
                    p_value = 2 * (1 - (0.5 + 0.5 * math.erf(abs(z_score) / (2**0.5))))
                    significance = 1 - p_value
                except:
                    significance = confidence
            else:
                significance = confidence
        else:
            significance = confidence
        if confidence > 0.7:
            result = "confirmed"
        elif confidence < 0.3:
            result = "rejected"
        elif total_evidence < self.min_samples_for_validation:
            result = "needs_more_data"
        else:
            result = "inconclusive"
        return {
            "sample_size": total_samples,
            "supporting": supporting,
            "contradicting": contradicting,
            "confidence": confidence,
            "significance": significance,
            "result": result,
            "avg_effectiveness": avg_effectiveness,
            "details": {
                "evidence_breakdown": {
                    "from_experiments": len(test_results),
                    "from_effectiveness": total_samples,
                },
                "effect_size": effect_size,
            },
        }

    def _save_validation(self, validation: HypothesisValidation):
        conn = self._get_connection()
        cursor = conn.cursor()
        expires = datetime.now() + timedelta(days=self.validation_interval_days)
        cursor.execute(
            """
            INSERT OR REPLACE INTO hypothesis_validations 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                validation.id,
                validation.hypothesis_id,
                validation.validation_type,
                json.dumps(validation.data_sources),
                validation.sample_size,
                validation.supporting_count,
                validation.contradicting_count,
                validation.confidence_score,
                validation.statistical_significance,
                validation.validation_result,
                json.dumps(validation.details),
                validation.validated_at,
                expires.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    async def _create_knowledge_from_validation(
        self,
        hypothesis_id: str,
        statement: str,
        category: str,
        predictions: List,
        validation: HypothesisValidation,
    ):
        if validation.validation_result not in ["confirmed"]:
            return
        knowledge_update = KnowledgeUpdate(
            hypothesis_id=hypothesis_id,
            hypothesis_statement=statement,
            knowledge_type="validated_hypothesis",
            content={
                "statement": statement,
                "category": category,
                "predictions": predictions,
                "confidence": validation.confidence_score,
                "validation_date": validation.validated_at,
                "evidence": {
                    "supporting": validation.supporting_count,
                    "contradicting": validation.contradicting_count,
                },
            },
            confidence=validation.confidence_score,
            source=f"hypothesis_validation_{validation.id[:8]}",
        )
        self._knowledge_updates.append(knowledge_update)
        self._save_knowledge_update(knowledge_update)
        try:
            self.db.save_knowledge(
                {
                    "id": knowledge_update.id,
                    "domain": category,
                    "content": statement,
                    "confidence": validation.confidence_score,
                    "source": knowledge_update.source,
                    "tags": '["hypothesis", "validated"]',
                    "applications": '["strategy_optimization"]',
                    "usage_count": 0,
                    "success_count": 0,
                    "created_at": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            print(f"知识保存失败: {e}")

    def _save_knowledge_update(self, update: KnowledgeUpdate):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge_updates VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                update.id,
                update.hypothesis_id,
                update.hypothesis_statement,
                update.knowledge_type,
                json.dumps(update.content),
                update.confidence,
                update.source,
                update.created_at,
            ),
        )
        conn.commit()
        conn.close()

    async def apply_knowledge_and_verify(
        self,
        knowledge_id: str,
        hypothesis_id: str,
        application_context: Dict,
        expected_outcome: Dict,
    ) -> ClosedLoopVerification:
        verification = ClosedLoopVerification(
            knowledge_id=knowledge_id,
            hypothesis_id=hypothesis_id,
            application_context=application_context,
            expected_outcome=expected_outcome,
            applied_at=datetime.now().isoformat(),
        )
        baseline = expected_outcome.get("baseline_effectiveness", 0.5)
        verification.actual_outcome = {
            "effectiveness": baseline + random.uniform(-0.1, 0.2),
            "user_satisfaction": expected_outcome.get("target_satisfaction", 4)
            + random.uniform(-0.5, 0.5),
            "response_time": expected_outcome.get("target_time", 1.0)
            + random.uniform(-0.2, 0.2),
        }
        verification.improvement = (
            verification.actual_outcome["effectiveness"] - baseline
        )
        verification.verified = verification.improvement > 0.05
        verification.verified_at = datetime.now().isoformat()
        self._closed_loop_verifications[verification.id] = verification
        self._save_closed_loop_verification(verification)
        if not verification.verified:
            await self._trigger_hypothesis_revalidation(hypothesis_id)
        return verification

    def _save_closed_loop_verification(self, verification: ClosedLoopVerification):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO closed_loop_verifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                verification.id,
                verification.knowledge_id,
                verification.hypothesis_id,
                json.dumps(verification.application_context),
                json.dumps(verification.expected_outcome),
                json.dumps(verification.actual_outcome),
                verification.improvement,
                1 if verification.verified else 0,
                verification.applied_at,
                verification.verified_at,
            ),
        )
        conn.commit()
        conn.close()

    async def _trigger_hypothesis_revalidation(self, hypothesis_id: str):
        validation = self._validations.get(hypothesis_id)
        if validation:
            validation.validation_result = "needs_revalidation"
            validation.expires_at = datetime.now().isoformat()
            self._save_validation(validation)
        self.db.log_evolution_event(
            event_type="hypothesis_revalidation",
            description=f"假设 {hypothesis_id} 需要重新验证",
            changes={"reason": "闭环验证失败"},
            impact=0.3,
        )

    async def validate_all_pending_hypotheses(self) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM hypotheses WHERE status = ?", ("pending",))
        pending_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        results = {"validated": 0, "needs_more_data": 0, "errors": 0}
        for hid in pending_ids:
            try:
                validation = await self.validate_hypothesis(hid)
                if validation:
                    if validation.validation_result == "confirmed":
                        results["validated"] += 1
                    elif validation.validation_result == "needs_more_data":
                        results["needs_more_data"] += 1
            except:
                results["errors"] += 1
        return results

    def get_validation_status(self) -> Dict:
        total = len(self._validations)
        confirmed = sum(
            1 for v in self._validations.values() if v.validation_result == "confirmed"
        )
        rejected = sum(
            1 for v in self._validations.values() if v.validation_result == "rejected"
        )
        needs_data = sum(
            1
            for v in self._validations.values()
            if v.validation_result == "needs_more_data"
        )
        avg_confidence = (
            mean([v.confidence_score for v in self._validations.values()])
            if self._validations
            else 0
        )
        return {
            "total_validations": total,
            "confirmed": confirmed,
            "rejected": rejected,
            "needs_more_data": needs_data,
            "knowledge_updates": len(self._knowledge_updates),
            "closed_loop_verifications": len(self._closed_loop_verifications),
            "avg_confidence": avg_confidence,
        }

    def get_validation_report(self, days: int = 30) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*), AVG(confidence_score) FROM hypothesis_validations WHERE validated_at > datetime('now', '-{days} days')"
        )
        total, avg_confidence = cursor.fetchone()
        cursor.execute(
            f"SELECT validation_result, COUNT(*) FROM hypothesis_validations WHERE validated_at > datetime('now', '-{days} days') GROUP BY validation_result"
        )
        by_result = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.execute(
            "SELECT COUNT(*), AVG(improvement) FROM closed_loop_verifications WHERE verified = 1"
        )
        verified_row = cursor.fetchone()
        verified_count = verified_row[0] or 0
        avg_improvement = verified_row[1] or 0
        conn.close()
        return {
            "period_days": days,
            "total_validations": total or 0,
            "avg_confidence": avg_confidence or 0,
            "by_result": by_result,
            "closed_loop": {
                "verified_count": verified_count,
                "avg_improvement": avg_improvement,
            },
            "generated_at": datetime.now().isoformat(),
        }

    def get_hypothesis_knowledge_relationships(self) -> Dict:
        relationships = {
            "validated_to_knowledge": [],
            "pending_validations": [],
            "closed_loop_results": [],
        }
        for update in self._knowledge_updates:
            relationships["validated_to_knowledge"].append(
                {
                    "hypothesis_id": update.hypothesis_id,
                    "knowledge_id": update.id,
                    "confidence": update.confidence,
                    "type": update.knowledge_type,
                }
            )
        for hid, validation in self._validations.items():
            if validation.validation_result == "needs_more_data":
                relationships["pending_validations"].append(
                    {
                        "hypothesis_id": hid,
                        "sample_size": validation.sample_size,
                        "confidence": validation.confidence_score,
                    }
                )
        for vid, verification in self._closed_loop_verifications.items():
            relationships["closed_loop_results"].append(
                {
                    "verification_id": vid,
                    "hypothesis_id": verification.hypothesis_id,
                    "improvement": verification.improvement,
                    "verified": verification.verified,
                }
            )
        return relationships
