"""
Continuous Data Collector - 持续数据收集器
==========================================

实现持续数据收集：
1. 定时任务 - 周期性数据收集
2. 实时收集 - 用户交互数据实时收集
3. 模式覆盖 - 提高模式覆盖率
4. 数据质量 - 数据质量监控

核心理念：
- 数据是进化的基础
- 持续收集确保系统不断学习
- 质量监控保证数据有效性
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import uuid
import random


class DataSource(Enum):
    INTERACTION = "interaction"
    FEEDBACK = "feedback"
    PATTERN = "pattern"
    STRATEGY = "strategy"
    HYPOTHESIS = "hypothesis"
    EXTERNAL = "external"


@dataclass
class DataPoint:
    """数据点"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = DataSource.INTERACTION.value
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    data_type: str = ""
    payload: Dict = field(default_factory=dict)

    quality_score: float = 0.8
    processed: bool = False

    user_id: str = ""
    session_id: str = ""
    interaction_id: str = ""


@dataclass
class CollectionTask:
    """收集任务"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source: str = DataSource.INTERACTION.value
    interval_seconds: int = 3600
    enabled: bool = True
    last_run: str = ""
    next_run: str = ""
    count: int = 0
    errors: int = 0


@dataclass
class PatternCoverage:
    """模式覆盖"""

    pattern_type: str = ""
    covered: bool = False
    examples: List[Dict] = field(default_factory=list)
    last_seen: str = ""
    count: int = 0
    confidence_sum: float = 0.0


class ContinuousDataCollector:
    """
    持续数据收集器

    功能：
    - 定时收集任务
    - 实时数据收集
    - 模式覆盖追踪
    - 数据质量监控
    """

    def __init__(self, db, evolution_engine=None):
        self.db = db
        self.evolution_engine = evolution_engine
        self._init_db()

        self.tasks: List[CollectionTask] = []
        self.coverage: Dict[str, PatternCoverage] = {}

        self.quality_thresholds = {
            "min_confidence": 0.5,
            "min_examples": 3,
            "max_age_days": 7,
        }

        self._setup_default_tasks()

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
            CREATE TABLE IF NOT EXISTS data_points (
                id TEXT PRIMARY KEY,
                source TEXT,
                timestamp TEXT,
                data_type TEXT,
                payload TEXT,
                quality_score REAL,
                processed INTEGER,
                user_id TEXT,
                session_id TEXT,
                interaction_id TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_tasks (
                id TEXT PRIMARY KEY,
                name TEXT,
                source TEXT,
                interval_seconds INTEGER,
                enabled INTEGER,
                last_run TEXT,
                next_run TEXT,
                count INTEGER,
                errors INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_coverage (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                covered INTEGER,
                examples TEXT,
                last_seen TEXT,
                count INTEGER,
                confidence_sum REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                check_type TEXT,
                status TEXT,
                details TEXT,
                score REAL
            )
        """)

        conn.commit()
        conn.close()

    def _setup_default_tasks(self):
        default_tasks = [
            CollectionTask(
                name="交互数据收集",
                source=DataSource.INTERACTION.value,
                interval_seconds=300,
                enabled=True,
            ),
            CollectionTask(
                name="反馈数据收集",
                source=DataSource.FEEDBACK.value,
                interval_seconds=600,
                enabled=True,
            ),
            CollectionTask(
                name="模式覆盖分析",
                source=DataSource.PATTERN.value,
                interval_seconds=1800,
                enabled=True,
            ),
            CollectionTask(
                name="策略效果收集",
                source=DataSource.STRATEGY.value,
                interval_seconds=3600,
                enabled=True,
            ),
            CollectionTask(
                name="假设验证数据",
                source=DataSource.HYPOTHESIS.value,
                interval_seconds=7200,
                enabled=True,
            ),
        ]

        for task in default_tasks:
            if not self._has_task(task.name):
                self.tasks.append(task)

    def _has_task(self, name: str) -> bool:
        """检查任务是否存在"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM collection_tasks WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        return row is not None

    def _get_task(self, name: str) -> Optional[CollectionTask]:
        """获取任务"""
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def collect_realtime(
        self,
        source: str,
        data_type: str,
        payload: Dict,
        user_id: str = "",
        session_id: str = "",
        interaction_id: str = "",
    ) -> DataPoint:
        """实时数据收集"""
        point = DataPoint(
            source=source,
            data_type=data_type,
            payload=payload,
            user_id=user_id,
            session_id=session_id,
            interaction_id=interaction_id,
            quality_score=self._calculate_quality(payload),
        )

        self._save_datapoint(point)

        if source == DataSource.INTERACTION.value:
            self._update_coverage(payload)

        return point

    def _calculate_quality(self, payload: Dict) -> float:
        """计算数据质量分数"""
        score = 0.8

        required_fields = ["input", "output", "context"]
        missing = sum(1 for f in required_fields if f not in payload)
        score -= missing * 0.1

        if "confidence" in payload:
            score = (score + payload["confidence"]) / 2

        if "feedback" in payload:
            score += 0.1

        return min(max(score, 0.0), 1.0)

    def _update_coverage(self, interaction: Dict):
        """更新模式覆盖"""
        patterns = interaction.get("patterns", [])

        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")

            if pattern_type not in self.coverage:
                self.coverage[pattern_type] = PatternCoverage(pattern_type=pattern_type)

            cov = self.coverage[pattern_type]
            cov.count += 1
            cov.covered = True
            cov.last_seen = datetime.now().isoformat()
            cov.confidence_sum += pattern.get("confidence", 0.5)

            if len(cov.examples) < 10:
                cov.examples.append(
                    {
                        "timestamp": interaction.get("timestamp", ""),
                        "input": interaction.get("input", "")[:100],
                        "output": interaction.get("output", "")[:100],
                    }
                )

            self._save_coverage(cov)

    def _save_datapoint(self, point: DataPoint):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO data_points 
            (id, source, timestamp, data_type, payload, quality_score, 
             processed, user_id, session_id, interaction_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                point.id,
                point.source,
                point.timestamp,
                point.data_type,
                json.dumps(point.payload),
                point.quality_score,
                1 if point.processed else 0,
                point.user_id,
                point.session_id,
                point.interaction_id,
            ),
        )

        conn.commit()
        conn.close()

    def _save_coverage(self, coverage: PatternCoverage):
        conn = self._get_connection()
        cursor = conn.cursor()

        avg_confidence = (
            coverage.confidence_sum / coverage.count if coverage.count > 0 else 0
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO pattern_coverage 
            (id, pattern_type, covered, examples, last_seen, count, confidence_sum)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                coverage.pattern_type,
                coverage.pattern_type,
                1 if coverage.covered else 0,
                json.dumps(coverage.examples),
                coverage.last_seen,
                coverage.count,
                coverage.confidence_sum,
            ),
        )

        conn.commit()
        conn.close()

    async def run_scheduled_collection(self, task_name: str) -> Dict:
        """执行定时收集任务"""
        task = next((t for t in self.tasks if t.name == task_name), None)
        if not task or not task.enabled:
            return {"status": "skipped", "reason": "Task not found or disabled"}

        try:
            collected = await self._collect_from_source(task.source)

            task.last_run = datetime.now().isoformat()
            task.count += collected.get("count", 0)

            next_run = datetime.now() + timedelta(seconds=task.interval_seconds)
            task.next_run = next_run.isoformat()

            self._save_task(task)

            return {
                "status": "success",
                "task": task_name,
                "collected": collected,
            }
        except Exception as e:
            task.errors += 1
            self._save_task(task)
            return {"status": "error", "task": task_name, "error": str(e)}

    async def _collect_from_source(self, source: str) -> Dict:
        """从数据源收集数据"""
        if source == DataSource.INTERACTION.value:
            return await self._collect_interactions()
        elif source == DataSource.FEEDBACK.value:
            return await self._collect_feedback()
        elif source == DataSource.PATTERN.value:
            return await self._collect_patterns()
        elif source == DataSource.STRATEGY.value:
            return await self._collect_strategies()
        elif source == DataSource.HYPOTHESIS.value:
            return await self._collect_hypotheses()
        else:
            return {"count": 0}

    async def _collect_interactions(self) -> Dict:
        """收集交互数据"""
        if self.evolution_engine:
            patterns = await self.evolution_engine.run_pattern_mining()
            return {"count": patterns.get("total_patterns", 0), "patterns": patterns}
        return {"count": 0}

    async def _collect_feedback(self) -> Dict:
        """收集反馈数据"""
        from evolution.feedback_collector import FeedbackCollector

        collector = FeedbackCollector(self.db)
        stats = collector.get_feedback_statistics()
        return {"count": stats.get("total_feedback", 0), "stats": stats}

    async def _collect_patterns(self) -> Dict:
        """收集模式数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM patterns")
        count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT pattern_type, COUNT(*) as cnt, AVG(confidence) as avg_conf
            FROM patterns GROUP BY pattern_type
        """)

        by_type = {
            row[0]: {"count": row[1], "avg_confidence": row[2]}
            for row in cursor.fetchall()
        }

        conn.close()

        return {"count": count, "by_type": by_type}

    async def _collect_strategies(self) -> Dict:
        """收集策略数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM strategies")
        count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT type, COUNT(*) as cnt, AVG(effectiveness) as avg_eff
            FROM strategies GROUP BY type
        """)

        by_type = {
            row[0]: {"count": row[1], "avg_effectiveness": row[2]}
            for row in cursor.fetchall()
        }

        conn.close()

        return {"count": count, "by_type": by_type}

    async def _collect_hypotheses(self) -> Dict:
        """收集假设数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM hypotheses")
        count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT status, COUNT(*) as cnt FROM hypotheses GROUP BY status
        """)

        by_status = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {"count": count, "by_status": by_status}

    def _save_task(self, task: CollectionTask):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO collection_tasks 
            (id, name, source, interval_seconds, enabled, last_run, 
             next_run, count, errors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task.id,
                task.name,
                task.source,
                task.interval_seconds,
                1 if task.enabled else 0,
                task.last_run,
                task.next_run,
                task.count,
                task.errors,
            ),
        )

        conn.commit()
        conn.close()

    def get_coverage_report(self) -> Dict:
        """获取覆盖报告"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM pattern_coverage")
        rows = cursor.fetchall()

        columns = [
            "id",
            "pattern_type",
            "covered",
            "examples",
            "last_seen",
            "count",
            "confidence_sum",
        ]

        coverage_data = []
        for row in rows:
            data = dict(zip(columns, row))
            data["examples"] = json.loads(data["examples"] or "[]")
            coverage_data.append(data)

        conn.close()

        total_types = len(coverage_data)
        covered_types = sum(1 for c in coverage_data if c["covered"])

        return {
            "total_pattern_types": total_types,
            "covered_types": covered_types,
            "coverage_rate": covered_types / total_types if total_types > 0 else 0,
            "pattern_types": coverage_data,
            "timestamp": datetime.now().isoformat(),
        }

    def get_quality_report(self) -> Dict:
        """获取质量报告"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT AVG(quality_score), COUNT(*) FROM data_points 
            WHERE timestamp > datetime('now', '-7 days')
        """)

        avg_score, total_points = cursor.fetchone()

        cursor.execute("""
            SELECT source, AVG(quality_score) as avg_score, COUNT(*) as count
            FROM data_points
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY source
        """)

        by_source = {
            row[0]: {"avg_score": row[1], "count": row[2]} for row in cursor.fetchall()
        }

        conn.close()

        return {
            "overall_score": avg_score or 0,
            "total_points": total_points or 0,
            "by_source": by_source,
            "timestamp": datetime.now().isoformat(),
        }

    def get_collection_status(self) -> Dict:
        """获取收集状态"""
        return {
            "tasks": [
                {
                    "name": t.name,
                    "source": t.source,
                    "enabled": t.enabled,
                    "interval_seconds": t.interval_seconds,
                    "last_run": t.last_run,
                    "count": t.count,
                    "errors": t.errors,
                }
                for t in self.tasks
            ],
            "coverage": self.get_coverage_report(),
            "quality": self.get_quality_report(),
        }

    def generate_synthetic_data(
        self,
        count: int = 10,
        source: str = DataSource.INTERACTION.value,
    ) -> List[DataPoint]:
        """生成合成数据用于测试"""
        points = []

        interaction_templates = [
            {
                "input": "解释量子纠缠",
                "output": "量子纠缠是两个粒子...",
                "patterns": [
                    {"type": "time_pattern", "confidence": 0.85},
                    {"type": "causality_pattern", "confidence": 0.72},
                ],
            },
            {
                "input": "如何学习编程",
                "output": "学习编程可以从...",
                "patterns": [
                    {"type": "sequence_pattern", "confidence": 0.91},
                    {"type": "clustering_pattern", "confidence": 0.68},
                ],
            },
            {
                "input": "推荐一本书",
                "output": "我推荐《百年孤独》...",
                "patterns": [
                    {"type": "anomaly_pattern", "confidence": 0.45},
                    {"type": "causality_pattern", "confidence": 0.78},
                ],
            },
            {
                "input": "什么是机器学习",
                "output": "机器学习是AI的...",
                "patterns": [
                    {"type": "time_pattern", "confidence": 0.88},
                    {"type": "sequence_pattern", "confidence": 0.82},
                ],
            },
            {
                "input": "帮我写代码",
                "output": "这是一个Python函数...",
                "patterns": [
                    {"type": "clustering_pattern", "confidence": 0.75},
                    {"type": "causality_pattern", "confidence": 0.69},
                ],
            },
        ]

        for i in range(count):
            template = random.choice(interaction_templates)

            point = DataPoint(
                source=source,
                data_type="interaction",
                payload={
                    "input": template["input"],
                    "output": template["output"],
                    "patterns": template["patterns"],
                    "timestamp": datetime.now().isoformat(),
                    "confidence": random.uniform(0.6, 0.95),
                },
                quality_score=random.uniform(0.7, 0.95),
                user_id=f"user_{random.randint(1, 100)}",
                session_id=f"session_{random.randint(1, 50)}",
                interaction_id=f"int_{uuid.uuid4().hex[:8]}",
            )

            points.append(point)
            self._save_datapoint(point)

        return points
