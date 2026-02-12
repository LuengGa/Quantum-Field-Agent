"""
Evolution Layer - 自我学习进化层
================================

核心理念：
- 不是预设规则，而是从经验中涌现模式
- 不是静态能力，而是在交互中持续进化
- 不是证明什么，而是在实践中验证假设

功能模块：
1. PatternMiner - 模式挖掘：从交互历史中发现隐藏模式
2. StrategyEvolver - 策略进化：根据效果调整协作策略
3. HypothesisTester - 假设验证：系统化验证关于协作的假设
4. KnowledgeSynthesizer - 知识综合：将碎片经验整合为可复用的知识
5. CapabilityBuilder - 能力构建：基于需求动态构建新能力
"""

import os
import json
import asyncio
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import random
import uuid


class EvolutionDatabase:
    """
    进化数据库 - 存储交互模式、学习成果、进化历史

    表结构：
    - patterns: 发现的模式
    - strategies: 协作策略及其效果
    - hypotheses: 待验证的假设
    - experiments: 实验记录
    - knowledge: 综合知识
    - capabilities: 动态能力
    - evolution_log: 进化日志
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "evolution.db"
        else:
            db_path = Path(db_path)

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                pattern_type TEXT,
                trigger_conditions TEXT,
                description TEXT,
                occurrences INTEGER DEFAULT 0,
                success_rate REAL,
                confidence REAL,
                first_observed TEXT,
                last_observed TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                strategy_type TEXT,
                conditions TEXT,
                actions TEXT,
                success_metrics TEXT,
                total_uses INTEGER DEFAULT 0,
                success_rate REAL,
                avg_effectiveness REAL,
                evolution_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_used TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                category TEXT,
                predictions TEXT,
                test_results TEXT,
                status TEXT DEFAULT 'pending',
                test_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                evidence_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_tested TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                name TEXT NOT NULL,
                setup TEXT,
                variables TEXT,
                control_group TEXT,
                results TEXT,
                analysis TEXT,
                conclusions TEXT,
                started_at TEXT,
                completed_at TEXT,
                success BOOLEAN
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                domain TEXT,
                content TEXT NOT NULL,
                source_patterns TEXT,
                evidence TEXT,
                applicability TEXT,
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT,
                validated_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capabilities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                trigger_conditions TEXT,
                implementation TEXT,
                dependencies TEXT,
                performance_metrics TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL,
                created_at TEXT,
                last_used TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                description TEXT,
                changes TEXT,
                before_state TEXT,
                after_state TEXT,
                trigger TEXT,
                timestamp TEXT,
                impact REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                input TEXT,
                output TEXT,
                interaction_type TEXT,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_history (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                user_id TEXT,
                session_id TEXT,
                interaction_type TEXT,
                input_summary TEXT,
                output_summary TEXT,
                outcome TEXT,
                pattern_matches TEXT,
                strategy_used TEXT,
                effectiveness REAL,
                feedback TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_effectiveness (
                id TEXT PRIMARY KEY,
                strategy_id TEXT,
                strategy_name TEXT,
                context_type TEXT,
                effectiveness REAL,
                success INTEGER,
                response_time REAL,
                user_id TEXT,
                session_id TEXT,
                interaction_id TEXT,
                context TEXT,
                result TEXT,
                feedback_score INTEGER,
                feedback_comment TEXT,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                group_a TEXT,
                group_b TEXT,
                control_group TEXT,
                traffic_split REAL,
                min_sample_size INTEGER,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                results_a TEXT,
                results_b TEXT,
                winner TEXT,
                confidence REAL,
                p_value REAL,
                auto_apply_winner INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def log_interaction(
        self,
        user_id: str,
        session_id: str,
        interaction_type: str,
        input_summary: str,
        output_summary: str,
        outcome: str,
        pattern_matches: Optional[List[str]] = None,
        strategy_used: Optional[str] = None,
        effectiveness: Optional[float] = None,
        feedback: Optional[str] = None,
    ):
        """记录一次交互"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO interaction_history 
            (id, timestamp, user_id, session_id, interaction_type, 
             input_summary, output_summary, outcome, pattern_matches,
             strategy_used, effectiveness, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(uuid.uuid4()),
                datetime.now().isoformat(),
                user_id,
                session_id,
                interaction_type,
                input_summary,
                output_summary,
                outcome,
                json.dumps(pattern_matches or []),
                strategy_used,
                effectiveness,
                feedback,
            ),
        )

        conn.commit()
        conn.close()

    def get_recent_interactions(self, days: int = 7, limit: int = 1000) -> List[Dict]:
        """获取最近的交互历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM interaction_history
            WHERE timestamp > datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (f"-{days} days", limit),
        )

        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "timestamp",
            "user_id",
            "session_id",
            "interaction_type",
            "input_summary",
            "output_summary",
            "outcome",
            "pattern_matches",
            "strategy_used",
            "effectiveness",
            "feedback",
        ]

        return [dict(zip(columns, row)) for row in rows]

    def save_pattern(self, pattern: Dict):
        """保存发现的模式"""
        name = pattern.get("name", "")
        if not name:
            name = pattern.get("cause", "") + "_" + pattern.get("effect", "")
        if not name:
            name = "pattern_" + pattern.get("id", str(uuid.uuid4()))[:8]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO patterns 
            (id, name, pattern_type, trigger_conditions, description,
             occurrences, success_rate, confidence, first_observed, 
             last_observed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pattern.get("id", str(uuid.uuid4())),
                name,
                pattern.get("pattern_type", "unknown"),
                json.dumps(pattern.get("trigger_conditions", {})),
                pattern.get("description", ""),
                pattern.get("occurrences", 1),
                pattern.get("success_rate", 0.5),
                pattern.get("confidence", 0.5),
                pattern.get("first_observed", datetime.now().isoformat()),
                pattern.get("last_observed", datetime.now().isoformat()),
                json.dumps(pattern.get("metadata", {})),
            ),
        )

        conn.commit()
        conn.close()

    def get_pattern(self, pattern_id: str) -> Optional[Dict]:
        """根据ID获取模式"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        columns = [
            "id",
            "name",
            "pattern_type",
            "trigger_conditions",
            "description",
            "occurrences",
            "success_rate",
            "confidence",
            "first_observed",
            "last_observed",
            "metadata",
        ]
        return dict(zip(columns, row))

    def get_patterns_by_type(self, pattern_type: str) -> List[Dict]:
        """按类型获取模式"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM patterns WHERE pattern_type = ?
            ORDER BY confidence DESC
        """,
            (pattern_type,),
        )

        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "name",
            "pattern_type",
            "trigger_conditions",
            "description",
            "occurrences",
            "success_rate",
            "confidence",
            "first_observed",
            "last_observed",
            "metadata",
        ]

        return [dict(zip(columns, row)) for row in rows]

    def save_strategy(self, strategy: Dict):
        """保存协作策略"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO strategies 
            (id, name, strategy_type, conditions, actions, success_metrics,
             total_uses, success_rate, avg_effectiveness, evolution_count,
             created_at, last_used, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                strategy.get("id", str(uuid.uuid4())),
                strategy["name"],
                strategy.get("strategy_type", "unknown"),
                json.dumps(strategy.get("conditions", {})),
                json.dumps(strategy.get("actions", {})),
                json.dumps(strategy.get("success_metrics", {})),
                strategy.get("total_uses", 0),
                strategy.get("success_rate", 0.5),
                strategy.get("avg_effectiveness", 0.5),
                strategy.get("evolution_count", 0),
                strategy.get("created_at", datetime.now().isoformat()),
                strategy.get("last_used", datetime.now().isoformat()),
                strategy.get("is_active", 1),
            ),
        )

        conn.commit()
        conn.close()

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """根据ID获取策略"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        columns = [
            "id",
            "name",
            "strategy_type",
            "conditions",
            "actions",
            "success_metrics",
            "total_uses",
            "success_rate",
            "avg_effectiveness",
            "evolution_count",
            "created_at",
            "last_used",
            "is_active",
        ]
        return dict(zip(columns, row))

    def update_strategy_metrics(
        self, strategy_id: str, success: bool, effectiveness: float
    ):
        """更新策略效果指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT total_uses, success_rate, avg_effectiveness FROM strategies WHERE id = ?
        """,
            (strategy_id,),
        )

        row = cursor.fetchone()
        if row:
            total_uses, old_success_rate, old_effectiveness = row
            new_uses = total_uses + 1
            new_success_rate = (
                (old_success_rate or 0) * total_uses + (1 if success else 0)
            ) / new_uses
            new_effectiveness = (
                (old_effectiveness or 0) * total_uses + effectiveness
            ) / new_uses

            cursor.execute(
                """
                UPDATE strategies 
                SET total_uses = ?, success_rate = ?, avg_effectiveness = ?, last_used = ?
                WHERE id = ?
            """,
                (
                    new_uses,
                    new_success_rate,
                    new_effectiveness,
                    datetime.now().isoformat(),
                    strategy_id,
                ),
            )

        conn.commit()
        conn.close()

    def get_active_strategies(self) -> List[Dict]:
        """获取活跃策略"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM strategies WHERE is_active = 1
            ORDER BY avg_effectiveness DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "name",
            "strategy_type",
            "conditions",
            "actions",
            "success_metrics",
            "total_uses",
            "success_rate",
            "avg_effectiveness",
            "evolution_count",
            "created_at",
            "last_used",
            "is_active",
        ]

        return [dict(zip(columns, row)) for row in rows]

    def save_hypothesis(self, hypothesis: Dict):
        """保存假设"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO hypotheses 
            (id, statement, category, predictions, test_results, status,
             test_count, confidence, evidence_count, created_at, last_tested)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                hypothesis.get("id", str(uuid.uuid4())),
                hypothesis["statement"],
                hypothesis.get("category", "unknown"),
                json.dumps(hypothesis.get("predictions", [])),
                json.dumps(hypothesis.get("test_results", [])),
                hypothesis.get("status", "pending"),
                hypothesis.get("test_count", 0),
                hypothesis.get("confidence", 0.0),
                hypothesis.get("evidence_count", 0),
                hypothesis.get("created_at", datetime.now().isoformat()),
                hypothesis.get("last_tested", datetime.now().isoformat()),
            ),
        )

        conn.commit()
        conn.close()

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict]:
        """根据ID获取假设"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        columns = [
            "id",
            "statement",
            "category",
            "predictions",
            "test_results",
            "status",
            "test_count",
            "confidence",
            "evidence_count",
            "created_at",
            "last_tested",
        ]
        return dict(zip(columns, row))

    def get_hypotheses_by_status(self, status: str) -> List[Dict]:
        """按状态获取假设"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM hypotheses WHERE status = ?
            ORDER BY confidence DESC
        """,
            (status,),
        )

        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "statement",
            "category",
            "predictions",
            "test_results",
            "status",
            "test_count",
            "confidence",
            "evidence_count",
            "created_at",
            "last_tested",
        ]

        return [dict(zip(columns, row)) for row in rows]

    def save_knowledge(self, knowledge: Dict):
        """保存综合知识"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge 
            (id, title, domain, content, source_patterns, evidence,
             applicability, confidence, usage_count, created_at, validated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                knowledge.get("id", str(uuid.uuid4())),
                knowledge["title"],
                knowledge.get("domain", "general"),
                knowledge["content"],
                json.dumps(knowledge.get("source_patterns", [])),
                json.dumps(knowledge.get("evidence", [])),
                json.dumps(knowledge.get("applicability", {})),
                knowledge.get("confidence", 0.5),
                knowledge.get("usage_count", 0),
                knowledge.get("created_at", datetime.now().isoformat()),
                knowledge.get("validated_at", datetime.now().isoformat()),
            ),
        )

        conn.commit()
        conn.close()

    def get_knowledge_by_domain(self, domain: str) -> List[Dict]:
        """按领域获取知识"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM knowledge WHERE domain = ?
            ORDER BY confidence DESC
        """,
            (domain,),
        )

        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "title",
            "domain",
            "content",
            "source_patterns",
            "evidence",
            "applicability",
            "confidence",
            "usage_count",
            "created_at",
            "validated_at",
        ]

        return [dict(zip(columns, row)) for row in rows]

    def log_evolution_event(
        self,
        event_type: str,
        description: str,
        changes: Optional[Dict] = None,
        before_state: Optional[Dict] = None,
        after_state: Optional[Dict] = None,
        trigger: Optional[str] = None,
        impact: float = 0.5,
    ):
        """记录进化事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO evolution_log 
            (id, event_type, description, changes, before_state, after_state,
             trigger, timestamp, impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(uuid.uuid4()),
                event_type,
                description,
                json.dumps(changes or {}),
                json.dumps(before_state or {}),
                json.dumps(after_state or {}),
                trigger,
                datetime.now().isoformat(),
                impact,
            ),
        )

        conn.commit()
        conn.close()

    def get_evolution_history(self, limit: int = 100) -> List[Dict]:
        """获取进化历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM evolution_log
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        columns = [
            "id",
            "event_type",
            "description",
            "changes",
            "before_state",
            "after_state",
            "trigger",
            "timestamp",
            "impact",
        ]

        return [dict(zip(columns, row)) for row in rows]


@dataclass
class Pattern:
    """模式数据类"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    pattern_type: str = "unknown"
    trigger_conditions: Dict = field(default_factory=dict)
    description: str = ""
    occurrences: int = 0
    success_rate: float = 0.5
    confidence: float = 0.5
    first_observed: str = field(default_factory=lambda: datetime.now().isoformat())
    last_observed: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Strategy:
    """策略数据类"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    strategy_type: str = "unknown"
    conditions: Dict = field(default_factory=dict)
    actions: List[Dict] = field(default_factory=list)
    success_metrics: Dict = field(default_factory=dict)
    total_uses: int = 0
    success_rate: float = 0.5
    avg_effectiveness: float = 0.5
    evolution_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Hypothesis:
    """假设数据类"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    category: str = "unknown"
    predictions: List[str] = field(default_factory=list)
    test_results: List[Dict] = field(default_factory=list)
    status: str = "pending"
    test_count: int = 0
    confidence: float = 0.0
    evidence_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_tested: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Knowledge:
    """知识数据类"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    domain: str = "general"
    content: str = ""
    source_patterns: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    applicability: Dict = field(default_factory=dict)
    confidence: float = 0.5
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)
