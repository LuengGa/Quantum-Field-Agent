"""
Strategy Tracker - 策略效果追踪器
=================================

实现策略效果追踪：
1. 效果记录 - 记录每次策略使用效果
2. A/B测试 - 比较不同策略效果
3. 效果分析 - 分析策略效果趋势
4. 自动选择 - 自动选择最优策略
5. 闭环验证 - 验证策略选择效果

核心理念：
- 数据驱动策略选择
- 持续追踪效果改进
- 自动化优化决策
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean, stdev
from enum import Enum
import uuid
import random


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class StrategyEffectivenessRecord:
    """策略效果记录"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    strategy_name: str = ""
    context_type: str = ""

    effectiveness: float = 0.5
    success: bool = False
    response_time: float = 0.0

    user_id: str = ""
    session_id: str = ""
    interaction_id: str = ""

    context: Dict = field(default_factory=dict)
    result: Dict = field(default_factory=dict)

    feedback_score: Optional[int] = None
    feedback_comment: str = ""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ABTestExperiment:
    """A/B测试实验"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    group_a: str = ""
    group_b: str = ""
    control_group: Optional[str] = None

    traffic_split: float = 0.5
    min_sample_size: int = 50

    status: str = ExperimentStatus.PENDING.value
    start_time: str = ""
    end_time: str = ""

    results_a: Dict = field(default_factory=dict)
    results_b: Dict = field(default_factory=dict)

    winner: Optional[str] = None
    confidence: float = 0.0
    p_value: float = 1.0

    auto_apply_winner: bool = True


@dataclass
class StrategyMetrics:
    """策略指标"""

    strategy_id: str = ""
    strategy_name: str = ""

    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0

    avg_effectiveness: float = 0.0
    effectiveness_std: float = 0.0
    min_effectiveness: float = 0.0
    max_effectiveness: float = 0.0

    avg_response_time: float = 0.0
    avg_feedback_score: float = 0.0

    by_context: Dict = field(default_factory=dict)
    trend: str = "stable"

    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class StrategyTracker:
    """
    策略效果追踪器

    功能：
    - 记录策略使用效果
    - 执行A/B测试
    - 分析效果趋势
    - 自动选择最优策略
    """

    def __init__(self, db):
        self.db = db
        self._init_db()

        self._records: List[StrategyEffectivenessRecord] = []
        self._experiments: List[ABTestExperiment] = []
        self._metrics_cache: Dict[str, StrategyMetrics] = {}

        self._load_experiments()

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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_metrics (
                strategy_id TEXT PRIMARY KEY,
                strategy_name TEXT,
                total_uses INTEGER,
                successful_uses INTEGER,
                failed_uses INTEGER,
                avg_effectiveness REAL,
                effectiveness_std REAL,
                min_effectiveness REAL,
                max_effectiveness REAL,
                avg_response_time REAL,
                avg_feedback_score REAL,
                by_context TEXT,
                trend TEXT,
                last_updated TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_experiments(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM ab_experiments WHERE status = ?",
            (ExperimentStatus.RUNNING.value,),
        )
        rows = cursor.fetchall()
        conn.close()

    def record_effectiveness(
        self,
        strategy_id: str,
        strategy_name: str,
        effectiveness: float,
        success: bool,
        context_type: str,
        context: Optional[Dict] = None,
        result: Optional[Dict] = None,
        user_id: str = "",
        session_id: str = "",
        interaction_id: str = "",
        response_time: float = 0.0,
        feedback_score: Optional[int] = None,
        feedback_comment: str = "",
    ) -> StrategyEffectivenessRecord:
        """记录策略效果"""
        record = StrategyEffectivenessRecord(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            context_type=context_type,
            effectiveness=effectiveness,
            success=success,
            response_time=response_time,
            user_id=user_id,
            session_id=session_id,
            interaction_id=interaction_id,
            context=context or {},
            result=result or {},
            feedback_score=feedback_score,
            feedback_comment=feedback_comment,
        )

        self._save_record(record)
        self._records.append(record)

        self._update_metrics(record)

        return record

    def _save_record(self, record: StrategyEffectivenessRecord):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO strategy_effectiveness 
            (id, strategy_id, strategy_name, context_type, effectiveness, success,
             response_time, user_id, session_id, interaction_id, context, result,
             feedback_score, feedback_comment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record.id,
                record.strategy_id,
                record.strategy_name,
                record.context_type,
                record.effectiveness,
                1 if record.success else 0,
                record.response_time,
                record.user_id,
                record.session_id,
                record.interaction_id,
                json.dumps(record.context),
                json.dumps(record.result),
                record.feedback_score,
                record.feedback_comment,
                record.timestamp,
            ),
        )

        conn.commit()
        conn.close()

    def _update_metrics(self, record: StrategyEffectivenessRecord):
        """更新策略指标"""
        strategy_id = record.strategy_id

        if strategy_id not in self._metrics_cache:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM strategy_metrics WHERE strategy_id = ?", (strategy_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                self._metrics_cache[strategy_id] = StrategyMetrics(
                    strategy_id=row[0],
                    strategy_name=row[1],
                    total_uses=row[2],
                    successful_uses=row[3],
                    failed_uses=row[4],
                    avg_effectiveness=row[5],
                    effectiveness_std=row[6],
                    min_effectiveness=row[7],
                    max_effectiveness=row[8],
                    avg_response_time=row[9],
                    avg_feedback_score=row[10],
                    by_context=json.loads(row[11] or "{}"),
                    trend=row[12],
                    last_updated=row[13],
                )
            else:
                self._metrics_cache[strategy_id] = StrategyMetrics(
                    strategy_id=strategy_id,
                    strategy_name=record.strategy_name,
                )

        metrics = self._metrics_cache[strategy_id]

        metrics.total_uses += 1
        if record.success:
            metrics.successful_uses += 1
        else:
            metrics.failed_uses += 1

        all_effectiveness = self._get_effectiveness_history(strategy_id, limit=50)
        all_effectiveness.append(record.effectiveness)

        metrics.avg_effectiveness = mean(all_effectiveness)
        if len(all_effectiveness) > 1:
            try:
                metrics.effectiveness_std = stdev(all_effectiveness)
            except:
                metrics.effectiveness_std = 0.0
        metrics.min_effectiveness = min(all_effectiveness)
        metrics.max_effectiveness = max(all_effectiveness)

        if record.response_time > 0:
            metrics.avg_response_time = (
                metrics.avg_response_time * (metrics.total_uses - 1)
                + record.response_time
            ) / metrics.total_uses

        if record.feedback_score is not None:
            metrics.avg_feedback_score = (
                metrics.avg_feedback_score * (metrics.total_uses - 1)
                + record.feedback_score
            ) / metrics.total_uses

        if record.context_type not in metrics.by_context:
            metrics.by_context[record.context_type] = {
                "total": 0,
                "success": 0,
                "avg_effectiveness": 0.0,
            }

        ctx_data = metrics.by_context[record.context_type]
        ctx_data["total"] += 1
        if record.success:
            ctx_data["success"] += 1
        ctx_data["avg_effectiveness"] = (
            ctx_data["avg_effectiveness"] * (ctx_data["total"] - 1)
            + record.effectiveness
        ) / ctx_data["total"]

        metrics.trend = self._calculate_trend(strategy_id)
        metrics.last_updated = datetime.now().isoformat()

        self._save_metrics(metrics)

    def _get_effectiveness_history(
        self, strategy_id: str, limit: int = 50
    ) -> List[float]:
        """获取策略效果历史"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT effectiveness FROM strategy_effectiveness
            WHERE strategy_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (strategy_id, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def _calculate_trend(self, strategy_id: str) -> str:
        """计算效果趋势"""
        history = self._get_effectiveness_history(strategy_id, limit=20)

        if len(history) < 5:
            return "stable"

        recent = history[:10]
        older = history[10:20]

        if len(older) == 0:
            return "stable"

        recent_avg = mean(recent)
        older_avg = mean(older)

        change = recent_avg - older_avg

        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "declining"
        else:
            return "stable"

    def _save_metrics(self, metrics: StrategyMetrics):
        """保存策略指标"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO strategy_metrics 
            (strategy_id, strategy_name, total_uses, successful_uses, failed_uses,
             avg_effectiveness, effectiveness_std, min_effectiveness, max_effectiveness,
             avg_response_time, avg_feedback_score, by_context, trend, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.strategy_id,
                metrics.strategy_name,
                metrics.total_uses,
                metrics.successful_uses,
                metrics.failed_uses,
                metrics.avg_effectiveness,
                metrics.effectiveness_std,
                metrics.min_effectiveness,
                metrics.max_effectiveness,
                metrics.avg_response_time,
                metrics.avg_feedback_score,
                json.dumps(metrics.by_context),
                metrics.trend,
                metrics.last_updated,
            ),
        )

        conn.commit()
        conn.close()

    def create_ab_experiment(
        self,
        name: str,
        group_a: str,
        group_b: str,
        description: str = "",
        control_group: Optional[str] = None,
        traffic_split: float = 0.5,
        min_sample_size: int = 50,
        auto_apply_winner: bool = True,
    ) -> ABTestExperiment:
        """创建A/B测试实验"""
        experiment = ABTestExperiment(
            name=name,
            description=description,
            group_a=group_a,
            group_b=group_b,
            control_group=control_group,
            traffic_split=traffic_split,
            min_sample_size=min_sample_size,
            auto_apply_winner=auto_apply_winner,
        )

        self._experiments.append(experiment)

        self._save_experiment(experiment)

        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """启动实验"""
        for exp in self._experiments:
            if exp.id == experiment_id:
                exp.status = ExperimentStatus.RUNNING.value
                exp.start_time = datetime.now().isoformat()
                self._save_experiment(exp)
                return True
        return False

    def end_experiment(self, experiment_id: str) -> Dict:
        """结束实验并返回分析结果"""
        for exp in self._experiments:
            if exp.id == experiment_id:
                exp.status = ExperimentStatus.COMPLETED.value
                exp.end_time = datetime.now().isoformat()

                exp.results_a = self._calculate_group_results(exp.group_a)
                exp.results_b = self._calculate_group_results(exp.group_b)

                comparison = self._compare_results(exp.results_a, exp.results_b)
                exp.winner = comparison["winner"]
                exp.confidence = comparison["confidence"]
                exp.p_value = comparison["p_value"]

                self._save_experiment(exp)

                if exp.auto_apply_winner and exp.winner:
                    self._apply_winner(exp)

                return comparison
        return {"error": "Experiment not found"}

    def _calculate_group_results(self, strategy_id: str) -> Dict:
        """计算策略组结果"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*), AVG(effectiveness), AVG(response_time),
                   SUM(success), COUNT(feedback_score), AVG(feedback_score)
            FROM strategy_effectiveness
            WHERE strategy_id = ?
        """,
            (strategy_id,),
        )

        row = cursor.fetchone()
        conn.close()

        return {
            "strategy_id": strategy_id,
            "sample_size": row[0] or 0,
            "avg_effectiveness": row[1] or 0.0,
            "avg_response_time": row[2] or 0.0,
            "success_count": row[3] or 0,
            "feedback_count": row[4] or 0,
            "avg_feedback": row[5] or 0.0,
        }

    def _compare_results(self, results_a: Dict, results_b: Dict) -> Dict:
        """比较两组结果"""
        n_a = results_a["sample_size"]
        n_b = results_b["sample_size"]

        if n_a < 10 or n_b < 10:
            return {
                "winner": None,
                "confidence": 0.0,
                "p_value": 1.0,
                "message": "样本量不足",
            }

        effect_a = results_a["avg_effectiveness"]
        effect_b = results_b["avg_effectiveness"]

        if effect_b > effect_a:
            winner = results_b["strategy_id"]
            improvement = (effect_b - effect_a) / max(effect_a, 0.01)
        else:
            winner = results_a["strategy_id"]
            improvement = (effect_a - effect_b) / max(effect_b, 0.01)

        effect_diff = abs(effect_a - effect_b)
        pooled_std = (
            (n_a - 1) * results_a.get("std", 0.1)
            + (n_b - 1) * results_b.get("std", 0.1)
        ) / (n_a + n_b - 2)

        if pooled_std > 0:
            se = pooled_std * (1 / n_a + 1 / n_b) ** 0.5
            z_score = effect_diff / max(se, 0.001)

            from scipy import stats

            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            confidence = (1 - p_value) * 100
        else:
            p_value = 1.0
            confidence = 0.0

        return {
            "winner": winner,
            "confidence": confidence,
            "p_value": p_value,
            "improvement": improvement,
            "effect_a": effect_a,
            "effect_b": effect_b,
            "sample_a": n_a,
            "sample_b": n_b,
        }

    def _apply_winner(self, experiment: ABTestExperiment):
        """应用优胜策略"""
        if not experiment.winner:
            return

        self.db.log_evolution_event(
            event_type="ab_test_winner",
            description=f"A/B测试获胜: {experiment.name}",
            changes={
                "experiment_id": experiment.id,
                "winner_strategy": experiment.winner,
                "confidence": experiment.confidence,
            },
            impact=0.4,
        )

    def _save_experiment(self, experiment: ABTestExperiment):
        """保存实验"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO ab_experiments 
            (id, name, description, group_a, group_b, control_group,
             traffic_split, min_sample_size, status, start_time, end_time,
             results_a, results_b, winner, confidence, p_value, auto_apply_winner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                experiment.id,
                experiment.name,
                experiment.description,
                experiment.group_a,
                experiment.group_b,
                experiment.control_group,
                experiment.traffic_split,
                experiment.min_sample_size,
                experiment.status,
                experiment.start_time,
                experiment.end_time,
                json.dumps(experiment.results_a),
                json.dumps(experiment.results_b),
                experiment.winner,
                experiment.confidence,
                experiment.p_value,
                1 if experiment.auto_apply_winner else 0,
            ),
        )

        conn.commit()
        conn.close()

    async def select_best_strategy(
        self,
        strategy_ids: List[str],
        context_type: str,
        require_min_uses: int = 5,
    ) -> Optional[str]:
        """选择最佳策略"""
        candidates = []

        for sid in strategy_ids:
            if sid in self._metrics_cache:
                metrics = self._metrics_cache[sid]
                if metrics.total_uses >= require_min_uses:
                    ctx_data = metrics.by_context.get(context_type, {})
                    if ctx_data.get("total", 0) >= 3:
                        candidates.append((metrics.avg_effectiveness, sid))

        if not candidates:
            for sid in strategy_ids:
                if sid in self._metrics_cache:
                    candidates.append((self._metrics_cache[sid].avg_effectiveness, sid))

        if not candidates:
            return strategy_ids[0] if strategy_ids else None

        candidates.sort(key=lambda x: x[0], reverse=True)

        best_effectiveness, best_id = candidates[0]

        best_metrics = self._metrics_cache.get(best_id)
        if best_metrics and best_metrics.trend == "declining":
            exploration_rate = 0.3
            if random.random() < exploration_rate:
                return candidates[1][1] if len(candidates) > 1 else best_id

        return best_id

    def get_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """获取策略指标"""
        return self._metrics_cache.get(strategy_id)

    def get_all_metrics(self) -> List[StrategyMetrics]:
        """获取所有策略指标"""
        return list(self._metrics_cache.values())

    def get_experiment_status(self) -> Dict:
        """获取实验状态"""
        return {
            "total_experiments": len(self._experiments),
            "running_experiments": sum(
                1
                for e in self._experiments
                if e.status == ExperimentStatus.RUNNING.value
            ),
            "completed_experiments": sum(
                1
                for e in self._experiments
                if e.status == ExperimentStatus.COMPLETED.value
            ),
            "experiments": [
                {
                    "id": e.id,
                    "name": e.name,
                    "status": e.status,
                    "winner": e.winner,
                    "confidence": e.confidence,
                }
                for e in self._experiments[-10:]
            ],
        }

    def get_effectiveness_report(self, days: int = 30) -> Dict:
        """获取效果报告"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT strategy_id, strategy_name, COUNT(*) as cnt,
                   AVG(effectiveness), AVG(success * 1.0)
            FROM strategy_effectiveness
            WHERE timestamp > datetime('now', '-{days} days')
            GROUP BY strategy_id
            ORDER BY cnt DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        strategies = []
        for row in rows:
            strategies.append(
                {
                    "strategy_id": row[0],
                    "strategy_name": row[1],
                    "sample_size": row[2],
                    "avg_effectiveness": row[3] or 0.0,
                    "success_rate": row[4] or 0.0,
                }
            )

        return {
            "period_days": days,
            "total_records": sum(s["sample_size"] for s in strategies),
            "strategies": strategies,
            "best_strategy": strategies[0] if strategies else None,
            "generated_at": datetime.now().isoformat(),
        }
