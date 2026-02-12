"""
Real Data Collector - 真实数据收集模块
=====================================

实现真实用户数据收集：
1. API数据收集 - 从FastAPI收集用户请求数据
2. 交互数据收集 - 收集用户交互详细信息
3. 性能数据收集 - 收集系统性能指标
4. 错误数据收集 - 收集系统错误和异常
5. 会话追踪 - 追踪用户会话流程

核心理念：
- 真实数据是系统进化的基础
- 收集用户行为模式
- 追踪系统性能瓶颈
- 持续优化用户体验
"""

import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import uuid
import hashlib
import logging

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    FEEDBACK = "feedback"
    TIMING = "timing"


class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


@dataclass
class UserInteraction:
    """用户交互数据"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    interaction_type: str = InteractionType.REQUEST.value

    request_path: str = ""
    request_method: str = ""
    request_body: Optional[Dict] = None
    request_headers: Optional[Dict] = None

    response_status: int = 200
    response_body: Optional[Dict] = None
    response_time_ms: float = 0

    user_agent: str = ""
    ip_address: str = ""
    location: str = ""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0

    success: bool = True
    error_message: str = ""
    error_trace: str = ""

    context: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """性能指标"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_type: str = MetricType.LATENCY.value
    name: str = ""

    value: float = 0
    unit: str = "ms"
    tags: Dict = field(default_factory=dict)

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "system"

    min_value: float = 0
    max_value: float = 0
    avg_value: float = 0
    count: int = 0


@dataclass
class SessionData:
    """会话数据"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""

    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    duration_ms: float = 0

    page_views: int = 0
    interactions: int = 0
    errors: int = 0

    entry_point: str = ""
    exit_point: str = ""

    events: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class RealDataCollector:
    """
    真实数据收集器

    功能：
    - 用户交互数据收集
    - 性能指标收集
    - 会话追踪
    - 错误追踪
    - 数据聚合分析
    """

    def __init__(self, db, config: Optional[Dict] = None):
        self.db = db
        self.config = config or {}

        self.interactions: List[UserInteraction] = []
        self.metrics: List[PerformanceMetric] = []
        self.sessions: Dict[str, SessionData] = {}

        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._collection_tasks: List[asyncio.Task] = []

        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                interaction_type TEXT,
                request_path TEXT,
                request_method TEXT,
                request_body TEXT,
                request_headers TEXT,
                response_status INTEGER,
                response_body TEXT,
                response_time_ms REAL,
                user_agent TEXT,
                ip_address TEXT,
                location TEXT,
                timestamp TEXT,
                duration_ms REAL,
                success INTEGER,
                error_message TEXT,
                error_trace TEXT,
                context TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                metric_type TEXT,
                name TEXT,
                value REAL,
                unit TEXT,
                tags TEXT,
                timestamp TEXT,
                source TEXT,
                min_value REAL,
                max_value REAL,
                avg_value REAL,
                count INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_ms REAL,
                page_views INTEGER,
                interactions INTEGER,
                errors INTEGER,
                entry_point TEXT,
                exit_point TEXT,
                events TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_aggregation (
                id TEXT PRIMARY KEY,
                aggregation_type TEXT,
                time_period TEXT,
                metric_name TEXT,
                value REAL,
                count INTEGER,
                min_value REAL,
                max_value REAL,
                avg_value REAL,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _get_connection(self):
        import sqlite3

        if hasattr(self.db, "db_path"):
            return sqlite3.connect(str(self.db.db_path))
        return sqlite3.connect(self.db)

    def _get_db_path(self) -> str:
        if hasattr(self.db, "db_path"):
            return str(self.db.db_path)
        return str(self.db)

    def collect_interaction(
        self,
        user_id: str,
        session_id: str,
        interaction_type: str,
        request_path: str = "",
        request_method: str = "",
        request_body: Optional[Dict] = None,
        response_status: int = 200,
        response_time_ms: float = 0,
        success: bool = True,
        error_message: str = "",
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> UserInteraction:
        """收集用户交互"""
        interaction = UserInteraction(
            user_id=user_id,
            session_id=session_id,
            interaction_type=interaction_type,
            request_path=request_path,
            request_method=request_method,
            request_body=request_body,
            response_status=response_status,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_message,
            context=context or {},
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            **kwargs,
        )

        self.interactions.append(interaction)
        self._save_interaction(interaction)

        if len(self.interactions) >= 100:
            self._flush_interactions()

        self._trigger_callbacks("interaction", interaction)

        return interaction

    def collect_metric(
        self,
        metric_type: str,
        name: str,
        value: float,
        unit: str = "ms",
        tags: Optional[Dict] = None,
        source: str = "system",
    ) -> PerformanceMetric:
        """收集性能指标"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            name=name,
            value=value,
            unit=unit,
            tags=tags or {},
            source=source,
            timestamp=datetime.now().isoformat(),
        )

        self.metrics.append(metric)
        self._save_metric(metric)

        if len(self.metrics) >= 100:
            self._flush_metrics()

        self._trigger_callbacks("metric", metric)

        return metric

    def track_session(
        self,
        user_id: str,
        session_id: str,
        event_type: str,
        event_data: Optional[Dict] = None,
    ) -> SessionData:
        """追踪用户会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(
                user_id=user_id,
                session_id=session_id,
                start_time=datetime.now().isoformat(),
            )

        session = self.sessions[session_id]
        session.interactions += 1
        session.events.append(
            {
                "type": event_type,
                "data": event_data or {},
                "timestamp": datetime.now().isoformat(),
            }
        )

        self._save_session(session)

        return session

    def end_session(self, session_id: str) -> Optional[SessionData]:
        """结束会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.end_time = datetime.now().isoformat()

            start = datetime.fromisoformat(session.start_time)
            end = datetime.fromisoformat(session.end_time)
            session.duration_ms = (end - start).total_seconds() * 1000

            self._save_session(session)
            del self.sessions[session_id]

            return session
        return None

    def _save_interaction(self, interaction: UserInteraction):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO user_interactions 
            (id, user_id, session_id, interaction_type, request_path, request_method,
             request_body, request_headers, response_status, response_body, response_time_ms,
             user_agent, ip_address, location, timestamp, duration_ms, success,
             error_message, error_trace, context, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                interaction.id,
                interaction.user_id,
                interaction.session_id,
                interaction.interaction_type,
                interaction.request_path,
                interaction.request_method,
                json.dumps(interaction.request_body)
                if interaction.request_body
                else None,
                json.dumps(interaction.request_headers)
                if interaction.request_headers
                else None,
                interaction.response_status,
                json.dumps(interaction.response_body)
                if interaction.response_body
                else None,
                interaction.response_time_ms,
                interaction.user_agent,
                interaction.ip_address,
                interaction.location,
                interaction.timestamp,
                interaction.duration_ms,
                1 if interaction.success else 0,
                interaction.error_message,
                interaction.error_trace,
                json.dumps(interaction.context),
                json.dumps(interaction.metadata),
            ),
        )

        conn.commit()
        conn.close()

    def _save_metric(self, metric: PerformanceMetric):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO performance_metrics 
            (id, metric_type, name, value, unit, tags, timestamp, source,
             min_value, max_value, avg_value, count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metric.id,
                metric.metric_type,
                metric.name,
                metric.value,
                metric.unit,
                json.dumps(metric.tags),
                metric.timestamp,
                metric.source,
                metric.min_value,
                metric.max_value,
                metric.avg_value,
                metric.count,
            ),
        )

        conn.commit()
        conn.close()

    def _save_session(self, session: SessionData):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO sessions 
            (id, user_id, session_id, start_time, end_time, duration_ms,
             page_views, interactions, errors, entry_point, exit_point, events, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session.id,
                session.user_id,
                session.session_id,
                session.start_time,
                session.end_time,
                session.duration_ms,
                session.page_views,
                session.interactions,
                session.errors,
                session.entry_point,
                session.exit_point,
                json.dumps(session.events),
                json.dumps(session.metadata),
            ),
        )

        conn.commit()
        conn.close()

    def _flush_interactions(self):
        if self.interactions:
            for interaction in self.interactions:
                self._save_interaction(interaction)
            self.interactions.clear()

    def _flush_metrics(self):
        if self.metrics:
            for metric in self.metrics:
                self._save_metric(metric)
            self.metrics.clear()

    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数"""
        self._callbacks[event_type].append(callback)

    def _trigger_callbacks(self, event_type: str, data: Any):
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    async def start_collection(self, interval_seconds: int = 60):
        """开始数据收集"""
        self._running = True

        self._collection_tasks.append(
            asyncio.create_task(self._periodic_flush(interval_seconds))
        )
        self._collection_tasks.append(
            asyncio.create_task(self._periodic_aggregation(interval_seconds * 5))
        )

        logger.info("Real data collection started")

    async def stop_collection(self):
        """停止数据收集"""
        self._running = False

        for task in self._collection_tasks:
            task.cancel()

        self._flush_interactions()
        self._flush_metrics()

        for session_id in list(self.sessions.keys()):
            self.end_session(session_id)

        logger.info("Real data collection stopped")

    async def _periodic_flush(self, interval_seconds: int):
        """定期刷新数据"""
        while self._running:
            await asyncio.sleep(interval_seconds)
            self._flush_interactions()
            self._flush_metrics()

    async def _periodic_aggregation(self, interval_seconds: int):
        """定期聚合数据"""
        while self._running:
            await asyncio.sleep(interval_seconds)
            self._aggregate_metrics()

    def _aggregate_metrics(self):
        """聚合指标数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT metric_type, name, COUNT(*), MIN(value), MAX(value), AVG(value)
            FROM performance_metrics
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY metric_type, name
        """)

        for row in cursor.fetchall():
            metric_type, name, count, min_val, max_val, avg_val = row
            self._save_aggregation(
                aggregation_type="hourly",
                metric_name=f"{metric_type}_{name}",
                value=avg_val,
                count=count,
                min_value=min_val,
                max_value=max_val,
                avg_value=avg_val,
            )

        conn.close()

    def _save_aggregation(
        self,
        aggregation_type: str,
        metric_name: str,
        value: float,
        count: int,
        min_value: float,
        max_value: float,
        avg_value: float,
    ):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO data_aggregation
            (id, aggregation_type, time_period, metric_name, value, count,
             min_value, max_value, avg_value, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(uuid.uuid4()),
                aggregation_type,
                "last_hour",
                metric_name,
                value,
                count,
                min_value,
                max_value,
                avg_value,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def get_interaction_report(self, days: int = 7) -> Dict:
        """获取交互报告"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT interaction_type, COUNT(*), AVG(duration_ms), COUNT(CASE WHEN success=0 THEN 1 END)
            FROM user_interactions
            WHERE timestamp > datetime('now', ?)
            GROUP BY interaction_type
        """,
            (f"-{days} days",),
        )

        by_type = {
            row[0]: {
                "count": row[1],
                "avg_duration_ms": row[2],
                "error_count": row[3],
            }
            for row in cursor.fetchall()
        }

        cursor.execute(
            """
            SELECT user_id, COUNT(*) as cnt
            FROM user_interactions
            WHERE timestamp > datetime('now', ?)
            GROUP BY user_id
            ORDER BY cnt DESC
            LIMIT 10
        """,
            (f"-{days} days",),
        )

        top_users = [
            {"user_id": row[0], "interactions": row[1]} for row in cursor.fetchall()
        ]

        cursor.execute(
            """
            SELECT request_path, COUNT(*) as cnt, AVG(duration_ms)
            FROM user_interactions
            WHERE timestamp > datetime('now', ?)
            GROUP BY request_path
            ORDER BY cnt DESC
            LIMIT 10
        """,
            (f"-{days} days",),
        )

        top_endpoints = [
            {"path": row[0], "count": row[1], "avg_duration_ms": row[2]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "period_days": days,
            "total_interactions": sum(t["count"] for t in by_type.values()),
            "by_type": by_type,
            "top_users": top_users,
            "top_endpoints": top_endpoints,
            "generated_at": datetime.now().isoformat(),
        }

    def get_performance_report(self, hours: int = 24) -> Dict:
        """获取性能报告"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT metric_type, name, COUNT(*), MIN(value), MAX(value), AVG(value)
            FROM performance_metrics
            WHERE timestamp > datetime('now', ?)
            GROUP BY metric_type, name
        """,
            (f"-{hours} hours",),
        )

        metrics = {
            row[0]: {
                "name": row[1],
                "count": row[2],
                "min": row[3],
                "max": row[4],
                "avg": row[5],
            }
            for row in cursor.fetchall()
        }

        cursor.execute(
            """
            SELECT value FROM performance_metrics
            WHERE metric_type = ? AND name = ?
            AND timestamp > datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT 100
        """,
            (MetricType.LATENCY.value, "api_response_time", f"-{hours} hours"),
        )

        latencies = [row[0] for row in cursor.fetchall()]
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

        conn.close()

        return {
            "period_hours": hours,
            "metrics": metrics,
            "latency_percentiles": {
                "p50": p50,
                "p95": p95,
                "p99": p99,
            },
            "generated_at": datetime.now().isoformat(),
        }

    def get_session_report(self, days: int = 7) -> Dict:
        """获取会话报告"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(DISTINCT user_id), COUNT(*), AVG(duration_ms), SUM(errors)
            FROM sessions
            WHERE start_time > datetime('now', ?)
        """,
            (f"-{days} days",),
        )

        row = cursor.fetchone()
        total_users = row[0] or 0
        total_sessions = row[1] or 0
        avg_session_duration = row[2] or 0
        total_errors = row[3] or 0

        cursor.execute(
            """
            SELECT user_id, COUNT(*) as cnt, AVG(duration_ms)
            FROM sessions
            WHERE start_time > datetime('now', ?)
            GROUP BY user_id
            ORDER BY cnt DESC
            LIMIT 10
        """,
            (f"-{days} days",),
        )

        top_users = [
            {"user_id": row[0], "sessions": row[1], "avg_duration_ms": row[2]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "period_days": days,
            "total_users": total_users,
            "total_sessions": total_sessions,
            "avg_session_duration_ms": avg_session_duration,
            "total_errors": total_errors,
            "error_rate": total_errors / total_sessions if total_sessions > 0 else 0,
            "top_users": top_users,
            "generated_at": datetime.now().isoformat(),
        }


class DataCollectorMiddleware:
    """FastAPI中间件用于数据收集"""

    def __init__(self, collector: RealDataCollector):
        self.collector = collector

    async def collect_request(self, request, user_id: str = "", session_id: str = ""):
        """收集请求数据"""
        self.collector.collect_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_type=InteractionType.REQUEST.value,
            request_path=str(request.url.path),
            request_method=request.method,
            request_body=await self._get_request_body(request),
            user_agent=request.headers.get("user-agent", ""),
            ip_address=self._get_client_ip(request),
        )

    async def collect_response(
        self,
        request,
        response,
        duration_ms: float,
        user_id: str = "",
        session_id: str = "",
    ):
        """收集响应数据"""
        self.collector.collect_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_type=InteractionType.RESPONSE.value,
            request_path=str(request.url.path),
            request_method=request.method,
            response_status=response.status_code,
            response_time_ms=duration_ms,
            success=response.status_code < 400,
        )

    def collect_error(
        self,
        request,
        error: Exception,
        user_id: str = "",
        session_id: str = "",
    ):
        """收集错误数据"""
        import traceback

        self.collector.collect_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_type=InteractionType.ERROR.value,
            request_path=str(request.url.path) if request else "",
            request_method=request.method if request else "",
            success=False,
            error_message=str(error),
            error_trace=traceback.format_exc(),
        )

    async def _get_request_body(self, request) -> Optional[Dict]:
        """获取请求体"""
        try:
            body = await request.body()
            if body:
                return json.loads(body.decode())
        except Exception:
            pass
        return None

    def _get_client_ip(self, request) -> str:
        """获取客户端IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else ""


async def demo_real_data_collection():
    """演示真实数据收集"""
    from evolution.database import EvolutionDatabase

    db = EvolutionDatabase()
    collector = RealDataCollector(db)

    print("=" * 60)
    print("Real Data Collector - 演示")
    print("=" * 60)

    for i in range(10):
        collector.collect_interaction(
            user_id=f"user_{i % 3}",
            session_id=f"session_{i % 2}",
            interaction_type=InteractionType.REQUEST.value,
            request_path=f"/api/v1/endpoint_{i % 5}",
            request_method="POST",
            request_body={"data": f"test_{i}"},
            response_status=200 if i % 10 != 9 else 500,
            response_time_ms=50 + i * 10,
            success=i % 10 != 9,
            error_message="Server error" if i % 10 == 9 else "",
        )

        collector.collect_metric(
            metric_type=MetricType.LATENCY.value,
            name="api_response_time",
            value=50 + i * 10 + random.random() * 20,
            unit="ms",
            tags={"endpoint": f"/api/v1/endpoint_{i % 5}"},
        )

        collector.track_session(
            user_id=f"user_{i % 3}",
            session_id=f"session_{i % 2}",
            event_type="page_view",
            event_data={"page": f"/page_{i % 3}"},
        )

    interaction_report = collector.get_interaction_report()
    print("\n📊 交互报告:")
    print(f"  总交互数: {interaction_report['total_interactions']}")
    print(f"  按类型统计:")
    for itype, stats in interaction_report["by_type"].items():
        print(
            f"    {itype}: {stats['count']}次, 平均耗时{stats['avg_duration_ms']:.1f}ms"
        )

    performance_report = collector.get_performance_report()
    print("\n📊 性能报告:")
    print(f"  延迟P50: {performance_report['latency_percentiles']['p50']:.1f}ms")
    print(f"  延迟P95: {performance_report['latency_percentiles']['p95']:.1f}ms")
    print(f"  延迟P99: {performance_report['latency_percentiles']['p99']:.1f}ms")

    print("\n✅ 真实数据收集演示完成")
    return collector
