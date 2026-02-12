"""
Performance Monitor - 性能监控模块
=================================

系统性能监控：
1. 实时指标收集
2. 健康检查
3. 告警管理
4. 性能报告

核心理念：
- 性能是用户体验的基础
- 实时监控及时发现问题
- 历史分析指导优化方向
"""

import asyncio
import psutil
import time
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """健康检查"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    status: str = HealthStatus.UNKNOWN.value
    message: str = ""
    latency_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict = field(default_factory=dict)


@dataclass
class Alert:
    """告警"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    severity: str = "warning"
    message: str = ""
    metric_name: str = ""
    current_value: float = 0
    threshold: float = 0
    status: str = "triggered"
    triggered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: str = ""
    acknowledged_at: str = ""
    acknowledged_by: str = ""
    labels: Dict = field(default_factory=dict)


@dataclass
class MetricSnapshot:
    """指标快照"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    value: float = 0
    unit: str = ""
    labels: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PerformanceMonitor:
    """
    性能监控器

    功能：
    - 系统资源监控
    - 应用性能监控
    - 健康检查
    - 告警管理
    - 性能报告
    """

    def __init__(self, db=None, config: Optional[Dict] = None):
        self.db = db
        self.config = config or {}

        self._running = False
        self._monitor_tasks: List[asyncio.Task] = []

        self._health_checks: Dict[str, HealthCheck] = {}
        self._alerts: List[Alert] = []
        self._alert_rules: Dict[str, Dict] = {}
        self._triggered_alerts: Dict[str, Alert] = {}

        self._metrics_history: List[MetricSnapshot] = []

        self._callbacks: Dict[str, List[Callable]] = {}

        self._thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "latency_ms": 1000.0,
            "error_rate": 0.05,
        }

        self._setup_default_alert_rules()
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        if not self.db:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_checks (
                id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                message TEXT,
                latency_ms REAL,
                timestamp TEXT,
                details TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                name TEXT,
                severity TEXT,
                message TEXT,
                metric_name TEXT,
                current_value REAL,
                threshold REAL,
                status TEXT,
                triggered_at TEXT,
                resolved_at TEXT,
                acknowledged_at TEXT,
                acknowledged_by TEXT,
                labels TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_history (
                id TEXT PRIMARY KEY,
                name TEXT,
                value REAL,
                unit TEXT,
                labels TEXT,
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

    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        self._alert_rules = {
            "high_cpu": {
                "name": "High CPU Usage",
                "severity": "warning",
                "metric": "system.cpu_percent",
                "operator": ">",
                "threshold": 80.0,
                "message": "CPU usage is above 80%",
            },
            "high_memory": {
                "name": "High Memory Usage",
                "severity": "warning",
                "metric": "system.memory_percent",
                "operator": ">",
                "threshold": 85.0,
                "message": "Memory usage is above 85%",
            },
            "high_latency": {
                "name": "High API Latency",
                "severity": "warning",
                "metric": "api.latency_ms",
                "operator": ">",
                "threshold": 1000.0,
                "message": "API latency is above 1000ms",
            },
            "high_error_rate": {
                "name": "High Error Rate",
                "severity": "critical",
                "metric": "api.error_rate",
                "operator": ">",
                "threshold": 0.05,
                "message": "Error rate is above 5%",
            },
        }

    async def start_monitoring(self, interval_seconds: int = 10):
        """开始监控"""
        self._running = True

        self._monitor_tasks.append(
            asyncio.create_task(self._monitor_system_resources(interval_seconds))
        )
        self._monitor_tasks.append(
            asyncio.create_task(self._check_health(interval_seconds))
        )
        self._monitor_tasks.append(
            asyncio.create_task(self._evaluate_alerts(interval_seconds))
        )

        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """停止监控"""
        self._running = False

        for task in self._monitor_tasks:
            task.cancel()

        self._monitor_tasks.clear()
        logger.info("Performance monitoring stopped")

    async def _monitor_system_resources(self, interval_seconds: int):
        """监控系统资源"""
        while self._running:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                process = psutil.Process()
                process_memory = process.memory_info()

                metrics = {
                    "system.cpu_percent": cpu_percent,
                    "system.memory_percent": memory.percent,
                    "system.memory_available_mb": memory.available / (1024 * 1024),
                    "system.disk_percent": disk.percent,
                    "system.disk_free_gb": disk.free / (1024 * 1024 * 1024),
                    "process.memory_rss_mb": process_memory.rss / (1024 * 1024),
                    "process.memory_vms_mb": process_memory.vms / (1024 * 1024),
                    "process.cpu_percent": process.cpu_percent(),
                    "process.open_files": len(process.open_files()),
                    "process.connections": len(process.connections()),
                }

                for name, value in metrics.items():
                    self._record_metric(name, value)
                    self._check_threshold(name, value)

                self._trigger_callbacks("metric", {"name": name, "value": value})

            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")

            await asyncio.sleep(interval_seconds)

    async def _check_health(self, interval_seconds: int):
        """检查健康状态"""
        while self._running:
            health_checks = [
                ("database", self._check_database),
                ("disk_space", self._check_disk_space),
                ("memory", self._check_memory),
            ]

            for name, check_func in health_checks:
                try:
                    result = await check_func()
                    self._health_checks[name] = result
                    self._save_health_check(result)

                    self._trigger_callbacks("health_check", result)

                except Exception as e:
                    logger.error(f"Health check failed for {name}: {e}")

            await asyncio.sleep(interval_seconds)

    async def _check_database(self) -> HealthCheck:
        """检查数据库健康"""
        start = time.time()
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()

            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY.value,
                message="Database is responsive",
                latency_ms=(time.time() - start) * 1000,
                details={"connected": True},
            )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY.value,
                message=f"Database connection failed: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
                details={"error": str(e)},
            )

    async def _check_disk_space(self) -> HealthCheck:
        """检查磁盘空间"""
        disk = psutil.disk_usage("/")
        percent = disk.percent

        if percent > 95:
            status = HealthStatus.UNHEALTHY.value
        elif percent > 90:
            status = HealthStatus.DEGRADED.value
        else:
            status = HealthStatus.HEALTHY.value

        return HealthCheck(
            name="disk_space",
            status=status,
            message=f"Disk usage: {percent:.1f}%",
            details={
                "total_gb": disk.total / (1024 * 1024 * 1024),
                "used_gb": disk.used / (1024 * 1024 * 1024),
                "free_gb": disk.free / (1024 * 1024 * 1024),
            },
        )

    async def _check_memory(self) -> HealthCheck:
        """检查内存状态"""
        memory = psutil.virtual_memory()
        percent = memory.percent

        if percent > 95:
            status = HealthStatus.UNHEALTHY.value
        elif percent > 85:
            status = HealthStatus.DEGRADED.value
        else:
            status = HealthStatus.HEALTHY.value

        return HealthCheck(
            name="memory",
            status=status,
            message=f"Memory usage: {percent:.1f}%",
            details={
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
            },
        )

    async def _evaluate_alerts(self, interval_seconds: int):
        """评估告警"""
        while self._running:
            for rule_id, rule in self._alert_rules.items():
                metric_name = rule["metric"]
                threshold = rule["threshold"]

                if metric_name in self._metrics_history:
                    latest = self._metrics_history[-1]
                    current_value = latest.value

                    should_trigger = False
                    if rule["operator"] == ">" and current_value > threshold:
                        should_trigger = True
                    elif rule["operator"] == "<" and current_value < threshold:
                        should_trigger = True

                    if should_trigger and rule_id not in self._triggered_alerts:
                        alert = self._create_alert(rule, current_value)
                        self._triggered_alerts[rule_id] = alert
                        self._alerts.append(alert)
                        self._save_alert(alert)

                        self._trigger_callbacks("alert_triggered", alert)

                    elif not should_trigger and rule_id in self._triggered_alerts:
                        alert = self._triggered_alerts[rule_id]
                        alert.status = "resolved"
                        alert.resolved_at = datetime.now().isoformat()
                        del self._triggered_alerts[rule_id]

                        self._save_alert(alert)

                        self._trigger_callbacks("alert_resolved", alert)

            await asyncio.sleep(interval_seconds)

    def _create_alert(self, rule: Dict, current_value: float) -> Alert:
        """创建告警"""
        return Alert(
            name=rule["name"],
            severity=rule["severity"],
            message=rule["message"],
            metric_name=rule["metric"],
            current_value=current_value,
            threshold=rule["threshold"],
            status="triggered",
            labels={"rule_id": rule.get("id", "")},
        )

    def _record_metric(self, name: str, value: float, unit: str = ""):
        """记录指标"""
        metric = MetricSnapshot(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now().isoformat(),
        )

        self._metrics_history.append(metric)

        if len(self._metrics_history) > 10000:
            self._metrics_history = self._metrics_history[-10000:]

        if self.db:
            self._save_metric(metric)

    def _check_threshold(self, name: str, value: float):
        """检查阈值"""
        if name in self._thresholds:
            threshold = self._thresholds[name]
            if value > threshold:
                logger.warning(
                    f"Metric {name} value {value} exceeds threshold {threshold}"
                )

    def _save_health_check(self, check: HealthCheck):
        """保存健康检查"""
        if not self.db:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO health_checks
            (id, name, status, message, latency_ms, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                check.id,
                check.name,
                check.status,
                check.message,
                check.latency_ms,
                check.timestamp,
                json.dumps(check.details),
            ),
        )

        conn.commit()
        conn.close()

    def _save_alert(self, alert: Alert):
        """保存告警"""
        if not self.db:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO alerts
            (id, name, severity, message, metric_name, current_value, threshold,
             status, triggered_at, resolved_at, acknowledged_at, acknowledged_by, labels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                alert.id,
                alert.name,
                alert.severity,
                alert.message,
                alert.metric_name,
                alert.current_value,
                alert.threshold,
                alert.status,
                alert.triggered_at,
                alert.resolved_at,
                alert.acknowledged_at,
                alert.acknowledged_by,
                json.dumps(alert.labels),
            ),
        )

        conn.commit()
        conn.close()

    def _save_metric(self, metric: MetricSnapshot):
        """保存指标"""
        if not self.db:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO metrics_history
            (id, name, value, unit, labels, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                metric.id,
                metric.name,
                metric.value,
                metric.unit,
                json.dumps(metric.labels),
                metric.timestamp,
            ),
        )

        conn.commit()
        conn.close()

    def register_callback(self, event_type: str, callback: Callable):
        """注册回调"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def _trigger_callbacks(self, event_type: str, data: Any):
        """触发回调"""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def record_custom_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        labels: Optional[Dict] = None,
    ):
        """记录自定义指标"""
        self._record_metric(name, value, unit)
        self._check_threshold(name, value)

    def acknowledge_alert(self, alert_id: str, user: str = ""):
        """确认告警"""
        for alert in self._alerts:
            if alert.id == alert_id and alert.status == "triggered":
                alert.status = "acknowledged"
                alert.acknowledged_at = datetime.now().isoformat()
                alert.acknowledged_by = user
                self._save_alert(alert)
                return True
        return False

    def get_health_status(self) -> Dict:
        """获取健康状态"""
        overall = HealthStatus.HEALTHY.value
        checks = []

        for name, check in self._health_checks.items():
            checks.append(
                {
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                    "latency_ms": check.latency_ms,
                }
            )

            if check.status == HealthStatus.UNHEALTHY.value:
                overall = HealthStatus.UNHEALTHY.value
            elif (
                check.status == HealthStatus.DEGRADED.value
                and overall != HealthStatus.UNHEALTHY.value
            ):
                overall = HealthStatus.DEGRADED.value

        return {
            "overall_status": overall,
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }

    def get_active_alerts(self) -> List[Dict]:
        """获取活跃告警"""
        return [
            {
                "id": a.id,
                "name": a.name,
                "severity": a.severity,
                "message": a.message,
                "metric_name": a.metric_name,
                "current_value": a.current_value,
                "threshold": a.threshold,
                "triggered_at": a.triggered_at,
                "labels": a.labels,
            }
            for a in self._alerts
            if a.status in ["triggered", "acknowledged"]
        ]

    def get_performance_report(self, hours: int = 24) -> Dict:
        """获取性能报告"""
        metrics_by_name: Dict[str, List[float]] = {}

        for metric in self._metrics_history:
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric.value)

        metrics_summary = {}
        for name, values in metrics_by_name.items():
            metrics_summary[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        cpu_values = metrics_by_name.get("system.cpu_percent", [])
        memory_values = metrics_by_name.get("system.memory_percent", [])

        return {
            "period_hours": hours,
            "metrics": metrics_summary,
            "cpu": {
                "avg_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max_percent": max(cpu_values) if cpu_values else 0,
            },
            "memory": {
                "avg_percent": sum(memory_values) / len(memory_values)
                if memory_values
                else 0,
                "max_percent": max(memory_values) if memory_values else 0,
            },
            "active_alerts": len(self.get_active_alerts()),
            "timestamp": datetime.now().isoformat(),
        }

    def get_metrics_history(self, name: str, hours: int = 1) -> List[Dict]:
        """获取指标历史"""
        cutoff = datetime.now() - timedelta(hours=hours)

        return [
            {"name": m.name, "value": m.value, "timestamp": m.timestamp}
            for m in self._metrics_history
            if m.name == name and datetime.fromisoformat(m.timestamp) > cutoff
        ]


async def demo_performance_monitor():
    """演示性能监控"""
    from evolution.database import EvolutionDatabase

    db = EvolutionDatabase()
    monitor = PerformanceMonitor(db)

    print("=" * 60)
    print("Performance Monitor - 演示")
    print("=" * 60)

    for i in range(10):
        monitor.record_custom_metric("api.latency_ms", 100 + i * 10 + (i % 3) * 20)
        monitor.record_custom_metric("api.request_count", i + 1)
        monitor.record_custom_metric("api.error_count", i % 3)

    health = monitor.get_health_status()
    print("\n💚 健康状态:")
    print(f"  总体状态: {health['overall_status']}")
    for check in health["checks"]:
        print(f"    {check['name']}: {check['status']} ({check['message']})")

    alerts = monitor.get_active_alerts()
    print(f"\n⚠️  活跃告警: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"  {alert['severity']}: {alert['message']}")

    report = monitor.get_performance_report(hours=1)
    print("\n📊 性能报告:")
    print(f"  API延迟: {report['metrics'].get('api.latency_ms', {})}")
    print(f"  CPU平均: {report['cpu']['avg_percent']:.1f}%")
    print(f"  内存平均: {report['memory']['avg_percent']:.1f}%")

    print("\n✅ 性能监控演示完成")
    return monitor
