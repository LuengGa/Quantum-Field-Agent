"""
Prometheus Metrics - Prometheus 指标端点
=======================================

提供 Prometheus 格式的指标数据：
1. HTTP 请求指标
2. 业务指标
3. 系统指标

核心理念：
- 可观测性是生产环境的基础
- 指标驱动优化
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Response
from dataclasses import dataclass, field
import time
import uuid

router = APIRouter(prefix="/metrics", tags=["metrics"])


@dataclass
class Metric:
    """指标"""

    name: str = ""
    value: float = 0
    labels: Dict = field(default_factory=dict)
    description: str = ""
    metric_type: str = "gauge"


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self._metrics: List[Metric] = []
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}

    def counter(self, name: str, value: float = 1, labels: Optional[Dict] = None):
        """增加计数器"""
        key = f"{name}{str(sorted(labels.items())) if labels else ''}"
        self._counters[key] = self._counters.get(key, 0) + value

    def gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """设置仪表值"""
        key = f"{name}{str(sorted(labels.items())) if labels else ''}"
        self._gauges[key] = value

    def histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """记录直方图值"""
        key = f"{name}{str(sorted(labels.items())) if labels else ''}"
        if key not in self._gauges:
            self._gauges[key] = 0
        self._gauges[key] = value

    def get_metrics(self) -> str:
        """获取 Prometheus 格式的指标"""
        lines = []

        lines.append("# HELP quantum_field_info Quantum Field Agent Info")
        lines.append("# TYPE quantum_field_info gauge")
        lines.append(f'quantum_field_info{{version="4.0.0"}} 1')

        lines.append("")
        lines.append("# HELP http_requests_total Total HTTP requests")
        lines.append("# TYPE http_requests_total counter")
        for key, value in self._counters.items():
            name = key.split("{")[0]
            labels = key[len(name) :] if "{" in key else ""
            lines.append(f"http_requests_total{labels} {value}")

        lines.append("")
        lines.append("# HELP http_request_duration_seconds HTTP request duration")
        lines.append("# TYPE http_request_duration_seconds histogram")

        lines.append("")
        lines.append("# HELP system_uptime_seconds System uptime")
        lines.append("# TYPE system_uptime_seconds gauge")
        lines.append(f"system_uptime_seconds {time.time()}")

        lines.append("")
        lines.append("# HELP patterns_total Total patterns discovered")
        lines.append("# TYPE patterns_total gauge")
        lines.append("patterns_total 0")

        lines.append("")
        lines.append("# HELP strategies_total Total strategies")
        lines.append("# TYPE strategies_total gauge")
        lines.append("strategies_total 0")

        lines.append("")
        lines.append("# HELP hypotheses_total Total hypotheses")
        lines.append("# TYPE hypotheses_total gauge")
        lines.append("hypotheses_total 0")

        lines.append("")
        lines.append("# HELP knowledge_units_total Total knowledge units")
        lines.append("# TYPE knowledge_units_total gauge")
        lines.append("knowledge_units_total 0")

        lines.append("")
        lines.append("# HELP capabilities_total Total capabilities")
        lines.append("# TYPE capabilities_total gauge")
        lines.append("capabilities_total 0")

        lines.append("")
        lines.append("# HELP evolution_cycles_total Total evolution cycles")
        lines.append("# TYPE evolution_cycles_total counter")
        lines.append("evolution_cycles_total 0")

        lines.append("")
        lines.append("# HELP data_points_total Total data points collected")
        lines.append("# TYPE data_points_total counter")
        lines.append("data_points_total 0")

        return "\n".join(lines)


_metrics_collector = MetricsCollector()


@router.get("")
async def get_metrics():
    """获取 Prometheus 指标"""
    content = _metrics_collector.get_metrics()
    return Response(content=content, media_type="text/plain")


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
    }


@router.get("/ready")
async def readiness_check():
    """就绪检查端点"""
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/live")
async def liveness_check():
    """存活检查端点"""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
    }


class MetricsMiddleware:
    """指标中间件"""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0

    def record_request(self, path: str, method: str, duration: float, status_code: int):
        """记录请求"""
        self.request_count += 1
        self.total_duration += duration

        if status_code >= 400:
            self.error_count += 1

        _metrics_collector.counter(
            "http_requests_total",
            labels={"method": method, "path": path, "status": str(status_code)},
        )
        _metrics_collector.histogram(
            "http_request_duration_seconds",
            duration,
            labels={"method": method, "path": path},
        )

    def get_summary(self) -> Dict:
        """获取请求摘要"""
        avg_duration = (
            self.total_duration / self.request_count if self.request_count > 0 else 0
        )
        error_rate = (
            self.error_count / self.request_count if self.request_count > 0 else 0
        )

        return {
            "total_requests": self.request_count,
            "average_duration_seconds": avg_duration,
            "error_count": self.error_count,
            "error_rate": error_rate,
        }


_metrics_middleware = MetricsMiddleware()


def record_business_metric(name: str, value: float):
    """记录业务指标"""
    _metrics_collector.gauge(name, value)


def increment_counter(name: str, labels: Optional[Dict] = None):
    """增加计数器"""
    _metrics_collector.counter(name, labels=labels)


async def demo_metrics():
    """演示指标端点"""
    print("=" * 60)
    print("Prometheus Metrics - 演示")
    print("=" * 60)

    for i in range(10):
        _metrics_collector.counter(
            "http_requests_total",
            labels={"method": "GET", "path": "/api/v1/resource", "status": "200"},
        )

    _metrics_collector.gauge("patterns_total", 5)
    _metrics_collector.gauge("strategies_total", 13)
    _metrics_collector.gauge("hypotheses_total", 271)
    _metrics_collector.gauge("knowledge_units_total", 45)
    _metrics_collector.gauge("capabilities_total", 8)

    print("\n指标收集完成!")
    print("访问 /metrics 获取 Prometheus 格式指标")

    return _metrics_collector
