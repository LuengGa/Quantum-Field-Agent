#!/usr/bin/env python3
"""Real Data Collector & Performance Monitor - 演示"""

import asyncio
import random
import sys

sys.path.insert(
    0, "/Volumes/J ZAO 9 SER 1/Python/Open Code/QUANTUM_FIELD_GUIDE/backend"
)

from evolution.database import EvolutionDatabase
from evolution.real_data_collector import RealDataCollector, InteractionType
from evolution.performance_monitor import PerformanceMonitor


async def demo_modules():
    print("=" * 70)
    print("Real Data Collector & Performance Monitor - 演示")
    print("=" * 70)

    db = EvolutionDatabase()

    print("\n📊 1. 真实数据收集器演示")
    print("-" * 50)

    collector = RealDataCollector(db)

    for i in range(15):
        collector.collect_interaction(
            user_id=f"user_{i % 5}",
            session_id=f"session_{i % 3}",
            interaction_type=InteractionType.REQUEST.value,
            request_path=f"/api/v1/resource_{i % 7}",
            request_method=["GET", "POST", "PUT"][i % 3],
            request_body={"id": i, "data": f"test_{i}"},
            response_status=200 if i % 12 != 11 else 500,
            response_time_ms=30 + i * 5 + (i % 4) * 10,
            success=i % 12 != 11,
            error_message="Internal Server Error" if i % 12 == 11 else "",
        )

        collector.collect_metric(
            metric_type="api",
            name="request_duration",
            value=30 + i * 5 + (i % 4) * 10,
            unit="ms",
            tags={"endpoint": f"/api/v1/resource_{i % 7}"},
        )

        collector.track_session(
            user_id=f"user_{i % 5}",
            session_id=f"session_{i % 3}",
            event_type="action",
            event_data={"action": f"action_{i}"},
        )

    interaction_report = collector.get_interaction_report()
    print(f"  总交互数: {interaction_report['total_interactions']}")
    print(f"  活跃用户: {len(interaction_report['top_users'])}")
    print(f"  热门端点:")
    for ep in interaction_report["top_endpoints"][:3]:
        print(f"    {ep['path']}: {ep['count']}次 ({ep['avg_duration_ms']:.1f}ms)")

    session_report = collector.get_session_report()
    print(f"\n  会话统计:")
    print(f"    总用户: {session_report['total_users']}")
    print(f"    总会话: {session_report['total_sessions']}")
    print(f"    平均会话时长: {session_report['avg_session_duration_ms']:.1f}ms")

    print("\n\n📈 2. 性能监控演示")
    print("-" * 50)

    monitor = PerformanceMonitor(db)

    for i in range(20):
        monitor.record_custom_metric(
            name="api.latency_ms",
            value=50 + i * 3 + (i % 5) * 15,
        )
        monitor.record_custom_metric(
            name="api.request_count",
            value=100 + i * 10,
        )
        monitor.record_custom_metric(
            name="api.error_count",
            value=i % 5,
        )

    health = monitor.get_health_status()
    print(f"  健康状态: {health['overall_status']}")
    print(f"  检查项:")
    for check in health["checks"]:
        print(f"    {check['name']}: {check['status']} ({check['message']})")

    alerts = monitor.get_active_alerts()
    print(f"  活跃告警: {len(alerts)}")

    perf_report = monitor.get_performance_report()
    print(f"\n  性能指标:")
    print(
        f"    API延迟 - 最小: {perf_report['metrics'].get('api.latency_ms', {}).get('min', 0):.1f}ms"
    )
    print(
        f"    API延迟 - 最大: {perf_report['metrics'].get('api.latency_ms', {}).get('max', 0):.1f}ms"
    )
    print(
        f"    API延迟 - 平均: {perf_report['metrics'].get('api.latency_ms', {}).get('avg', 0):.1f}ms"
    )

    print("\n\n✅ 模块演示完成!")
    print("核心理念：过程即幻觉，I/O即实相")


if __name__ == "__main__":
    asyncio.run(demo_modules())
