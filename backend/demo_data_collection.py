#!/usr/bin/env python3
"""
持续数据收集演示脚本
====================

演示持续数据收集功能：
1. 实时数据收集
2. 定时任务模拟
3. 模式覆盖追踪
4. 数据质量监控
"""

import asyncio
from evolution.database import EvolutionDatabase
from evolution.evolution_engine import EvolutionEngine
from evolution.data_collector import ContinuousDataCollector


async def main():
    print("=" * 60)
    print("持续数据收集演示")
    print("=" * 60)

    # 初始化组件
    print("\n[1] 初始化组件...")
    db = EvolutionDatabase()
    engine = EvolutionEngine(db=db)
    collector = ContinuousDataCollector(db, engine)
    print("✓ 组件初始化完成")

    # 生成合成交互数据
    print("\n[2] 生成合成交互数据...")
    points = collector.generate_synthetic_data(count=20)
    print(f"✓ 生成了 {len(points)} 个交互数据点")

    # 更新模式覆盖
    print("\n[3] 更新模式覆盖...")
    for point in points[:5]:
        if point.payload.get("patterns"):
            for pattern in point.payload["patterns"]:
                ptype = pattern.get("type", "unknown")
                if ptype not in collector.coverage:
                    collector.coverage[ptype] = type(
                        "PatternCoverage",
                        (),
                        {
                            "pattern_type": ptype,
                            "covered": True,
                            "examples": [],
                            "last_seen": point.timestamp,
                            "count": 0,
                            "confidence_sum": 0,
                        },
                    )()
                cov = collector.coverage[ptype]
                cov.count += 1
                cov.confidence_sum += pattern.get("confidence", 0.5)
                cov.last_seen = point.timestamp
                cov.examples.append(
                    {
                        "timestamp": point.timestamp,
                        "input": point.payload.get("input", "")[:50],
                    }
                )
                collector._save_coverage(cov)

    print(f"✓ 更新了 {len(collector.coverage)} 种模式覆盖")

    # 运行定时收集任务
    print("\n[4] 执行定时收集任务...")
    result = await collector.run_scheduled_collection("交互数据收集")
    print(f"✓ 任务结果: {result['status']}")
    if result.get("collected"):
        print(f"  - 收集数量: {result['collected'].get('count', 0)}")

    result = await collector.run_scheduled_collection("模式覆盖分析")
    print(f"✓ 模式分析任务: {result['status']}")

    # 获取收集状态
    print("\n[5] 获取收集状态...")
    status = collector.get_collection_status()
    print(f"✓ 任务数量: {len(status['tasks'])}")
    print(f"✓ 启用任务: {sum(1 for t in status['tasks'] if t['enabled'])}")

    # 获取覆盖报告
    print("\n[6] 获取覆盖报告...")
    coverage = collector.get_coverage_report()
    print(f"✓ 总模式类型: {coverage['total_pattern_types']}")
    print(f"✓ 已覆盖类型: {coverage['covered_types']}")
    print(f"✓ 覆盖率: {coverage['coverage_rate']:.1%}")

    if coverage.get("pattern_types"):
        print("\n模式类型详情:")
        for pt in coverage["pattern_types"]:
            avg_conf = pt["confidence_sum"] / pt["count"] if pt["count"] > 0 else 0
            print(
                f"  - {pt['pattern_type']}: {pt['count']} 个, 平均置信度: {avg_conf:.2f}"
            )

    # 获取质量报告
    print("\n[7] 获取质量报告...")
    quality = collector.get_quality_report()
    print(f"✓ 总体质量分数: {quality['overall_score']:.2f}")
    print(f"✓ 总数据点: {quality['total_points']}")

    if quality.get("by_source"):
        print("\n数据源质量:")
        for source, data in quality["by_source"].items():
            print(f"  - {source}: 分数 {data['avg_score']:.2f}, 数量 {data['count']}")

    # 实时数据收集演示
    print("\n[8] 实时数据收集演示...")
    realtime_point = collector.collect_realtime(
        source="interaction",
        data_type="question_answer",
        payload={
            "input": "什么是量子计算？",
            "output": "量子计算利用量子力学原理...",
            "context": {"user_level": "beginner"},
            "confidence": 0.92,
        },
        user_id="demo_user",
        session_id="demo_session",
    )
    print(f"✓ 实时收集数据点 ID: {realtime_point.id[:8]}...")
    print(f"  - 质量分数: {realtime_point.quality_score:.2f}")

    # 最终状态
    print("\n" + "=" * 60)
    print("最终系统状态")
    print("=" * 60)
    final_status = collector.get_collection_status()
    print(f"活跃收集任务: {sum(1 for t in final_status['tasks'] if t['enabled'])}")
    print(f"数据点总数: {final_status['quality']['total_points']}")
    print(f"平均数据质量: {final_status['quality']['overall_score']:.2f}")
    print(f"模式覆盖率: {final_status['coverage']['coverage_rate']:.1%}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
