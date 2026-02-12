#!/usr/bin/env python3
"""
策略效果追踪演示脚本
====================

演示策略效果追踪功能：
1. 效果记录
2. A/B测试
3. 效果分析
4. 自动选择最优策略
"""

import asyncio
from statistics import mean
from evolution.database import EvolutionDatabase
from evolution.strategy_tracker import StrategyTracker


def generate_sample_effectiveness():
    """生成随机效果数据"""
    strategies = [
        ("strategy_001", "渐进式解释"),
        ("strategy_002", "类比说明"),
        ("strategy_003", "示例驱动"),
        ("strategy_004", "交互式问答"),
        ("strategy_005", "结构化总结"),
    ]

    effects = []
    for sid, sname in strategies:
        for _ in range(20):
            effectiveness = 0.6 + (hash(sid + str(_)) % 100) / 500
            success = effectiveness > 0.65
            effects.append(
                {
                    "strategy_id": sid,
                    "strategy_name": sname,
                    "effectiveness": round(effectiveness, 2),
                    "success": success,
                    "context_type": ["question_answer", "explanation", "creative"][
                        _ % 3
                    ],
                }
            )
    return effects


async def main():
    print("=" * 60)
    print("策略效果追踪演示")
    print("=" * 60)

    # 初始化
    print("\n[1] 初始化追踪器...")
    db = EvolutionDatabase()
    tracker = StrategyTracker(db)
    print("✓ 追踪器初始化完成")

    # 记录效果数据
    print("\n[2] 记录策略效果数据...")
    effects_data = generate_sample_effectiveness()

    for i, data in enumerate(effects_data):
        tracker.record_effectiveness(
            strategy_id=data["strategy_id"],
            strategy_name=data["strategy_name"],
            effectiveness=data["effectiveness"],
            success=data["success"],
            context_type=data["context_type"],
            user_id=f"user_{i % 10}",
            session_id=f"session_{i % 5}",
            interaction_id=f"int_{i:04d}",
            response_time=0.1 + (i % 10) * 0.02,
            feedback_score=3 + int(data["effectiveness"] * 2),
        )

    print(f"✓ 记录了 {len(effects_data)} 条效果数据")

    # 获取所有策略指标
    print("\n[3] 获取策略指标...")
    all_metrics = tracker.get_all_metrics()
    print(f"✓ 追踪了 {len(all_metrics)} 个策略")

    for metrics in all_metrics[:5]:
        print(f"\n  策略: {metrics.strategy_name}")
        print(f"    使用次数: {metrics.total_uses}")
        print(f"    成功率: {metrics.successful_uses}/{metrics.total_uses}")
        print(f"    平均效果: {metrics.avg_effectiveness:.2f}")
        print(f"    效果趋势: {metrics.trend}")
        print(f"    响应时间: {metrics.avg_response_time:.3f}s")

    # 创建A/B测试
    print("\n[4] 创建A/B测试实验...")

    strategy_ids = [m.strategy_id for m in all_metrics]

    if len(strategy_ids) >= 2:
        exp1 = tracker.create_ab_experiment(
            name="解释策略对比",
            group_a=strategy_ids[0],
            group_b=strategy_ids[1],
            description="比较渐进式解释和类比说明的效果",
            traffic_split=0.5,
            min_sample_size=10,
        )
        print(f"✓ 创建实验: {exp1.name}")
        print(f"  实验ID: {exp1.id[:8]}...")

        # 模拟实验数据
        tracker.start_experiment(exp1.id)

        # 添加更多数据用于测试
        for i in range(15):
            tracker.record_effectiveness(
                strategy_id=exp1.group_a,
                strategy_name="渐进式解释",
                effectiveness=0.7 + (i % 10) / 100,
                success=True,
                context_type="question_answer",
            )
            tracker.record_effectiveness(
                strategy_id=exp1.group_b,
                strategy_name="类比说明",
                effectiveness=0.75 + (i % 10) / 100,
                success=True,
                context_type="question_answer",
            )

        # 结束实验
        print("\n[5] 运行A/B测试分析...")
        result = tracker.end_experiment(exp1.id)
        print(f"✓ 实验完成!")
        print(f"  胜出策略: {result.get('winner', 'N/A')}")
        print(f"  置信度: {result.get('confidence', 0):.1f}%")
        print(f"  A组效果: {result.get('effect_a', 0):.3f}")
        print(f"  B组效果: {result.get('effect_b', 0):.3f}")
        print(
            f"  样本A: {result.get('sample_a', 0)}, 样本B: {result.get('sample_b', 0)}"
        )
    else:
        print("  策略数量不足，跳过A/B测试")

    # 获取实验状态
    print("\n[6] 获取实验状态...")
    exp_status = tracker.get_experiment_status()
    print(f"✓ 总实验数: {exp_status['total_experiments']}")
    print(f"✓ 运行中: {exp_status['running_experiments']}")
    print(f"✓ 已完成: {exp_status['completed_experiments']}")

    # 获取效果报告
    print("\n[7] 生成效果报告...")
    report = tracker.get_effectiveness_report(days=30)
    print(f"✓ 报告周期: {report['period_days']}天")
    print(f"✓ 总记录数: {report['total_records']}")

    if report.get("best_strategy"):
        best = report["best_strategy"]
        print(f"\n  最佳策略: {best['strategy_name']}")
        print(f"    样本量: {best['sample_size']}")
        print(f"    平均效果: {best['avg_effectiveness']:.3f}")
        print(f"    成功率: {best['success_rate']:.1%}")

    # 测试自动选择策略
    print("\n[8] 测试自动策略选择...")
    if strategy_ids:
        best = await tracker.select_best_strategy(
            strategy_ids[:3],
            context_type="question_answer",
        )
        print(f"✓ 为question_answer上下文选择: {best}")

    # 展示按效果排序的策略
    print("\n[9] 策略效果排名...")
    sorted_metrics = sorted(
        all_metrics, key=lambda x: x.avg_effectiveness, reverse=True
    )

    for i, metrics in enumerate(sorted_metrics[:5], 1):
        trend_icon = {"improving": "↑", "declining": "↓", "stable": "→"}[metrics.trend]
        print(
            f"  {i}. {metrics.strategy_name}: {metrics.avg_effectiveness:.3f} {trend_icon}"
        )

    # 最终状态
    print("\n" + "=" * 60)
    print("最终追踪状态")
    print("=" * 60)
    final_metrics = tracker.get_all_metrics()
    total_uses = sum(m.total_uses for m in final_metrics)
    avg_effectiveness = (
        mean([m.avg_effectiveness for m in final_metrics]) if final_metrics else 0
    )

    print(f"追踪策略数: {len(final_metrics)}")
    print(f"总使用次数: {total_uses}")
    print(f"平均效果: {avg_effectiveness:.3f}")
    print(f"活跃实验: {exp_status['running_experiments']}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
