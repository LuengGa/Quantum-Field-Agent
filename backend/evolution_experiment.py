"""
协作实验脚本 - 验证进化层功能
============================

运行以下实验：
1. 交互记录实验
2. 模式挖掘实验
3. 策略进化实验
4. 假设验证实验
5. 知识综合实验
6. 完整进化周期实验
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution import EvolutionEngine, EvolutionDatabase, EvolutionConfig
from evolution.pattern_miner import PatternMiner
from evolution.strategy_evolver import StrategyEvolver
from evolution.hypothesis_tester import HypothesisTester
from evolution.knowledge_synthesizer import KnowledgeSynthesizer
from evolution.capability_builder import CapabilityBuilder


async def run_interaction_experiment(engine: EvolutionEngine):
    """实验1: 交互记录"""
    print("\n" + "=" * 50)
    print("实验1: 交互记录")
    print("=" * 50)

    interactions = [
        {
            "user_id": "user_001",
            "session_id": "session_001",
            "interaction_type": "question_answering",
            "input_summary": "用户询问量子场理论",
            "output_summary": "解释了量子场理论的基本概念",
            "outcome": "success",
            "effectiveness": 0.85,
        },
        {
            "user_id": "user_001",
            "session_id": "session_001",
            "interaction_type": "question_answering",
            "input_summary": "用户询问波粒二象性",
            "output_summary": "解释了波粒二象性的实验基础",
            "outcome": "success",
            "effectiveness": 0.90,
        },
        {
            "user_id": "user_002",
            "session_id": "session_002",
            "interaction_type": "code_generation",
            "input_summary": "用户请求编写量子模拟器",
            "output_summary": "提供了Python量子模拟器代码",
            "outcome": "success",
            "effectiveness": 0.75,
        },
        {
            "user_id": "user_002",
            "session_id": "session_002",
            "interaction_type": "code_generation",
            "input_summary": "用户请求添加新功能",
            "output_summary": "为模拟器添加了纠缠功能",
            "outcome": "success",
            "effectiveness": 0.80,
        },
        {
            "user_id": "user_003",
            "session_id": "session_003",
            "interaction_type": "analysis",
            "input_summary": "用户请求分析复杂系统",
            "output_summary": "提供了系统分析报告",
            "outcome": "partial",
            "effectiveness": 0.60,
        },
    ]

    for interaction in interactions:
        await engine.process_interaction(**interaction)

    print(f"✓ 记录了 {len(interactions)} 个交互")
    print(f"  - 成功: {sum(1 for i in interactions if i['outcome'] == 'success')}")
    print(f"  - 部分成功: {sum(1 for i in interactions if i['outcome'] == 'partial')}")

    return len(interactions)


async def run_pattern_mining_experiment(engine: EvolutionEngine):
    """实验2: 模式挖掘"""
    print("\n" + "=" * 50)
    print("实验2: 模式挖掘")
    print("=" * 50)

    result = await engine.run_pattern_mining()

    print(f"状态: {result.get('status', 'unknown')}")
    print(f"发现模式: {result.get('total_found', 0)}")

    if "time_patterns" in result:
        print(f"  - 时间模式: {result['time_patterns']}")
    if "causality_patterns" in result:
        print(f"  - 因果模式: {result['causality_patterns']}")
    if "sequence_patterns" in result:
        print(f"  - 序列模式: {result['sequence_patterns']}")
    if "clustering_patterns" in result:
        print(f"  - 聚类模式: {result['clustering_patterns']}")
    if "anomaly_patterns" in result:
        print(f"  - 异常模式: {result['anomaly_patterns']}")

    return result.get("total_found", 0)


async def run_strategy_evolution_experiment(engine: EvolutionEngine):
    """实验3: 策略进化"""
    print("\n" + "=" * 50)
    print("实验3: 策略进化")
    print("=" * 50)

    stats_before = engine.strategy_evolver.get_strategy_statistics()
    print(f"进化前策略数: {stats_before.get('total_strategies', 0)}")
    print(f"活跃策略数: {stats_before.get('active_strategies', 0)}")
    print(f"平均效果: {stats_before.get('avg_effectiveness', 0):.2f}")

    result = await engine.run_strategy_evolution()

    stats_after = engine.strategy_evolver.get_strategy_statistics()
    print(f"\n进化后策略数: {stats_after.get('total_strategies', 0)}")
    print(f"进化数量: {result.get('evolved_count', 0)}")

    return result.get("evolved_count", 0)


async def run_hypothesis_experiment(engine: EvolutionEngine):
    """实验4: 假设验证"""
    print("\n" + "=" * 50)
    print("实验4: 假设验证")
    print("=" * 50)

    stats = engine.hypothesis_tester.get_hypothesis_statistics()
    print(f"假设总数: {stats.get('total_hypotheses', 0)}")
    print(f"待验证: {stats.get('pending', 0)}")
    print(f"已确认: {stats.get('confirmed', 0)}")
    print(f"已拒绝: {stats.get('rejected', 0)}")

    hypotheses = engine.hypothesis_tester.get_pending_hypotheses()
    print(f"\n待验证假设数: {len(hypotheses)}")

    for h in hypotheses[:3]:
        print(f"  - {h.get('statement', '')[:50]}...")

    return len(hypotheses)


async def run_knowledge_experiment(engine: EvolutionEngine):
    """实验5: 知识综合"""
    print("\n" + "=" * 50)
    print("实验5: 知识综合")
    print("=" * 50)

    stats_before = engine.knowledge_synthesizer.get_knowledge_statistics()
    print(f"综合前知识数: {stats_before.get('total_knowledge', 0)}")
    print(f"平均置信度: {stats_before.get('avg_confidence', 0):.2f}")

    result = await engine.run_knowledge_synthesis()

    stats_after = engine.knowledge_synthesizer.get_knowledge_statistics()
    print(f"\n综合后知识数: {stats_after.get('total_knowledge', 0)}")
    print(f"新增知识: {result.get('synthesized_count', 0)}")

    return result.get("synthesized_count", 0)


async def run_capability_experiment(engine: EvolutionEngine):
    """实验6: 能力构建"""
    print("\n" + "=" * 50)
    print("实验6: 能力构建")
    print("=" * 50)

    stats_before = engine.capability_builder.get_capability_statistics()
    print(f"构建前能力数: {stats_before.get('total_capabilities', 0)}")
    print(f"活跃能力: {stats_before.get('active_capabilities', 0)}")

    result = await engine.run_capability_building()

    stats_after = engine.capability_builder.get_capability_statistics()
    print(f"\n构建后能力数: {stats_after.get('total_capabilities', 0)}")
    print(f"新增能力: {result.get('built_count', 0)}")

    capabilities = engine.capability_builder.list_capabilities()
    print(f"\n可用能力:")
    for cap in capabilities[:5]:
        print(f"  - {cap.name} ({cap.category})")

    return result.get("built_count", 0)


async def run_full_evolution_cycle(engine: EvolutionEngine):
    """实验7: 完整进化周期"""
    print("\n" + "=" * 50)
    print("实验7: 完整进化周期")
    print("=" * 50)

    print("运行完整进化周期...")
    cycle = await engine.run_full_evolution_cycle()

    print(f"\n周期ID: {cycle.id}")
    print(f"状态: {cycle.status}")
    print(f"开始时间: {cycle.start_time}")
    print(f"结束时间: {cycle.end_time}")
    print(f"\n发现模式: {cycle.patterns_discovered}")
    print(f"进化策略: {cycle.strategies_evolved}")
    print(f"验证假设: {cycle.hypotheses_tested}")
    print(f"综合知识: {cycle.knowledge_synthesized}")
    print(f"构建能力: {cycle.capabilities_built}")
    print(f"\n整体得分: {cycle.overall_score:.2f}")

    return cycle.overall_score


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Meta Quantum Field Agent - 协作实验")
    print("=" * 60)

    config = EvolutionConfig(
        auto_mine_patterns=False,
        auto_evolve_strategies=False,
        auto_test_hypotheses=False,
        auto_synthesize_knowledge=False,
        enabled=True,
    )

    engine = EvolutionEngine(config=config)

    print("\n进化引擎初始化完成")
    status = engine.get_evolution_status()
    print(f"进化层状态: {status['enabled']}")

    results = {}

    try:
        results["interactions"] = await run_interaction_experiment(engine)

        results["patterns"] = await run_pattern_mining_experiment(engine)

        results["strategies"] = await run_strategy_evolution_experiment(engine)

        results["hypotheses"] = await run_hypothesis_experiment(engine)

        results["knowledge"] = await run_knowledge_experiment(engine)

        results["capabilities"] = await run_capability_experiment(engine)

        results["cycle_score"] = await run_full_evolution_cycle(engine)

        print("\n" + "=" * 60)
        print("实验总结")
        print("=" * 60)
        print(f"记录交互: {results.get('interactions', 0)}")
        print(f"发现模式: {results.get('patterns', 0)}")
        print(f"进化策略: {results.get('strategies', 0)}")
        print(f"待验证假设: {results.get('hypotheses', 0)}")
        print(f"综合知识: {results.get('knowledge', 0)}")
        print(f"构建能力: {results.get('capabilities', 0)}")
        print(f"进化周期得分: {results.get('cycle_score', 0):.2f}")

        print("\n" + "=" * 60)
        print("✓ 所有协作实验完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n实验过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
