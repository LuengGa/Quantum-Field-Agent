#!/usr/bin/env python3
"""Meta Quantum Field Agent - 完整系统演示"""

import asyncio, sqlite3, random
from datetime import datetime
from evolution.database import EvolutionDatabase
from evolution.pattern_miner import PatternMiner
from evolution.strategy_evolver import StrategyEvolver
from evolution.hypothesis_tester import HypothesisTester
from evolution.data_collector import ContinuousDataCollector
from evolution.strategy_tracker import StrategyTracker
from evolution.hypothesis_validator import HypothesisValidator


def gen(db):
    c = sqlite3.connect(db.db_path)
    cur = c.cursor()
    ts = datetime.now().isoformat()
    for name, sid in [
        ("渐进式解释", "str_001"),
        ("类比说明", "str_002"),
        ("示例驱动", "str_003"),
    ]:
        eff = 0.8 + random.random() * 0.1
        cur.execute(
            "INSERT OR REPLACE INTO strategies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [sid, name, "explanation", "{}", "[]", "{}", 0, 0.8, eff, 0, ts, ts, 1],
        )
    for hid, stmt in [
        ("hyp_001", "渐进式解释能提高理解度"),
        ("hyp_002", "类比说明效果更好"),
    ]:
        cur.execute(
            "INSERT OR REPLACE INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [hid, stmt, "explanation", "[]", "[]", "pending", 3, 0.5, 3, ts, ts],
        )
    for pid, name, ptype in [
        ("pat_001", "时间模式", "time_pattern"),
        ("pat_002", "因果模式", "causality_pattern"),
    ]:
        cur.execute(
            "INSERT OR REPLACE INTO patterns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [pid, name, ptype, "{}", name, 10, 0.8, 0.85, ts, ts, "{}"],
        )
    c.commit()
    c.close()


async def main():
    print("=" * 70)
    print("Meta Quantum Field Agent - 完整系统演示")
    print("过程即幻觉，I/O即实相")
    print("=" * 70)
    db = EvolutionDatabase()
    gen(db)
    collector = ContinuousDataCollector(db)
    miner = PatternMiner(db)
    evolver = StrategyEvolver(db)
    tester = HypothesisTester(db)
    tracker = StrategyTracker(db)
    validator = HypothesisValidator(db, collector)
    print("\n✓ 初始化完成")
    print("\n[1] 模式挖掘...")
    p = await miner.mine_patterns()
    print(f"✓ 发现 {p.get('total_patterns', 0)} 个模式")
    s = evolver.get_strategy_statistics()
    print(
        f"\n[2] 策略: {s['total_strategies']}个, 活跃{s['active_strategies']}, 效果{s['avg_effectiveness']:.2f}"
    )
    for i in range(30):
        tracker.record_effectiveness(
            f"str_00{(i % 3) + 1}",
            ["渐进式解释", "类比说明", "示例驱动"][i % 3],
            0.6 + random.random() * 0.3,
            random.random() > 0.3,
            "test",
        )
    m = sorted(
        tracker.get_all_metrics(), key=lambda x: x.avg_effectiveness, reverse=True
    )[:3]
    print(f"\n[3] 策略效果 (Top 3):")
    [
        print(f"  {x.strategy_name}: {x.avg_effectiveness:.2f} ({x.total_uses}次)")
        for x in m
    ]
    h = tester.get_hypothesis_statistics()
    print(
        f"\n[4] 假设: {h['total_hypotheses']}, 待验证{h['pending']}, 已确认{h['confirmed']}"
    )
    cur = sqlite3.connect(db.db_path).cursor()
    cur.execute("SELECT id FROM hypotheses LIMIT 2")
    for (hid,) in cur.fetchall():
        v = await validator.validate_hypothesis(hid, "automatic")
        print(f"\n[5] 验证: 置信度{v.confidence_score:.2f}, 结果:{v.validation_result}")
    points = collector.generate_synthetic_data(20)
    cov = collector.get_coverage_report()
    q = collector.get_quality_report()
    print(
        f"\n[6] 数据: {len(points)}点, 覆盖{cov['coverage_rate']:.1%}, 质量{q['overall_score']:.2f}"
    )
    if len(m) >= 2:
        e = tracker.create_ab_experiment(
            "对比",
            m[0].strategy_id,
            m[1].strategy_id,
            traffic_split=0.5,
            min_sample_size=10,
        )
        print(f"\n[7] A/B测试: {e.name}")
        tracker.start_experiment(e.id)
        for i in range(20):
            for sid, name in [(e.group_a, "A"), (e.group_b, "B")]:
                tracker.record_effectiveness(
                    sid,
                    f"策略{name}",
                    0.7 + random.uniform(-0.1, 0.15)
                    if name == "B"
                    else 0.7 + random.uniform(-0.1, 0.1),
                    random.random() > 0.3,
                    "test",
                )
        r = tracker.end_experiment(e.id)
        print(f"  胜出:{r.get('winner', 'N/A')}, 置信度:{r.get('confidence', 0):.1f}%")
    v = await validator.apply_knowledge_and_verify(
        "know_test",
        "hyp_001",
        {"s": m[0].strategy_name if m else "test"},
        {"be": 0.7, "te": 0.8},
    )
    print(
        f"\n[8] 闭环验证: 改进{v.improvement:+.3f}, 通过:{'✓' if v.verified else '✗'}"
    )
    print("\n" + "=" * 70)
    print("最终状态")
    print("=" * 70)
    print(f"\n📊 模式: {p.get('total_patterns', 0)}")
    print(f"📊 策略: {s['active_strategies']} (效果:{s['avg_effectiveness']:.2f})")
    print(f"📊 假设: {h['total_hypotheses']} (确认:{h['confirmed']})")
    print(f"📊 数据点: {q['total_points']} (质量:{q['overall_score']:.2f})")
    print(f"📊 模式覆盖: {cov['coverage_rate']:.1%}")
    exp = tracker.get_experiment_status()
    val = validator.get_validation_status()
    print(f"📊 A/B实验: {exp['completed_experiments']} 完成")
    print(f"📊 闭环验证: {val['closed_loop_verifications']}")
    print("\n✅ Meta Quantum Field Agent 演示完成!")
    print("核心理念：过程即幻觉，I/O即实相")


if __name__ == "__main__":
    asyncio.run(main())
