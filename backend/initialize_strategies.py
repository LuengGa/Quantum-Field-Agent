#!/usr/bin/env python3
"""
策略初始化脚本
==============

初始化预定义协作策略到进化系统中
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution import EvolutionEngine, EvolutionDatabase
from evolution.strategies import initialize_strategy_evolvers


def main():
    print("=" * 60)
    print("策略初始化")
    print("=" * 60)

    engine = EvolutionEngine()

    count = initialize_strategy_evolvers(engine)

    print(f"\n✓ 初始化了 {count} 个预定义策略")

    stats = engine.strategy_evolver.get_strategy_statistics()
    print(f"\n策略统计:")
    print(f"  - 总策略: {stats.get('total_strategies', 0)}")
    print(f"  - 活跃策略: {stats.get('active_strategies', 0)}")
    print(f"  - 平均效果: {stats.get('avg_effectiveness', 0):.2f}")

    print(f"\n按类型分布:")
    for stype, num in stats.get("by_type", {}).items():
        print(f"  - {stype}: {num}")

    print("\n" + "=" * 60)
    print("初始化完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
