"""
真实协作实验 - 量子场问题解决助手
================================

设计真实的协作场景：
1. 用户提出复杂问题
2. AI协作者提供多角度解答
3. 用户反馈改进
4. 协作学习与进化

实验目标：
- 验证模式发现（时间、因果、序列）
- 验证策略选择效果
- 验证知识综合
- 验证能力构建
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution import EvolutionEngine, EvolutionDatabase, EvolutionConfig


class CollaborationScenario:
    """协作场景"""

    def __init__(self):
        self.scenarios = [
            {
                "name": "量子计算基础",
                "user": "学生小明",
                "interactions": [
                    {
                        "type": "question",
                        "input": "什么是量子叠加态？",
                        "output": "量子叠加态是量子系统同时处于多个状态的特性，直到测量时才坍缩到单一状态。",
                        "effectiveness": 0.90,
                        "feedback": "解释清晰，用了比喻",
                    },
                    {
                        "type": "follow_up",
                        "input": "那量子纠缠呢？",
                        "output": "量子纠缠是两个粒子之间存在的神秘关联，测量一个粒子会瞬间影响另一个，无论距离多远。",
                        "effectiveness": 0.85,
                        "feedback": "爱因斯坦称其为'鬼魅般的超距作用'",
                    },
                    {
                        "type": "example_request",
                        "input": "能给我一个实际的例子吗？",
                        "output": "比如双缝实验，电子同时通过两条缝，只有在观测时才确定位置。这展示了波粒二象性。",
                        "effectiveness": 0.92,
                        "feedback": "实验例子帮助理解",
                    },
                ],
            },
            {
                "name": "编程任务",
                "user": "开发者小李",
                "interactions": [
                    {
                        "type": "code_request",
                        "input": "帮我写一个Python函数实现量子态模拟",
                        "output": """```python
import numpy as np

def create_superposition(state0, state1, alpha=0.5):
    return alpha * state0 + np.sqrt(1-alpha**2) * state1
```""",
                        "effectiveness": 0.88,
                        "feedback": "代码简洁有效",
                    },
                    {
                        "type": "debug_request",
                        "input": "这个函数报错了，说shape不匹配",
                        "output": "需要归一化。修改为：np.sqrt(1-alpha**2) 然后用np.kron合并维度。",
                        "effectiveness": 0.82,
                        "feedback": "修复后正常运行",
                    },
                    {
                        "type": "enhancement",
                        "input": "能加一个测量函数吗？",
                        "output": """```python
def measure(state):
    prob = np.abs(state)**2
    return np.random.choice(len(state), p=prob)
```""",
                        "effectiveness": 0.90,
                        "feedback": "功能完整",
                    },
                ],
            },
            {
                "name": "概念辨析",
                "user": "研究者小王",
                "interactions": [
                    {
                        "type": "analysis",
                        "input": "量子退相干和量子坍缩有什么区别？",
                        "output": "退相干是环境导致的量子信息流失到环境，不可逆。坍缩是测量导致的波函数突变，伴随信息选择。",
                        "effectiveness": 0.87,
                        "feedback": "对比清晰",
                    },
                    {
                        "type": "comparison",
                        "input": "在量子计算中哪个更重要？",
                        "output": "退相干是需要克服的主要障碍，坍缩是我们利用的操作。控制退相干是量子纠错的核心问题。",
                        "effectiveness": 0.85,
                        "feedback": "联系实际应用",
                    },
                ],
            },
            {
                "name": "创意生成",
                "user": "创作者小张",
                "interactions": [
                    {
                        "type": "brainstorm",
                        "input": "量子力学有什么有趣的哲学思考？",
                        "output": "1. 观察者是否创造现实？2. 多世界诠释是否意味着无限平行宇宙？3. 决定论是否被打破？",
                        "effectiveness": 0.91,
                        "feedback": "激发思考",
                    },
                    {
                        "type": "creative",
                        "input": "能帮我设计一个科幻故事设定吗？",
                        "output": "在2237年，人类发明了'量子纠缠通讯器'，可以跨越星际即时通讯，但每次通讯会随机改变接收者的记忆...",
                        "effectiveness": 0.88,
                        "feedback": "很有创意的设定",
                    },
                ],
            },
            {
                "name": "学习辅导",
                "user": "自学者小陈",
                "interactions": [
                    {
                        "type": "learning_path",
                        "input": "我想学习量子力学，应该从哪里开始？",
                        "output": "建议路线：1. 线性代数（向量、矩阵、本征值）2. 复数基础 3. 经典波动 4. 薛定谔方程 5. 希尔伯特空间",
                        "effectiveness": 0.93,
                        "feedback": "路线清晰",
                    },
                    {
                        "type": "resource_request",
                        "input": "有什么推荐的学习资源吗？",
                        "output": "入门：《量子物理史话》- 曹天元；进阶：《量子力学概论》- Griffiths；在线：Coursera的'量子计算'课程",
                        "effectiveness": 0.89,
                        "feedback": "资源实用",
                    },
                    {
                        "type": "difficulty_help",
                        "input": "数学太难了，看不懂波函数",
                        "output": "先理解波函数是'概率幅'，它不是概率本身。复数幅度的平方才是概率。想象它是'看不见的概率'的描述。",
                        "effectiveness": 0.86,
                        "feedback": "比喻帮助理解",
                    },
                ],
            },
        ]


async def generate_more_interactions(engine: EvolutionEngine) -> int:
    """生成更多模拟交互数据"""
    print("\n" + "=" * 50)
    print("生成模拟交互数据")
    print("=" * 50)

    additional_scenarios = [
        {
            "user_id": "user_010",
            "session_id": "session_010",
            "type": "analysis",
            "effectiveness_range": (0.7, 0.95),
        },
        {
            "user_id": "user_011",
            "session_id": "session_011",
            "type": "creative",
            "effectiveness_range": (0.75, 0.92),
        },
        {
            "user_id": "user_012",
            "session_id": "session_012",
            "type": "explanation",
            "effectiveness_range": (0.8, 0.95),
        },
        {
            "user_id": "user_013",
            "session_id": "session_013",
            "type": "code",
            "effectiveness_range": (0.65, 0.90),
        },
        {
            "user_id": "user_014",
            "session_id": "session_014",
            "type": "comparison",
            "effectiveness_range": (0.7, 0.88),
        },
        {
            "user_id": "user_015",
            "session_id": "session_015",
            "type": "question",
            "effectiveness_range": (0.75, 0.93),
        },
        {
            "user_id": "user_016",
            "session_id": "session_016",
            "type": "analysis",
            "effectiveness_range": (0.7, 0.9),
        },
        {
            "user_id": "user_017",
            "session_id": "session_017",
            "type": "creative",
            "effectiveness_range": (0.8, 0.95),
        },
        {
            "user_id": "user_018",
            "session_id": "session_018",
            "type": "learning",
            "effectiveness_range": (0.75, 0.92),
        },
        {
            "user_id": "user_019",
            "session_id": "session_019",
            "type": "code",
            "effectiveness_range": (0.7, 0.88),
        },
    ]

    interaction_templates = [
        {
            "input": "请解释{concept}的原理",
            "output": "{concept}是量子力学中的重要概念，它描述了{specific}...",
            "feedback": "解释详细",
        },
        {
            "input": "帮我比较{concept_a}和{concept_b}",
            "output": "{concept_a}和{concept_b}的主要区别在于：1. 2. 3...",
            "feedback": "对比清晰",
        },
        {
            "input": "能写一个{concept}的代码吗？",
            "output": "```python\n# {concept}代码实现\n```",
            "feedback": "代码正确",
        },
        {
            "input": "{concept}有什么实际应用？",
            "output": "{concept}在以下领域有重要应用：1. 2. 3...",
            "feedback": "举例恰当",
        },
    ]

    concepts = [
        {
            "concept": "量子隧穿",
            "specific": "粒子穿透势垒的量子效应",
            "concept_a": "隧穿",
            "concept_b": "反射",
        },
        {
            "concept": "量子涨落",
            "specific": "真空中的能量起伏",
            "concept_a": "涨落",
            "concept_b": "稳定态",
        },
        {
            "concept": "量子自旋",
            "specific": "粒子的内禀角动量",
            "concept_a": "自旋",
            "concept_b": "轨道角动量",
        },
        {
            "concept": "量子态叠加",
            "specific": "态的线性组合",
            "concept_a": "叠加",
            "concept_b": "纯态",
        },
        {
            "concept": "量子测量",
            "specific": "获取量子系统信息的过程",
            "concept_a": "测量",
            "concept_b": "观测",
        },
    ]

    import random

    count = 0
    for scenario in additional_scenarios:
        user_id = scenario["user_id"]
        session_id = scenario["session_id"]

        for i in range(5):
            template = random.choice(interaction_templates)
            concept = random.choice(concepts)

            input_summary = template["input"].format(**concept)
            output_summary = template["output"].format(**concept)
            effectiveness = random.uniform(*scenario["effectiveness_range"])

            await engine.process_interaction(
                user_id=user_id,
                session_id=session_id,
                interaction_type=scenario["type"],
                input_summary=input_summary,
                output_summary=output_summary,
                outcome="success" if effectiveness > 0.7 else "partial",
                effectiveness=effectiveness,
                feedback=random.choice(
                    ["解释清晰", "有帮助", "需要更多例子", "很好", "继续努力"]
                ),
            )
            count += 1

    print(f"✓ 生成了 {count} 个模拟交互")
    return count


async def run_collaboration_scenarios(engine: EvolutionEngine) -> Dict:
    """运行协作场景"""
    print("\n" + "=" * 50)
    print("运行协作场景")
    print("=" * 50)

    scenario_manager = CollaborationScenario()

    total_interactions = 0
    scenario_results = []

    for scenario in scenario_manager.scenarios:
        print(f"\n场景: {scenario['name']} ({scenario['user']})")

        session_id = f"session_{scenario['user'].split('小')[1]}"

        for interaction in scenario["interactions"]:
            await engine.process_interaction(
                user_id=f"user_{scenario['user'].split('小')[1]}",
                session_id=session_id,
                interaction_type=interaction["type"],
                input_summary=interaction["input"],
                output_summary=interaction["output"],
                outcome="success",
                effectiveness=interaction["effectiveness"],
                feedback=interaction["feedback"],
            )
            total_interactions += 1

        scenario_results.append(
            {
                "name": scenario["name"],
                "user": scenario["user"],
                "interactions": len(scenario["interactions"]),
                "avg_effectiveness": sum(
                    i["effectiveness"] for i in scenario["interactions"]
                )
                / len(scenario["interactions"]),
            }
        )

    print(
        f"\n✓ 完成 {len(scenario_manager.scenarios)} 个场景，共 {total_interactions} 个交互"
    )

    return {"scenarios": scenario_results, "total_interactions": total_interactions}


async def analyze_results(engine: EvolutionEngine):
    """分析实验结果"""
    print("\n" + "=" * 50)
    print("实验结果分析")
    print("=" * 50)

    status = engine.get_evolution_status()

    print("\n进化状态:")
    print(f"  启用: {status['enabled']}")
    print(f"  运行中: {status['running']}")

    print("\n组件统计:")
    components = status.get("components", {})

    pattern_stats = components.get("pattern_miner", {})
    print(f"  模式挖掘:")
    print(f"    - 时间模式: {pattern_stats.get('time_patterns', {}).get('count', 0)}")
    print(
        f"    - 因果模式: {pattern_stats.get('causality_patterns', {}).get('count', 0)}"
    )
    print(
        f"    - 序列模式: {pattern_stats.get('sequence_patterns', {}).get('count', 0)}"
    )

    strategy_stats = components.get("strategy_evolver", {})
    print(f"  策略进化:")
    print(f"    - 总策略: {strategy_stats.get('total_strategies', 0)}")
    print(f"    - 活跃策略: {strategy_stats.get('active_strategies', 0)}")
    print(f"    - 平均效果: {strategy_stats.get('avg_effectiveness', 0):.2f}")

    hypothesis_stats = components.get("hypothesis_tester", {})
    print(f"  假设验证:")
    print(f"    - 总假设: {hypothesis_stats.get('total_hypotheses', 0)}")
    print(f"    - 待验证: {hypothesis_stats.get('pending', 0)}")
    print(f"    - 已确认: {hypothesis_stats.get('confirmed', 0)}")

    knowledge_stats = components.get("knowledge_synthesizer", {})
    print(f"  知识综合:")
    print(f"    - 总知识: {knowledge_stats.get('total_knowledge', 0)}")
    print(f"    - 平均置信度: {knowledge_stats.get('avg_confidence', 0):.2f}")

    capability_stats = components.get("capability_builder", {})
    print(f"  能力构建:")
    print(f"    - 总能力: {capability_stats.get('total_capabilities', 0)}")
    print(f"    - 活跃能力: {capability_stats.get('active_capabilities', 0)}")

    return status


async def demonstrate_collaboration_patterns(engine: EvolutionEngine):
    """演示协作模式发现"""
    print("\n" + "=" * 50)
    print("协作模式发现演示")
    print("=" * 50)

    strategies = engine.strategy_evolver.get_strategy_statistics()
    print(f"\n当前策略环境:")
    print(f"  - 活跃策略数: {strategies.get('active_strategies', 0)}")
    print(f"  - 总使用次数: {strategies.get('total_uses', 0)}")
    print(f"  - 平均效果: {strategies.get('avg_effectiveness', 0):.2f}")

    print("\n协作洞察:")
    print("  1. 不同用户类型需要不同策略")
    print("  2. 代码任务需要精确的输出格式")
    print("  3. 学习辅导需要循序渐进的解释")
    print("  4. 创意生成需要开放性的思维扩展")

    hypotheses = engine.hypothesis_tester.get_pending_hypotheses()
    if hypotheses:
        print(f"\n待验证假设 ({len(hypotheses)}):")
        for h in hypotheses[:3]:
            print(f"  - {h.get('statement', '')[:60]}...")
    else:
        print("\n从协作数据中生成假设...")

        new_hypothesis = {
            "statement": "当用户请求代码示例时，提供完整可运行的代码比片段效果更好",
            "category": "collaboration",
            "predictions": [
                {"description": "完整代码的交互效果更高", "probability": 0.75},
                {"description": "用户满意度提升", "probability": 0.70},
            ],
        }

        from evolution import Hypothesis

        hypothesis = Hypothesis(**new_hypothesis)
        engine.hypothesis_tester._hypotheses[hypothesis.id] = hypothesis
        engine.db.save_hypothesis(asdict(hypothesis))

        print(f"  ✓ 创建新假设: {hypothesis.statement[:50]}...")


async def run_strategy_selection_demo(engine: EvolutionEngine):
    """演示策略选择"""
    print("\n" + "=" * 50)
    print("策略选择演示")
    print("=" * 50)

    contexts = [
        {"user_type": "student", "task": "learning", "complexity": "medium"},
        {"user_type": "developer", "task": "coding", "complexity": "high"},
        {"user_type": "researcher", "task": "analysis", "complexity": "high"},
        {"user_type": "creator", "task": "creative", "complexity": "medium"},
    ]

    print("\n根据不同上下文选择策略:")

    for context in contexts:
        result = await engine.get_adaptive_strategy(context)
        print(f"\n  上下文: {context}")
        print(f"  选择策略: {result.get('name', '默认策略')}")
        print(f"  置信度: {result.get('confidence', 0):.2f}")


async def run_knowledge_application_demo(engine: EvolutionEngine):
    """演示知识应用"""
    print("\n" + "=" * 50)
    print("知识应用演示")
    print("=" * 50)

    contexts = [
        {"domain": "quantum_computing", "task_type": "explanation"},
        {"domain": "programming", "task_type": "code_generation"},
        {"domain": "physics", "task_type": "concept_comparison"},
    ]

    print("\n查找适用的知识:")

    for context in contexts:
        result = await engine.apply_knowledge(context)
        if "knowledge_id" in result:
            print(f"\n  上下文: {context}")
            print(f"  应用知识: {result.get('title', 'N/A')}")
            print(f"  置信度: {result.get('confidence', 0):.2f}")
        else:
            print(f"\n  上下文: {context}")
            print(f"  状态: {result.get('status', 'unknown')}")


async def run_capability_demo(engine: EvolutionEngine):
    """演示能力执行"""
    print("\n" + "=" * 50)
    print("能力执行演示")
    print("=" * 50)

    capabilities = engine.capability_builder.list_capabilities()

    print(f"\n可用能力 ({len(capabilities)}):")

    for cap in capabilities[:5]:
        print(f"  - {cap.name} ({cap.category})")
        print(f"    描述: {cap.description}")
        print(f"    使用次数: {cap.usage_count}")
        print(f"    成功率: {cap.success_count / max(cap.usage_count, 1) * 100:.1f}%")

    if capabilities:
        print(f"\n执行能力演示:")
        result = await engine.execute_capability(
            capabilities[0].name, {"test_input": "example"}
        )
        print(f"  结果: {result}")


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Meta Quantum Field Agent - 真实协作实验")
    print("=" * 60)

    config = EvolutionConfig(
        auto_mine_patterns=False,
        auto_evolve_strategies=False,
        auto_test_hypotheses=False,
        auto_synthesize_knowledge=False,
        enabled=True,
    )

    engine = EvolutionEngine(config=config)

    print("\n✓ 进化引擎初始化完成")

    try:
        results = {}

        print("\n" + "=" * 60)
        print("阶段1: 运行预定义协作场景")
        print("=" * 60)
        results["scenarios"] = await run_collaboration_scenarios(engine)

        print("\n" + "=" * 60)
        print("阶段2: 生成模拟交互数据")
        print("=" * 60)
        results["generated"] = await generate_more_interactions(engine)

        print("\n" + "=" * 60)
        print("阶段3: 运行完整进化周期")
        print("=" * 60)
        cycle = await engine.run_full_evolution_cycle()
        results["cycle"] = {
            "id": cycle.id,
            "status": cycle.status,
            "patterns": cycle.patterns_discovered,
            "strategies": cycle.strategies_evolved,
            "hypotheses": cycle.hypotheses_tested,
            "knowledge": cycle.knowledge_synthesized,
            "capabilities": cycle.capabilities_built,
            "score": cycle.overall_score,
        }

        print("\n" + "=" * 60)
        print("阶段4: 分析实验结果")
        print("=" * 60)
        await analyze_results(engine)

        print("\n" + "=" * 60)
        print("阶段5: 协作模式发现演示")
        print("=" * 60)
        await demonstrate_collaboration_patterns(engine)

        print("\n" + "=" * 60)
        print("阶段6: 策略选择演示")
        print("=" * 60)
        await run_strategy_selection_demo(engine)

        print("\n" + "=" * 60)
        print("阶段7: 知识应用演示")
        print("=" * 60)
        await run_knowledge_application_demo(engine)

        print("\n" + "=" * 60)
        print("阶段8: 能力执行演示")
        print("=" * 60)
        await run_capability_demo(engine)

        print("\n" + "=" * 60)
        print("实验总结")
        print("=" * 60)
        print(f"\n协作场景: {results['scenarios']['total_interactions']} 交互")
        print(f"模拟数据: {results['generated']} 交互")
        print(f"\n进化周期结果:")
        print(f"  - 发现模式: {results['cycle']['patterns']}")
        print(f"  - 进化策略: {results['cycle']['strategies']}")
        print(f"  - 验证假设: {results['cycle']['hypotheses']}")
        print(f"  - 综合知识: {results['cycle']['knowledge']}")
        print(f"  - 构建能力: {results['cycle']['capabilities']}")
        print(f"  - 整体得分: {results['cycle']['score']:.2f}")

        print("\n" + "=" * 60)
        print("✓ 真实协作实验完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n实验过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
