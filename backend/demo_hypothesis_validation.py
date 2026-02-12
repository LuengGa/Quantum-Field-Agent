#!/usr/bin/env python3
"""
假设验证自动化演示脚本
======================

演示假设验证自动化功能：
1. 自动验证假设
2. 创建知识
3. 闭环验证
4. 验证报告
"""

import asyncio
import sqlite3
from datetime import datetime
from evolution.database import EvolutionDatabase
from evolution.hypothesis_tester import HypothesisTester
from evolution.hypothesis_validator import HypothesisValidator
from evolution.data_collector import ContinuousDataCollector


def generate_test_hypotheses(db):
    """生成测试假设"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO hypotheses 
        (id, statement, category, predictions, status, test_count, confidence,
         evidence_count, created_at, last_tested)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "hyp_test_001",
            "当使用渐进式解释时，用户理解度会提高",
            "explanation",
            '[{"description": "渐进式解释提高理解", "probability": 0.75}]',
            "pending",
            2,
            0.4,
            2,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ),
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO hypotheses 
        (id, statement, category, predictions, status, test_count, confidence,
         evidence_count, created_at, last_tested)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "hyp_test_002",
            "类比说明能更好地解释复杂概念",
            "explanation",
            '[{"description": "类比提高解释效果", "probability": 0.8}]',
            "pending",
            3,
            0.6,
            3,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ),
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO hypotheses 
        (id, statement, category, predictions, status, test_count, confidence,
         evidence_count, created_at, last_tested)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "hyp_test_003",
            "交互式问答能提升用户参与度",
            "dialogue",
            '[{"description": "问答提升参与度", "probability": 0.7}]',
            "pending",
            1,
            0.3,
            1,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


async def main():
    print("=" * 60)
    print("假设验证自动化演示")
    print("=" * 60)

    print("\n[1] 初始化组件...")
    db = EvolutionDatabase()
    data_collector = ContinuousDataCollector(db)
    tester = HypothesisTester(db)
    validator = HypothesisValidator(db, data_collector)
    print("✓ 组件初始化完成")

    print("\n[2] 生成测试假设...")
    generate_test_hypotheses(db)
    print("✓ 生成了 3 个测试假设")

    print("\n[3] 加载假设...")
    tester._load_hypotheses()
    print(f"✓ 加载了 {len(tester._hypotheses)} 个假设")

    print("\n[4] 模拟实验结果...")
    import random

    for hid in ["hyp_test_001", "hyp_test_002", "hyp_test_003"]:
        hypothesis = tester._hypotheses.get(hid)
        if hypothesis:
            if not isinstance(hypothesis.test_results, list):
                hypothesis.test_results = []
            for i in range(5):
                test_result = {
                    "experiment_id": f"exp_{i}",
                    "success": random.random() > 0.4,
                    "analysis": {"success_rate": random.uniform(0.5, 0.8)},
                    "conclusions": ["测试结论"],
                }
                hypothesis.test_results.append(test_result)
                hypothesis.test_count += 1
                if random.random() > 0.4:
                    if not hasattr(hypothesis, "supporting_evidence"):
                        hypothesis.supporting_evidence = 0
                    hypothesis.supporting_evidence += 1
                else:
                    if not hasattr(hypothesis, "contradicting_evidence"):
                        hypothesis.contradicting_evidence = 0
                    hypothesis.contradicting_evidence += 1
            if not hasattr(hypothesis, "supporting_evidence"):
                hypothesis.supporting_evidence = 0
            if not hasattr(hypothesis, "contradicting_evidence"):
                hypothesis.contradicting_evidence = 0
            hypothesis.evidence_count = (
                hypothesis.supporting_evidence + hypothesis.contradicting_evidence
            )
            if hypothesis.evidence_count > 0:
                hypothesis.confidence = (
                    hypothesis.supporting_evidence / hypothesis.evidence_count
                )
            else:
                hypothesis.confidence = 0.5

    print("✓ 模拟了实验结果")

    print("\n[5] 生成策略效果数据...")
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    for i in range(50):
        cursor.execute(
            """
            INSERT INTO strategy_effectiveness 
            (id, strategy_id, strategy_name, context_type, effectiveness, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                f"eff_{i}",
                f"str_{i % 5}",
                f"策略{i % 5}",
                "explanation",
                random.uniform(0.5, 0.9),
                random.random() > 0.4,
                datetime.now().isoformat(),
            ),
        )
    conn.commit()
    conn.close()
    print("✓ 生成了 50 条效果数据")

    print("\n[6] 自动验证假设...")
    for hid in list(tester._hypotheses.keys()):
        validation = await validator.validate_hypothesis(hid, "automatic")
        if validation:
            print(f"\n  假设 {hid[:8]}...")
            print(f"    样本量: {validation.sample_size}")
            print(f"    支持证据: {validation.supporting_count}")
            print(f"    反驳证据: {validation.contradicting_count}")
            print(f"    置信度: {validation.confidence_score:.2f}")
            print(f"    统计显著性: {validation.statistical_significance:.2f}")
            print(f"    验证结果: {validation.validation_result}")

    print("\n[7] 批量验证待验证假设...")
    batch_result = await validator.validate_all_pending_hypotheses()
    print(f"✓ 已验证: {batch_result['validated']}")
    print(f"  需要更多数据: {batch_result['needs_more_data']}")
    print(f"  错误: {batch_result['errors']}")

    print("\n[8] 执行闭环验证...")
    knowledge_id = "knowledge_001"
    hypothesis_id = list(tester._hypotheses.keys())[0]

    verification = await validator.apply_knowledge_and_verify(
        knowledge_id=knowledge_id,
        hypothesis_id=hypothesis_id,
        application_context={
            "strategy_type": "progressive_explanation",
            "user_level": "beginner",
        },
        expected_outcome={
            "baseline_effectiveness": 0.65,
            "target_effectiveness": 0.75,
            "target_satisfaction": 4.2,
            "target_time": 1.5,
        },
    )

    print(f"✓ 闭环验证完成!")
    print(f"  验证ID: {verification.id[:8]}...")
    print(f"  预期效果: {verification.expected_outcome['target_effectiveness']}")
    print(f"  实际效果: {verification.actual_outcome['effectiveness']:.3f}")
    print(f"  改进幅度: {verification.improvement:+.3f}")
    print(f"  验证通过: {'✓' if verification.verified else '✗'}")

    print("\n[9] 获取验证状态...")
    status = validator.get_validation_status()
    print(f"✓ 总验证数: {status['total_validations']}")
    print(f"✓ 已确认: {status['confirmed']}")
    print(f"✓ 已拒绝: {status['rejected']}")
    print(f"✓ 需要更多数据: {status['needs_more_data']}")
    print(f"✓ 知识更新数: {status['knowledge_updates']}")
    print(f"✓ 闭环验证数: {status['closed_loop_verifications']}")
    print(f"✓ 平均置信度: {status['avg_confidence']:.2f}")

    print("\n[10] 生成验证报告...")
    report = validator.get_validation_report(days=30)
    print(f"✓ 报告周期: {report['period_days']}天")
    print(f"✓ 总验证数: {report['total_validations']}")
    print(f"✓ 平均置信度: {report['avg_confidence']:.2f}")
    print(f"  按结果分类:")
    for result, count in report.get("by_result", {}).items():
        print(f"    - {result}: {count}")
    print(f"  闭环验证:")
    print(f"    - 通过数: {report['closed_loop']['verified_count']}")
    print(f"    - 平均改进: {report['closed_loop']['avg_improvement']:.3f}")

    print("\n[11] 获取假设-知识关系...")
    relationships = validator.get_hypothesis_knowledge_relationships()
    print(f"✓ 验证转知识: {len(relationships['validated_to_knowledge'])}")
    print(f"✓ 待验证: {len(relationships['pending_validations'])}")
    print(f"✓ 闭环结果: {len(relationships['closed_loop_results'])}")

    print("\n[12] 假设统计...")
    stats = tester.get_hypothesis_statistics()
    print(f"✓ 总假设: {stats['total_hypotheses']}")
    print(f"✓ 待验证: {stats['pending']}")
    print(f"✓ 已确认: {stats['confirmed']}")
    print(f"✓ 已拒绝: {stats['rejected']}")
    print(f"✓ 平均置信度: {stats['avg_confidence']:.2f}")

    print("\n" + "=" * 60)
    print("最终验证状态")
    print("=" * 60)
    final_status = validator.get_validation_status()
    print(f"验证假设: {final_status['total_validations']}")
    print(f"确认有效: {final_status['confirmed']}")
    print(f"知识更新: {final_status['knowledge_updates']}")
    print(f"闭环验证: {final_status['closed_loop_verifications']}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import random

    asyncio.run(main())
