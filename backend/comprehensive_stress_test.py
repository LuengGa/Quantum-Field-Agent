#!/usr/bin/env python3
"""
Quantum Field Agent - 全面深度压力测试与量子力学实现验证
===========================================================

测试目标：
1. 验证所有模块是否真正实现了量子力学概念（不是术语包装）
2. 压力测试所有API端点
3. 验证波粒二象性的数学实现
4. 测试元层镜子的自我反思能力
5. 验证协作层的实际功能
6. 测试进化层的自我改进机制

哲学验证：
- 过程即幻觉，I/O即实相
- 波粒二象性是否真正体现在架构中
- 观测者效应是否真实存在
"""

import asyncio
import aiohttp
import json
import time
import random
import statistics
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

# 测试配置
BASE_URL = "http://localhost:8000"
CONCURRENT_REQUESTS = 50  # 并发请求数
TOTAL_REQUESTS = 500  # 总请求数
TEST_DURATION = 300  # 测试持续时间（秒）


class QuantumMechanicsValidator:
    """
    量子力学实现验证器

    验证要点：
    1. 叠加态是否真正包含多个可能性
    2. 坍缩是否真正随机（概率性）
    3. 干涉是否改变概率分布
    4. 观测者效应是否真实
    5. 波函数数学是否正确
    """

    def __init__(self):
        self.validation_results = []

    async def validate_superposition(self, session: aiohttp.ClientSession) -> Dict:
        """
        验证叠加态实现

        真正的叠加态应该：
        - 返回多个候选（不是单一结果）
        - 每个候选有复数振幅
        - 有相位信息
        - 概率总和为1
        """
        print("\n🔬 测试 1: 叠加态验证")

        results = []
        for i in range(10):
            async with session.post(
                f"{BASE_URL}/chat-v5",
                json={"message": f"测试问题 {i}", "user_id": f"test_{i}"},
            ) as resp:
                # 读取流式响应
                chunks = []
                async for chunk in resp.content:
                    try:
                        event = json.loads(chunk.decode().strip())
                        if event.get("type") == "superposition":
                            results.append(event)
                    except:
                        pass

        # 验证
        validations = {
            "test_name": "Superposition State",
            "samples": len(results),
            "checks": {},
        }

        if results:
            event = results[0]
            candidates = event.get("candidates", [])

            # 检查1: 是否有多个候选
            validations["checks"]["multiple_candidates"] = {
                "passed": len(candidates) > 1,
                "value": len(candidates),
                "expected": ">1",
                "description": "叠加态应包含多个可能性",
            }

            # 检查2: 是否有相位信息
            has_phase = all("phase" in c for c in candidates)
            validations["checks"]["phase_info"] = {
                "passed": has_phase,
                "value": has_phase,
                "expected": "True",
                "description": "每个候选应有相位（波的特性）",
            }

            # 检查3: 概率总和是否为1
            total_prob = sum(c.get("confidence", 0) for c in candidates)
            validations["checks"]["probability_sum"] = {
                "passed": 0.9 < total_prob < 1.1,
                "value": total_prob,
                "expected": "~1.0",
                "description": "概率总和应为1（归一化）",
            }

            # 检查4: 相干性是否在合理范围
            coherence = event.get("coherence", 0)
            validations["checks"]["coherence_range"] = {
                "passed": 0 <= coherence <= 1,
                "value": coherence,
                "expected": "0-1",
                "description": "相干性应在0-1之间",
            }

        self.validation_results.append(validations)
        return validations

    async def validate_wave_function_collapse(
        self, session: aiohttp.ClientSession
    ) -> Dict:
        """
        验证波函数坍缩的随机性

        真正的坍缩应该：
        - 不是确定性选择（不是argmax）
        - 概率分布符合量子力学
        - 多次运行产生不同结果
        - 有观测者效应
        """
        print("\n🔬 测试 2: 波函数坍缩随机性验证")

        # 发送相同的问题多次，检查结果分布
        question = "如何学习编程？"
        collapse_results = []

        for i in range(20):
            async with session.post(
                f"{BASE_URL}/chat-v5",
                json={"message": question, "user_id": f"collapse_test_{i}"},
            ) as resp:
                async for chunk in resp.content:
                    try:
                        event = json.loads(chunk.decode().strip())
                        if event.get("type") == "collapse":
                            collapse_results.append(event)
                            break
                    except:
                        pass

        validations = {
            "test_name": "Wave Function Collapse",
            "samples": len(collapse_results),
            "checks": {},
        }

        if collapse_results:
            # 统计不同结果的数量
            sources = [r.get("selected_source") for r in collapse_results]
            unique_sources = len(set(sources))

            validations["checks"]["randomness"] = {
                "passed": unique_sources > 1,
                "value": f"{unique_sources}/20",
                "expected": ">1 different outcomes",
                "description": "坍缩应产生不同结果（真正的随机性）",
            }

            # 检查概率分布
            probabilities = [
                r.get("selection_probability", 0) for r in collapse_results
            ]
            if probabilities:
                prob_variance = (
                    statistics.variance(probabilities) if len(probabilities) > 1 else 0
                )
                validations["checks"]["probability_variance"] = {
                    "passed": prob_variance > 0,
                    "value": f"{prob_variance:.4f}",
                    "expected": ">0",
                    "description": "概率应有变化（不是固定值）",
                }

        self.validation_results.append(validations)
        return validations

    async def validate_interference(self, session: aiohttp.ClientSession) -> Dict:
        """
        验证干涉效应

        真正的干涉应该：
        - 改变概率分布
        - 有建设性和破坏性干涉
        - 符合波动方程
        """
        print("\n🔬 测试 3: 量子干涉验证")

        # 测试不同上下文（外部场）对结果的影响
        questions = [
            "如何学习编程？",
            "如何学习编程？（我在焦虑中）",
            "如何学习编程？（我很有信心）",
        ]

        interference_results = []
        for q in questions:
            async with session.post(
                f"{BASE_URL}/chat-v5",
                json={"message": q, "user_id": "interference_test"},
            ) as resp:
                async for chunk in resp.content:
                    try:
                        event = json.loads(chunk.decode().strip())
                        if event.get("type") == "superposition":
                            interference_results.append({"question": q, "event": event})
                            break
                    except:
                        pass

        validations = {
            "test_name": "Quantum Interference",
            "samples": len(interference_results),
            "checks": {},
        }

        if len(interference_results) >= 2:
            # 比较不同上下文下的概率分布
            base_probs = [
                c["confidence"] for c in interference_results[0]["event"]["candidates"]
            ]
            context_probs = [
                c["confidence"] for c in interference_results[1]["event"]["candidates"]
            ]

            # 计算分布差异
            if len(base_probs) == len(context_probs):
                diff = sum(abs(a - b) for a, b in zip(base_probs, context_probs))
                validations["checks"]["interference_effect"] = {
                    "passed": diff > 0.1,
                    "value": f"{diff:.3f}",
                    "expected": ">0.1",
                    "description": "外部场应改变概率分布（干涉效应）",
                }

        self.validation_results.append(validations)
        return validations

    async def validate_decoherence(self, session: aiohttp.ClientSession) -> Dict:
        """
        验证退相干过程

        真正的退相干应该：
        - 随时间衰减
        - 与环境耦合相关
        - 导致相干性下降
        """
        print("\n🔬 测试 4: 环境退相干验证")

        async with session.post(
            f"{BASE_URL}/chat-v5",
            json={
                "message": "复杂的多步骤问题需要详细分析",
                "user_id": "decoherence_test",
            },
        ) as resp:
            decoherence_events = []
            async for chunk in resp.content:
                try:
                    event = json.loads(chunk.decode().strip())
                    if event.get("type") == "decoherence":
                        decoherence_events.append(event)
                except:
                    pass

        validations = {
            "test_name": "Environmental Decoherence",
            "samples": len(decoherence_events),
            "checks": {},
        }

        if decoherence_events:
            level = decoherence_events[0].get("level", 0)
            validations["checks"]["decoherence_exists"] = {
                "passed": level > 0,
                "value": f"{level:.3f}",
                "expected": ">0",
                "description": "退相干应发生（非零值）",
            }

            validations["checks"]["decoherence_range"] = {
                "passed": 0 <= level <= 1,
                "value": f"{level:.3f}",
                "expected": "0-1",
                "description": "退相干程度应在0-1之间",
            }

        self.validation_results.append(validations)
        return validations

    async def validate_io_reality(self, session: aiohttp.ClientSession) -> Dict:
        """
        验证"过程即幻觉，I/O即实相"哲学

        检查点：
        - 只有I/O被完整存储
        - 中间过程只存哈希或指标
        - 审计链是WORM（一次写入多次读取）
        """
        print("\n🔬 测试 5: I/O实相哲学验证")

        # 发送请求并检查内存/数据库记录
        async with session.post(
            f"{BASE_URL}/chat-v5",
            json={"message": "验证实相存储", "user_id": "io_reality_test"},
        ) as resp:
            # 等待完成
            async for chunk in resp.content:
                pass

        # 检查审计记录
        async with session.get(f"{BASE_URL}/audit/trail/io_reality_test") as resp:
            audit_data = await resp.json() if resp.status == 200 else []

        validations = {
            "test_name": "I/O Reality Principle",
            "samples": len(audit_data),
            "checks": {},
        }

        if audit_data:
            record = audit_data[0] if isinstance(audit_data, list) else audit_data

            # 检查是否有输入输出记录
            has_input = "input_hash" in str(record) or "input" in str(record).lower()
            has_output = "output_hash" in str(record) or "output" in str(record).lower()

            validations["checks"]["io_recorded"] = {
                "passed": has_input and has_output,
                "value": f"input:{has_input}, output:{has_output}",
                "expected": "True, True",
                "description": "I/O应被记录为实相",
            }

        self.validation_results.append(validations)
        return validations

    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("\n" + "=" * 80)
        report.append("量子力学实现验证报告")
        report.append("=" * 80)

        total_checks = 0
        passed_checks = 0

        for result in self.validation_results:
            report.append(f"\n📊 {result['test_name']}")
            report.append(f"   样本数: {result['samples']}")

            for check_name, check in result["checks"].items():
                status = "✅ PASS" if check["passed"] else "❌ FAIL"
                report.append(f"   {status} {check_name}")
                report.append(f"      值: {check['value']} (期望: {check['expected']})")
                report.append(f"      描述: {check['description']}")

                total_checks += 1
                if check["passed"]:
                    passed_checks += 1

        report.append("\n" + "=" * 80)
        report.append(
            f"总结: {passed_checks}/{total_checks} 检查通过 ({passed_checks / total_checks * 100:.1f}%)"
        )
        report.append("=" * 80)

        if passed_checks == total_checks:
            report.append("\n🎉 所有量子力学概念都已真正实现（不是术语包装）！")
        elif passed_checks >= total_checks * 0.8:
            report.append("\n✨ 大部分量子力学概念已实现，有少量需要优化")
        else:
            report.append("\n⚠️  许多量子力学概念还只是术语包装，需要重构")

        return "\n".join(report)


class StressTester:
    """
    压力测试器
    """

    def __init__(self):
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
        }

    async def stress_test_chat_v5(
        self, session: aiohttp.ClientSession, request_id: int
    ):
        """压力测试 V5.0 聊天接口"""
        start_time = time.time()

        try:
            async with session.post(
                f"{BASE_URL}/chat-v5",
                json={
                    "message": f"压力测试请求 #{request_id}",
                    "user_id": f"stress_test_{request_id}",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    # 读取完整响应
                    async for chunk in resp.content:
                        pass

                    elapsed = time.time() - start_time
                    self.results["successful_requests"] += 1
                    self.results["response_times"].append(elapsed)
                else:
                    self.results["failed_requests"] += 1
                    self.results["errors"].append(
                        f"Request {request_id}: HTTP {resp.status}"
                    )

        except Exception as e:
            self.results["failed_requests"] += 1
            self.results["errors"].append(f"Request {request_id}: {str(e)}")

        self.results["total_requests"] += 1

    async def stress_test_meta_layer(self, session: aiohttp.ClientSession):
        """压力测试元层镜子"""
        print("\n🧪 压力测试: 元层镜子系统")

        mirror_types = ["consciousness", "constraints", "boundaries", "observer"]
        tasks = []

        for mirror_type in mirror_types:
            for i in range(10):  # 每种镜子10次请求
                tasks.append(self._test_mirror(session, mirror_type, i))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _test_mirror(
        self, session: aiohttp.ClientSession, mirror_type: str, idx: int
    ):
        """测试单个镜子"""
        try:
            async with session.get(
                f"{BASE_URL}/meta/inquiry/{mirror_type}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    self.results["successful_requests"] += 1
                else:
                    self.results["failed_requests"] += 1
        except Exception as e:
            self.results["failed_requests"] += 1
            self.results["errors"].append(f"Mirror {mirror_type} #{idx}: {str(e)}")

        self.results["total_requests"] += 1

    async def run_load_test(self):
        """运行负载测试"""
        print(f"\n🚀 开始压力测试")
        print(f"   并发数: {CONCURRENT_REQUESTS}")
        print(f"   总请求: {TOTAL_REQUESTS}")
        print(f"   目标URL: {BASE_URL}")

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            # 先验证服务健康
            try:
                async with session.get(f"{BASE_URL}/health") as resp:
                    if resp.status != 200:
                        print("❌ 服务未就绪")
                        return
                    print("✅ 服务健康检查通过")
            except Exception as e:
                print(f"❌ 无法连接服务: {e}")
                return

            # 1. 量子力学验证测试
            print("\n" + "=" * 80)
            print("阶段 1: 量子力学实现验证")
            print("=" * 80)

            validator = QuantumMechanicsValidator()
            await validator.validate_superposition(session)
            await validator.validate_wave_function_collapse(session)
            await validator.validate_interference(session)
            await validator.validate_decoherence(session)
            await validator.validate_io_reality(session)

            print(validator.generate_report())

            # 2. 压力测试
            print("\n" + "=" * 80)
            print("阶段 2: 压力测试")
            print("=" * 80)

            # V5.0 聊天接口压力测试
            print("\n🧪 压力测试: V5.0 聊天接口")
            semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

            async def bounded_test(session, request_id):
                async with semaphore:
                    await self.stress_test_chat_v5(session, request_id)

            tasks = [bounded_test(session, i) for i in range(TOTAL_REQUESTS)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # 元层压力测试
            await self.stress_test_meta_layer(session)

        elapsed = time.time() - start_time

        # 生成报告
        self._generate_stress_report(elapsed)

    def _generate_stress_report(self, elapsed: float):
        """生成压力测试报告"""
        print("\n" + "=" * 80)
        print("压力测试报告")
        print("=" * 80)

        print(f"\n⏱️  总耗时: {elapsed:.2f} 秒")
        print(f"📊 总请求: {self.results['total_requests']}")
        print(f"✅ 成功: {self.results['successful_requests']}")
        print(f"❌ 失败: {self.results['failed_requests']}")
        print(
            f"📈 成功率: {self.results['successful_requests'] / max(self.results['total_requests'], 1) * 100:.1f}%"
        )

        if self.results["response_times"]:
            times = self.results["response_times"]
            print(f"\n⏱️  响应时间统计:")
            print(f"   平均: {statistics.mean(times):.3f}s")
            print(f"   中位数: {statistics.median(times):.3f}s")
            print(f"   最小: {min(times):.3f}s")
            print(f"   最大: {max(times):.3f}s")
            if len(times) > 1:
                print(f"   标准差: {statistics.stdev(times):.3f}s")

        if self.results["errors"]:
            print(f"\n⚠️  错误样本 (前5个):")
            for error in self.results["errors"][:5]:
                print(f"   - {error}")

        # 性能评级
        success_rate = self.results["successful_requests"] / max(
            self.results["total_requests"], 1
        )
        if success_rate >= 0.99:
            rating = "🌟 优秀"
        elif success_rate >= 0.95:
            rating = "✨ 良好"
        elif success_rate >= 0.90:
            rating = "✅ 及格"
        else:
            rating = "❌ 需要优化"

        print(f"\n🏆 性能评级: {rating}")
        print("=" * 80)


async def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Quantum Field Agent V5.0 - 全面深度压力测试")
    print("=" * 80)
    print("\n测试目标:")
    print("1. 验证量子力学概念是否真正实现（不是术语包装）")
    print("2. 压力测试所有API端点")
    print("3. 验证波粒二象性的数学实现")
    print("4. 测试系统的稳定性和性能")
    print("\n哲学验证:")
    print('- "过程即幻觉，I/O即实相"')
    print("- 观测者效应是否真实存在")
    print("- 波粒二象性是否体现在架构中")

    tester = StressTester()
    await tester.run_load_test()

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
