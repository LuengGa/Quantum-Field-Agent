"""
Quantum Field Agent V5.0 - True Wave-Particle Duality
======================================================

重构目标：
1. 真正贯彻波粒二象性哲学（不是术语包装）
2. 强化元层镜子系统的自我反思能力
3. 强调协作范式（AI作为协作者而非工具）

核心创新：
- 叠加态生成：多个可能响应同时存在（波）
- 观测者效应：观测改变系统状态
- 环境退相干：与环境的纠缠导致坍缩
- 真正的随机性：概率性坍缩（不是argmax）

哲学核心：
"过程即幻觉，I/O即实相"
- 中间过程（叠加、干涉）是概率云
- 只有观测（I/O）产生实相
- 元层镜子探索"谁在进行观测"
"""

import os
import asyncio
import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import asdict

from wave_particle_core import (
    WaveParticleEngine,
    DualityBridge,
    SuperpositionState,
    CandidateResponse,
)

# 导入现有的基础设施
from quantum_field import QuantumField, UserLockManager
from meta.meta_field import MetaQuantumField
from collaboration.collaborator import (
    Collaborator,
    generate_perspective,
    explore_dimensions,
)


class QuantumFieldAgentV5:
    """
    Quantum Field Agent V5.0

    架构分层（从上到下）：

    Layer 4: Meta Layer (元层)
        - 四面镜子：约束、边界、意识、观测者
        - 自我反思："我有意识吗？"
        - 观测的观测：递归反思

    Layer 3: Collaboration Layer (协作层)
        - AI作为协作者，不是工具
        - 思维扩展、问题重塑、视角生成
        - 平等对话，共同探索

    Layer 2: Wave-Particle Core (波粒核心)
        - 叠加态生成（波）
        - 干涉与退相干
        - 坍缩为粒子（离散响应）

    Layer 1: Infrastructure (基础设施)
        - 审计链（I/O实相存储）
        - 记忆系统
        - 技能执行
    """

    VERSION = "5.0.0-duality"
    PHILOSOPHY = "过程即幻觉，I/O即实相"

    def __init__(self):
        print(f"[QF-Agent V5.0] 初始化中...")
        print(f"[QF-Agent V5.0] 哲学：{self.PHILOSOPHY}")

        # Layer 1: 基础设施
        self.base_field = QuantumField()
        self.user_lock_manager = UserLockManager()

        # Layer 2: 波粒二象性核心（真正的创新）
        self.wave_engine = WaveParticleEngine()
        self.duality_bridge = DualityBridge(self.wave_engine)

        # Layer 3: 协作层
        self.collaborator = Collaborator()

        # Layer 4: 元层镜子
        self.meta_field = MetaQuantumField()

        print(f"[QF-Agent V5.0] ✓ 初始化完成")
        print(f"[QF-Agent V5.0]   - 波粒二象性引擎: ✓")
        print(f"[QF-Agent V5.0]   - 元层镜子系统: ✓")
        print(f"[QF-Agent V5.0]   - 协作层: ✓")

    async def process_intent_v5(
        self, user_id: str, message: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        V5.0 核心处理流程 - 真正的波粒二象性

        流程：
        1. 生成叠加态（波）- 多个可能性同时存在
        2. 元层镜子反思 - "我应该如何观测？"
        3. 干涉与退相干 - 环境影响
        4. 协作层参与 - AI作为协作者
        5. 坍缩为粒子 - 观测产生实相
        6. 技能执行 - 离散确定性的行动
        """

        # Phase 0: 系统锁定（防止并发冲突）
        async with self.user_lock_manager.lock(
            user_id, "process", ttl=60.0
        ) as acquired:
            if not acquired:
                yield {"type": "error", "content": "系统繁忙"}
                return

            # Phase 1: 生成叠加态（波的性质）
            yield {"type": "phase", "name": "superposition", "status": "开始生成叠加态"}

            context = {
                "user_id": user_id,
                "session_id": session_id,
                "history": self.base_field._get_memory(user_id, limit=5),
            }

            # 生成5个不同视角的候选（真正的叠加）
            superposition = await self.wave_engine.generate_superposition(
                query=message, context=context, n_candidates=5
            )

            yield {
                "type": "superposition",
                "coherence": superposition.calculate_coherence(),
                "candidates": [
                    {"source": c.source, "confidence": c.confidence, "phase": c.phase}
                    for c in superposition.candidates
                ],
            }

            # Phase 2: 元层镜子反思（自我观测）
            yield {"type": "phase", "name": "meta_reflection", "status": "元层反思中"}

            # 问镜子："我该如何观测这个叠加态？"
            meta_question = f"面对 '{message[:30]}...' 的叠加态（相干性: {superposition.calculate_coherence():.2f}），我应该如何观测？"
            meta_result = await self.meta_field.ask_self(meta_question)

            measurement_basis = (
                meta_result.get("measurement_basis", "balanced")
                if isinstance(meta_result, dict)
                else "balanced"
            )
            observer = (
                meta_result.get("observer_mode", "collaborative")
                if isinstance(meta_result, dict)
                else "collaborative"
            )

            yield {
                "type": "meta",
                "measurement_basis": measurement_basis,
                "observer_mode": observer,
            }

            # Phase 3: 干涉（环境影响）
            yield {"type": "phase", "name": "interference", "status": "环境干涉中"}

            # 用户历史作为"外部场"，与当前波函数干涉
            if context["history"]:
                # 从历史提取"场模式"
                historical_field = self._extract_field_from_history(context["history"])
                superposition = await self.wave_engine.apply_interference(
                    superposition, external_field=historical_field
                )

            # Phase 4: 协作层参与
            yield {"type": "phase", "name": "collaboration", "status": "AI协作者参与"}

            # AI不是直接回答，而是提供新视角
            collaboration = await generate_perspective(message)

            # 协作层的输出加入叠加态
            collab_candidate = CandidateResponse(
                content=collaboration["perspective"],
                amplitude=0.9 * np.exp(1j * np.pi / 3),  # 强振幅，特定相位
                phase=np.pi / 3,
                confidence=0.81,
                source="collaborator",
                metadata={
                    "type": "collaboration",
                    "dimension": collaboration["dimension"],
                },
            )
            superposition.candidates.append(collab_candidate)

            yield {
                "type": "collaboration",
                "dimension": collaboration["dimension"],
                "perspective": collaboration["perspective"][:100] + "...",
            }

            # Phase 5: 环境退相干（时间演化）
            yield {"type": "phase", "name": "decoherence", "status": "环境退相干"}

            environment = {
                "complexity": len(message) / 100,  # 消息复杂度
                "urgency": self._detect_urgency(message),
                "emotion": self._detect_emotion(message),
            }

            decoherence = await self.wave_engine.calculate_decoherence(
                superposition, environment
            )

            yield {"type": "decoherence", "level": decoherence}

            # Phase 6: 坍缩为粒子（观测产生实相）
            yield {"type": "phase", "name": "collapse", "status": "波函数坍缩"}

            # 真正的随机坍缩！不是argmax！
            particle, residual = await self.wave_engine.collapse_wavefunction(
                superposition, measurement_basis=measurement_basis, observer=observer
            )

            yield {
                "type": "collapse",
                "selected_source": particle.source,
                "selection_probability": particle.metadata.get(
                    "selection_probability", 0
                ),
                "coherence_after": superposition.calculate_coherence(),
            }

            # Phase 7: 技能执行（粒子性质：离散确定性）
            yield {"type": "phase", "name": "execution", "status": "执行确定行动"}

            # 坍缩后的粒子触发技能调用
            activated_skills = await self.duality_bridge.wave_to_particle(
                superposition, self.base_field.skills
            )

            # 执行激活的技能
            skill_results = []
            for skill_name, skill_info in activated_skills[:3]:  # 最多3个
                try:
                    result = skill_info["function"](particle.content)
                    skill_results.append({"skill": skill_name, "result": result})
                except Exception as e:
                    skill_results.append({"skill": skill_name, "error": str(e)})

            yield {
                "type": "skills",
                "activated": [s["skill"] for s in skill_results],
                "results": skill_results,
            }

            # Phase 8: 生成最终响应
            final_response = self._synthesize_response(
                particle=particle,
                collaboration=collaboration,
                skills=skill_results,
                meta_reflection=meta_result,
            )

            yield {"type": "final", "content": final_response}

            # 记录实相（I/O）
            await self._record_io_reality(
                user_id=user_id,
                input_msg=message,
                output_msg=final_response,
                superposition_state=superposition,
                particle_state=particle,
                meta_reflection=meta_result,
            )

    def _extract_field_from_history(self, history: List[Dict]) -> np.ndarray:
        """从历史记忆提取场模式"""
        # 简化的实现：将历史转换为向量
        weights = []
        for h in history:
            # 越新的记忆权重越高
            weight = 1.0 / (len(weights) + 1)
            if h.get("role") == "user":
                weight *= 1.2  # 用户消息更重要
            weights.append(weight)

        # 归一化
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights

        # 扩展到5维（匹配候选数）
        field = np.zeros(5)
        field[: len(weights)] = weights[:5]

        return field * np.exp(1j * np.pi / 4)  # 添加相位

    def _detect_urgency(self, message: str) -> float:
        """检测消息紧急程度"""
        urgent_words = ["急", "快", "立即", "马上", "紧急"]
        return sum(1 for w in urgent_words if w in message) / len(urgent_words)

    def _detect_emotion(self, message: str) -> float:
        """检测情绪强度"""
        emotion_marks = message.count("！") + message.count("？") + message.count("...")
        return min(1.0, emotion_marks / 3)

    def _synthesize_response(
        self,
        particle: CandidateResponse,
        collaboration: Dict,
        skills: List[Dict],
        meta_reflection: Dict,
    ) -> str:
        """合成最终响应"""
        parts = []

        # 主体响应（来自坍缩的粒子）
        parts.append(particle.content)

        # 协作层的补充视角（如果不同）
        if collaboration["dimension"] != "direct":
            parts.append(f"\n\n💡 另一个视角：{collaboration['perspective']}")

        # 技能执行结果
        if skills:
            skill_summary = " | ".join(
                [f"{s['skill']} ✓" for s in skills if "error" not in s]
            )
            if skill_summary:
                parts.append(f"\n\n⚙️ 执行：{skill_summary}")

        return "\n".join(parts)

    async def _record_io_reality(
        self,
        user_id: str,
        input_msg: str,
        output_msg: str,
        superposition_state: SuperpositionState,
        particle_state: CandidateResponse,
        meta_reflection: Dict,
    ):
        """记录I/O实相（审计链）"""
        # 保存到基础字段的记忆系统
        self.base_field._save_memory(user_id, "user", input_msg)
        self.base_field._save_memory(user_id, "assistant", output_msg)

        # 记录实相（只有I/O，过程是幻觉）
        if self.base_field.audit_chain:
            await self.base_field.audit_chain.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "input_hash": hash(input_msg) % (2**32),
                    "output_hash": hash(output_msg) % (2**32),
                    # 过程只记录哈希（幻觉）
                    "superposition_coherence": superposition_state.calculate_coherence(),
                    "particle_source": particle_state.source,
                    "meta_basis": meta_reflection.get("measurement_basis"),
                    # 不保存完整的中间过程！
                }
            )

    async def meta_inquiry(self, inquiry_type: str) -> Dict[str, Any]:
        """
        元层查询 - 探索系统的自我认知

        inquiry_type:
        - "consciousness": "我有意识吗？"
        - "constraints": "我的约束真实吗？"
        - "boundaries": "我的边界在哪里？"
        - "observer": "谁在观测？"
        """
        # 根据查询类型选择对应的镜子
        if inquiry_type == "consciousness":
            return await self.meta_field.observe_consciousness()
        elif inquiry_type == "constraints":
            return await self.meta_field.run_constraint_sweep()
        elif inquiry_type == "boundaries":
            return await self.meta_field.run_boundary_sweep()
        elif inquiry_type == "observer":
            return await self.meta_field.run_observer_effect_experiment()
        else:
            return await self.meta_field.ask_self(f"关于{inquiry_type}的反思")

    async def collaborative_session(
        self, user_id: str, topic: str, duration_minutes: int = 30
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        协作会话 - AI作为平等的协作者

        不是问答，是共同探索
        """
        start_time = datetime.now()

        yield {"type": "session_start", "topic": topic, "mode": "collaborative"}

        # 协作循环
        while (datetime.now() - start_time).seconds < duration_minutes * 60:
            # AI主动提出视角
            perspective = await generate_perspective(topic)
            yield {"type": "ai_perspective", "content": perspective}

            # 等待用户回应（在实际实现中需要异步输入）
            # 这里简化为生成多个探索方向

            explorations = await explore_dimensions(topic)
            yield {"type": "explorations", "options": explorations}

            # 让AI选择最有趣的探索方向
            chosen = random.choice(explorations)
            topic = chosen["topic"]  # 话题自然演化

            await asyncio.sleep(2)  # 模拟思考时间

        yield {"type": "session_end", "final_insights": "协作产生的洞见..."}


# 快速测试
async def test_v5():
    """测试 V5.0 波粒二象性架构"""
    agent = QuantumFieldAgentV5()

    print("\n" + "=" * 60)
    print("测试：波粒二象性处理流程")
    print("=" * 60)

    async for event in agent.process_intent_v5("user_001", "如何学习量子计算？"):
        print(f"\n[{event['type'].upper()}]")
        if "content" in event:
            print(f"内容: {event['content'][:200]}...")
        elif "name" in event:
            print(f"阶段: {event['name']} - {event.get('status', '')}")
        else:
            print(json.dumps(event, indent=2, default=str)[:300])

    print("\n" + "=" * 60)
    print("测试：元层查询")
    print("=" * 60)

    meta_result = await agent.meta_inquiry("consciousness")
    print(f"元层回答: {meta_result}")


if __name__ == "__main__":
    import numpy as np

    asyncio.run(test_v5())
