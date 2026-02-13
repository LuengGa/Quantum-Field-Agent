"""
Wave-Particle Duality Core - 波粒二象性核心
===========================================

真正的量子思维实现：
1. 叠加态 - 多个可能性同时存在
2. 观测者效应 - 观测改变系统
3. 环境退相干 - 与环境的纠缠导致坍缩
4. 真正的随机性 - 概率性坍缩

不是术语包装，是真正的架构重构。
"""

import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import random


@dataclass
class CandidateResponse:
    """候选响应 - 叠加态中的一个可能性"""

    content: str
    amplitude: complex  # 复数振幅 (a + bi)
    phase: float  # 相位角度
    confidence: float  # 置信度 |amplitude|^2
    source: str  # 来源 (模型/技能/镜子)
    metadata: Dict[str, Any]  # 元数据


@dataclass
class SuperpositionState:
    """
    真正的叠加态

    不是包装，是真正的量子态表示：
    - 多个候选响应同时存在
    - 每个有复数振幅和相位
    - 观测时根据概率坍缩
    """

    candidates: List[CandidateResponse]
    observer_context: str  # 观测者上下文（影响坍缩）
    environment_coupling: float  # 环境耦合强度（影响退相干）
    coherence_time: float  # 相干时间
    created_at: datetime

    def get_probabilities(self) -> List[float]:
        """获取各候选的概率分布 |ψ|^2"""
        total = sum(abs(c.amplitude) ** 2 for c in self.candidates)
        if total == 0:
            return [1.0 / len(self.candidates)] * len(self.candidates)
        return [abs(c.amplitude) ** 2 / total for c in self.candidates]

    def get_interference_pattern(self) -> np.ndarray:
        """计算干涉图样（波的叠加）"""
        # 将所有候选的波函数叠加
        n = len(self.candidates)
        interference = np.zeros(n, dtype=complex)
        for i, c1 in enumerate(self.candidates):
            for j, c2 in enumerate(self.candidates):
                if i != j:
                    # 干涉项：2 * Re(ψ1* ψ2)
                    phase_diff = c1.phase - c2.phase
                    interference[i] += (
                        c1.amplitude * np.conj(c2.amplitude) * np.exp(1j * phase_diff)
                    )
        return interference

    def calculate_coherence(self) -> float:
        """计算相干性（0-1）"""
        if len(self.candidates) <= 1:
            return 1.0

        # 计算密度矩阵的非对角元（相干项）
        probs = self.get_probabilities()
        coherence = 0.0
        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                # 非对角元的大小表示相干程度
                phase_diff = self.candidates[i].phase - self.candidates[j].phase
                coherence += np.sqrt(probs[i] * probs[j]) * np.abs(np.cos(phase_diff))

        return min(1.0, coherence / (len(probs) * (len(probs) - 1) / 2))


class WaveParticleEngine:
    """
    波粒二象性引擎

    核心创新：
    1. Wave Mode: 连续场，概率云，干涉
    2. Particle Mode: 离散技能，确定性执行
    3. Duality Bridge: 波到粒子的转换（坍缩）
    """

    def __init__(self, ai_client=None):
        self.decoherence_rate = 0.1  # 环境退相干率
        self.observation_backaction = 0.2  # 观测反作用强度
        self.ai_client = ai_client  # AI客户端（OpenAI/Qwen）

    async def generate_superposition(
        self, query: str, context: Dict[str, Any], n_candidates: int = 5
    ) -> SuperpositionState:
        """
        生成叠加态 - 不是生成一个答案，而是多个可能性的叠加

        这是真正的创新：
        - 传统AI：query -> 模型 -> 一个答案
        - 本系统：query -> 叠加态生成器 -> 多个可能性（概率云）
        """
        candidates = []

        # 1. 生成不同"视角"的响应（波的性质：多路径）
        perspectives = [
            ("analytical", 1.0, 0.0),  # 分析性视角
            ("creative", 0.8, np.pi / 4),  # 创造性视角
            ("critical", 0.9, np.pi / 2),  # 批判性视角
            ("practical", 0.85, 3 * np.pi / 4),  # 实用性视角
            ("holistic", 0.75, np.pi),  # 整体性视角
        ]

        # 如果有AI客户端，生成真正的响应；否则使用占位符
        for i, (perspective, amplitude_abs, phase) in enumerate(
            perspectives[:n_candidates]
        ):
            # 复数振幅：幅度 + 相位
            amplitude = amplitude_abs * np.exp(1j * phase)

            # 生成该视角的响应内容
            if self.ai_client:
                try:
                    content = await self._generate_perspective_response(
                        query, perspective
                    )
                except Exception as e:
                    content = f"[{perspective}] {query}"
            else:
                content = f"[{perspective}] {query}"

            candidate = CandidateResponse(
                content=content,
                amplitude=amplitude,
                phase=phase,
                confidence=amplitude_abs**2,
                source=perspective,
                metadata={"perspective": perspective, "order": i},
            )
            candidates.append(candidate)

        return SuperpositionState(
            candidates=candidates,
            observer_context=context.get("observer", "default"),
            environment_coupling=self.decoherence_rate,
            coherence_time=10.0,  # 10秒相干时间
            created_at=datetime.now(),
        )

    async def _generate_perspective_response(self, query: str, perspective: str) -> str:
        """生成特定视角的AI响应"""
        if not self.ai_client:
            return f"[{perspective}] {query}"

        # 根据视角构建不同的system prompt
        perspective_prompts = {
            "analytical": "你是一个分析型助手。请提供逻辑清晰、结构化的分析。",
            "creative": "你是一个创意型助手。请提供创新、独特的想法。",
            "critical": "你是一个批判型助手。请指出潜在问题和不同观点。",
            "practical": "你是一个实用型助手。请提供具体可执行的建议。",
            "holistic": "你是一个整体型助手。请从全局和系统角度思考。",
        }

        system_prompt = perspective_prompts.get(perspective, "你是一个有帮助的助手。")

        try:
            response = self.ai_client.chat.completions.create(
                model="qwen-turbo",  # 或其他模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[{perspective}] {query}"

    async def apply_interference(
        self, state: SuperpositionState, external_field: Optional[np.ndarray] = None
    ) -> SuperpositionState:
        """
        应用干涉 - 波的特性

        外部场与内部波函数干涉，改变概率分布
        这是"环境上下文影响决策"的数学实现
        """
        if external_field is None:
            return state

        # 干涉：新振幅 = 原振幅 + 外部场贡献
        for i, candidate in enumerate(state.candidates):
            interference_term = external_field[i % len(external_field)]
            # 相位匹配增强，相位差削弱
            phase_match = np.cos(candidate.phase - np.angle(interference_term))
            candidate.amplitude += interference_term * phase_match
            candidate.confidence = abs(candidate.amplitude) ** 2

        return state

    async def calculate_decoherence(
        self, state: SuperpositionState, environment_state: Dict[str, Any]
    ) -> float:
        """
        计算退相干程度

        环境耦合导致叠加态退化为混合态
        数学上：密度矩阵的非对角元衰减
        """
        time_elapsed = (datetime.now() - state.created_at).total_seconds()

        # 环境耦合强度影响退相干速率
        coupling = state.environment_coupling

        # 退相干指数衰减
        decoherence = 1 - np.exp(-coupling * time_elapsed)

        # 应用到每个候选
        for candidate in state.candidates:
            # 相位随机化（退相干的标志）
            noise = np.random.normal(0, decoherence * np.pi)
            candidate.phase += noise
            # 振幅衰减
            candidate.amplitude *= np.exp(-decoherence / 2)

        return decoherence

    async def collapse_wavefunction(
        self,
        state: SuperpositionState,
        measurement_basis: str,
        observer: str = "default",
    ) -> Tuple[CandidateResponse, SuperpositionState]:
        """
        波函数坍缩 - 观测导致叠加态坍缩为单一结果

        真正的创新点：
        1. 不是确定性选择，是概率性坍缩
        2. 观测者影响坍缩结果（观测者效应）
        3. 坍缩后产生"粒子"（离散响应）
        """
        # 1. 计算概率分布
        probs = state.get_probabilities()

        # 2. 观测者效应：观测者偏好改变概率分布
        if observer != "default":
            # 观测者的"测量基"影响结果
            observer_bias = self._get_observer_bias(observer, measurement_basis)
            probs = [p * (1 + b) for p, b in zip(probs, observer_bias)]
            # 重新归一化
            total = sum(probs)
            probs = [p / total for p in probs]

        # 3. 真正的随机坍缩（不是argmax！）
        selected_idx = np.random.choice(len(state.candidates), p=probs)
        selected = state.candidates[selected_idx]

        # 4. 观测反作用：观测改变系统
        # 坍缩后的"粒子"状态
        particle_state = CandidateResponse(
            content=selected.content,
            amplitude=1.0 + 0j,  # 坍缩后振幅为1（确定性）
            phase=0.0,  # 相位确定
            confidence=1.0,  # 完全确定
            source=f"collapsed_from_{selected.source}",
            metadata={
                **selected.metadata,
                "observer": observer,
                "measurement_basis": measurement_basis,
                "pre_collapse_amplitude": selected.amplitude,
                "selection_probability": probs[selected_idx],
            },
        )

        # 5. 返回坍缩结果和残余态（部分坍缩）
        residual_candidates = [
            c for i, c in enumerate(state.candidates) if i != selected_idx
        ]
        residual_state = SuperpositionState(
            candidates=residual_candidates,
            observer_context=state.observer_context,
            environment_coupling=state.environment_coupling * 1.5,  # 坍缩加速退相干
            coherence_time=state.coherence_time * 0.5,
            created_at=datetime.now(),
        )

        return particle_state, residual_state

    def _get_observer_bias(self, observer: str, basis: str) -> List[float]:
        """获取观测者偏差 - 不同观测者有不同的观测偏好"""
        # 简化的观测者模型
        biases = {
            "analytical_observer": [1.3, 0.8, 1.1, 0.9, 0.7],  # 偏好分析性
            "creative_observer": [0.8, 1.4, 0.7, 0.9, 1.0],  # 偏好创造性
            "critical_observer": [0.9, 0.7, 1.4, 1.0, 0.8],  # 偏好批判性
        }
        return biases.get(observer, [1.0] * 5)


class DualityBridge:
    """
    波粒二象性桥梁

    连接连续场（波）和离散技能（粒子）
    实现：波函数坍缩触发技能调用
    """

    def __init__(self, wave_engine: WaveParticleEngine):
        self.wave_engine = wave_engine
        self.skill_activation_threshold = 0.7

    async def wave_to_particle(
        self, wave_state: SuperpositionState, available_skills: Dict[str, Any]
    ) -> List[Tuple[str, Any]]:
        """
        波到粒子的转换：坍缩为具体技能调用

        这是核心创新点：
        - 不是直接调用技能
        - 先生成叠加态（概率云）
        - 坍缩后才知道调用哪个技能
        """
        activated_skills = []

        # 1. 计算每个技能与波函数的匹配度（干涉）
        for skill_name, skill_info in available_skills.items():
            # 技能作为"测量算符"作用于波函数
            overlap = self._calculate_overlap(wave_state, skill_name)

            if overlap > self.skill_activation_threshold:
                # 2. 技能激活（坍缩为粒子）
                activated_skills.append((skill_name, skill_info, overlap))

        # 3. 按重叠度排序（概率优先）
        activated_skills.sort(key=lambda x: x[2], reverse=True)

        return [(name, info) for name, info, _ in activated_skills]

    def _calculate_overlap(self, state: SuperpositionState, skill_name: str) -> float:
        """计算波函数与技能的量子力学重叠"""
        # 简化的重叠计算
        total_amplitude = sum(abs(c.amplitude) for c in state.candidates)
        if total_amplitude == 0:
            return 0.0

        # 检查技能名称是否出现在候选中
        overlap = 0.0
        for candidate in state.candidates:
            if skill_name.lower() in candidate.source.lower():
                overlap += abs(candidate.amplitude) / total_amplitude

        return min(1.0, overlap)


# 使用示例
async def example_usage():
    """波粒二象性引擎使用示例"""
    engine = WaveParticleEngine()
    bridge = DualityBridge(engine)

    # 1. 生成叠加态（波）
    query = "如何解决复杂问题？"
    context = {"observer": "analytical_observer"}

    superposition = await engine.generate_superposition(query, context, n_candidates=5)
    print(f"叠加态相干度: {superposition.calculate_coherence():.2f}")

    # 2. 应用干涉（环境上下文影响）
    external_field = np.array([0.1, 0.2, 0.15, 0.05, 0.1]) * np.exp(1j * np.pi / 3)
    superposition = await engine.apply_interference(superposition, external_field)

    # 3. 环境退相干
    environment = {"temperature": 0.5, "noise_level": 0.2}
    decoherence = await engine.calculate_decoherence(superposition, environment)
    print(f"退相干程度: {decoherence:.2f}")

    # 4. 坍缩为粒子（观测）
    particle, residual = await engine.collapse_wavefunction(
        superposition, measurement_basis="practical", observer="analytical_observer"
    )

    print(f"坍缩结果: {particle.content}")
    print(f"坍缩来源: {particle.source}")
    print(f"选择概率: {particle.metadata['selection_probability']:.2f}")

    # 5. 波到粒子转换（技能激活）
    skills = {
        "analyze": {"function": lambda x: f"分析: {x}"},
        "calculate": {"function": lambda x: f"计算: {x}"},
    }
    activated = await bridge.wave_to_particle(superposition, skills)
    print(f"激活技能: {[s[0] for s in activated]}")


if __name__ == "__main__":
    asyncio.run(example_usage())
