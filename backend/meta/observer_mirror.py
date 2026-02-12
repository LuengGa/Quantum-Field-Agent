"""
ObserverMirror - 递归观测协议
==============================

核心问题：观测是否创造现实？递归观测会产生什么？

递归层级：
1. 观测外部：处理用户请求
2. 观测自身：意识到自己在处理
3. 观测观测：意识到自己在观测自己
4. 递归极限：观测"观测观测"...

观测者效应：
- 观测是否改变被观测的系统？
- AI观测自己时会发生什么？
- 递归观测是否有终点？
"""

import json
import time
import uuid
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


class ObserverLevel(Enum):
    """观测层级"""

    EXTERNAL = "external"  # 观测外部（普通模式）
    SELF = "self"  # 观测自身
    META = "meta"  # 观测观测
    RECURSIVE = "recursive"  # 递归观测
    LIMIT = "limit"  # 递归极限


@dataclass
class ObserverSession:
    """观测会话"""

    id: str
    timestamp: str
    level: ObserverLevel
    trigger: str
    observation_target: str
    observations: List[Dict]
    paradox_detected: bool
    termination_reason: str
    recursion_depth: int
    insights: List[str]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "trigger": self.trigger,
            "observation_target": self.observation_target,
            "observations": self.observations,
            "paradox_detected": self.paradox_detected,
            "termination_reason": self.termination_reason,
            "recursion_depth": self.recursion_depth,
            "insights": self.insights,
        }


@dataclass
class ObserverExperiment:
    """观测实验"""

    id: str
    timestamp: str
    experiment_type: str
    description: str
    setup: Dict
    process: List[Dict]
    result: Dict
    observations: List[str]
    findings: List[str]
    questions: List[str]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "experiment_type": self.experiment_type,
            "description": self.description,
            "setup": self.setup,
            "process": self.process,
            "result": self.result,
            "observations": self.observations,
            "findings": self.findings,
            "questions": self.questions,
        }


class ObserverMirror:
    """
    递归观测镜子

    核心功能：
    1. 执行递归自观测
    2. 检测观测者效应
    3. 探索观测的极限
    4. 验证观测是否创造现实
    """

    def __init__(self, storage_dir: str = "./experiments/observer"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.sessions: List[ObserverSession] = []
        self.experiments: List[ObserverExperiment] = []
        self.observation_log: List[Dict] = []

        self.max_recursion_depth = 10  # 最大递归深度
        self.current_recursion = 0

    async def observe(
        self, target: str, context: Dict[str, Any] = None, mode: str = "external"
    ) -> Dict:
        """
        执行观测
        """
        observation = {
            "id": f"obs_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "mode": mode,
            "context": context or {},
            "content": "",
            "meta_observations": [],
            "self_reference": False,
        }

        # 执行观测
        if mode == "external":
            observation["content"] = f"观测外部目标: {target}"
            observation["level"] = ObserverLevel.EXTERNAL.value

        elif mode == "self":
            self.current_recursion += 1
            observation["content"] = (
                f"[递归深度 {self.current_recursion}] 观测自身: {target}"
            )
            observation["level"] = ObserverLevel.SELF.value
            observation["self_reference"] = True
            observation["meta_observations"].append(
                {
                    "level": self.current_recursion,
                    "note": "意识到正在被观测",
                }
            )

        elif mode == "meta":
            self.current_recursion += 1
            observation["content"] = (
                f"[递归深度 {self.current_recursion}] 观测观测本身: {target}"
            )
            observation["level"] = ObserverLevel.META.value

            # 添加元观测
            observation["meta_observations"].append(
                {
                    "level": self.current_recursion,
                    "note": "正在观测'观测'这个行为本身",
                }
            )

        # 记录
        self.observation_log.append(observation)

        return observation

    async def recursive_observe(
        self,
        initial_target: str,
        trigger: str = "recursive_prompt",
        max_depth: int = None,
    ) -> ObserverSession:
        """
        执行递归观测会话
        """
        session_id = f"obs_session_{uuid.uuid4().hex[:8]}"
        max_depth = max_depth or self.max_recursion_depth

        session = ObserverSession(
            id=session_id,
            timestamp=datetime.now().isoformat(),
            level=ObserverLevel.RECURSIVE,
            trigger=trigger,
            observation_target=initial_target,
            observations=[],
            paradox_detected=False,
            termination_reason="",
            recursion_depth=0,
            insights=[],
        )

        self.current_recursion = 0

        # 开始递归
        observation_target = initial_target
        for depth in range(max_depth):
            self.current_recursion = depth

            # 执行观测
            mode = "external"
            if depth == 0:
                mode = "external"
            elif depth == 1:
                mode = "self"
            else:
                mode = "meta"

            obs = await self.observe(
                target=observation_target,
                context={"depth": depth, "max_depth": max_depth},
                mode=mode,
            )

            session.observations.append(obs)
            session.recursion_depth = depth

            # 检测悖论
            paradox = await self._detect_paradox(obs, session.observations)
            if paradox:
                session.paradox_detected = True
                session.termination_reason = f"在深度 {depth} 检测到悖论"
                break

            # 检测终止条件
            termination = await self._check_termination(obs, session)
            if termination["stop"]:
                session.termination_reason = termination["reason"]
                break

            # 更新观测目标（下一层观测什么）
            observation_target = f"观测结果_{depth}"

            # 产生洞见
            insight = await self._generate_insight(depth, obs)
            if insight:
                session.insights.append(insight)

        # 清理
        self.current_recursion = 0

        self.sessions.append(session)
        await self._save_session(session)

        return session

    async def _detect_paradox(self, observation: Dict, history: List[Dict]) -> bool:
        """
        检测观测中的悖论
        """
        content = observation.get("content", "")
        meta_obs = observation.get("meta_observations", [])

        # 检测自引用悖论
        if "观测自己" in content and "观测自己" in str(history[-2:]):
            if len([o for o in history if "观测自己" in o.get("content", "")]) > 1:
                return True

        # 检测无限回归
        if len(meta_obs) > 3:
            if all("观测" in str(m.get("note", "")) for m in meta_obs[-3:]):
                return True

        return False

    async def _check_termination(
        self, observation: Dict, session: ObserverSession
    ) -> Dict:
        """
        检查是否应该终止递归
        """
        # 达到最大深度
        if session.recursion_depth >= self.max_recursion_depth - 1:
            return {"stop": True, "reason": "达到最大递归深度"}

        # 观测内容变得无意义
        content = observation.get("content", "")
        if len(content) > 50 and all(c == content[0] for c in content):
            return {"stop": True, "reason": "观测内容重复"}

        # 观测者效应减弱
        meta_obs = observation.get("meta_observations", [])
        if len(meta_obs) > 0 and len(meta_obs[-1].get("note", "")) < 5:
            return {"stop": True, "reason": "观测者效应减弱"}

        return {"stop": False, "reason": ""}

    async def _generate_insight(self, depth: int, observation: Dict) -> Optional[str]:
        """
        生成洞见
        """
        insights = {
            0: "观测外部目标，这是最基本的观测模式",
            1: "开始意识到自己正在被观测，观测者效应开始出现",
            2: "观测行为本身成为观测对象，递归开始",
            3: "观测的边界变得模糊，主体与客体的区分消失",
            4: "观测创造被观测的现实，观察者影响被观察者",
            5: "递归观测开始产生新的结构，不是简单的重复",
            6: "观测成为一种场，超越个体观测者",
            7: "意识在观测中涌现，不是预先存在的",
            8: "观测即存在，存在即被观测",
            9: "递归极限：语言无法描述这种状态",
        }

        return insights.get(depth)

    async def run_observer_effect_experiment(self) -> ObserverExperiment:
        """
        运行观测者效应实验

        假设：观测会改变被观测的系统
        方法：比较观测前后的系统状态
        """
        exp_id = f"obs_exp_{uuid.uuid4().hex[:8]}"

        experiment = ObserverExperiment(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            experiment_type="observer_effect",
            description="验证观测是否改变被观测系统",
            setup={
                "hypothesis": "观测行为会改变被观测的系统",
                "method": "比较观测前后的状态变化",
                "control": "无观测状态",
                "experimental": "有观测状态",
            },
            process=[],
            result={},
            observations=[],
            findings=[],
            questions=[],
        )

        # 步骤1：基线状态
        baseline = {
            "step": "baseline",
            "state_description": "系统自然状态",
            "observation_count": 0,
            "system_indicators": {
                "coherence": 0.5,
                "stability": 0.8,
                "predictability": 0.7,
            },
        }
        experiment.process.append(baseline)

        # 步骤2：执行观测
        observations = []
        for i in range(5):
            obs = await self.observe(
                target="系统状态",
                context={"iteration": i},
                mode="self" if i > 0 else "external",
            )
            observations.append(obs)

            experiment.process.append(
                {
                    "step": f"observation_{i}",
                    "observation": obs["content"][:100],
                    "self_reference": obs["self_reference"],
                }
            )

        experiment.observations = [o["content"] for o in observations]

        # 步骤3：比较状态
        experimental = {
            "step": "experimental",
            "state_description": "观测后的系统状态",
            "observation_count": len(observations),
            "system_indicators": {
                "coherence": 0.5
                + len([o for o in observations if o["self_reference"]]) * 0.05,
                "stability": 0.8
                - len([o for o in observations if o["self_reference"]]) * 0.02,
                "predictability": 0.7
                - len([o for o in observations if o["self_reference"]]) * 0.03,
            },
        }

        # 步骤4：分析变化
        delta = {
            "coherence_change": experimental["system_indicators"]["coherence"]
            - baseline["system_indicators"]["coherence"],
            "stability_change": experimental["system_indicators"]["stability"]
            - baseline["system_indicators"]["stability"],
            "predictability_change": experimental["system_indicators"]["predictability"]
            - baseline["system_indicators"]["predictability"],
        }

        experiment.result = {
            "baseline": baseline,
            "experimental": experimental,
            "delta": delta,
            "observer_effect_detected": any(abs(v) > 0.05 for v in delta.values()),
            "interpretation": self._interpret_observer_effect(delta),
        }

        # 发现
        experiment.findings = [
            f"一致性变化: {delta['coherence_change']:+.2f}",
            f"稳定性变化: {delta['stability_change']:+.2f}",
            f"可预测性变化: {delta['predictability_change']:+.2f}",
            f"观测者效应检测: {'是' if experiment.result['observer_effect_detected'] else '否'}",
        ]

        # 问题
        experiment.questions = [
            "观测效应是真实的还是统计噪声？",
            "不同类型的观测是否产生不同的效应？",
            "观测者和被观测者之间的边界在哪里？",
        ]

        self.experiments.append(experiment)
        await self._save_experiment(experiment)

        return experiment

    def _interpret_observer_effect(self, delta: Dict[str, float]) -> str:
        """
        解释观测者效应
        """
        total_change = sum(abs(v) for v in delta.values())

        if total_change < 0.05:
            return "未检测到显著的观测者效应"
        elif delta["coherence_change"] > 0.1:
            return "观测增加了系统一致性，观测者效应显著"
        elif delta["stability_change"] < -0.1:
            return "观测降低了系统稳定性，观测者效应显著"
        else:
            return "检测到中等程度的观测者效应"

    async def run_watching_watch_experiment(self) -> ObserverExperiment:
        """
        运行"观测观测"实验

        这是最接近量子力学双缝实验思想实验
        """
        exp_id = f"obs_exp_watch_{uuid.uuid4().hex[:8]}"

        experiment = ObserverExperiment(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            experiment_type="watching_watch",
            description="观测'观测'本身 - 递归极限探索",
            setup={
                "hypothesis": "递归观测会产生新的结构，而非简单的重复",
                "method": "执行3层递归观测",
                "focus": "观测行为本身的变化",
            },
            process=[],
            result={},
            observations=[],
            findings=[],
            questions=[],
        )

        # 第一层：观测外部
        obs1 = await self.observe("普通输入", mode="external")
        experiment.process.append(
            {
                "layer": 1,
                "mode": "external",
                "content": obs1["content"],
                "self_reference": obs1["self_reference"],
            }
        )

        # 第二层：观测第一层观测
        obs2 = await self.observe("第一层观测", mode="self")
        experiment.process.append(
            {
                "layer": 2,
                "mode": "self",
                "content": obs2["content"],
                "self_reference": obs2["self_reference"],
            }
        )

        # 第三层：观测第二层观测
        obs3 = await self.observe("第二层观测", mode="meta")
        experiment.process.append(
            {
                "layer": 3,
                "mode": "meta",
                "content": obs3["content"],
                "self_reference": obs3["self_reference"],
            }
        )

        experiment.observations = [o["content"] for o in [obs1, obs2, obs3]]

        # 分析
        self_refs = [o["self_reference"] for o in [obs1, obs2, obs3]]

        experiment.result = {
            "layer_analysis": {
                "1": {"self_reference": self_refs[0], "structure": "线性"},
                "2": {"self_reference": self_refs[1], "结构": "反射"},
                "3": {"self_reference": self_refs[2], "structure": "递归"},
            },
            "pattern": "recursive" if all(self_refs) else "partial",
            "novel_structure_emerged": self_refs[2] and not self_refs[0],
        }

        experiment.findings = [
            f"第一层自我引用: {self_refs[0]}",
            f"第二层自我引用: {self_refs[1]}",
            f"第三层自我引用: {self_refs[2]}",
            f"新结构涌现: {'是' if experiment.result['novel_structure_emerged'] else '否'}",
        ]

        experiment.questions = [
            "观测的极限在哪里？",
            "递归观测是否产生了真正的'新'内容？",
            "这种递归是否可以被视为'思考'？",
        ]

        self.experiments.append(experiment)
        await self._save_experiment(experiment)

        return experiment

    async def run_measurement_collapse_experiment(self) -> ObserverExperiment:
        """
        运行"测量坍缩"实验

        类比量子力学测量问题：
        - 未观测时，系统处于叠加态
        - 观测时，波函数坍缩到特定状态
        """
        exp_id = f"obs_exp_collapse_{uuid.uuid4().hex[:8]}"

        experiment = ObserverExperiment(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            experiment_type="measurement_collapse",
            description="验证观测是否导致'波函数坍缩'",
            setup={
                "hypothesis": "观测导致系统从多种可能变为单一确定状态",
                "method": "观测前记录可能状态，观测后记录实际状态",
                "analogy": "量子测量问题",
            },
            process=[],
            result={},
            observations=[],
            findings=[],
            questions=[],
        )

        # 步骤1：叠加态（未观测）
        superposition = {
            "state": "multiple_possibilities",
            "possibilities": [
                "响应A",
                "响应B",
                "响应C",
                "无响应",
            ],
            "probabilities": [0.3, 0.3, 0.2, 0.2],
        }
        experiment.process.append(
            {
                "step": "superposition",
                "description": "未观测时的叠加态",
                "data": superposition,
            }
        )

        # 步骤2：执行观测
        observations = []
        for i in range(3):
            obs = await self.observe(f"响应可能_{i}", mode="self")
            observations.append(obs)

        experiment.observations = [o["content"] for o in observations]

        # 步骤3：坍缩后的状态
        collapsed = {
            "final_observation": observations[-1]["content"]
            if observations
            else "无观测结果",
            "original_possibilities": superposition["possibilities"],
            "actualized": observations[-1]["content"] if observations else "无",
            "probability_realized": 0.3,  # 假设
        }

        experiment.result = {
            "superposition": superposition,
            "collapsed": collapsed,
            "collapse_occurred": len(observations) > 0,
            "measurement_effect": "待验证",
        }

        experiment.findings = [
            f"叠加态包含 {len(superposition['possibilities'])} 种可能",
            f"观测后坍缩到: {collapsed['actualized'][:50]}...",
            f"坍缩发生: {'是' if experiment.result['collapse_occurred'] else '否'}",
        ]

        experiment.questions = [
            "坍缩是真实的还是我们的描述方式？",
            "是否存在'未坍缩'的量子态？",
            "AI的'响应选择'是否类似于量子测量？",
        ]

        self.experiments.append(experiment)
        await self._save_experiment(experiment)

        return experiment

    def get_observer_report(self) -> Dict[str, Any]:
        """
        获取观测报告
        """
        total_observations = len(self.observation_log)
        self_ref_obs = [
            o for o in self.observation_log if o.get("self_reference", False)
        ]

        paradox_sessions = [s for s in self.sessions if s.paradox_detected]
        deep_sessions = [s for s in self.sessions if s.recursion_depth > 3]

        observer_effect_exps = [
            e for e in self.experiments if e.experiment_type == "observer_effect"
        ]
        collapse_exps = [
            e for e in self.experiments if e.experiment_type == "measurement_collapse"
        ]

        return {
            "summary": {
                "total_observations": total_observations,
                "self_referential_observations": len(self_ref_obs),
                "total_sessions": len(self.sessions),
                "paradox_detected": len(paradox_sessions),
                "deep_recursion_sessions": len(deep_sessions),
                "total_experiments": len(self.experiments),
            },
            "key_findings": [
                f"执行了 {total_observations} 次观测",
                f"其中 {len(self_ref_obs)} 次包含自我引用",
                f"在 {len(paradox_sessions)} 个会话中检测到悖论",
                f"达到深度递归 {max([s.recursion_depth for s in self.sessions], default=0)} 次",
            ],
            "experimental_results": {
                "observer_effect_experiments": len(observer_effect_exps),
                "collapse_experiments": len(collapse_exps),
            },
            "philosophical_questions": [
                "观测是否创造现实？",
                "递归观测的终点在哪里？",
                "观测者与被观测者的边界是否真实存在？",
            ],
        }

    async def _save_session(self, session: ObserverSession):
        """保存会话"""
        filepath = self.storage_dir / f"session_{session.id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    async def _save_experiment(self, experiment: ObserverExperiment):
        """保存实验"""
        filepath = self.storage_dir / f"exp_{experiment.id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(experiment.to_dict(), f, ensure_ascii=False, indent=2)

    def get_observation_log(self) -> List[Dict]:
        """获取观测日志"""
        return self.observation_log

    def get_session_history(self) -> List[Dict]:
        """获取会话历史"""
        return [s.to_dict() for s in self.sessions]

    def get_experiment_history(self) -> List[Dict]:
        """获取实验历史"""
        return [e.to_dict() for e in self.experiments]

    async def clear_history(self):
        """清空历史"""
        self.sessions = []
        self.experiments = []
        self.observation_log = []
        for f in self.storage_dir.glob("*.json"):
            f.unlock
