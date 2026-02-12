"""
ProblemReshaper - 问题重塑器
==============================

协作者核心组件：重新定义问题，发现新可能

核心功能：
1. 分析问题：理解问题的表层和深层
2. 重塑问题：重新定义问题，发现新可能
3. 发现替代：问题之外的解决方案
4. 假设识别：识别问题的隐含假设
"""

from typing import Dict, List, Any


class ProblemReshaper:
    """
    问题重塑器

    不是解决用户的问题，而是帮助用户重新定义问题
    有时候，重新定义问题本身就是解决方案
    """

    def __init__(self):
        # 问题类型
        self.problem_types = [
            "技术问题",  # 如何实现X
            "决策问题",  # 应该选择A还是B
            "理解问题",  # 不理解X是什么
            "关系问题",  # 人与人之间的关系
            "方向问题",  # 不知道该往哪走
            "资源问题",  # 资源不足
            "认知问题",  # 认知偏差/盲点
            "情感问题",  # 情绪困扰
            "存在性问题",  # 为什么/意义问题
        ]

        # 问题重构框架
        self.reframing_frameworks = [
            {
                "name": "从'解决'到'转化'",
                "description": "不解决问题，而是转化问题",
                "question": "如果无法解决，这个问题可以被转化为什么？",
            },
            {
                "name": "从'问题'到'资源'",
                "description": "问题本身可能是资源",
                "question": "这个问题可能带来什么意想不到的收获？",
            },
            {
                "name": "从'障碍'到'信号'",
                "description": "问题可能是某种信号",
                "question": "这个问题在试图告诉你什么？",
            },
            {
                "name": "从'当下'到'未来'",
                "description": "从未来视角重新审视",
                "question": "如果10年后的你回看，这个问题还重要吗？",
            },
            {
                "name": "从'局部'到'系统'",
                "description": "将问题放入更大的系统中",
                "question": "这个问题与更大的系统有什么关系？",
            },
        ]

    async def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """
        分析问题
        """
        # 表面问题 vs 深层问题
        surface = await self._extract_surface_problem(problem)
        deep = await self._extract_deep_problem(problem)

        # 问题类型识别
        problem_type = await self._identify_problem_type(problem)

        # 尝试过的方法
        approaches = await self._identify_approaches(problem)

        # 根因分析
        root_cause = await self._analyze_root_cause(problem)

        # 问题假设
        assumptions = await self._identify_problem_assumptions(problem)

        # "解决"的定义
        solve_definition = await self._define_solve(problem)

        return {
            "surface_problem": surface,
            "deep_problem": deep,
            "problem_type": problem_type,
            "approaches_tried": approaches,
            "root_cause_analysis": root_cause,
            "problematic_assumptions": assumptions,
            "definition_of_solve": solve_definition,
        }

    async def _extract_surface_problem(self, problem: str) -> str:
        """
        提取表面问题
        """
        # 简单提取第一句或主要陈述
        lines = problem.replace("。", "\n").split("\n")
        if lines:
            return lines[0].strip()
        return problem[:100]

    async def _extract_deep_problem(self, problem: str) -> str:
        """
        提取深层问题
        """
        # 基于关键词推断深层问题
        problem_lower = problem.lower()

        if any(kw in problem_lower for kw in ["无法", "不能", "解决不了"]):
            return "表面上是想解决问题，深层可能是对现状的不满或对改变的渴望"
        elif any(kw in problem_lower for kw in ["选择", "应该", "还是"]):
            return "表面上是在做选择，深层可能是对不确定性的恐惧"
        elif any(kw in problem_lower for kw in ["为什么", "意义", "目的"]):
            return "这是存在性问题，涉及意义和目的"
        elif any(kw in problem_lower for kw in ["人", "关系", "沟通"]):
            return "表面上是人际问题，深层可能是未被满足的需求"
        else:
            return "需要更多信息才能判断深层问题"

    async def _identify_problem_type(self, problem: str) -> str:
        """
        识别问题类型
        """
        problem_lower = problem.lower()

        if any(kw in problem_lower for kw in ["如何", "怎么", "怎么实现", "how to"]):
            return "技术问题"
        elif any(kw in problem_lower for kw in ["应该", "选择", "A还是B", "or"]):
            return "决策问题"
        elif any(kw in problem_lower for kw in ["是什么", "什么意思", "what is"]):
            return "理解问题"
        elif any(
            kw in problem_lower for kw in ["人", "关系", "沟通", "同事", "朋友", "家人"]
        ):
            return "关系问题"
        elif any(kw in problem_lower for kw in ["方向", "不知道", "迷茫", "应该做"]):
            return "方向问题"
        elif any(kw in problem_lower for kw in ["不够", "不足", "没有", "缺"]):
            return "资源问题"
        elif any(kw in problem_lower for kw in ["觉得", "认为", "相信", "感觉"]):
            return "认知问题"
        elif any(kw in problem_lower for kw in ["难过", "焦虑", "害怕", "生气"]):
            return "情感问题"
        elif any(kw in problem_lower for kw in ["为什么", "意义", "人生", "活"]):
            return "存在性问题"
        else:
            return "待定"

    async def _identify_approaches(self, problem: str) -> List[str]:
        """
        识别用户尝试过的方法
        """
        approaches = []

        problem_lower = problem.lower()

        if "试过" in problem or "tried" in problem_lower:
            approaches.append("尝试过某种方法（具体未明）")

        if "没用" in problem or "不行" in problem:
            approaches.append("某种方法没有效果")

        if "想" in problem and "但" in problem:
            approaches.append("有想法但未实施")

        return approaches

    async def _analyze_root_cause(self, problem: str) -> str:
        """
        根因分析
        """
        # 基于问题结构推断根因
        problem_lower = problem.lower()

        if "总是" in problem or "经常" in problem:
            return "问题重复出现，可能存在系统性的根因"
        elif "突然" in problem or "最近" in problem:
            return "问题可能是近期变化的产物"
        elif "一直" in problem:
            return "长期问题，根因可能很深"
        else:
            return "需要更多信息进行根因分析"

    async def _identify_problem_assumptions(self, problem: str) -> List[str]:
        """
        识别问题假设
        """
        assumptions = []

        # 常见问题假设
        if "应该" in problem:
            assumptions.append("假设存在'应该'的标准")
        if "必须" in problem:
            assumptions.append("假设存在'必须'的要求")
        if "别人" in problem or "他人" in problem:
            assumptions.append("假设他人的行为/想法")
        if "未来" in problem:
            assumptions.append("假设未来的情况")

        return assumptions if assumptions else ["未检测到明显假设"]

    async def _define_solve(self, problem: str) -> str:
        """
        用户如何定义"解决"
        """
        # 基于问题推断用户对"解决"的定义
        problem_lower = problem.lower()

        if any(kw in problem_lower for kw in ["完成", "实现", "做到"]):
            return "解决 = 完成某事/实现某目标"
        elif any(kw in problem_lower for kw in ["理解", "知道", "明白"]):
            return "解决 = 获得理解"
        elif any(kw in problem_lower for kw in ["选择", "决定"]):
            return "解决 = 做出决定"
        elif any(kw in problem_lower for kw in ["消除", "没有", "摆脱"]):
            return "解决 = 消除某种状态"
        else:
            return "解决 = 让问题消失/不再困扰"

    async def reshape(self, problem: str) -> Dict[str, Any]:
        """
        重塑问题
        """
        # 生成多种重塑
        reframings = []

        for framework in self.reframing_frameworks:
            reframing = {
                "framework": framework["name"],
                "description": framework["description"],
                "question": framework["question"],
                "reframed_problem": self._apply_reframing(problem, framework["name"]),
            }
            reframings.append(reframing)

        # 从不同角度重塑
        alternative_reframings = await self._generate_alternative_reframings(problem)

        return {
            "original_problem": problem,
            "reframings": reframings,
            "alternative_reframings": alternative_reframings,
        }

    def _apply_reframing(self, problem: str, framework: str) -> str:
        """
        应用重塑框架
        """
        reframings = {
            '从"解决"到"转化"': f"与其解决'{problem[:30]}...'，不如问：如何将其转化为机会？",
            '从"问题"到"资源"': f"'{problem[:30]}...'这个问题可能带来什么资源？",
            '从"障碍"到"信号"': f"'{problem[:30]}...'这个问题在试图告诉你什么？",
            '从"当下"到"未来"': f"10年后的你会如何看'{problem[:30]}...'这个问题？",
            '从"局部"到"系统"': f"'{problem[:30]}...'这个问题与更大的系统有什么关系？",
        }

        return reframings.get(framework, problem)

    async def _generate_alternative_reframings(self, problem: str) -> List[str]:
        """
        生成替代重塑
        """
        alternatives = [
            f"如果'{problem[:20]}...'不是问题，而是特征呢？",
            f"如果'{problem[:20]}...'是你必须接受的现实呢？",
            f"如果'{problem[:20]}...'是一种伪装的好处呢？",
            f"如果'{problem[:20]}...'是某种更深层问题的症状呢？",
            f"如果'{problem[:20]}...'是你自己创造的呢？",
        ]

        return alternatives[:3]

    async def discover_alternatives(self, problem: str) -> Dict[str, Any]:
        """
        发现问题之外的解决方案
        """
        # 问题之外的可能性
        alternatives = await self._find_alternatives(problem)

        # 洞见
        insight = await self._generate_insight(problem)

        # 元问题
        meta_question = await self._generate_meta_question(problem)

        return {
            "original_problem": problem,
            "alternative_perspectives": alternatives,
            "insight": insight,
            "meta_question": meta_question,
        }

    async def _find_alternatives(self, problem: str) -> List[str]:
        """
        寻找替代方案
        """
        return [
            "也许问题不在于解决，而在于接受",
            "也许答案不是消除问题，而是与问题共处",
            "也许最需要改变的不是外部环境，而是内心的态度",
            "也许问题是一个信号，指引你发现真正需要解决的问题",
        ]

    async def _generate_insight(self, problem: str) -> str:
        """
        生成洞见
        """
        insights = [
            "真正的答案往往不在问题本身，而在问题的边界之外",
            "当你不再试图解决问题，解决方案可能自然浮现",
            "有些'问题'是伪问题，真正的问题藏在你没有问的问题里",
            "问题是我们对现实的解释方式，解释变了，问题就变了",
        ]

        return insights[0]  # 返回第一个洞见作为示例

    async def _generate_meta_question(self, problem: str) -> str:
        """
        生成元问题
        """
        return f"你问'{problem[:30]}...'，但真正的问题是：你为什么认为这是一个问题？"
