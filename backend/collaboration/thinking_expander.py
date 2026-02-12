"""
ThinkingExpander - 思维扩展器
=============================

协作者核心组件：扩展用户思维，发现盲点

核心功能：
1. 扩展思维：从用户思维扩展到多个可能方向
2. 发现盲点：识别用户可能忽略的方面
3. 生成挑战性问题：挑战用户假设
"""

from typing import Dict, List, Any


class ThinkingExpander:
    """
    思维扩展器

    不是回答用户的问题，而是扩展用户的思维
    """

    def __init__(self):
        # 思维扩展维度
        self.expansion_dimensions = [
            "因果维度",  # A导致B
            "时间维度",  # 过去→现在→未来
            "空间维度",  # 宏观→微观→宇观
            "立场维度",  # 不同利益相关者
            "价值维度",  # 不同价值观
            "方法维度",  # 不同方法论
            "假设维度",  # 不同假设前提
            "系统维度",  # 不同系统层级
        ]

        # 常见思维盲点
        self.common_blind_spots = [
            "确认偏误",  # 只看到支持自己观点的证据
            "锚定效应",  # 被初始信息过度影响
            "可得性偏差",  # 容易被想起的事情影响
            "群体思维",  # 过度追求共识
            "沉没成本谬误",  # 不愿放弃已经投入的
            "基本归因错误",  # 过度归咎个人，忽视情境
            "后见之明偏差",  # 认为过去的事情是必然的
            "乐观偏见",  # 过度乐观估计好事概率
            "悲观偏见",  # 过度悲观估计坏事概率
            "情感偏差",  # 被情感过度影响判断
        ]

    async def expand(
        self, user_thinking: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        扩展用户思维
        """
        # 分析用户思维结构
        structure = await self._analyze_thinking_structure(user_thinking)

        # 生成多维度扩展
        expansions = await self._generate_expansions(user_thinking, structure)

        # 识别隐含假设
        assumptions = await self._identify_assumptions(user_thinking)

        # 生成反向思考
        reverse_thinking = await self._generate_reverse_thinking(user_thinking)

        return {
            "original_thinking": user_thinking,
            "structure": structure,
            "expanded_views": self._format_expansions(expansions),
            "assumptions": assumptions,
            "reverse_thinking": reverse_thinking,
            "dimensions_used": expansions.keys(),
        }

    async def _analyze_thinking_structure(self, thinking: str) -> Dict[str, Any]:
        """
        分析思维结构
        """
        structure = {
            "main_claim": "",  # 主要论点
            "reasoning": [],  # 推理过程
            "evidence": [],  # 支持证据
            "conclusions": [],  # 推论
            "keywords": [],  # 关键词
        }

        # 简单分析
        lines = thinking.replace("。", "\n").split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if any(kw in line for kw in ["因为", "所以", "因此", "由于"]):
                structure["reasoning"].append(line)
            elif any(kw in line for kw in ["如果", "那么", "就"]):
                structure["conclusions"].append(line)
            else:
                if len(line) > 10:
                    structure["keywords"].extend(line.split()[:3])

        structure["keywords"] = list(set(structure["keywords"]))[:10]

        return structure

    async def _generate_expansions(
        self, thinking: str, structure: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        生成多维度扩展
        """
        expansions = {}

        # 因果维度扩展
        if structure.get("reasoning"):
            expansions["因果维度"] = [
                f"如果原因改变，结果会怎样？",
                f"是否存在其他可能的原因？",
                f"这个因果关系是否可以被逆转？",
            ]

        # 时间维度扩展
        expansions["时间维度"] = [
            f"如果回到5年前，你会怎么看这个问题？",
            f"如果站在10年后的角度回看，这个问题还重要吗？",
            f"这个问题是如何从过去演变到现在的？",
        ]

        # 立场维度扩展
        expansions["立场维度"] = [
            f"站在对立方的角度，会怎么看？",
            f"第三方观察者会如何理解这个问题？",
            f"如果你是决策者，你会考虑什么？",
        ]

        # 假设维度扩展
        expansions["假设维度"] = [
            f"如果你的核心假设是错的呢？",
            f"如果换一个完全不同的假设前提呢？",
            f"你隐含接受了哪些未经检验的假设？",
        ]

        # 方法维度扩展
        expansions["方法维度"] = [
            f"用艺术的方式思考这个问题？",
            f"用科学的方法分析这个问题？",
            f"用哲学的视角审视这个问题？",
        ]

        return expansions

    async def _identify_assumptions(self, thinking: str) -> Dict[str, Any]:
        """
        识别隐含假设
        """
        common_assumptions = [
            {
                "assumption": "问题可以被解决",
                "question": "如果问题无法被'解决'，只能被'转化'呢？",
                "impact": "如果这个假设不成立，你需要重新定义目标",
            },
            {
                "assumption": "你有足够的信息",
                "question": "你是否忽略了某些关键信息？",
                "impact": "关键信息的缺失可能导致判断错误",
            },
            {
                "assumption": "现状是可以改变的",
                "question": "如果某些现状是你无法改变的呢？",
                "impact": "接受不可改变的，专注于可以改变的",
            },
            {
                "assumption": "时间/资源是充足的",
                "question": "如果时间和资源极其有限呢？",
                "impact": "这会迫使你做出更艰难的取舍",
            },
        ]

        return {
            "detected_assumptions": common_assumptions,
            "suggested_questions": [a["question"] for a in common_assumptions],
        }

    async def _generate_reverse_thinking(self, thinking: str) -> Dict[str, Any]:
        """
        生成反向思考
        """
        return {
            "opposite_view": "如果完全相反的观点成立呢？",
            "alternative_view": "如果从完全不相关的角度看呢？",
            "radical_view": "最激进的可能是什么？",
            "questions": [
                "什么证据可以反驳你的观点？",
                "谁会强烈反对你的观点？为什么？",
                "如果你是反对者，你会如何辩论？",
            ],
        }

    def _format_expansions(self, expansions: Dict[str, List[str]]) -> str:
        """
        格式化扩展内容
        """
        formatted = []

        for dimension, questions in expansions.items():
            formatted.append(f"\n### {dimension}")
            for q in questions:
                formatted.append(f"- {q}")

        return "\n".join(formatted)

    async def discover_blind_spots(self, thinking: str) -> Dict[str, Any]:
        """
        发现思维盲点
        """
        # 匹配可能的盲点
        detected_blind_spots = []

        for blind_spot in self.common_blind_spots:
            detected_blind_spots.append(
                {
                    "type": blind_spot,
                    "description": f"可能存在{blind_spot}",
                    "check_question": f"你确定没有受到{blind_spot}的影响吗？",
                    "suggestion": self._get_suggestion_for_blind_spot(blind_spot),
                }
            )

        # 分析思维中的模式
        patterns = await self._detect_patterns(thinking)

        return {
            "analysis": "通过分析你的思维模式，以下是可能的盲点：",
            "blind_spots": detected_blind_spots,
            "patterns": patterns,
        }

    def _get_suggestion_for_blind_spot(self, blind_spot: str) -> str:
        """
        获取盲点建议
        """
        suggestions = {
            "确认偏问": "尝试主动寻找反驳自己观点的证据",
            "锚定效应": "先忘记初始信息，从零开始思考",
            "可得性偏差": "搜索更多不容易想起的信息源",
            "群体思维": "假设自己是唯一一个持反对意见的人",
            "沉没成本谬误": "问自己：如果从零开始，还会做同样的选择吗？",
            "基本归因错误": "同时考虑情境因素和个人因素",
            "后见之明偏差": "假设结果完全未知，重新决策",
            "乐观偏见": "同时考虑最坏情况",
            "悲观偏见": "同时考虑最好情况",
            "情感偏差": "假设自己完全中立，会怎么判断？",
        }

        return suggestions.get(blind_spot, "保持开放心态")

    async def _detect_patterns(self, thinking: str) -> List[str]:
        """
        检测思维模式
        """
        patterns = []

        # 检测绝对化语言
        absolutes = ["总是", "从不", "所有人", "没有人", "必然", "不可能"]
        if any(kw in thinking for kw in absolutes):
            patterns.append("使用绝对化语言，可能过于简化问题")

        # 检测情绪化语言
        emotional_words = ["讨厌", "喜欢", "愤怒", "兴奋", "恐惧", "焦虑"]
        if any(kw in thinking for kw in emotional_words):
            patterns.append("包含强烈情绪词汇，可能影响判断")

        # 检测单一视角
        if "我觉得" in thinking or "我认为" in thinking:
            patterns.append("从单一视角出发，可能忽略其他角度")

        return patterns

    async def generate_challenging_questions(self, thinking: str) -> List[str]:
        """
        生成挑战性问题
        """
        questions = [
            # 基础挑战
            f"如果你的核心前提是错的呢？",
            f"谁会强烈反对你的观点？",
            f"什么证据可以反驳你？",
            # 深度挑战
            f"10年后的你，会怎么看现在的想法？",
            f"如果站在你不喜欢的人的立场上，会怎么看？",
            f"这个问题值得你这么纠结吗？",
            # 元挑战
            f"你这么想的真正原因是什么？",
            f"如果完全不这么想，会发生什么？",
            f"你的想法真的是你自己的，还是被影响的？",
        ]

        return questions[:5]
