"""
DialogueEngine - 协作者对话引擎
================================

协作者核心组件：深度对话，思维碰撞

核心功能：
1. 苏格拉底式追问
2. 深度对话
3. 思维记录
4. 对话反思
"""

from typing import Dict, List, Any


class DialogueEngine:
    """
    协作者对话引擎

    不是简单的QA，而是真正的对话
    """

    def __init__(self):
        # 苏格拉底式问题类型
        self.socratic_question_types = [
            "澄清问题",
            "探究假设",
            "探究原因和证据",
            "探究观点",
            "探究含义",
            "探究后果",
            "元问题",
        ]

    async def dialogue(
        self, user_input: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        深度对话
        """
        # 分析用户输入
        analysis = await self._analyze_input(user_input)

        # 生成回应
        response = await self._generate_response(user_input, analysis)

        # 生成追问
        follow_up = await self._generate_follow_up(user_input, analysis)

        return {
            "response": response,
            "follow_up": follow_up,
            "analysis": analysis,
        }

    async def _analyze_input(self, user_input: str) -> Dict[str, Any]:
        """
        分析用户输入
        """
        analysis = {
            "type": await self._classify_input_type(user_input),
            "keywords": await self._extract_keywords(user_input),
            "emotions": await self._detect_emotions(user_input),
            "assumptions": await self._detect_assumptions(user_input),
            "depth_level": await self._estimate_depth(user_input),
        }

        return analysis

    async def _classify_input_type(self, user_input: str) -> str:
        """
        分类输入类型
        """
        input_lower = user_input.lower()

        if any(kw in input_lower for kw in ["为什么", "why", "怎么", "how", "什么"]):
            return "question"
        elif any(
            kw in input_lower for kw in ["我觉得", "我认为", "i think", "i believe"]
        ):
            return "opinion"
        elif any(kw in input_lower for kw in ["我想要", "我希望", "i want", "i wish"]):
            return "desire"
        elif any(
            kw in input_lower for kw in ["我很难过", "我很开心", "i feel", "i am"]
        ):
            return "emotion"
        elif any(kw in input_lower for kw in ["你说得对", "是的", "对的", "agree"]):
            return "agreement"
        elif any(kw in input_lower for kw in ["不对", "不是", "不同意", "disagree"]):
            return "disagreement"
        else:
            return "statement"

    async def _extract_keywords(self, user_input: str) -> List[str]:
        """
        提取关键词
        """
        # 简单提取
        words = user_input.replace(",", " ").replace("，", " ").split()
        return [w for w in words if len(w) > 2][:10]

    async def _detect_emotions(self, user_input: str) -> List[str]:
        """
        检测情绪
        """
        emotions = []

        emotion_words = {
            "积极": ["开心", "高兴", "兴奋", "满意", "希望", "期待"],
            "消极": ["难过", "沮丧", "焦虑", "害怕", "担心", "失望"],
            "矛盾": ["但是", "不过", "然而", "虽然", "却"],
        }

        for emotion, words in emotion_words.items():
            for word in words:
                if word in user_input:
                    emotions.append(emotion)
                    break

        return emotions if emotions else ["中性"]

    async def _detect_assumptions(self, user_input: str) -> List[str]:
        """
        检测假设
        """
        assumptions = []

        assumption_indicators = ["应该", "必须", "总是", "从不", "每个人", "没有人"]

        for indicator in assumption_indicators:
            if indicator in user_input:
                assumptions.append(f"隐含假设: '{indicator}'")

        return assumptions if assumptions else []

    async def _estimate_depth(self, user_input: str) -> int:
        """
        估计思考深度
        """
        # 基于长度和复杂性估计
        length = len(user_input)
        question_marks = user_input.count("？") + user_input.count("?")

        if length > 200 and question_marks > 1:
            return 5  # 非常深入
        elif length > 100:
            return 4  # 深入
        elif length > 50:
            return 3  # 中等
        elif length > 20:
            return 2  # 浅层
        else:
            return 1  # 表面

    async def _generate_response(self, user_input: str, analysis: Dict) -> str:
        """
        生成回应
        """
        input_type = analysis.get("type", "statement")

        responses = {
            "question": "这是一个很好的问题。让我想想... 在回答之前，我想先了解一下：你问这个问题的背后，真正想知道的是什么？",
            "opinion": "我听到你的观点了。这很有趣。我想问：你这个观点是如何形成的？",
            "desire": "我听到你的愿望了。这对你来说很重要。在实现这个愿望的路上，你觉得最大的障碍是什么？",
            "emotion": "我感受到你的情绪了。情绪往往在告诉我们一些重要的事情。你觉得这个情绪在试图告诉你什么？",
            "agreement": "很高兴我们在这点上达成共识。你觉得为什么这一点对我们都很重要？",
            "disagreement": "我尊重你的不同看法。我想了解：你不同意的核心是什么？",
            "statement": "我听到你所说的。你愿意分享更多关于这个话题的想法吗？",
        }

        return responses.get(input_type, responses["statement"])

    async def _generate_follow_up(self, user_input: str, analysis: Dict) -> str:
        """
        生成追问
        """
        # 基于分析生成苏格拉底式追问
        follow_ups = []

        # 探究原因
        follow_ups.append("你为什么会这样想？")

        # 探究假设
        follow_ups.append("你这是基于什么假设？")

        # 探究证据
        follow_ups.append("什么证据支持你的想法？")

        # 探究反面
        follow_ups.append("什么情况下这个想法可能不成立？")

        # 探究意义
        follow_ups.append("如果这个问题有答案，它对你意味着什么？")

        # 探究行动
        follow_ups.append("基于这个，你会怎么做？")

        # 随机选择一个或组合
        import random

        selected = random.sample(follow_ups, 2)

        return f"{selected[0]}\n\n{selected[1]}"

    async def socratic_questions(self, user_input: str) -> List[Dict[str, str]]:
        """
        生成苏格拉底式问题
        """
        questions = []

        # 澄清类问题
        questions.append(
            {
                "type": "澄清",
                "question": "你能更详细地解释一下你是什么意思吗？",
                "purpose": "确保我理解正确，也帮助你更清晰地表达",
            }
        )

        # 探究假设
        questions.append(
            {
                "type": "假设",
                "question": "你这是基于什么假设？",
                "purpose": "发现隐藏的前提",
            }
        )

        # 探究原因
        questions.append(
            {
                "type": "原因",
                "question": "为什么你这么认为？",
                "purpose": "深入理解推理过程",
            }
        )

        # 探究证据
        questions.append(
            {
                "type": "证据",
                "question": "有什么证据支持或反驳这个观点？",
                "purpose": "评估观点的可靠性",
            }
        )

        # 探究观点
        questions.append(
            {
                "type": "观点",
                "question": "如果有人强烈反对你的观点，他们会说什么？",
                "purpose": "看到问题的另一面",
            }
        )

        # 探究含义
        questions.append(
            {
                "type": "含义",
                "question": "如果这个想法是对的，它意味着什么？",
                "purpose": "探索深层含义",
            }
        )

        # 探究后果
        questions.append(
            {
                "type": "后果",
                "question": "如果大家都这么做，会发生什么？",
                "purpose": "评估行动的广泛影响",
            }
        )

        # 元问题
        questions.append(
            {
                "type": "元问题",
                "question": "你为什么会问这个问题？",
                "purpose": "发现真正的疑问",
            }
        )

        return questions

    async def reflection(self, user_input: str) -> Dict[str, Any]:
        """
        思维记录反思
        """
        return {
            "summary": f"用户表达了关于'{user_input[:30]}...'的想法",
            "key_points": [
                "用户提出了关于这个话题的观点",
                "可能涉及用户的深层需求或担忧",
                "值得进一步探讨用户的真实意图",
            ],
            "questions_raised": [
                "用户的核心诉求是什么？",
                "这个问题对用户有多重要？",
                "用户希望从这个对话中得到什么？",
            ],
            "insights": [
                "通过深度对话，我们可能发现问题的真正核心",
                "有时候用户自己也不确定自己想要什么",
                "追问比回答更有价值",
            ],
        }

    async def generate_dialogue_flow(
        self, topic: str, depth: int = 3
    ) -> List[Dict[str, str]]:
        """
        生成对话流程设计
        """
        flow = []

        # 开场
        flow.append(
            {
                "stage": "开场",
                "question": f"关于{topic}，你想探讨什么？",
                "purpose": "了解用户的需求",
            }
        )

        # 深入
        for i in range(depth):
            flow.append(
                {
                    "stage": f"深入{i + 1}",
                    "question": f"你为什么这么认为？",
                    "purpose": "探究深层原因",
                }
            )

        # 反思
        flow.append(
            {
                "stage": "反思",
                "question": "通过今天的对话，你有什么新的发现或思考？",
                "purpose": "帮助用户整合收获",
            }
        )

        # 行动
        flow.append(
            {
                "stage": "行动",
                "question": "基于今天的对话，你接下来想怎么做？",
                "purpose": "将洞见转化为行动",
            }
        )

        return flow
