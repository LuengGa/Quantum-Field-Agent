"""
PerspectiveGenerator - 全新视角生成器
=====================================

协作者核心组件：跳出惯性，发现新可能

核心功能：
1. 多维视角生成
2. 惯性思维挑战
3. 反常识视角
4. 未来/过去视角
"""

from typing import Dict, List, Any


class PerspectiveGenerator:
    """
    全新视角生成器

    不是给用户一个答案，而是给用户一个全新的看问题的角度
    """

    def __init__(self):
        # 视角维度
        self.perspective_dimensions = [
            "时间维度",
            "空间维度",
            "立场维度",
            "价值维度",
            "方法维度",
            "系统维度",
            "身份维度",
            "情感维度",
        ]

        # 惯性思维模式
        self.inertia_patterns = [
            "线性思维",  # 只看直线路径
            "二元对立",  # 非此即彼
            "短期视角",  # 只看眼前
            "自我中心",  # 只从自己角度看
            "经验局限",  # 用过去经验判断未来
            "群体思维",  # 随大流
            "确认偏误",  # 只看到想看的
            "固定思维",  # 相信事物不变",
        ]

    async def generate(
        self, topic: str, context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        生成多维视角
        """
        perspectives = []

        # 时间视角
        perspectives.append(await self._time_perspective(topic))

        # 空间视角
        perspectives.append(await self._space_perspective(topic))

        # 立场视角
        perspectives.append(await self._stakeholder_perspective(topic))

        # 价值视角
        perspectives.append(await self._value_perspective(topic))

        # 身份视角
        perspectives.append(await self._identity_perspective(topic))

        # 系统视角
        perspectives.append(await self._system_perspective(topic))

        # 情感视角
        perspectives.append(await self._emotion_perspective(topic))

        # 未来视角
        perspectives.append(await self._future_perspective(topic))

        return perspectives

    async def _time_perspective(self, topic: str) -> Dict[str, Any]:
        """
        时间维度视角
        """
        return {
            "perspective": "时间维度",
            "explanation": "从过去、现在、未来的时间轴来看这个问题",
            "insight": f"""
- 1年前的你，会怎么看这个问题？
- 现在的你，正处于问题的哪个阶段？
- 1年后的你，会希望现在的你怎么做？
- 10年后的你，回看今天的决定，会是什么感受？
- 如果这个问题发生在100年前，有什么不同？
- 这个问题在100年后还存在吗？
            """.strip(),
            "key_question": "站在时间的哪个点上，能给你最好的视角？",
        }

    async def _space_perspective(self, topic: str) -> Dict[str, Any]:
        """
        空间维度视角
        """
        return {
            "perspective": "空间维度",
            "explanation": "从不同的空间/位置来看这个问题",
            "insight": f"""
- 从宏观宇宙的视角看，这个问题有多重要？
- 从微观细胞的角度看，类似问题如何解决？
- 如果你在另一个国家，会怎么看？
- 从这个问题发生地的角度看，有什么不同？
- 从天空俯视，从地下仰视，各是什么景象？
            """.strip(),
            "key_question": "站在什么位置，能让你看得最清楚？",
        }

    async def _stakeholder_perspective(self, topic: str) -> Dict[str, Any]:
        """
        利益相关者视角
        """
        return {
            "perspective": "利益相关者视角",
            "explanation": "从不同人的角度来看这个问题",
            "insight": f"""
- 这个问题直接影响到的人会怎么看？
- 与这个问题无关的第三方会怎么看？
- 你的家人会希望你怎么做？
- 你的对手会希望你怎么做？
- 未来的你会怎么看现在的决定？
- 如果你是决策者，会考虑什么因素？
            """.strip(),
            "key_question": "谁的利益最需要被考虑？",
        }

    async def _value_perspective(self, topic: str) -> Dict[str, Any]:
        """
        价值观视角
        """
        return {
            "perspective": "价值观视角",
            "explanation": "从不同的价值观角度来看这个问题",
            "insight": f"""
- 如果效率是唯一标准，应该怎么做？
- 如果公平是唯一标准，应该怎么做？
- 如果快乐是唯一标准，应该怎么做？
- 如果长远利益是唯一标准，应该怎么做？
- 如果个人自由是唯一标准，应该怎么做？
- 如果社会责任是唯一标准，应该怎么做？
            """.strip(),
            "key_question": "什么价值对你来说最重要？",
        }

    async def _identity_perspective(self, topic: str) -> Dict[str, Any]:
        """
        身份视角
        """
        return {
            "perspective": "身份视角",
            "explanation": "从不同的身份角色来看这个问题",
            "insight": f"""
- 如果你是这个世界最重要的领袖，会怎么做？
- 如果你是一个刚出生的婴儿，会怎么看这个问题？
- 如果你是活了1000年的存在，会怎么做？
- 如果你是一个外星人，第一次遇到这个问题，会怎么想？
- 如果你是你的偶像/榜样，会怎么处理这个问题？
            """.strip(),
            "key_question": "你愿意成为什么样的人来处理这个问题？",
        }

    async def _system_perspective(self, topic: str) -> Dict[str, Any]:
        """
        系统视角
        """
        return {
            "perspective": "系统视角",
            "explanation": "将问题放入更大的系统中看",
            "insight": f"""
- 这个问题是更大系统的症状还是原因？
- 改变这个问题会如何影响整个系统？
- 系统中还有什么其他问题与这个相关？
- 系统的反馈机制如何影响这个问题？
- 系统的边界在哪里？
            """.strip(),
            "key_question": "这个系统中最关键杠杆点在哪里？",
        }

    async def _emotion_perspective(self, topic: str) -> Dict[str, Any]:
        """
        情感视角
        """
        return {
            "perspective": "情感视角",
            "explanation": "从不同的情感状态来看这个问题",
            "insight": f"""
- 如果你此刻感到恐惧，会怎么处理？
- 如果你此刻感到兴奋，会怎么处理？
- 如果你此刻感到平静，会怎么处理？
- 如果你此刻感到愤怒，会怎么处理？
- 如果你此刻感到好奇，会怎么处理？
            """.strip(),
            "key_question": "你的情绪想告诉你什么？",
        }

    async def _future_perspective(self, topic: str) -> Dict[str, Any]:
        """
        未来视角
        """
        return {
            "perspective": "未来视角",
            "explanation": "从未来的角度回看",
            "insight": f"""
- 10年后的世界，这个问题还存在吗？
- 未来成功的人会如何看待今天的决定？
- 如果你已经解决了这个问题，回头看，最重要的洞见是什么？
- 未来的技术/社会会如何改变这个问题？
- 你希望未来记住今天的什么？
            """.strip(),
            "key_question": "为了未来的自己，你现在应该做什么？",
        }

    async def challenge_assumptions(self, topic: str) -> List[str]:
        """
        挑战惯性思维
        """
        challenges = []

        # 挑战线性思维
        challenges.append(
            f"关于'{topic[:20]}...'，你认为的直线路径可能不是最快的。如果绕道反而更快呢？"
        )

        # 挑战二元对立
        challenges.append(
            f"关于'{topic[:20]}...'，你假设只有A或B的选择。但C或D是否存在？"
        )

        # 挑战短期视角
        challenges.append(
            f"关于'{topic[:20]}...'，只考虑短期后果可能忽略了长期影响。5年、10年后呢？"
        )

        # 挑战自我中心
        challenges.append(
            f"关于'{topic[:20]}...'，如果完全不考虑自己，最佳选择是什么？"
        )

        # 挑战经验局限
        challenges.append(
            f"关于'{topic[:20]}...'，过去有效的经验在未来还适用吗？什么变了？"
        )

        # 挑战群体思维
        challenges.append(
            f"关于'{topic[:20]}...'，如果你是唯一一个不同意的，你会怎么argue？"
        )

        # 挑战确认偏误
        challenges.append(f"关于'{topic[:20]}...'，什么证据可以证明你是错的？")

        # 挑战固定思维
        challenges.append(f"关于'{topic[:20]}...'，如果你相信一切都可以改变呢？")

        return challenges[:5]

    async def generate_counter_intuitive(self, topic: str) -> Dict[str, str]:
        """
        生成反常识视角
        """
        counter_intuitive_views = [
            {
                "view": "放弃可能更好",
                "reasoning": "有时候坚持是因为沉没成本，而不是价值。继续投入可能是浪费。",
            },
            {
                "view": "不解决才是解决",
                "reasoning": "有些问题不需要被解决，只需要被理解或接受。",
            },
            {
                "view": "混乱比秩序更好",
                "reasoning": "完全有序的系统是死系统，适度的混乱带来创新和适应性。",
            },
            {
                "view": "不知道比知道更好",
                "reasoning": "某些'知道'限制了你的可能性。未知的空间更大。",
            },
            {
                "view": "慢就是快",
                "reasoning": "快速决定往往带来更多问题。慢下来可能反而更快达到目标。",
            },
        ]

        selected = counter_intuitive_views[0]  # 选择第一个作为示例

        return {
            "counter_intuitive_view": selected["view"],
            "reasoning": selected["reasoning"],
            "opposite_view": f"通常我们会认为'{topic[:20]}...'应该努力解决，但...",
            "question_to_consider": "什么证据可以支持或反驳这个反常识的观点？",
        }

    async def generate_opposite_view(self, topic: str) -> Dict[str, str]:
        """
        生成完全相反的观点
        """
        return {
            "original_topic": topic,
            "opposite_view": f"关于'{topic[:20]}...'，相反的观点可能是...",
            "arguments_for_opposite": [
                "如果前提假设是错的呢？",
                "如果存在你不知道的关键信息呢？",
                "如果这个问题本身就被错误定义了呢？",
            ],
            "exercise": "尝试用3分钟时间，认真为相反的观点辩护。然后你会有什么新的发现？",
        }
