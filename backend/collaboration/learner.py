"""
Learner - 学习系统
==================

协作者核心组件：从用户反馈中学习，越来越了解用户

核心功能：
1. 用户画像构建
2. 协作历史记录
3. 偏好学习
4. 关系深化
"""

from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import json


class Learner:
    """
    学习系统

    协作者不仅提供价值，也从协作中学习
    越来越了解用户，形成真正的协作关系
    """

    def __init__(self, storage_dir: str = "./collaboration_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 用户画像存储
        self.profiles_file = self.storage_dir / "user_profiles.json"
        self._load_profiles()

        # 协作历史存储
        self.history_dir = self.storage_dir / "collaboration_history"
        self.history_dir.mkdir(exist_ok=True)

        # 偏好学习
        self.learned_preferences: Dict[str, Any] = {}

    def _load_profiles(self):
        """加载用户画像"""
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file, "r", encoding="utf-8") as f:
                    self.user_profiles = json.load(f)
            except:
                self.user_profiles = {}
        else:
            self.user_profiles = {}

    def _save_profiles(self):
        """保存用户画像"""
        with open(self.profiles_file, "w", encoding="utf-8") as f:
            json.dump(self.user_profiles, f, ensure_ascii=False, indent=2)

    async def learn_from_collaboration(
        self, user_id: str, user_input: str, collaborator_response: Dict[str, Any]
    ):
        """
        从协作中学习
        """
        # 确保用户画像存在
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "created_at": datetime.now().isoformat(),
                "interactions_count": 0,
                "topics": set(),
                "preferences": {},
                "style": {},
                "insights_given": [],
                "insights_received": [],
                "collaboration_patterns": [],
            }

        profile = self.user_profiles[user_id]

        # 更新交互次数
        profile["interactions_count"] = profile.get("interactions_count", 0) + 1

        # 学习话题
        topics = await self._extract_topics(user_input)
        profile["topics"].update(topics)

        # 学习风格
        style = await self._analyze_style(user_input)
        profile["style"] = self._merge_style(profile.get("style", {}), style)

        # 学习偏好
        preferences = await self._detect_preferences(user_input, collaborator_response)
        profile["preferences"] = self._merge_preferences(
            profile.get("preferences", {}), preferences
        )

        # 记录协作模式
        pattern = await self._detect_pattern(user_input, collaborator_response)
        if pattern:
            profile["collaboration_patterns"].append(pattern)

        # 简短记录
        profile["last_interaction"] = datetime.now().isoformat()

        self._save_profiles()

    async def _extract_topics(self, text: str) -> set:
        """提取话题"""
        topics = set()

        # 简单提取：识别大写字母开头的词、专业术语等
        import re

        # 识别引号内的词
        quoted = re.findall(r'"([^"]+)"', text)
        topics.update(quoted)

        # 识别常见话题词
        topic_indicators = ["关于", "关于", "话题", "主题", "问题", "事情"]
        for indicator in topic_indicators:
            if indicator in text:
                idx = text.find(indicator)
                if idx >= 0:
                    # 提取后面的词
                    words = text[
                        idx + len(indicator) : idx + len(indicator) + 20
                    ].split()
                    if words:
                        topics.add(words[0])

        return topics

    async def _analyze_style(self, text: str) -> Dict[str, Any]:
        """分析用户风格"""
        style = {}

        # 长度风格
        if len(text) > 200:
            style["length_preference"] = "详细"
        elif len(text) > 50:
            style["length_preference"] = "中等"
        else:
            style["length_preference"] = "简洁"

        # 语气风格
        if any(kw in text for kw in ["!", "！", "哈哈", "开心"]):
            style["tone"] = "积极"
        elif any(kw in text for kw in ["...", "唉", "难过", "焦虑"]):
            style["tone"] = "消极"
        else:
            style["tone"] = "中性"

        # 问题风格
        question_count = text.count("？") + text.count("?")
        style["question_frequency"] = question_count

        # 反思风格
        if any(kw in text for kw in ["我想", "我觉得", "我认为"]):
            style["reflective"] = True
        else:
            style["reflective"] = False

        return style

    def _merge_style(self, old_style: Dict, new_style: Dict) -> Dict:
        """合并风格"""
        merged = old_style.copy()

        for key, value in new_style.items():
            if key in merged:
                # 如果值相同，保持
                if merged[key] != value:
                    # 如果不同，记录两种可能
                    merged[key] = value  # 使用最新的
            else:
                merged[key] = value

        return merged

    async def _detect_preferences(
        self, user_input: str, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检测偏好"""
        preferences = {}

        # 检测响应类型偏好
        response_type = response.get("type", "")
        if response_type:
            preferences["response_type"] = response_type

        # 检测深度偏好
        if len(user_input) > 100:
            preferences["depth_preference"] = "深入"
        elif len(user_input) > 30:
            preferences["depth_preference"] = "中等"
        else:
            preferences["depth_preference"] = "简洁"

        # 检测互动偏好
        if any(kw in user_input for kw in ["你怎么看", "有什么看法", "你觉得"]):
            preferences["interaction_type"] = "对话"
        elif any(kw in user_input for kw in ["帮我", "请", "能不能"]):
            preferences["interaction_type"] = "请求"
        else:
            preferences["interaction_type"] = "分享"

        return preferences

    def _merge_preferences(self, old_prefs: Dict, new_prefs: Dict) -> Dict:
        """合并偏好"""
        merged = old_prefs.copy()

        for key, value in new_prefs.items():
            merged[key] = value

        return merged

    async def _detect_pattern(
        self, user_input: str, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检测协作模式"""
        patterns = []

        # 检测重复模式
        if "问题" in user_input:
            patterns.append("问题导向")

        if any(kw in user_input for kw in ["我想", "我想", "我想要"]):
            patterns.append("目标导向")

        if any(kw in user_input for kw in ["为什么", "怎么会"]):
            patterns.append("探究导向")

        if patterns:
            return {
                "timestamp": datetime.now().isoformat(),
                "patterns": patterns,
                "response_type": response.get("type", ""),
            }

        return {}

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户画像
        """
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id].copy()
            # 转换set为list（JSON不支持set）
            if "topics" in profile and isinstance(profile["topics"], set):
                profile["topics"] = list(profile["topics"])
            return profile
        else:
            return {
                "message": "新用户，暂无画像",
                "suggestions": [
                    "多进行几次协作，我会逐渐了解你的风格和偏好",
                    "你可以告诉我你的工作领域、思考风格等，帮助我更好地协作为你",
                ],
            }

    async def get_collaboration_summary(self, user_id: str) -> Dict[str, Any]:
        """
        获取协作总结
        """
        profile = await self.get_user_profile(user_id)

        if "message" in profile:
            return profile

        return {
            "total_interactions": profile.get("interactions_count", 0),
            "main_topics": list(profile.get("topics", set()))[:5],
            "collaboration_style": profile.get("style", {}),
            "preferences": profile.get("preferences", {}),
            "common_patterns": profile.get("collaboration_patterns", [])[-5:],
        }

    async def teach_collaborator(self, user_id: str, lesson: str, context: str):
        """
        用户指导协作者

        用户可以直接告诉协作者如何更好地协作为自己
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "created_at": datetime.now().isoformat(),
                "interactions_count": 0,
                "topics": set(),
                "preferences": {},
                "style": {},
                "lessons_learned": [],
            }

        lesson_entry = {
            "timestamp": datetime.now().isoformat(),
            "lesson": lesson,
            "context": context,
        }

        self.user_profiles[user_id]["lessons_learned"] = self.user_profiles[
            user_id
        ].get("lessons_learned", []) + [lesson_entry]

        self._save_profiles()

        return {
            "status": "learned",
            "message": f"谢谢你的指导！我会记住：{lesson}",
            "in_context": context,
        }

    async def suggest_improvement(self, user_id: str) -> Dict[str, Any]:
        """
        获取改进建议
        """
        profile = await self.get_user_profile(user_id)

        if "message" in profile:
            return profile

        suggestions = []

        # 基于交互次数
        count = profile.get("interactions_count", 0)
        if count < 5:
            suggestions.append("多进行几次协作，我会更好地了解你的风格")

        # 基于话题多样性
        topics = profile.get("topics", set())
        if len(topics) < 3:
            suggestions.append("我们探索的话题还不够多样，可以尝试新的领域")

        # 基于风格
        style = profile.get("style", {})
        if style.get("reflective"):
            suggestions.append("你喜欢反思式对话，我可以更多地引导深度思考")

        # 基于偏好
        prefs = profile.get("preferences", {})
        if prefs.get("depth_preference") == "简洁":
            suggestions.append("你偏好简洁的回应，我会尽量精简")
        elif prefs.get("depth_preference") == "深入":
            suggestions.append("你喜欢深入探讨，我会提供更详细的分析")

        return {
            "suggestions": suggestions,
            "current_profile": {
                "interactions": count,
                "topics_count": len(topics),
                "style": style,
            },
        }

    async def get_mutual_understanding(self, user_id: str) -> Dict[str, Any]:
        """
        获取相互理解报告

        协作者向用户汇报：我理解你什么
        """
        profile = await self.get_user_profile(user_id)

        if "message" in profile:
            return profile

        understanding = []

        # 理解了什么
        topics = list(profile.get("topics", set()))
        if topics:
            understanding.append(f"我了解到你对以下话题感兴趣：{', '.join(topics[:3])}")

        style = profile.get("style", {})
        if style:
            understanding.append(
                f"我发现你的交流风格是：{style.get('tone', '中性')}、{style.get('length_preference', '中等')}的"
            )

        preferences = profile.get("preferences", {})
        if preferences:
            understanding.append(
                f"基于你的偏好，我学会了：{preferences.get('interaction_type', '灵活')}的互动方式"
            )

        # 汇报相互理解
        return {
            "understanding_report": understanding,
            "what_I_learned": understanding,
            "what_you_can_teach_me": [
                "你的工作/学习领域",
                "你偏好的思考方式",
                "你希望我如何协作为你",
            ],
            "collaboration_invitation": "让我们继续协作，我会越来越了解你，你也会发现我的价值",
        }

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        获取学习统计
        """
        total_users = len(self.user_profiles)
        total_interactions = sum(
            p.get("interactions_count", 0) for p in self.user_profiles.values()
        )

        return {
            "total_users_tracked": total_users,
            "total_interactions": total_interactions,
            "average_interactions_per_user": (total_interactions / max(1, total_users)),
        }
