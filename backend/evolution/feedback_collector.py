"""
Feedback Collector - 用户反馈收集器
==================================

收集和分析用户反馈：
1. 反馈收集 - 记录用户对协作的反馈
2. 反馈分析 - 分析反馈的情感和主题
3. 反馈应用 - 将反馈应用到系统改进

核心理念：
- 用户反馈是系统进化的重要输入
- 反馈需要被量化分析
- 反馈驱动系统改进
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict, dataclass, field
from collections import defaultdict
import uuid


@dataclass
class UserFeedback:
    """用户反馈"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    interaction_id: str = ""

    feedback_type: str = "rating"
    rating: int = 5
    sentiment: str = "positive"
    sentiment_score: float = 0.5

    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    comment: str = ""
    suggestion: str = ""

    is_positive: bool = True
    actionable: bool = False
    priority: int = 1

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processed_at: str = ""


@dataclass
class FeedbackAnalysis:
    """反馈分析结果"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback_id: str = ""

    sentiment_analysis: Dict = field(default_factory=dict)
    topic_classification: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    impact_score: float = 0.0
    confidence: float = 0.5

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FeedbackCollector:
    """
    用户反馈收集器

    收集、分析和应用用户反馈：
    - 收集多种类型的反馈
    - 分析反馈情感和主题
    - 生成改进建议
    """

    def __init__(self, db):
        self.db = db
        self._init_db()

        self.sentiment_keywords = {
            "positive": [
                "好",
                "满意",
                "有帮助",
                "清晰",
                "棒",
                "优秀",
                "准确",
                "详细",
                "感谢",
                "赞",
            ],
            "negative": [
                "差",
                "不满意",
                "没用",
                "模糊",
                "错",
                "糟糕",
                "不准确",
                "缺少",
                "困惑",
                "慢",
            ],
            "neutral": ["一般", "普通", "还行", "还可以"],
        }

    def _get_db_path(self) -> str:
        """获取数据库路径"""
        if hasattr(self.db, "db_path"):
            path = self.db.db_path
            return str(path)
        return str(self.db)

    def _get_connection(self):
        """获取数据库连接"""
        import sqlite3

        return sqlite3.connect(self._get_db_path())

    def _init_db(self):
        """初始化反馈表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                interaction_id TEXT,
                feedback_type TEXT,
                rating INTEGER,
                sentiment TEXT,
                sentiment_score REAL,
                categories TEXT,
                tags TEXT,
                comment TEXT,
                suggestion TEXT,
                is_positive INTEGER,
                actionable INTEGER,
                priority INTEGER,
                created_at TEXT,
                processed_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_analysis (
                id TEXT PRIMARY KEY,
                feedback_id TEXT,
                sentiment_analysis TEXT,
                topic_classification TEXT,
                action_items TEXT,
                impact_score REAL,
                confidence REAL,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def collect_feedback(
        self,
        user_id: str,
        session_id: str,
        interaction_id: str = "",
        feedback_type: str = "rating",
        rating: int = 5,
        comment: str = "",
        suggestion: str = "",
        tags: Optional[List[str]] = None,
    ) -> UserFeedback:
        """收集用户反馈"""
        sentiment, sentiment_score = self._analyze_sentiment(comment)

        categories = self._classify_feedback(comment, tags or [])

        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            interaction_id=interaction_id,
            feedback_type=feedback_type,
            rating=rating,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            categories=categories,
            tags=tags or [],
            comment=comment,
            suggestion=suggestion,
            is_positive=sentiment in ["positive", "neutral"],
            actionable=bool(suggestion),
            priority=self._calculate_priority(rating, sentiment),
        )

        self._save_feedback(feedback)

        analysis = self._analyze_feedback(feedback)
        self._save_analysis(feedback, analysis)

        return feedback

    def _analyze_sentiment(self, text: str) -> tuple:
        """分析情感"""
        if not text:
            return "neutral", 0.5

        text_lower = text.lower()

        positive_count = sum(
            1 for word in self.sentiment_keywords["positive"] if word in text_lower
        )
        negative_count = sum(
            1 for word in self.sentiment_keywords["negative"] if word in text_lower
        )

        total = positive_count + negative_count
        if total == 0:
            return "neutral", 0.5

        score = positive_count / total

        if score > 0.6:
            sentiment = "positive"
        elif score < 0.4:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment, score

    def _classify_feedback(self, text: str, tags: List[str]) -> List[str]:
        """分类反馈"""
        categories = []

        text_lower = text.lower()

        category_keywords = {
            "accuracy": ["准确", "正确", "对", "错", "误差", "错误"],
            "clarity": ["清晰", "清楚", "明白", "模糊", "困惑"],
            "completeness": ["完整", "详细", "缺少", "不够"],
            "relevance": ["相关", "跑题", "偏离"],
            "speed": ["快", "慢", "延迟", "卡"],
            "usefulness": ["有用", "没用", "帮助"],
            "creativity": ["创意", "新颖", "独特", "普通"],
        }

        for category, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                categories.append(category)

        if not categories:
            categories = ["general"]

        categories.extend(tags[:2])

        return list(set(categories))

    def _calculate_priority(self, rating: int, sentiment: str) -> int:
        """计算优先级"""
        if rating <= 2 or sentiment == "negative":
            return 3
        elif rating == 3:
            return 2
        else:
            return 1

    def _analyze_feedback(self, feedback: UserFeedback) -> FeedbackAnalysis:
        """分析反馈"""
        analysis = FeedbackAnalysis(
            feedback_id=feedback.id,
            sentiment_analysis={
                "sentiment": feedback.sentiment,
                "score": feedback.sentiment_score,
                "confidence": min(0.5 + feedback.sentiment_score * 0.5, 0.95),
            },
            topic_classification=feedback.categories,
            action_items=self._generate_action_items(feedback),
            impact_score=self._calculate_impact(feedback),
            confidence=0.7,
        )

        return analysis

    def _generate_action_items(self, feedback: UserFeedback) -> List[str]:
        """生成改进建议"""
        items = []

        if "clarity" in feedback.categories:
            items.append("提高回答的清晰度")
        if "accuracy" in feedback.categories:
            items.append("检查和修正可能的错误")
        if "completeness" in feedback.categories:
            items.append("增加更多详细信息")
        if "speed" in feedback.categories:
            items.append("优化响应速度")

        if feedback.suggestion:
            items.append(f"考虑建议: {feedback.suggestion}")

        return items

    def _calculate_impact(self, feedback: UserFeedback) -> float:
        """计算影响分数"""
        impact = 0.5

        if not feedback.is_positive:
            impact += 0.3

        if feedback.actionable:
            impact += 0.1

        if feedback.priority == 3:
            impact += 0.1

        return min(impact, 1.0)

    def _save_feedback(self, feedback: UserFeedback):
        """保存反馈"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO user_feedback 
            (id, user_id, session_id, interaction_id, feedback_type, rating,
             sentiment, sentiment_score, categories, tags, comment, suggestion,
             is_positive, actionable, priority, created_at, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                feedback.id,
                feedback.user_id,
                feedback.session_id,
                feedback.interaction_id,
                feedback.feedback_type,
                feedback.rating,
                feedback.sentiment,
                feedback.sentiment_score,
                json.dumps(feedback.categories),
                json.dumps(feedback.tags),
                feedback.comment,
                feedback.suggestion,
                1 if feedback.is_positive else 0,
                1 if feedback.actionable else 0,
                feedback.priority,
                feedback.created_at,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def _save_analysis(self, feedback: UserFeedback, analysis: FeedbackAnalysis):
        """保存分析"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO feedback_analysis 
            (id, feedback_id, sentiment_analysis, topic_classification,
             action_items, impact_score, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                analysis.id,
                analysis.feedback_id,
                json.dumps(analysis.sentiment_analysis),
                json.dumps(analysis.topic_classification),
                json.dumps(analysis.action_items),
                analysis.impact_score,
                analysis.confidence,
                analysis.created_at,
            ),
        )

        conn.commit()
        conn.close()

    def get_feedback_statistics(self, days: int = 30) -> Dict:
        """获取反馈统计"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM user_feedback
            WHERE created_at > datetime('now', ?)
        """,
            (f"-{days} days",),
        )

        total = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*) FROM user_feedback
            WHERE created_at > datetime('now', ?) AND is_positive = 1
        """,
            (f"-{days} days",),
        )

        positive = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT AVG(rating) FROM user_feedback
            WHERE created_at > datetime('now', ?)
        """,
            (f"-{days} days",),
        )

        avg_rating = cursor.fetchone()[0] or 0

        cursor.execute(
            """
            SELECT COUNT(*) FROM user_feedback
            WHERE created_at > datetime('now', ?) AND actionable = 1
        """,
            (f"-{days} days",),
        )

        actionable = cursor.fetchone()[0]

        if hasattr(self.db, "db_path") and isinstance(self.db.db_path, str):
            conn.close()

        return {
            "total_feedback": total,
            "positive_feedback": positive,
            "negative_feedback": total - positive,
            "satisfaction_rate": positive / total if total > 0 else 0,
            "avg_rating": avg_rating,
            "actionable_feedback": actionable,
        }

    def get_actionable_feedback(self) -> List[Dict]:
        """获取可操作的反馈"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM user_feedback
            WHERE actionable = 1 AND processed_at = ''
            ORDER BY priority DESC, created_at DESC
        """)

        rows = cursor.fetchall()

        columns = [
            "id",
            "user_id",
            "session_id",
            "interaction_id",
            "feedback_type",
            "rating",
            "sentiment",
            "sentiment_score",
            "categories",
            "tags",
            "comment",
            "suggestion",
            "is_positive",
            "actionable",
            "priority",
            "created_at",
            "processed_at",
        ]

        if hasattr(self.db, "db_path") and isinstance(self.db.db_path, str):
            conn.close()

        return [dict(zip(columns, row)) for row in rows]
