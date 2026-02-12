"""
Knowledge Synthesizer - 知识综合器
==================================

将碎片经验整合为可复用的知识：
1. 模式抽象 - 从具体模式中提取通用知识
2. 知识验证 - 验证知识的正确性和实用性
3. 知识组织 - 建立知识之间的关系
4. 知识检索 - 快速找到相关知识
5. 知识应用 - 在实践中应用知识

核心理念：
- 知识不是灌输的，而是从经验中涌现的
- 知识需要验证，不是所有经验都是知识
- 知识是可复用的模式
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import uuid


@dataclass
class KnowledgeUnit:
    """知识单元"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    domain: str = "general"
    content: str = ""
    source_patterns: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    applicability: Dict = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    related_knowledge: List[str] = field(default_factory=list)
    confidence: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    last_validated: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KnowledgeGraph:
    """知识图谱"""

    nodes: Dict[str, KnowledgeUnit] = field(default_factory=dict)
    edges: Dict[str, List[Dict]] = field(default_factory=dict)
    domains: Set[str] = field(default_factory=set)

    def add_node(self, knowledge: KnowledgeUnit):
        """添加知识节点"""
        self.nodes[knowledge.id] = knowledge
        self.domains.add(knowledge.domain)

        if knowledge.id not in self.edges:
            self.edges[knowledge.id] = []

    def add_edge(self, source_id: str, target_id: str, relation_type: str = "related"):
        """添加知识边"""
        if source_id not in self.edges:
            self.edges[source_id] = []

        self.edges[source_id].append(
            {
                "target": target_id,
                "relation": relation_type,
                "created_at": datetime.now().isoformat(),
            }
        )

    def get_related(
        self, knowledge_id: str, relation_type: Optional[str] = None, max_depth: int = 2
    ) -> List[KnowledgeUnit]:
        """获取相关知识"""
        if knowledge_id not in self.nodes:
            return []

        related = []
        visited = set()
        queue = [(knowledge_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            if current_id in self.nodes and current_id != knowledge_id:
                related.append(self.nodes[current_id])

            if current_id in self.edges:
                for edge in self.edges[current_id]:
                    target = edge["target"]
                    if target not in visited:
                        if relation_type is None or edge["relation"] == relation_type:
                            queue.append((target, depth + 1))

        return related

    def get_domain_knowledge(self, domain: str) -> List[KnowledgeUnit]:
        """获取特定领域的知识"""
        return [node for node in self.nodes.values() if node.domain == domain]

    def get_knowledge_by_domain(self, domain: str) -> List[KnowledgeUnit]:
        """获取特定领域的知识（兼容方法）"""
        return self.get_domain_knowledge(domain)


class KnowledgeSynthesizer:
    """
    知识综合器

    将碎片经验整合为可复用的知识：
    - 从模式中提取知识
    - 验证和更新知识
    - 建立知识图谱
    - 检索和应用知识
    """

    def __init__(self, db):
        self.db = db

        self.min_confidence_for_synthesis = 0.4  # 降低阈值
        self.min_evidence_for_knowledge = 1  # 降低到1个模式

        self.knowledge_graph = KnowledgeGraph()
        self._load_existing_knowledge()

    def _load_existing_knowledge(self):
        """加载现有知识"""
        all_knowledge = self.db.get_knowledge_by_domain("all")

        for k_data in all_knowledge:
            knowledge = KnowledgeUnit(
                id=k_data.get("id", str(uuid.uuid4())),
                title=k_data.get("title", ""),
                domain=k_data.get("domain", "general"),
                content=k_data.get("content", ""),
                source_patterns=json.loads(k_data.get("source_patterns", "[]")),
                evidence=json.loads(k_data.get("evidence", "[]")),
                applicability=json.loads(k_data.get("applicability", "{}")),
                confidence=k_data.get("confidence", 0.5),
                usage_count=k_data.get("usage_count", 0),
                created_at=k_data.get("created_at", datetime.now().isoformat()),
                last_validated=k_data.get("validated_at", datetime.now().isoformat()),
            )

            self.knowledge_graph.add_node(knowledge)

    async def synthesize_from_patterns(
        self, patterns: List[Dict], domain: str = "general"
    ) -> List[KnowledgeUnit]:
        """
        从模式中综合知识

        Args:
            patterns: 模式列表
            domain: 知识领域

        Returns:
            综合出的知识列表
        """
        knowledge_units = []

        patterns_by_type = defaultdict(list)

        for pattern in patterns:
            p_type = pattern.get("pattern_type", "unknown")
            patterns_by_type[p_type].append(pattern)

        for p_type, type_patterns in patterns_by_type.items():
            if len(type_patterns) >= self.min_evidence_for_knowledge:
                knowledge = self._extract_knowledge(type_patterns, domain, p_type)
                if knowledge:
                    knowledge_units.append(knowledge)
                    self.knowledge_graph.add_node(knowledge)
                    self.db.save_knowledge(asdict(knowledge))

        for i, knowledge in enumerate(knowledge_units):
            for j, other in enumerate(knowledge_units):
                if i != j:
                    self._link_knowledge(knowledge, other)

        return knowledge_units

    def _extract_knowledge(
        self, patterns: List[Dict], domain: str, base_type: str
    ) -> Optional[KnowledgeUnit]:
        """从模式中提取知识"""
        if not patterns:
            return None

        if len(patterns) < self.min_evidence_for_knowledge:
            return None

        avg_confidence = sum(p.get("confidence", 0.5) for p in patterns) / len(patterns)

        if avg_confidence < self.min_confidence_for_synthesis:
            return None

        common_features = self._find_common_features(patterns)

        if common_features:
            title = f"知识-{common_features[:30]}"
        else:
            title = f"{base_type}模式知识"

        content_parts = []

        descriptions = [
            p.get("description", "") for p in patterns if p.get("description")
        ]
        if descriptions:
            content_parts.append(f"描述：{'; '.join(descriptions[:3])}")

        success_rate = sum(p.get("success_rate", 0.5) for p in patterns) / len(patterns)
        content_parts.append(f"成功率：{success_rate:.1%}")

        applicability = {
            "scope": "wide" if avg_confidence > 0.6 else "narrow",
            "conditions": list(
                {k for p in patterns for k in p.get("trigger_conditions", {}).keys()}
            )
            if any("trigger_conditions" in p for p in patterns)
            else [base_type],
        }

        evidence = [
            {
                "pattern_id": p.get("id", str(uuid.uuid4())),
                "description": p.get("description", "") or p.get("name", ""),
                "confidence": p.get("confidence", 0.5),
                "success_rate": p.get("success_rate", 0.5),
            }
            for p in patterns[:10]
        ]

        return KnowledgeUnit(
            title=title,
            domain=domain,
            content=". ".join(content_parts),
            source_patterns=[p.get("id", "") for p in patterns],
            evidence=evidence,
            applicability=applicability,
            confidence=avg_confidence,
            tags=[base_type, domain],
        )

    def _find_common_features(self, patterns: List[Dict]) -> str:
        """找出共同特征"""
        all_features = []

        for pattern in patterns:
            description = pattern.get("description", "")
            if description:
                words = description.split()
                all_features.append(words)

        if not all_features:
            return ""

        common_words = set(all_features[0])
        for features in all_features[1:]:
            common_words = common_words.intersection(set(features))

        if common_words:
            return " ".join(list(common_words)[:5])

        first_desc = patterns[0].get("description", "")
        return first_desc[:30] if first_desc else ""

    def _link_knowledge(self, knowledge_a: KnowledgeUnit, knowledge_b: KnowledgeUnit):
        """建立知识之间的关联"""
        common_tags = set(knowledge_a.tags).intersection(set(knowledge_b.tags))

        if common_tags:
            self.knowledge_graph.add_edge(knowledge_a.id, knowledge_b.id, "shares_tag")

        if knowledge_a.domain == knowledge_b.domain:
            self.knowledge_graph.add_edge(knowledge_a.id, knowledge_b.id, "same_domain")

        knowledge_a.related_knowledge.append(knowledge_b.id)
        knowledge_b.related_knowledge.append(knowledge_a.id)

    async def validate_knowledge(
        self, knowledge_id: str, success: bool, context: Optional[Dict] = None
    ):
        """验证知识的有效性"""
        if knowledge_id not in self.knowledge_graph.nodes:
            return

        knowledge = self.knowledge_graph.nodes[knowledge_id]

        knowledge.usage_count += 1

        if success:
            knowledge.success_count += 1
        else:
            knowledge.failure_count += 1

        new_confidence = knowledge.success_count / max(knowledge.usage_count, 1)

        if abs(new_confidence - knowledge.confidence) > 0.1:
            knowledge.confidence = new_confidence
            knowledge.last_validated = datetime.now().isoformat()

        self.db.save_knowledge(asdict(knowledge))

        if context:
            evidence = {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "context": context,
            }
            knowledge.evidence.append(evidence)

    async def find_applicable_knowledge(
        self, context: Dict, domain: Optional[str] = None
    ) -> List[KnowledgeUnit]:
        """
        查找适用的知识

        Args:
            context: 当前上下文
            domain: 限定领域

        Returns:
            适用的知识列表
        """
        candidates = list(self.knowledge_graph.nodes.values())

        if domain:
            candidates = [k for k in candidates if k.domain == domain]

        scored = []

        for knowledge in candidates:
            score = self._calculate_applicability_score(knowledge, context)
            scored.append((score, knowledge))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored[:5]]

    def _calculate_applicability_score(
        self, knowledge: KnowledgeUnit, context: Dict
    ) -> float:
        """计算知识适用性得分"""
        base_score = knowledge.confidence

        applicability = knowledge.applicability
        scope = applicability.get("scope", "unknown")

        if scope == "wide":
            scope_bonus = 1.0
        elif scope == "narrow":
            scope_bonus = 0.7
        else:
            scope_bonus = 0.5

        usage_bonus = min(knowledge.usage_count / 100, 0.2)

        domain_match = 1.0
        if "domain" in context:
            if knowledge.domain == context["domain"]:
                domain_match = 1.2

        recency = 0.8
        try:
            last_validated = datetime.fromisoformat(
                knowledge.last_validated.replace("Z", "+00:00")
            )
            days_ago = (datetime.now() - last_validated).days
            if days_ago < 7:
                recency = 1.0
            elif days_ago < 30:
                recency = 0.9
            else:
                recency = 0.7
        except:
            recency = 0.8

        total_score = (
            base_score * 0.4
            + scope_bonus * 0.2
            + usage_bonus * 0.1
            + domain_match * 0.1
            + recency * 0.2
        )

        return total_score

    def query_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeUnit]:
        """
        查询知识

        Args:
            query: 查询文本
            domain: 限定领域
            tags: 限定标签
            min_confidence: 最低置信度

        Returns:
            匹配的知识列表
        """
        candidates = list(self.knowledge_graph.nodes.values())

        if domain:
            candidates = [k for k in candidates if k.domain == domain]

        if tags:
            candidates = [k for k in candidates if any(t in k.tags for t in tags)]

        if min_confidence > 0:
            candidates = [k for k in candidates if k.confidence >= min_confidence]

        query_lower = query.lower()

        scored = []

        for knowledge in candidates:
            score = 0

            if query_lower in knowledge.title.lower():
                score += 3

            if query_lower in knowledge.content.lower():
                score += 2

            for tag in knowledge.tags:
                if query_lower in tag.lower():
                    score += 1

            scored.append((score, knowledge))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored if _ > 0]

    def get_knowledge_statistics(self) -> Dict:
        """获取知识统计"""
        all_knowledge = list(self.knowledge_graph.nodes.values())

        if all_knowledge:
            avg_confidence = sum(k.confidence for k in all_knowledge) / len(
                all_knowledge
            )
            total_usage = sum(k.usage_count for k in all_knowledge)
            avg_success_rate = sum(
                k.success_count / max(k.usage_count, 1) for k in all_knowledge
            ) / len(all_knowledge)
        else:
            avg_confidence = 0
            total_usage = 0
            avg_success_rate = 0

        domain_counts = defaultdict(int)
        tag_counts = defaultdict(int)

        for knowledge in all_knowledge:
            domain_counts[knowledge.domain] += 1
            for tag in knowledge.tags:
                tag_counts[tag] += 1

        return {
            "total_knowledge": len(all_knowledge),
            "domains": list(self.knowledge_graph.domains),
            "avg_confidence": avg_confidence,
            "total_usage": total_usage,
            "avg_success_rate": avg_success_rate,
            "by_domain": dict(domain_counts),
            "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[
                :10
            ],
        }

    def export_knowledge_graph(self) -> Dict:
        """导出知识图谱"""
        return {
            "nodes": {
                k.id: {
                    "title": k.title,
                    "domain": k.domain,
                    "confidence": k.confidence,
                    "tags": k.tags,
                }
                for k in self.knowledge_graph.nodes.values()
            },
            "edges": self.knowledge_graph.edges,
            "domains": list(self.knowledge_graph.domains),
            "exported_at": datetime.now().isoformat(),
        }
