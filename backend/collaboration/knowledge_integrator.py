"""
KnowledgeIntegrator - 知识整合器
================================

协作者核心组件：跨领域连接，底层逻辑贯通

核心功能：
1. 识别知识领域
2. 发现领域间连接
3. 整合不同领域的知识
4. 提供跨领域新视角
"""

from typing import Dict, List, Any


class KnowledgeIntegrator:
    """
    知识整合器

    不是告诉用户答案，而是帮助用户看到领域之间的连接
    """

    def __init__(self):
        # 知识领域映射
        self.domain_mapping = {
            "科技": ["计算机科学", "人工智能", "互联网", "生物技术"],
            "商业": ["经济学", "管理学", "市场营销", "创业"],
            "设计": ["艺术", "用户体验", "建筑", "时尚"],
            "科学": ["物理学", "化学", "生物学", "天文学"],
            "哲学": ["伦理学", "认识论", "形而上学", "美学"],
            "心理学": ["认知心理学", "社会心理学", "发展心理学", "临床心理学"],
            "历史": ["世界史", "文化史", "科技史", "思想史"],
            "文学": ["小说", "诗歌", "戏剧", "散文"],
            "艺术": ["绘画", "音乐", "舞蹈", "电影"],
            "数学": ["代数", "几何", "分析", "概率统计"],
        }

        # 跨领域类比
        self.cross_domain_analogies = [
            {
                "domain_a": "软件工程",
                "domain_b": "生物学",
                "analogy": "软件架构 ~ 生物进化",
                "connection": "两者都是适应性系统的演化过程",
            },
            {
                "domain_a": "经济学",
                "domain_b": "物理学",
                "analogy": "市场供需 ~ 物理场",
                "connection": "看不见的手 ~ 看不见的力场",
            },
            {
                "domain_a": "心理学",
                "domain_b": "计算机科学",
                "analogy": "记忆 ~ 存储系统",
                "connection": "人类记忆和计算机存储都是信息编码和检索系统",
            },
            {
                "domain_a": "哲学",
                "domain_b": "数学",
                "analogy": "逻辑推理 ~ 数学证明",
                "connection": "两者都是关于真理和有效性的探索",
            },
            {
                "domain_a": "艺术",
                "domain_b": "商业",
                "analogy": "创意 ~ 商业模式创新",
                "connection": "两者都源于对现有框架的突破",
            },
        ]

    async def identify_domains(self, content: str) -> List[Dict[str, Any]]:
        """
        识别内容涉及的知识领域
        """
        domains = []

        content_lower = content.lower()

        # 检测关键词，映射到领域
        domain_keywords = {
            "技术/计算机": ["代码", "程序", "软件", "互联网", "AI", "算法", "数据"],
            "商业/经济": ["市场", "利润", "用户", "增长", "战略", "商业模式"],
            "设计/创意": ["用户", "体验", "视觉", "界面", "产品", "创新"],
            "科学/研究": ["实验", "理论", "数据", "验证", "假设"],
            "哲学/思考": ["意义", "价值", "存在", "思考", "本质"],
            "心理学/情感": ["感受", "情绪", "心理", "关系", "沟通"],
            "历史/文化": ["历史", "传统", "文化", "过去", "背景"],
            "艺术/创作": ["艺术", "创作", "表达", "灵感", "审美"],
        }

        for domain, keywords in domain_keywords.items():
            if isinstance(keywords, list):
                matches = [kw for kw in keywords if kw in content_lower]
            else:
                matches = [keywords] if keywords in content_lower else []

            if matches:
                # 识别具体概念
                concepts = await self._extract_concepts(content, domain)

                domains.append(
                    {
                        "domain": domain,
                        "matched_keywords": matches,
                        "concepts": concepts,
                        "confidence": len(matches) / max(len(keywords), 1),
                    }
                )

        # 如果没有检测到，标记为通用
        if not domains:
            domains.append(
                {
                    "domain": "通用",
                    "matched_keywords": [],
                    "concepts": [],
                    "confidence": 0.5,
                }
            )

        return domains

    async def _extract_concepts(self, content: str, domain: str) -> List[str]:
        """
        从内容中提取具体概念
        """
        # 简单提取：识别引号内的词、专有名词等
        import re

        patterns = [
            r'"([^"]+)"',  # 引号内的词
            r"「([^」]+)」",  # 中文引号
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # 英文专有名词
        ]

        concepts = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            concepts.extend(matches)

        return concepts[:5]  # 最多返回5个概念

    async def find_connections(
        self, domains: List[Dict[str, Any]], content: str
    ) -> List[Dict[str, Any]]:
        """
        发现领域间的连接
        """
        connections = []

        # 获取领域名称列表
        domain_names = [d["domain"] for d in domains]

        # 检查预定义的跨领域类比
        for analogy in self.cross_domain_analogies:
            if (
                analogy["domain_a"] in domain_names
                and analogy["domain_b"] in domain_names
            ):
                connections.append(
                    {
                        "type": "predefined_analogy",
                        "domain_a": analogy["domain_a"],
                        "domain_b": analogy["domain_b"],
                        "analogy": analogy["analogy"],
                        "connection": analogy["connection"],
                    }
                )

        # 生成动态连接
        if len(domains) >= 2:
            # 连接前两个领域
            connections.append(
                {
                    "type": "dynamic_connection",
                    "domain_a": domains[0]["domain"],
                    "domain_b": domains[1]["domain"],
                    "analogy": f"{domains[0]['domain']}中的问题 ~ {domains[1]['domain']}中的模式",
                    "connection": self._generate_dynamic_connection(
                        domains[0], domains[1], content
                    ),
                }
            )

        return connections

    def _generate_dynamic_connection(
        self, domain_a: Dict[str, Any], domain_b: Dict[str, Any], content: str
    ) -> str:
        """
        生成动态连接
        """
        return f"'{content[:30]}...'这个问题同时涉及'{domain_a['domain']}'和'{domain_b['domain']}'，两个领域可能有共同的底层逻辑"

    async def integrate(
        self, domains: List[Dict[str, Any]], connections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        整合不同领域的知识
        """
        # 发现核心模式
        core_pattern = await self._find_core_pattern(domains, connections)

        # 生成整合洞见
        insight = await self._generate_insight(domains, connections)

        # 生成整合视角
        integrated_perspectives = await self._generate_integrated_perspectives(
            domains, connections
        )

        return {
            "core_pattern": core_pattern,
            "insight": insight,
            "integrated_perspectives": integrated_perspectives,
            "domains_integrated": len(domains),
            "connections_found": len(connections),
        }

    async def _find_core_pattern(
        self, domains: List[Dict[str, Any]], connections: List[Dict[str, Any]]
    ) -> str:
        """
        发现核心模式
        """
        if not domains:
            return "没有检测到明确的领域模式"

        if len(domains) == 1:
            return f"核心模式：'{domains[0]['domain']}'领域的{', '.join(domains[0].get('concepts', [])[:3])}"

        if connections:
            return f"核心模式：'{connections[0]['domain_a']}'与'{connections[0]['domain_b']}'之间的{connections[0]['analogy']}"

        return f"核心模式：{len(domains)}个不同领域的交叉点"

    async def _generate_insight(
        self, domains: List[Dict[str, Any]], connections: List[Dict[str, Any]]
    ) -> str:
        """
        生成整合洞见
        """
        insights = [
            "跨领域视角往往能揭示单一领域无法看到的模式",
            "不同领域的底层逻辑往往是相通的",
            "创新的关键往往在于将一个领域的原理应用到另一个领域",
            "当你从多个角度看一个问题时，答案往往自然浮现",
        ]

        if connections:
            return f"我发现：{connections[0]['connection']}"

        return insights[0]

    async def _generate_integrated_perspectives(
        self, domains: List[Dict[str, Any]], connections: List[Dict[str, Any]]
    ) -> List[str]:
        """
        生成整合视角
        """
        perspectives = []

        for d in domains:
            perspectives.append(
                f"从{d['domain']}的角度看：{d.get('concepts', [])[0] if d.get('concepts') else '这个问题'}可以被理解为..."
            )

        if len(domains) >= 2:
            perspectives.append(
                f"从跨领域角度看：'{domains[0]['domain']}'和'{domains[1]['domain']}'可能有共同的底层逻辑"
            )

        return perspectives[:3]

    async def generate_perspectives(
        self, domains: List[Dict[str, Any]], content: str
    ) -> List[Dict[str, Any]]:
        """
        生成跨领域新视角
        """
        perspectives = []

        # 从每个领域生成视角
        for domain in domains[:3]:  # 最多3个领域
            perspective = {
                "domain": domain["domain"],
                "perspective": f"从{domain['domain']}的角度重新审视这个问题",
                "explanation": f"{domain['domain']}的核心理念是{self._get_domain_principle(domain['domain'])}",
                "key_question": self._get_domain_question(domain["domain"], content),
            }
            perspectives.append(perspective)

        # 生成整合视角
        if len(domains) >= 2:
            integrated = {
                "domain": "跨领域",
                "perspective": "整合多个领域的视角",
                "explanation": "当从多个领域同时看一个问题时，往往能发现新的可能性",
                "key_question": f"如果用{domains[0]['domain']}和{domains[1]['domain']}的视角同时看这个问题，会得到什么？",
            }
            perspectives.append(integrated)

        return perspectives

    def _get_domain_principle(self, domain: str) -> str:
        """
        获取领域的核心理念
        """
        principles = {
            "技术/计算机": "系统化和自动化的思维方式",
            "商业/经济": "资源优化和价值创造",
            "设计/创意": "以人为本的解决问题方式",
            "科学/研究": "实证和验证的方法论",
            "哲学/思考": "对本质和意义的追问",
            "心理学/情感": "对人类行为的深入理解",
            "历史/文化": "时间维度的考量",
            "艺术/创作": "表达和美的追求",
        }

        return principles.get(domain, "独特的思维方式")

    def _get_domain_question(self, domain: str, content: str) -> str:
        """
        从领域角度提出关键问题
        """
        questions = {
            "技术/计算机": "这个问题可以被系统化/自动化吗？",
            "商业/经济": "这个问题如何创造价值？",
            "设计/创意": "用户体验的角度这个问题是什么？",
            "科学/研究": "如何验证这个问题的解决方案？",
            "哲学/思考": "这个问题的本质是什么？",
            "心理学/情感": "人的因素如何影响这个问题？",
            "历史/文化": "这个问题在历史上有先例吗？",
            "艺术/创作": "如何用创造性的方式表达这个问题？",
        }

        return questions.get(domain, f"从{domain}的角度，这个问题意味着什么？")
