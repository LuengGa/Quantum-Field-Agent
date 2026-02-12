"""文档撰写技能"""


def register_skill():
    """注册文档撰写技能"""
    return {
        "name": "write_document",
        "description": "撰写文档，参数：topic（主题）、doc_type（类型：email/report/notes）、length（长度short/medium/long）",
        "domain": "office",
        "function": write_document,
    }


def write_document(topic: str, doc_type: str = "notes", length: str = "medium") -> str:
    """撰写文档"""
    try:
        if not topic:
            return "错误：没有提供文档主题"

        length_map = {"short": 50, "medium": 150, "long": 300}
        word_count = length_map.get(length, 150)

        templates = {
            "email": f"关于：{topic}\n\n尊敬的收件人：\n\n我写这封邮件是为了讨论{topic}相关事宜。请尽快回复。\n\n此致\n敬礼",
            "report": f"# {topic}\n\n## 概述\n本报告旨在分析{topic}的关键要点。\n\n## 主要内容\n1. 背景介绍\n2. 现状分析\n3. 建议措施\n\n## 结论\n{topic}需要进一步关注和行动。",
            "notes": f"【{topic}笔记】\n\n要点：\n1. {topic}的核心概念\n2. 实施方法\n3. 注意事项\n\n待办事项：\n- [ ] 进一步研究\n- [ ] 制定计划\n- [ ] 执行验证",
        }

        content = templates.get(doc_type, templates["notes"])
        return f"【文档已生成】\n类型：{doc_type}\n主题：{topic}\n\n{content}"
    except Exception as e:
        return f"文档撰写失败：{str(e)}"
