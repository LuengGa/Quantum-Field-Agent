"""总结技能 - 文本摘要"""


def register_skill():
    """注册总结技能"""
    return {
        "name": "summarize",
        "description": "总结长文本，参数：text（要总结的内容）、max_length（摘要长度，默认100）",
        "domain": "office",
        "function": summarize,
    }


def summarize(text: str, max_length: int = 100) -> str:
    """总结文本"""
    try:
        if not text:
            return "错误：没有提供要总结的内容"

        words = text.split()
        if len(words) <= max_length:
            return f"【摘要】\n{text}"

        summary = " ".join(words[:max_length]) + "..."
        return f"【摘要】(原文本{len(words)}字，摘要{max_length}字)\n{summary}"
    except Exception as e:
        return f"总结失败：{str(e)}"
