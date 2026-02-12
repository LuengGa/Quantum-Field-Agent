"""网络搜索技能 - 使用DuckDuckGo"""


def register_skill():
    """注册网络搜索技能"""
    return {
        "name": "websearch",
        "description": "网络搜索，参数：query（搜索关键词）",
        "domain": "general",
        "function": websearch,
    }


def websearch(query: str) -> str:
    """执行网络搜索"""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            if not results:
                return f"未找到关于「{query}」的结果"

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. {r.get('title', '无标题')}\n   {r.get('url', '')}\n   {r.get('body', '')[:100]}..."
                )

            return f"搜索「{query}」结果：\n\n" + "\n".join(formatted)
    except ImportError:
        return "错误：duckduckgo_search库未安装（pip install duckduckgo-search）"
    except Exception as e:
        return f"搜索失败：{str(e)}"
