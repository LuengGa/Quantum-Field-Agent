"""推荐技能 - 智能推荐"""


def register_skill():
    """注册推荐技能"""
    return {
        "name": "get_recommendation",
        "description": "获取推荐，参数：category（类别：books/movies/music/tools/places）、preference（用户偏好描述）",
        "domain": "life",
        "function": get_recommendation,
    }


def get_recommendation(category: str, preference: str = "") -> str:
    """获取推荐"""
    try:
        recommendations = {
            "books": [
                "《思考，快与慢》- 丹尼尔·卡尼曼",
                "《原子习惯》- 詹姆斯·克利尔",
                "《深度工作》- 卡尔·纽波特",
            ],
            "movies": [
                "《盗梦空间》- 克里斯托弗·诺兰",
                "《黑客帝国》- 沃卓斯基姐妹",
                "《星际穿越》- 克里斯托弗·诺兰",
            ],
            "music": [
                "Electronic/Lo-Fi 学习工作",
                "Classical 放松身心",
                "Jazz 咖啡时光",
            ],
            "tools": [
                "Notion - 笔记和项目管理",
                "Obsidian - 双向链接笔记",
                "VS Code - 代码编辑器",
            ],
            "places": [
                "咖啡馆 - 适合思考和写作",
                "图书馆 - 适合深度学习",
                "公园 - 适合放松和散步",
            ],
        }

        category_list = recommendations.get(category.lower(), [])

        if not category_list:
            return f"错误：未知的推荐类别 '{category}'，可选：books/movies/music/tools/places"

        pref_text = f"\n根据你的偏好：{preference}" if preference else ""

        return f"【{category.title()}推荐】{pref_text}\n\n" + "\n".join(
            [f"{i + 1}. {item}" for i, item in enumerate(category_list)]
        )
    except Exception as e:
        return f"推荐失败：{str(e)}"
