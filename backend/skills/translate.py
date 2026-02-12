"""翻译技能 - 多语言翻译"""


def register_skill():
    """注册翻译技能"""
    return {
        "name": "translate",
        "description": "翻译文本到指定语言，参数：text（原文）、target_lang（目标语言如en/zh/ja）",
        "domain": "office",
        "function": translate,
    }


def translate(text: str, target_lang: str) -> str:
    """翻译文本（模拟实现）"""
    try:
        if not text or not target_lang:
            return "错误：原文和目标语言不能为空"

        lang_names = {
            "en": "English",
            "zh": "中文",
            "ja": "日本語",
            "ko": "한국어",
            "fr": "Français",
            "de": "Deutsch",
            "es": "Español",
            "ru": "Русский",
        }

        lang = lang_names.get(target_lang.lower(), target_lang)

        return (
            f"【翻译完成】\n原文：{text}\n目标语言：{lang}\n翻译结果：{text} (模拟翻译)"
        )
    except Exception as e:
        return f"翻译失败：{str(e)}"
