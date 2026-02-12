"""天气技能 - 查询城市天气"""


def register_skill():
    """注册天气技能"""
    return {
        "name": "search_weather",
        "description": "查询指定城市的天气情况，参数：city（城市名）",
        "domain": "life",
        "function": lambda city: f"{city}今天晴天，25°C，微风，空气质量良好。",
    }
