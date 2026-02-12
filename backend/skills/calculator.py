"""计算器技能 - 数学计算"""

import math
import re


def register_skill():
    """注册计算器技能"""
    return {
        "name": "calculate",
        "description": "进行数学计算，参数：expression（数学表达式如25*4）",
        "domain": "math",
        "function": calculate,
    }


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        sanitized = re.sub(r"[^0-9+\-*/().sqrtlogsinconscos^]", "", expression)
        if not sanitized:
            return "无效的数学表达式"

        if "sqrt" in sanitized.lower():
            match = re.search(r"sqrt\((\d+)\)", sanitized, re.IGNORECASE)
            if match:
                num = float(match.group(1))
                result = math.sqrt(num)
                return f"√{num} = {result:.6f}"

        result = eval(sanitized)
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            return f"{result:.6f}"
        return str(result)
    except ZeroDivisionError:
        return "错误：除以零"
    except Exception as e:
        return f"计算错误：{str(e)}"
