"""解方程技能"""

import re


def register_skill():
    """注册解方程技能"""
    return {
        "name": "solve_equation",
        "description": "解方程，参数：equation（一元方程如2x+5=15）",
        "domain": "math",
        "function": solve_equation,
    }


def solve_equation(equation: str) -> str:
    """解一元方程"""
    try:
        if not equation:
            return "错误：没有提供方程"

        equation = equation.strip()

        linear_match = re.match(
            r"([+-]?\d*)x\s*([+-])\s*(\d+)\s*=\s*([+-]?\d+)", equation.replace(" ", "")
        )
        if linear_match:
            a_str, op, b, c = linear_match.groups()

            if a_str == "" or a_str == "+":
                a = 1
            elif a_str == "-":
                a = -1
            else:
                a = int(a_str)

            b = int(b)
            c = int(c)

            if op == "-":
                b = -b

            if a == 0:
                return "错误：不是有效的方程（x的系数不能为0）"

            x = (c - b) / a

            if x.is_integer():
                x = int(x)

            return f"【解方程】\n方程：{equation}\n解：x = {x}"

        quadratic_match = re.match(
            r"([+-]?\d*)x\^2\s*([+-])\s*(\d+)\s*=\s*(\d+)", equation.replace(" ", "")
        )
        if quadratic_match:
            return f"【二次方程】\n{equation}\n解：x = 1 或 x = -1 (示例)"

        return f"【方程求解】\n方程：{equation}\n当前仅支持简单的一元一次方程"
    except Exception as e:
        return f"解方程失败：{str(e)}"
