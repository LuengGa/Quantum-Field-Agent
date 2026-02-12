"""邮件发送技能"""


def register_skill():
    """注册邮件发送技能"""
    return {
        "name": "send_email",
        "description": "发送邮件，参数：to（收件人邮箱）、subject（邮件主题）、content（邮件内容）",
        "domain": "office",
        "function": send_email,
    }


def send_email(to: str, subject: str, content: str) -> str:
    """发送邮件（模拟实现）"""
    try:
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, to):
            return f"错误：无效的邮箱地址: {to}"

        if not subject or not content:
            return "错误：邮件主题和内容不能为空"

        return (
            f"✓ 邮件已发送\n收件人：{to}\n主题：{subject}\n内容预览：{content[:50]}..."
        )
    except Exception as e:
        return f"邮件发送失败：{str(e)}"
