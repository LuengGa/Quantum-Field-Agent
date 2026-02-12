#!/usr/bin/env python3
"""
代码质量修复脚本
================

批量修复：
1. LSP 警告
2. 类型注解问题
3. to_dict 方法缺失
"""

import os
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).parent

# 需要修复的文件列表
FILES_TO_FIX = [
    "main.py",
    "evolution/database.py",
    "evolution/pattern_miner.py",
    "evolution/strategy_evolver.py",
    "evolution/hypothesis_tester.py",
    "evolution/knowledge_synthesizer.py",
    "evolution/capability_builder.py",
    "evolution/evolution_engine.py",
    "evolution/feedback_collector.py",
    "evolution/hypothesis_generator.py",
    "evolution_router.py",
    "collaboration_router.py",
]


def fix_imports_add_asdict(file_path: Path) -> int:
    """为需要 dataclass asdict 的文件添加导入"""
    changes = 0

    with open(file_path, "r") as f:
        content = f.read()

    # 检查是否已导入 dataclasses
    if "from dataclasses import" in content:
        if "asdict" not in content and "@dataclass" in content:
            # 查找 dataclasses 导入行并添加 asdict
            pattern = r"(from dataclasses import\s+)"
            replacement = r"\1asdict, "
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                changes += count

    if changes > 0:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"  ✓ {file_path.name}: 添加 asdict 导入")

    return changes


def fix_to_dict_calls(file_path: Path) -> int:
    """将 to_dict() 替换为 asdict()"""
    changes = 0

    with open(file_path, "r") as f:
        content = f.read()

    # 替换 pattern.to_dict() -> asdict(pattern)
    pattern1 = r"(\w+)\.to_dict\(\)"
    replacement1 = r"asdict(\1)"
    new_content, count1 = re.subn(pattern1, replacement1, content)
    changes += count1

    if count1 > 0:
        print(f"  ✓ {file_path.name}: 修复 {count1} 处 to_dict() 调用")

    # 写入文件
    if changes > 0:
        with open(file_path, "w") as f:
            f.write(new_content)

    return changes


def fix_none_default_annotations(file_path: Path) -> int:
    """修复类型注解中的 None 默认值"""
    changes = 0

    with open(file_path, "r") as f:
        content = f.read()

    # 修复 List[str] = None
    patterns = [
        (r"(\w+): List\[str\] = None", r"\1: Optional[List[str]] = None"),
        (r"(\w+): Dict\[.*?\] = None", r"\1: Optional[Dict] = None"),
        (r"(\w+): str = None", r"\1: Optional[str] = None"),
        (r"(\w+): int = None", r"\1: Optional[int] = None"),
        (r"(\w+): float = None", r"\1: Optional[float] = None"),
    ]

    for pattern, replacement in patterns:
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            changes += count

    # 添加 Optional 导入
    if changes > 0:
        if "from typing import" in content and "Optional" not in content:
            content = re.sub(r"(from typing import\s+)", r"\1Optional, ", content)

    if changes > 0:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"  ✓ {file_path.name}: 修复 {changes} 处类型注解")

    return changes


def fix_dataclass_field_defaults(file_path: Path) -> int:
    """修复 dataclass field 默认值问题"""
    changes = 0

    with open(file_path, "r") as f:
        content = f.read()

    # 修复 field(default_factory=list) 缺失导入
    if (
        "field(default_factory=list)" in content
        or "field(default_factory=dict)" in content
    ):
        if "from dataclasses import field" in content:
            if "from dataclasses import" in content:
                new_content = re.sub(
                    r"(from dataclasses import\s+)field(\s*)",
                    r"\1field, asdict\2",
                    content,
                )
                if new_content != content:
                    content = new_content
                    changes += 1
                    print(f"  ✓ {file_path.name}: 添加 asdict 到 dataclasses 导入")

    if changes > 0:
        with open(file_path, "w") as f:
            f.write(content)

    return changes


def main():
    print("=" * 60)
    print("代码质量修复")
    print("=" * 60)

    total_changes = 0

    for filename in FILES_TO_FIX:
        file_path = BACKEND_DIR / filename
        if not file_path.exists():
            print(f"  ⚠ {filename}: 文件不存在")
            continue

        print(f"\n处理: {filename}")

        # 1. 添加 asdict 导入
        changes = fix_imports_add_asdict(file_path)
        total_changes += changes

        # 2. 修复 to_dict 调用
        changes = fix_to_dict_calls(file_path)
        total_changes += changes

        # 3. 修复类型注解
        changes = fix_none_default_annotations(file_path)
        total_changes += changes

        # 4. 修复 dataclass field
        changes = fix_dataclass_field_defaults(file_path)
        total_changes += changes

    print("\n" + "=" * 60)
    print(f"修复完成! 共 {total_changes} 处更改")
    print("=" * 60)


if __name__ == "__main__":
    main()
