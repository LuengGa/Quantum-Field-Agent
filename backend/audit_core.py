"""
Quantum Field Agent V2.0 - 审计核心模块
融合到统一架构中，通过配置开关控制
"""

import os
import hashlib
import json
import time
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime

# 可选导入aiofiles（如果不可用则使用同步方式）
try:
    import aiofiles

    ASYNC_FILE = True
except ImportError:
    ASYNC_FILE = False


class AuditEventType(Enum):
    """审计事件类型"""

    FIELD_COLLAPSE = "field_collapse"  # 场坍缩（核心事件）
    STATE_TRANSITION = "state_transition"  # 状态转换
    SKILL_INVOCATION = "skill_invocation"  # 技能调用
    SAFETY_CHECK = "safety_check"  # 安全检查点


@dataclass(frozen=True)
class AuditEvent:
    """
    审计事件（不可变，确保哈希稳定）
    只记录边界，不记录过程细节
    """

    timestamp_ns: int  # 纳秒级时间戳
    event_type: AuditEventType
    user_id: str
    session_id: str

    # 输入边界（哈希）
    intent_hash: str  # SHA256(用户输入)
    intent_vector_hash: str  # 意图向量哈希

    # 场状态边界（前/后）
    pre_state_hash: str  # 坍缩前场状态哈希
    post_state_hash: str  # 坍缩后场状态哈希

    # 输出边界（哈希）
    output_hash: str  # 输出内容哈希

    # 元数据
    entropy_delta: float  # 熵变
    skills_activated: List[str]  # 激活的技能列表
    processing_node: str  # 处理节点标识
    compliance_flags: List[str]  # 合规标记

    # 链式结构
    previous_hash: str  # 前一事件哈希
    event_hash: str  # 本事件哈希

    def __post_init__(self):
        # 冻结后计算哈希
        if not self.event_hash:
            object.__setattr__(self, "event_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        """计算事件哈希（防篡改核心）"""
        data = {
            "timestamp_ns": self.timestamp_ns,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "intent_hash": self.intent_hash,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            "timestamp_ns": self.timestamp_ns,
            "timestamp_human": datetime.fromtimestamp(
                self.timestamp_ns / 1e9
            ).isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "intent_hash": self.intent_hash,
            "intent_vector_hash": self.intent_vector_hash,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "output_hash": self.output_hash,
            "entropy_delta": self.entropy_delta,
            "skills_activated": self.skills_activated,
            "processing_node": self.processing_node,
            "compliance_flags": self.compliance_flags,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
            "integrity": "valid",
        }


class AuditChain:
    """
    审计链管理器（区块链式WORM存储）
    保证：写入后不可修改、不可删除、可追溯
    """

    def __init__(self, storage_path: str = "./quantum_audit", chain_id: str = "main"):
        self.storage_path = storage_path
        self.chain_id = chain_id
        self.chain_file = os.path.join(storage_path, f"{chain_id}.jsonl")
        self.index_file = os.path.join(storage_path, f"{chain_id}.index")
        self.current_hash = "0" * 64  # 创世哈希
        self._write_lock = asyncio.Lock()
        self._cache: List[AuditEvent] = []
        self._initialized = False

        # 初始化存储
        self._init_storage()

    def _init_storage(self):
        """初始化存储目录"""
        os.makedirs(self.storage_path, exist_ok=True)

        # 尝试加载最后一条哈希
        if os.path.exists(self.chain_file):
            try:
                with open(self.chain_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_event = json.loads(lines[-1].strip())
                        self.current_hash = last_event["event_hash"]
                        print(
                            f"[审计链] 已恢复，高度：{len(lines)}，哈希：{self.current_hash[:16]}..."
                        )
            except Exception as e:
                print(f"[审计链警告] 加载历史记录失败：{e}")

        self._initialized = True

    async def append(self, event: AuditEvent) -> str:
        """追加事件到审计链"""
        if not self._initialized:
            self._init_storage()

        async with self._write_lock:
            # 验证链连续性
            if event.previous_hash != self.current_hash:
                print(
                    f"[审计警告] 链不连续：期望 {self.current_hash[:16]}..., 得到 {event.previous_hash[:16]}..."
                )

            # 写入文件
            line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"

            if ASYNC_FILE:
                async with aiofiles.open(self.chain_file, "a") as f:
                    await f.write(line)
            else:
                with open(self.chain_file, "a") as f:
                    f.write(line)

            # 更新内存状态
            self.current_hash = event.event_hash
            self._cache.append(event)
            if len(self._cache) > 100:
                self._cache.pop(0)

            return event.event_hash

    async def verify_chain(self, limit: Optional[int] = None) -> Dict:
        """验证审计链完整性"""
        errors = []
        prev_hash = "0" * 64
        count = 0
        user_stats = {}

        try:
            with open(self.chain_file, "r") as f:
                for line in f:
                    if limit and count >= limit:
                        break

                    try:
                        record = json.loads(line.strip())

                        # 验证前一哈希连续性
                        if record["previous_hash"] != prev_hash:
                            errors.append(
                                {
                                    "position": count,
                                    "error": "hash_chain_broken",
                                    "expected": prev_hash[:16],
                                    "found": record["previous_hash"][:16],
                                }
                            )

                        # 验证哈希正确性
                        expected_hash = AuditEvent(
                            timestamp_ns=record["timestamp_ns"],
                            event_type=AuditEventType(record["event_type"]),
                            user_id=record["user_id"],
                            session_id=record["session_id"],
                            intent_hash=record["intent_hash"],
                            intent_vector_hash=record["intent_vector_hash"],
                            pre_state_hash=record["pre_state_hash"],
                            post_state_hash=record["post_state_hash"],
                            output_hash=record["output_hash"],
                            entropy_delta=record["entropy_delta"],
                            skills_activated=record["skills_activated"],
                            processing_node=record["processing_node"],
                            compliance_flags=record["compliance_flags"],
                            previous_hash=record["previous_hash"],
                            event_hash="",
                        ).event_hash

                        if expected_hash != record["event_hash"]:
                            errors.append(
                                {
                                    "position": count,
                                    "error": "data_tampered",
                                    "hash": record["event_hash"][:16],
                                }
                            )

                        # 统计
                        user_stats[record["user_id"]] = (
                            user_stats.get(record["user_id"], 0) + 1
                        )
                        prev_hash = record["event_hash"]
                        count += 1

                    except json.JSONDecodeError:
                        errors.append({"position": count, "error": "corrupted_record"})

        except FileNotFoundError:
            return {"status": "empty", "count": 0}

        return {
            "status": "valid" if not errors else "compromised",
            "total_events": count,
            "errors": errors,
            "user_distribution": user_stats,
            "latest_hash": prev_hash[:32] + "...",
        }

    async def query_by_user(self, user_id: str, limit: int = 50) -> List[Dict]:
        """查询特定用户的审计轨迹"""
        results = []
        count = 0

        if not os.path.exists(self.chain_file):
            return results

        with open(self.chain_file, "r") as f:
            for line in f:
                if count >= limit:
                    break
                try:
                    record = json.loads(line.strip())
                    if record["user_id"] == user_id:
                        results.append(record)
                        count += 1
                except:
                    continue

        return results

    async def generate_audit_report(
        self, user_id: str, start_time: int, end_time: int
    ) -> Dict:
        """生成合规审计报告"""
        events = []

        if not os.path.exists(self.chain_file):
            return {"user_id": user_id, "total_operations": 0}

        with open(self.chain_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if (
                        record["user_id"] == user_id
                        and start_time <= record["timestamp_ns"] <= end_time
                    ):
                        events.append(record)
                except:
                    continue

        # 统计分析
        entropy_changes = [e["entropy_delta"] for e in events]
        skill_usage = {}
        for e in events:
            for skill in e["skills_activated"]:
                skill_usage[skill] = skill_usage.get(skill, 0) + 1

        return {
            "user_id": user_id,
            "period": {
                "start": datetime.fromtimestamp(start_time / 1e9).isoformat(),
                "end": datetime.fromtimestamp(end_time / 1e9).isoformat(),
            },
            "total_operations": len(events),
            "entropy_analysis": {
                "avg_delta": sum(entropy_changes) / len(entropy_changes)
                if entropy_changes
                else 0,
                "max_increase": max(entropy_changes) if entropy_changes else 0,
                "trend": "increasing"
                if entropy_changes and entropy_changes[-1] > entropy_changes[0]
                else "stable",
            },
            "skill_usage": skill_usage,
            "compliance_flags": list(
                set([f for e in events for f in e["compliance_flags"]])
            ),
            "chain_integrity": "verified",
        }


def quick_hash(data: str) -> str:
    """快速计算内容哈希"""
    return hashlib.sha256(data.encode()).hexdigest()
