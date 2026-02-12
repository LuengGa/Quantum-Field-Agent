#!/usr/bin/env python3
import os

# 文件路径
file_path = "/Volumes/J ZAO 9 SER 1/Python/Open Code/QUANTUM_FIELD_GUIDE/backend/evolution_router.py"

with open(file_path, "r") as f:
    content = f.read()

old_start = """from evolution import (
    EvolutionEngine,
    EvolutionDatabase,
    EvolutionConfig,
    PatternMiner,
    StrategyEvolver,
    HypothesisTester,
    KnowledgeSynthesizer,
    CapabilityBuilder,
)
from evolution.feedback_collector import FeedbackCollector

router = APIRouter(prefix="/evolution", tags=["evolution"])

_db = None
_engine = None


def get_db():
    global _db
    if _db is None:
        _db = EvolutionDatabase()
    return _db


def get_engine():
    global _engine
    if _engine is None:
        _engine = EvolutionEngine(db=get_db())
    return _engine"""

new_start = """from evolution import (
    EvolutionEngine,
    EvolutionConfig,
    PatternMiner,
    StrategyEvolver,
    HypothesisTester,
    KnowledgeSynthesizer,
    CapabilityBuilder,
)
from evolution.feedback_collector import FeedbackCollector

router = APIRouter(prefix="/evolution", tags=["evolution"])

_db = None
_engine = None
_NEON_DB = None

USE_NEON = os.getenv("DATABASE_TYPE", "sqlite") == "postgresql"


def get_db():
    global _db, _NEON_DB
    if USE_NEON:
        if _NEON_DB is None:
            from evolution.evolution_router_neon import get_neon_db
            _NEON_DB = get_neon_db()
        return _NEON_DB
    else:
        if _db is None:
            from evolution import EvolutionDatabase
            _db = EvolutionDatabase()
        return _db


def get_engine():
    global _engine
    if _engine is None:
        db = get_db()
        _engine = EvolutionEngine(db=db)
    return _engine"""

content = content.replace(old_start, new_start)

with open(file_path, "w") as f:
    f.write(content)

print("✅ evolution_router.py 已更新支持 Neon")
