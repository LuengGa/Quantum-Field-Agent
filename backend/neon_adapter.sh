#!/bin/bash
# Neon Database Adapter - Neon 数据库适配器
# ===========================================

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Meta Quantum Field Agent - Neon 适配          ║"
echo "╚══════════════════════════════════════════════════════════╝"

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "❌ .env 文件不存在"
    exit 1
fi

source .env

if [ "$DATABASE_TYPE" != "postgresql" ]; then
    echo "⚠️  DATABASE_TYPE 不是 postgresql，当前值: $DATABASE_TYPE"
    echo "切换到 Neon PostgreSQL..."
    sed -i '' 's/DATABASE_TYPE=.*/DATABASE_TYPE=postgresql/' .env
    source .env
fi

echo ""
echo "✅ 配置验证:"
echo "   DATABASE_TYPE: $DATABASE_TYPE"
echo "   DATABASE_URL: ${DATABASE_URL:0:50}..."

# 安装 psycopg2 (如果需要)
echo ""
echo "📦 检查依赖..."
pip3 install -q psycopg2-binary 2>/dev/null || true

# 创建 PostgreSQL 适配器
echo ""
echo "🔧 创建数据库适配器..."

cat > evolution/evolution_router_neon.py << 'ROUTER_EOF'
"""
Neon PostgreSQL Adapter - Neon 数据库适配器
============================================

修改 EvolutionDatabase 以支持 PostgreSQL (Neon)
"""
import os
import json
from typing import Optional, List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor


class NeonDatabaseAdapter:
    """Neon PostgreSQL 数据库适配器"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # 解析连接字符串
        # 格式: postgresql://user:pass@host:port/db?sslmode=require
        self.conn_params = self._parse_connection_string(self.db_url)
        self._init_db()
    
    def _parse_connection_string(self, url: str) -> dict:
        """解析 PostgreSQL 连接字符串"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path[1:] if parsed.path else "neondb",
            "user": parsed.username,
            "password": parsed.password,
            "sslmode": "require"
        }
    
    def _get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(**self.conn_params)
    
    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # patterns 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                trigger_conditions TEXT,
                description TEXT,
                occurrences INTEGER DEFAULT 0,
                success_rate REAL,
                confidence REAL,
                first_observed TEXT,
                last_observed TEXT,
                metadata TEXT
            )
        """)
        
        # strategies 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                conditions TEXT,
                actions TEXT,
                success_metrics TEXT,
                total_uses INTEGER DEFAULT 0,
                success_rate REAL,
                avg_effectiveness REAL,
                evolution_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_used TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # hypotheses 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                statement TEXT,
                category TEXT,
                predictions TEXT,
                test_results TEXT,
                status TEXT,
                test_count INTEGER DEFAULT 0,
                confidence REAL,
                evidence_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_tested TEXT
            )
        """)
        
        # knowledge 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                title TEXT,
                domain TEXT,
                content TEXT,
                source_patterns TEXT,
                evidence TEXT,
                applicability TEXT,
                prerequisites TEXT,
                related_knowledge TEXT,
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_used TEXT
            )
        """)
        
        # interactions 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                interaction_type TEXT,
                input_summary TEXT,
                output_summary TEXT,
                outcome TEXT,
                pattern_matches TEXT,
                strategy_used TEXT,
                effectiveness REAL,
                feedback TEXT,
                timestamp TEXT
            )
        """)
        
        # evolution_events 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_events (
                id TEXT PRIMARY KEY,
                event_type TEXT,
                description TEXT,
                changes TEXT,
                before_state TEXT,
                after_state TEXT,
                trigger TEXT,
                impact REAL,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Neon 数据库表初始化完成")
    
    def save_pattern(self, pattern: dict):
        """保存模式"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO patterns (id, name, type, trigger_conditions, description,
                               occurrences, success_rate, confidence, first_observed,
                               last_observed, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                type = EXCLUDED.type,
                occurrences = EXCLUDED.occurrences,
                success_rate = EXCLUDED.success_rate,
                confidence = EXCLUDED.confidence,
                last_observed = EXCLUDED.last_observed
        """, (
            pattern.get("id"), pattern.get("name"), pattern.get("type"),
            json.dumps(pattern.get("trigger_conditions", {})),
            pattern.get("description"), pattern.get("occurrences", 0),
            pattern.get("success_rate", 0), pattern.get("confidence", 0),
            pattern.get("first_observed"), pattern.get("last_observed"),
            json.dumps(pattern.get("metadata", {}))
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def save_strategy(self, strategy: dict):
        """保存策略"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO strategies (id, name, type, conditions, actions,
                                  success_metrics, total_uses, success_rate,
                                  avg_effectiveness, evolution_count, created_at,
                                  last_used, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                total_uses = EXCLUDED.total_uses,
                success_rate = EXCLUDED.success_rate,
                avg_effectiveness = EXCLUDED.avg_effectiveness,
                last_used = EXCLUDED.last_used
        """, (
            strategy.get("id"), strategy.get("name"), strategy.get("type"),
            json.dumps(strategy.get("conditions", {})),
            json.dumps(strategy.get("actions", [])),
            json.dumps(strategy.get("success_metrics", {})),
            strategy.get("total_uses", 0), strategy.get("success_rate", 0),
            strategy.get("avg_effectiveness", 0), strategy.get("evolution_count", 0),
            strategy.get("created_at"), strategy.get("last_used"),
            1 if strategy.get("is_active", True) else 0
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def save_hypothesis(self, hypothesis: dict):
        """保存假设"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO hypotheses (id, statement, category, predictions,
                                 test_results, status, test_count,
                                 confidence, evidence_count, created_at, last_tested)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                statement = EXCLUDED.statement,
                status = EXCLUDED.status,
                test_count = EXCLUDED.test_count,
                confidence = EXCLUDED.confidence
        """, (
            hypothesis.get("id"), hypothesis.get("statement"),
            hypothesis.get("category"),
            json.dumps(hypothesis.get("predictions", [])),
            json.dumps(hypothesis.get("test_results", [])),
            hypothesis.get("status", "pending"),
            hypothesis.get("test_count", 0), hypothesis.get("confidence", 0),
            hypothesis.get("evidence_count", 0),
            hypothesis.get("created_at"), hypothesis.get("last_tested")
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def log_interaction(self, interaction: dict):
        """记录交互"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO interactions (id, user_id, session_id, interaction_type,
                                    input_summary, output_summary, outcome,
                                    pattern_matches, strategy_used, effectiveness,
                                    feedback, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            interaction.get("id"), interaction.get("user_id"),
            interaction.get("session_id"), interaction.get("interaction_type"),
            interaction.get("input_summary"), interaction.get("output_summary"),
            interaction.get("outcome"),
            json.dumps(interaction.get("pattern_matches", [])),
            interaction.get("strategy_used"),
            interaction.get("effectiveness"),
            interaction.get("feedback"), interaction.get("timestamp")
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def log_evolution_event(self, event_type: str, description: str,
                           changes: dict = None, trigger: str = None,
                           impact: float = 0.5):
        """记录进化事件"""
        conn = self._get_connection()
        cursor = conn.cursor()
        from datetime import datetime
        
        cursor.execute("""
            INSERT INTO evolution_events (id, event_type, description, changes,
                                       before_state, after_state, trigger, impact, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            str(datetime.now().timestamp()),
            event_type, description,
            json.dumps(changes or {}),
            None, None, trigger, impact,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def get_patterns(self) -> List[Dict]:
        """获取所有模式"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM patterns ORDER BY confidence DESC")
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(row) for row in results]
    
    def get_strategies(self) -> List[Dict]:
        """获取所有策略"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM strategies WHERE is_active = 1")
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(row) for row in results]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM patterns")
        stats["patterns"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM strategies WHERE is_active = 1")
        stats["strategies"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM hypotheses")
        stats["hypotheses"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        stats["knowledge"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM interactions")
        stats["interactions"] = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return stats


# 便捷函数
def get_neon_db() -> NeonDatabaseAdapter:
    """获取 Neon 数据库实例"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is required")
    return NeonDatabaseAdapter(db_url)


if __name__ == "__main__":
    print("🧪 测试 Neon 数据库连接...")
    
    db = get_neon_db()
    stats = db.get_statistics()
    
    print("\n📊 数据库统计:")
    print(f"   模式: {stats.get('patterns', 0)}")
    print(f"   策略: {stats.get('strategies', 0)}")
    print(f"   假设: {stats.get('hypotheses', 0)}")
    print(f"   知识: {stats.get('knowledge', 0)}")
    print(f"   交互: {stats.get('interactions', 0)}")
    
    print("\n✅ Neon 数据库适配器测试完成！")
ROUTER_EOF

echo "✅ 适配器创建完成"

# 测试连接
echo ""
echo "🔗 测试 Neon 连接..."
python3 evolution/evolution_router_neon.py

echo ""
echo "✅ Neon 数据库适配器配置完成！"
echo ""
echo "📝 使用方法:"
echo "   from evolution.evolution_router_neon import get_neon_db"
echo "   db = get_neon_db()"
