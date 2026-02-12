"""
Production Config - 生产环境配置
==============================

生产环境优化：
1. PostgreSQL 数据库支持
2. Redis 缓存层
3. API 速率限制
4. 连接池配置
5. 安全配置

核心理念：
- 生产环境需要高可用
- 缓存提升性能
- 限流保护系统
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DatabaseConfig:
    """数据库配置"""

    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "quantum_field"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量加载配置"""
        db_type = os.getenv("DB_TYPE", "sqlite")

        if db_type == "postgresql":
            return cls(
                type="postgresql",
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                name=os.getenv("DB_NAME", "quantum_field"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            )
        return cls()


@dataclass
class RedisConfig:
    """Redis配置"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    key_prefix: str = "quantum_field:"
    cache_ttl: int = 3600
    max_connections: int = 50

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """从环境变量加载配置"""
        if os.getenv("REDIS_HOST"):
            return cls(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD", ""),
                key_prefix=os.getenv("REDIS_PREFIX", "quantum_field:"),
                cache_ttl=int(os.getenv("REDIS_CACHE_TTL", "3600")),
            )
        return None


@dataclass
class RateLimitConfig:
    """速率限制配置"""

    enabled: bool = True
    default_rate: int = 100
    default_period: int = 60
    api_rate: int = 1000
    api_period: int = 60
    auth_rate: int = 10
    auth_period: int = 60

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """从环境变量加载配置"""
        return cls(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            default_rate=int(os.getenv("RATE_LIMIT_DEFAULT", "100")),
            api_rate=int(os.getenv("RATE_LIMIT_API", "1000")),
        )


@dataclass
class SecurityConfig:
    """安全配置"""

    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    cors_origins: List[str] = field(default_factory=list)
    cors_methods: List[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """从环境变量加载配置"""
        return cls(
            secret_key=os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(
                os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
            ),
            cors_origins=os.getenv("CORS_ORIGINS", "").split(","),
        )


@dataclass
class LoggingConfig:
    """日志配置"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_size_mb: int = 100
    backup_count: int = 5
    json_format: bool = False


@dataclass
class ProductionConfig:
    """生产环境配置"""

    debug: bool = False
    environment: str = "development"

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: Optional[RedisConfig] = field(default_factory=lambda: None)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    api_version: str = "v1"
    title: str = "Meta Quantum Field Agent"
    description: str = "AI协作系统 - 过程即幻觉，I/O即实相"
    version: str = "4.0.0"

    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """从环境变量加载完整配置"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            environment=os.getenv("ENVIRONMENT", "development"),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            rate_limit=RateLimitConfig.from_env(),
            security=SecurityConfig.from_env(),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                json_format=os.getenv("LOG_JSON", "false").lower() == "true",
            ),
        )


class CacheManager:
    """缓存管理器"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._client = None

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self._client:
            return None
        try:
            value = await self._client.get(f"{self.config.key_prefix}{key}")
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存"""
        if not self._client:
            return
        try:
            await self._client.set(
                f"{self.config.key_prefix}{key}",
                json.dumps(value),
                ex=ttl or self.config.cache_ttl,
            )
        except Exception:
            pass

    async def delete(self, key: str):
        """删除缓存"""
        if not self._client:
            return
        try:
            await self._client.delete(f"{self.config.key_prefix}{key}")
        except Exception:
            pass

    async def clear_pattern(self, pattern: str):
        """清除匹配模式的缓存"""
        if not self._client:
            return
        try:
            keys = await self._client.keys(f"{self.config.key_prefix}{pattern}")
            if keys:
                await self._client.delete(*keys)
        except Exception:
            pass


class RateLimiter:
    """速率限制器"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: Dict[str, List[datetime]] = {}

    async def check_rate_limit(
        self, identifier: str, rate: int, period: int
    ) -> tuple[bool, int]:
        """
        检查速率限制

        Returns:
            (是否允许, 剩余请求数)
        """
        now = datetime.now()
        window_start = now.timestamp() - period

        if identifier not in self._requests:
            self._requests[identifier] = []

        self._requests[identifier] = [
            t for t in self._requests[identifier] if t.timestamp() > window_start
        ]

        remaining = rate - len(self._requests[identifier])

        if remaining > 0:
            self._requests[identifier].append(now)
            return True, remaining

        return False, 0

    async def check(self, identifier: str) -> tuple[bool, int]:
        """使用默认配置检查速率限制"""
        return await self.check_rate_limit(
            identifier,
            self.config.default_rate,
            self.config.default_period,
        )


async def demo_production_config():
    """演示生产配置"""
    print("=" * 60)
    print("Production Config - 演示")
    print("=" * 60)

    config = ProductionConfig.from_env()

    print(f"\n📦 数据库配置:")
    print(f"  类型: {config.database.type}")
    print(f"  主机: {config.database.host}")
    print(f"  端口: {config.database.port}")

    if config.redis:
        print(f"\n📦 Redis配置:")
        print(f"  主机: {config.redis.host}")
        print(f"  端口: {config.redis.port}")
        print(f"  缓存TTL: {config.redis.cache_ttl}s")
    else:
        print(f"\n📦 Redis: 未配置")

    print(f"\n⚡ 速率限制:")
    print(f"  启用: {config.rate_limit.enabled}")
    print(
        f"  默认限制: {config.rate_limit.default_rate}/{config.rate_limit.default_period}s"
    )
    print(f"  API限制: {config.rate_limit.api_rate}/{config.rate_limit.api_period}s")

    print(f"\n🔐 安全配置:")
    print(f"  JWT算法: {config.security.algorithm}")
    print(f"  Token过期: {config.security.access_token_expire_minutes}分钟")
    print(f"  CORS源: {config.security.cors_origins}")

    print(f"\n✅ 配置演示完成")
    return config
