"""
Security Auth - 安全认证模块
==========================

安全认证功能：
1. JWT 令牌生成和验证
2. 用户认证
3. 权限控制
4. API 密钥管理

核心理念：
- 安全是生产环境的基础
- 最小权限原则
- 纵深防御策略
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
import jwt
import bcrypt
import uuid


class UserRole(Enum):
    """用户角色"""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API = "api"


@dataclass
class User:
    """用户"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    hashed_password: str = ""
    role: str = UserRole.USER.value
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: str = ""
    permissions: List[str] = field(default_factory=list)


@dataclass
class Token:
    """访问令牌"""

    access_token: str = ""
    token_type: str = "bearer"
    expires_in: int = 0
    refresh_token: str = ""


@dataclass
class APIKey:
    """API密钥"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str = field(default_factory=lambda: secrets.token_hex(32))
    name: str = ""
    user_id: str = ""
    permissions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = ""
    expires_at: str = ""
    is_active: bool = True


class SecurityManager:
    """安全管理器"""

    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self.secret_key = secret_key or os.getenv("SECRET_KEY", "dev-secret-key")
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._revoked_tokens: set = set()

    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode(), hashed.encode())

    def create_user(
        self, username: str, password: str, email: str = "", role: str = "user"
    ) -> User:
        """创建用户"""
        user = User(
            username=username,
            email=email,
            hashed_password=self.hash_password(password),
            role=role,
            permissions=self._get_default_permissions(role),
        )
        self._users[username] = user
        return user

    def authenticate(self, username: str, password: str) -> Optional[Token]:
        """用户认证"""
        user = self._users.get(username)
        if not user or not user.is_active:
            return None

        if not self.verify_password(password, user.hashed_password):
            return None

        user.last_login = datetime.now().isoformat()
        return self.create_tokens(user)

    def _get_default_permissions(self, role: str) -> List[str]:
        """获取默认权限"""
        permissions = {
            UserRole.ADMIN.value: ["read", "write", "delete", "admin"],
            UserRole.USER.value: ["read", "write"],
            UserRole.VIEWER.value: ["read"],
            UserRole.API.value: ["read", "write:api"],
        }
        return permissions.get(role, ["read"])

    def create_tokens(self, user: User) -> Token:
        """创建令牌"""
        now = datetime.now()

        access_payload = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role,
            "permissions": user.permissions,
            "iat": now,
            "exp": now + timedelta(minutes=self.access_token_expire_minutes),
            "type": "access",
        }

        access_token = jwt.encode(
            access_payload, self.secret_key, algorithm=self.algorithm
        )

        refresh_payload = {
            "sub": user.username,
            "iat": now,
            "exp": now + timedelta(days=self.refresh_token_expire_days),
            "type": "refresh",
        }

        refresh_token = jwt.encode(
            refresh_payload, self.secret_key, algorithm=self.algorithm
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60,
            refresh_token=refresh_token,
        )

    def verify_token(self, token: str) -> Optional[Dict]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") == "refresh":
                return None

            if token in self._revoked_tokens:
                return None

            return payload
        except jwt.PyJWTError:
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[Token]:
        """刷新访问令牌"""
        try:
            payload = jwt.decode(
                refresh_token, self.secret_key, algorithms=[self.algorithm]
            )

            if payload.get("type") != "refresh":
                return None

            username = payload.get("sub")
            user = self._users.get(username)

            if not user or not user.is_active:
                return None

            return self.create_tokens(user)

        except jwt.PyJWTError:
            return None

    def revoke_token(self, token: str):
        """撤销令牌"""
        self._revoked_tokens.add(token)

    def create_api_key(
        self, name: str, user_id: str, permissions: List[str] = None
    ) -> APIKey:
        """创建API密钥"""
        api_key = APIKey(
            name=name,
            user_id=user_id,
            permissions=permissions or ["read"],
        )
        self._api_keys[api_key.key] = api_key
        return api_key

    def verify_api_key(self, key: str) -> Optional[APIKey]:
        """验证API密钥"""
        api_key = self._api_keys.get(key)
        if not api_key or not api_key.is_active:
            return None

        if (
            api_key.expires_at
            and datetime.fromisoformat(api_key.expires_at) < datetime.now()
        ):
            return None

        api_key.last_used = datetime.now().isoformat()
        return api_key

    def check_permission(self, user: User, permission: str) -> bool:
        """检查权限"""
        return permission in user.permissions or "admin" in user.permissions

    def require_permission(self, user: User, permission: str):
        """要求权限"""
        if not self.check_permission(user, permission):
            raise PermissionError(f"Permission '{permission}' required")


async def demo_security():
    """演示安全模块"""
    print("=" * 60)
    print("Security Auth - 演示")
    print("=" * 60)

    security = SecurityManager(
        secret_key="demo-secret-key-change-in-production",
        access_token_minutes=30,
        refresh_days=7,
    )

    admin = security.create_user(
        username="admin",
        password="admin123",
        email="admin@example.com",
        role="admin",
    )
    print(f"\n👤 创建用户: {admin.username} ({admin.role})")

    user = security.create_user(
        username="testuser",
        password="test123",
        email="test@example.com",
        role="user",
    )
    print(f"👤 创建用户: {user.username} ({user.role})")

    tokens = security.authenticate("admin", "admin123")
    print(f"\n🔑 认证成功!")
    print(f"  访问令牌: {tokens.access_token[:20]}...")
    print(f"  刷新令牌: {tokens.refresh_token[:20]}...")
    print(f"  过期时间: {tokens.expires_in}s")

    payload = security.verify_token(tokens.access_token)
    print(f"\n✅ 令牌验证成功!")
    print(f"  用户: {payload.get('sub')}")
    print(f"  权限: {payload.get('permissions')}")

    api_key = security.create_api_key(
        name="Production API",
        user_id=admin.id,
        permissions=["read", "write:api"],
    )
    print(f"\n🔐 创建API密钥: {api_key.key[:10]}...")

    verified = security.verify_api_key(api_key.key)
    print(f"  验证: {'成功' if verified else '失败'}")
    print(f"  权限: {verified.permissions}")

    print(f"\n✅ 安全模块演示完成")
    return security
