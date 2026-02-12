# Quantum Field Agent - 增量升级架构设计

## 📋 问题分析

### 当前问题
- ❌ 创建了独立的v1.5目录，取代了v1.0
- ❌ 数据不继承，需要重新配置
- ❌ 无法平滑回滚

### 正确做法
- ✅ 在v1.0基础上**增量添加**功能
- ✅ 保留所有数据和配置
- ✅ 支持版本切换和回滚
- ✅ 为2.0, 3.0, 4.0预留扩展接口

---

## 🏗️ 增量升级架构

```
quantum-field-agent/
├── backend/
│   ├── main.py                    # 核心入口 (保留v1.0)
│   ├── quantum_memory.db          # SQLite数据 (保留)
│   ├── .env                       # 配置 (保留)
│   ├── skills/                    # 技能库 (保留v1.0全部)
│   │   ├── __init__.py
│   │   ├── search_weather.py
│   │   ├── calculate.py
│   │   └── ...
│   │
│   ├── version/                   # 【新增】版本管理
│   │   ├── __init__.py
│   │   ├── base.py               # 版本基类
│   │   ├── v1_0.py               # V1.0实现
│   │   ├── v1_5.py               # V1.5实现 (分布式)
│   │   └── manager.py            # 版本管理器
│   │
│   ├── core/                      # 【新增】核心功能
│   │   ├── __init__.py
│   │   ├── field_state.py        # 场状态管理
│   │   ├── distributed.py        # 分布式计算
│   │   └── cache.py              # 缓存管理
│   │
│   ├── migration/                 # 【新增】数据迁移
│   │   ├── __init__.py
│   │   ├── v1_0_to_v1_5.py       # v1.0->v1.5迁移
│   │   └── rollback.py           # 回滚脚本
│   │
│   └── extensions/                # 【预留】未来扩展
│       ├── __init__.py
│       └── plugin_loader.py      # 插件加载器
│
├── frontend/
│   ├── index.html                 # 前端 (增量更新)
│   └── assets/                    # 【新增】静态资源
│
└── docker-compose.yml             # 【新增】可选部署
```

---

## 🔄 版本管理机制

### 1. 版本基类设计

```python
# backend/version/base.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseVersion(ABC):
    """版本基类 - 所有版本必须实现"""
    
    VERSION = "0.0.0"
    NAME = "base"
    
    @abstractmethod
    async def process_intent(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        """处理用户意图 - 核心方法"""
        pass
    
    @abstractmethod
    async def get_field_status(self, user_id: str) -> dict:
        """获取场状态"""
        pass
    
    @abstractmethod
    async def reset_field(self, user_id: str) -> dict:
        """重置场"""
        pass
```

### 2. V1.0 实现 (保留现有代码)

```python
# backend/version/v1_0.py
from .base import BaseVersion

class VersionV1_0(BaseVersion):
    """V1.0 实现 - 单节点架构"""
    
    VERSION = "1.0.0"
    NAME = "quantum-field-v1"
    
    def __init__(self):
        self.skills = {}  # 使用现有的skills
        self.memory_db = "quantum_memory.db"
    
    async def process_intent(self, user_id: str, message: str):
        """使用现有的main.py逻辑"""
        # 直接调用现有的处理逻辑
        # 保持100%兼容
        pass
```

### 3. V1.5 实现 (增量添加)

```python
# backend/version/v1_5.py
from .v1_0 import VersionV1_0
import redis

class VersionV1_5(VersionV1_0):
    """V1.5 实现 - 在V1.0基础上添加分布式功能"""
    
    VERSION = "1.5.0"
    NAME = "quantum-field-v1.5-distributed"
    
    def __init__(self):
        super().__init__()
        # 增量添加Redis支持
        self.redis_client = None
        self.field_cache = {}
    
    async def process_intent(self, user_id: str, message: str):
        """增强版处理 - 自动选择本地或分布式"""
        # 1. 先调用父类V1.0逻辑
        # 2. 如果场熵高，使用Redis缓存
        # 3. 支持分布式Worker
        pass
```

### 4. 版本管理器

```python
# backend/version/manager.py
class VersionManager:
    """版本管理器 - 动态切换版本"""
    
    VERSIONS = {
        "1.0.0": "v1_0.VersionV1_0",
        "1.5.0": "v1_5.VersionV1_5",
        # 预留未来版本
        # "2.0.0": "v2_0.VersionV2_0",
    }
    
    def __init__(self, target_version: str = "1.0.0"):
        self.current_version = None
        self.load_version(target_version)
    
    def load_version(self, version: str):
        """加载指定版本"""
        # 动态导入版本类
        # 支持热切换
        pass
    
    def upgrade(self, new_version: str):
        """升级到新版本"""
        # 1. 备份当前数据
        # 2. 运行迁移脚本
        # 3. 加载新版本
        # 4. 验证兼容性
        pass
    
    def rollback(self):
        """回滚到上一版本"""
        # 使用备份数据恢复
        pass
```

---

## 📊 数据迁移策略

### 1. 迁移脚本设计

```python
# backend/migration/v1_0_to_v1_5.py
class MigrationV1_0ToV1_5:
    """v1.0 到 v1.5 的迁移"""
    
    def __init__(self):
        self.source_db = "quantum_memory.db"
        self.backup_db = "backup/quantum_memory_v1.0.db"
    
    def backup(self):
        """备份v1.0数据"""
        import shutil
        shutil.copy(self.source_db, self.backup_db)
    
    def migrate(self):
        """执行迁移"""
        # 1. 备份数据
        self.backup()
        
        # 2. 添加新表(如果有)
        # SQLite -> Redis的数据转换
        
        # 3. 验证数据完整性
        
        return True
    
    def rollback(self):
        """回滚到v1.0"""
        import shutil
        shutil.copy(self.backup_db, self.source_db)
```

### 2. 兼容性层

```python
# backend/core/compatibility.py
class CompatibilityLayer:
    """兼容性层 - 处理不同版本间的差异"""
    
    @staticmethod
    def adapt_field_state(old_state: dict) -> dict:
        """适配旧版场状态到新格式"""
        # 添加新字段的默认值
        # 转换数据格式
        pass
    
    @staticmethod
    def adapt_response(old_response: str) -> str:
        """适配旧版响应格式"""
        # 处理格式差异
        pass
```

---

## 🚀 升级步骤 (增量式)

### 步骤1: 备份 (自动)
```python
# 升级前自动备份
version_manager.backup()
```

### 步骤2: 安装依赖 (增量)
```bash
# 只安装新增的依赖
pip install redis  # v1.5新增
```

### 步骤3: 数据迁移 (自动)
```python
# 自动迁移数据
migration = MigrationV1_0ToV1_5()
migration.migrate()
```

### 步骤4: 加载新版本
```python
# 切换到v1.5
version_manager.upgrade("1.5.0")
```

### 步骤5: 验证
```python
# 自动验证所有功能
version_manager.verify()
```

---

## 📝 为2.0, 3.0预留的扩展点

### 1. 插件系统
```python
# backend/extensions/plugin_loader.py
class PluginLoader:
    """插件加载器 - 支持未来版本的功能扩展"""
    
    def load_plugin(self, plugin_name: str):
        """动态加载插件"""
        pass
```

### 2. 配置系统
```python
# backend/core/config.py
class VersionConfig:
    """版本配置 - 支持多版本配置管理"""
    
    def get_config(self, version: str):
        """获取指定版本的配置"""
        pass
```

### 3. API版本控制
```python
# 支持不同版本的API端点
@app.post("/v1/chat")      # v1.0
@app.post("/v1.5/chat")    # v1.5
@app.post("/v2/chat")      # v2.0 (预留)
```

---

## ✅ 实施计划

### Phase 1: 重构现有代码
1. 创建version/目录结构
2. 将现有main.py提取为v1_0.py
3. 添加版本基类

### Phase 2: 添加v1.5功能
1. 创建v1_5.py (继承v1_0)
2. 添加Redis支持
3. 添加分布式功能

### Phase 3: 数据迁移
1. 创建迁移脚本
2. 添加兼容性层
3. 测试数据迁移

### Phase 4: 版本切换
1. 添加版本管理器
2. 支持动态切换
3. 添加回滚功能

**现在开始执行Phase 1？** 还是您有其他建议？
