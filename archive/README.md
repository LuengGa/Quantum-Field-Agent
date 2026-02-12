# Archive 目录说明

## v1.5 文件夹

此文件夹包含 **Quantum Field Agent V1.5** 的原始独立实现。

### 状态
- **已归档**: 2026-02-12
- **原因**: 已融合到主架构中

### 内容
- 完整的 V1.5 独立版本
- Docker 集群部署配置
- 分布式场核心实现

### 当前使用
现在使用 **融合版本** (backend/quantum_field.py):
```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001
```

### 如需独立 V1.5
```bash
cd archive/v1.5
docker-compose up -d
```

### 保留目的
1. 备份/参考原始 V1.5 实现
2. 如需完整集群部署，可直接使用
3. 历史版本追溯

---

**注意**: 主项目现在使用融合架构 (V1.0 + V1.5)，无需单独部署 V1.5。
