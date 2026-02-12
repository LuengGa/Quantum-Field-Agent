# Quantum Field Agent - Neon + 服务器部署指南
# ===========================================

## 目录

1. [方案架构](#方案架构)
2. [Neon数据库配置](#neon数据库配置)
3. [服务器配置](#服务器配置)
4. [本地开发配置](#本地开发配置)
5. [部署步骤](#部署步骤)
6. [运维命令](#运维命令)

---

## 🏗️ 方案架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         架构图                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────┐         ┌────────────────────┐      │
│  │    用户浏览器       │         │   CloudBase        │      │
│  │   (Vue3前端)       │         │   (前端托管+CDN)   │      │
│  └─────────┬──────────┘         └────────────────────┘      │
│            │                                                   │
│            ▼                                                   │
│  ┌────────────────────┐                                        │
│  │  腾讯云轻量服务器   │                                        │
│  │  ┌──────────────┐ │         ┌────────────────────┐      │
│  │  │ Nginx       │ │         │   Neon             │      │
│  │  │ (SSL/代理)  │ │────────▶│   PostgreSQL       │      │
│  │  └──────────────┘ │         │   (Serverless)     │      │
│  │  ┌──────────────┐ │         │   免费10GB        │      │
│  │  │ FastAPI     │ │         └────────────────────┘      │
│  │  │ (API服务)    │ │                                        │
│  │  └──────────────┘ │                                        │
│  │  ┌──────────────┐ │                                        │
│  │  │ Redis       │ │                                        │
│  │  │ (缓存)      │ │                                        │
│  │  └──────────────┘ │                                        │
│  └───────────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

费用:
- 轻量服务器: ¥30/月 (2核4G)
- Neon: 免费(10GB) / $25/月(无限)
- CloudBase: 免费 (静态托管)
- 域名: ¥5/月 (可选)
─────────────────────────────────────────────────────────────────
总计: ¥35-60/月
```

---

## 🗄️ Neon 数据库配置

### 1. 注册 Neon

```
1. 打开 https://neon.tech
2. 点击 "Sign Up" 注册 (可用GitHub登录)
3. 完成邮箱验证
```

### 2. 创建数据库

```
1. 点击 "Create Project"
2. 配置:
   - Name: quantum-field-agent
   - PostgreSQL Version: 15
   - Region: us-east-1 (美国) 或 eu-west-1 (欧洲)
   - Compute Size: Autoscale (0-2 vCPU)
3. 点击 "Create Project"
```

### 3. 获取连接字符串

```
项目创建完成后，你会看到:

Connection string:
postgresql://alex:AbC123@ep-xyz.us-east-1.aws.neon.tech/quantum-field-agent?sslmode=require

或者在 Settings → Connection 中查看
```

### 4. Neon 特点

```
✅ 免费额度:
   - 10GB 存储
   - 每月 300 IOPS
   - 3个数据库分支 (dev/test/prod)

✅ 高级功能 (付费):
   - 无限存储
   - 更高IOPS
   - 优先支持
```

---

## 🖥️ 服务器配置

### 1. 购买轻量服务器

```
腾讯云: https://cloud.tencent.com/product/lighthouse

推荐配置:
- 地域: 上海 (低延迟)
- CPU: 2核
- 内存: 4GB
- 磁盘: 50GB SSD
- 系统: Ubuntu 22.04 LTS
- 带宽: 5Mbps
- 费用: ¥30/月
```

### 2. 连接服务器

```bash
# SSH 连接
ssh root@你的服务器IP

# 密码在腾讯云控制台获取
```

### 3. 安装环境

```bash
# 更新系统
apt update && apt upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com | sh
usermod -aG docker $USER

# 安装 Docker Compose
pip install docker-compose
```

---

## 💻 本地开发配置

### 1. 环境变量

```bash
# .env 文件 (本地开发)
cd backend

cat > .env << EOF
# Neon 数据库 (生产)
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/quantum-field-agent?sslmode=require

# 或者本地 SQLite (开发)
DATABASE_TYPE=sqlite
DATABASE_URL=sqlite:///data/evolution.db

# Redis (可选)
REDIS_HOST=localhost
REDIS_PORT=6379

# 安全
SECRET_KEY=your-secret-key-here
EOF
```

### 2. 修改代码支持 Neon

```python
# evolution/database.py 中修改连接逻辑

def get_connection_string():
    import os
    db_type = os.getenv("DATABASE_TYPE", "sqlite")
    
    if db_type == "postgresql":
        return os.getenv("DATABASE_URL")
    else:
        return "sqlite:///data/evolution.db"
```

### 3. 本地测试 Neon

```bash
# 1. 安装 PostgreSQL 驱动
pip install psycopg2-binary

# 2. 测试连接
python3 -c "
import os
from evolution.database import EvolutionDatabase

db = EvolutionDatabase()
print('✅ 数据库连接成功')
"
```

---

## 🚀 部署步骤

### 方式一：一键部署

```bash
# 1. 上传项目到服务器
scp -r quantum-field-agent root@你的IP:/opt/

# 2. SSH 连接服务器
ssh root@你的IP

# 3. 执行部署
cd /opt/quantum-field-agent/backend
chmod +x deploy_neon.sh
./deploy_neon.sh full
```

### 方式二：手动部署

```bash
# 1. 安装依赖
apt update
apt install -y python3-pip docker.io

# 2. 配置环境变量
cat > /opt/quantum-field-agent/backend/.env << EOF
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/quantum-field-agent?sslmode=require
SECRET_KEY=your-production-secret-key
LOG_LEVEL=INFO
EOF

# 3. 安装 Python 依赖
cd /opt/quantum-field-agent/backend
pip install -r requirements.txt
pip install psycopg2-binary uvicorn

# 4. 启动服务
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &

# 5. 配置 Nginx (见下方)
```

### 3. 配置 Nginx

```bash
# 安装 Nginx
apt install -y nginx

# 配置
cat > /etc/nginx/sites-available/quantum-field << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/quantum-field /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

### 4. 配置域名 (可选)

```
腾讯云 → 域名注册 → 购买域名
DNS解析 → 添加记录:
  - A记录 @ 服务器IP
  - CNAME www 服务器IP
```

---

## 🔧 运维命令

### 服务管理

```bash
# 查看状态
./deploy_neon.sh status

# 查看日志
tail -f /opt/quantum-field-agent/backend/logs/app.log

# 重启服务
./deploy_neon.sh restart

# 停止服务
./deploy_neon.sh stop

# 更新部署
./deploy_neon.sh update
```

### 数据库备份

```bash
# Neon 自动备份，无需手动操作
# 如需手动备份:
pg_dump "postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/quantum-field-agent" > backup.sql
```

### 监控

```bash
# API 健康检查
curl http://localhost:8000/health

# 查看进程
ps aux | grep uvicorn

# 查看端口
netstat -tlnp | grep 8000
```

---

## 🔒 安全配置

### 1. 防火墙

```bash
# 腾讯云控制台 → 防火墙
开放端口:
- 22 (SSH)
- 80 (HTTP)
- 443 (HTTPS)
```

### 2. SSL 证书

```bash
# 安装 Certbot
apt install -y certbot python3-certbot-nginx

# 申请证书
certbot --nginx -d your-domain.com
```

### 3. 环境变量安全

```bash
# 不要提交 .env 到 Git
echo ".env" >> .gitignore
```

---

## 📊 监控告警

### 腾讯云云监控

```
1. 腾讯云控制台 → 云监控
2. 创建告警策略:
   - CPU使用率 > 80%
   - 内存使用率 > 85%
   - 磁盘使用率 > 90%
3. 设置通知: 短信/邮件
```

### Neon 控制台

```
Neon 控制台 → Project → Monitoring
查看:
- CPU 使用率
- IOPS 使用率
- 存储使用量
```

---

## 💰 费用总结

| 项目 | 免费 | 付费 |
|------|------|------|
| Neon | 10GB | $25/月起 |
| 轻量服务器 | - | ¥30/月 |
| 域名 | - | ¥5/月 |
| SSL证书 | 免费 | 免费 |
| **总计** | ¥35/月 | ¥60/月起 |

---

## ❓ 常见问题

### Q1: Neon 连接失败？

```bash
# 检查连接字符串
curl "postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/quantum-field-agent?sslmode=require" -o /dev/null -w "%{http_code}"

# 确认 IP 白名单
Neon → Settings → IP Allowlist → Add 0.0.0.0/0
```

### Q2: 性能差？

```
Neon:
- 升级到付费计划
- 增加 compute size

服务器:
- 升级配置
- 添加 Redis 缓存
```

### Q3: 如何迁移数据库？

```bash
# 从 SQLite 迁移
python3 -c "
import sqlite3
import psycopg2

# 读取 SQLite
conn1 = sqlite3.connect('data/evolution.db')
# 写入 PostgreSQL
conn2 = psycopg2.connect('postgresql://...')
# 迁移数据...
"
```

---

## ✅ 快速检查清单

- [ ] Neon 账户创建
- [ ] 数据库连接测试
- [ ] 轻量服务器购买
- [ ] Docker 安装
- [ ] 项目上传
- [ ] 环境变量配置
- [ ] 服务启动测试
- [ ] 域名解析 (可选)
- [ ] SSL 证书 (可选)
- [ ] 监控告警配置

---

## 📞 获取帮助

- Neon 文档: https://neon.tech/docs
- 腾讯云文档: https://cloud.tencent.com/document
- 项目问题: GitHub Issues
