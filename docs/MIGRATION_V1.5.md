# V1.0 升级到 V1.5 分布式版本 - 完整迁移指南

## 🎯 迁移概述

| 项目 | V1.0 | V1.5 (升级后) |
|------|------|---------------|
| 架构 | 单节点 | 分布式微服务 |
| 并发能力 | ~5用户 | 50+用户 |
| 响应时间 | 5-12秒 | 2-5秒 (有缓存) |
| 吞吐量 | 0.39 req/s | 10+ req/s |
| 场状态 | SQLite本地 | Redis集群 |
| 高可用 | ❌ | ✅ |

---

## 📋 前置要求

- Docker Desktop 或 Docker Engine 20.10+
- Docker Compose 2.0+
- 至少 4GB 可用内存
- OpenAI API Key (或其他LLM提供商)

---

## 🚀 快速升级 (3步完成)

### 步骤1: 备份现有数据

```bash
# 进入项目目录
cd /Volumes/J\ ZAO\ 9\ SER\ 1/Python/Open\ Code/QUANTUM_FIELD_GUIDE

# 备份SQLite数据库
cp backend/quantum_memory.db backup/quantum_memory_v1.0.db

# 备份配置文件
cp backend/.env backup/.env.v1.0

# 备份前端 (如有自定义)
cp frontend/index.html backup/frontend_v1.0.html

echo "✅ 备份完成"
```

### 步骤2: 配置环境

```bash
# 进入V1.5目录
cd v1.5/backend

# 复制环境配置模板
cp .env.example .env

# 编辑配置文件 (使用你喜欢的编辑器)
nano .env  # 或 vim .env 或 code .env
```

**.env 配置示例**:
```bash
# LLM API配置 (必填)
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
WORKER_MODEL_NAME=gpt-4o

# 可选: 其他LLM提供商
# DEEPSEEK_API_KEY=sk-your-deepseek-key
# GROQ_API_KEY=gsk-your-groq-key

# Redis配置 (默认即可)
REDIS_URL=redis://redis-master:6379

# 服务配置
PORT=8000
HOST=0.0.0.0
ENABLE_WORKER=true
```

### 步骤3: 启动分布式集群

```bash
# 回到V1.5根目录
cd ..

# 方式A: 基础部署 (推荐)
docker-compose up -d

# 方式B: 高可用部署 (2个API节点)
docker-compose up -d --scale api-node-1=1 --scale api-node-2=1

# 方式C: 完整部署 (含GPU Worker)
docker-compose --profile gpu up -d
```

**启动过程**:
```
[+] Running 4/4
 ✔ Container qf-redis      Started  3.2s
 ✔ Container qf-api-1      Started  4.1s
 ✔ Container qf-api-2      Started  4.5s
 ✔ Container qf-nginx      Started  5.0s
```

---

## ✅ 验证部署

### 1. 检查服务状态

```bash
# 查看运行中的容器
docker-compose ps

# 预期输出:
NAME        IMAGE          STATUS          PORTS
qf-redis    redis:7        Up 10 seconds   0.0.0.0:6379->6379/tcp
qf-api-1    v1.5_backend   Up 8 seconds    0.0.0.0:8001->8000/tcp
qf-api-2    v1.5_backend   Up 7 seconds    0.0.0.0:8002->8000/tcp
qf-nginx    nginx:alpine   Up 6 seconds    0.0.0.0:8000->80/tcp
```

### 2. 健康检查

```bash
# 测试API
curl http://localhost:8000/health

# 预期响应:
{
  "status": "healthy",
  "version": "1.5.0-distributed",
  "components": {
    "redis": "connected",
    "field_manager": "active",
    "worker": "active"
  }
}
```

### 3. 访问前端

打开浏览器访问: **http://localhost:8000/frontend**

应该看到:
- ✅ 场熵显示条 (实时更新)
- ✅ 技能列表 (8个技能)
- ✅ 分布式量子场界面

---

## 🧪 功能测试

### 测试1: 基础对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "user_id": "test_user"}'
```

### 测试2: 场状态查询

```bash
# 查询用户场状态
curl http://localhost:8000/field/status/test_user

# 预期响应:
{
  "user_id": "test_user",
  "entropy": 0.1,
  "activated_skills": [],
  "in_local_cache": true
}
```

### 测试3: 高熵任务 (触发分布式计算)

```bash
# 复杂查询，应该触发高熵场
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "分析量子计算在AI领域的应用前景，并搜索最新进展", "user_id": "high_entropy_user"}'
```

---

## 📊 性能对比

启动后可以进行对比测试:

```bash
# V1.0 性能 (之前测试)
# 平均响应: 5-12秒
# 并发: 5用户
# 吞吐量: 0.39 req/s

# V1.5 性能 (预期)
# 平均响应: 2-5秒 (Redis缓存)
# 并发: 50+用户
# 吞吐量: 10+ req/s
```

---

## 🔧 常见问题

### 问题1: Redis连接失败

**症状**: 服务启动后立刻退出

**解决**:
```bash
# 检查Redis状态
docker-compose logs redis-master

# 重启Redis
docker-compose restart redis-master

# 查看Redis是否就绪
docker-compose exec redis-master redis-cli ping
# 应该返回: PONG
```

### 问题2: API Key无效

**症状**: 返回 "Authentication Error"

**解决**:
```bash
# 检查.env文件
cat backend/.env | grep OPENAI_API_KEY

# 重新加载配置
docker-compose down
docker-compose up -d
```

### 问题3: 端口被占用

**症状**: "bind: address already in use"

**解决**:
```bash
# 查找占用端口的进程
lsof -i :8000
lsof -i :8001
lsof -i :8002
lsof -i :6379

# 停止占用进程
kill -9 <PID>

# 或修改docker-compose.yml中的端口映射
```

### 问题4: 内存不足

**症状**: 容器启动后OOM

**解决**:
```bash
# 查看内存使用
docker stats

# 限制容器内存 (修改docker-compose.yml)
services:
  api-node-1:
    deploy:
      resources:
        limits:
          memory: 512M
```

---

## 📈 监控和日志

### 查看实时日志

```bash
# 所有服务日志
docker-compose logs -f

# 特定服务日志
docker-compose logs -f api-node-1
docker-compose logs -f redis-master
docker-compose logs -f nginx
```

### 查看性能指标

```bash
# 系统统计
curl http://localhost:8000/stats

# 场状态分布
docker-compose exec redis-master redis-cli
> KEYS qf:field:*
> LLEN qf:compute_queue
```

### 监控面板

```bash
# 实时查看容器状态
watch -n 1 docker-compose ps

# 查看资源使用
watch -n 1 docker stats
```

---

## 🔄 回滚到V1.0 (如需)

如果升级后遇到问题，可以快速回滚:

```bash
# 停止V1.5服务
cd v1.5
docker-compose down

# 回到V1.0
cd ..
cd backend
source venv/bin/activate
python main.py

# 恢复前端
cp backup/frontend_v1.0.html ../frontend/index.html
```

---

## 🎉 升级完成检查清单

- [ ] Docker容器全部运行中 (`docker-compose ps`)
- [ ] 健康检查通过 (`curl localhost:8000/health`)
- [ ] 前端页面可访问 (`http://localhost:8000/frontend`)
- [ ] 基础对话测试成功
- [ ] 场状态查询正常
- [ ] 技能节点动画显示
- [ ] 日志无错误信息

---

## 💡 下一步优化

### 启用HTTPS (生产环境)

```bash
# 生成SSL证书
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/nginx.key \
  -out nginx/ssl/nginx.crt

# 修改nginx.conf启用SSL
# 端口改为443, 添加ssl配置
```

### 添加监控

```bash
# 安装Prometheus + Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# 查看面板: http://localhost:3000
```

### 扩展Worker节点

```bash
# 启动多个GPU Worker
docker-compose up -d --scale compute-worker-1=3
```

---

## 📞 支持

如果遇到问题:

1. 查看日志: `docker-compose logs`
2. 检查配置: `cat backend/.env`
3. 重启服务: `docker-compose restart`
4. 完全重置: `docker-compose down -v && docker-compose up -d`

**恭喜！您已成功升级到V1.5分布式量子场架构！** 🚀
