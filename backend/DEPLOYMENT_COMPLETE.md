# 完整开发与部署指南

## 1. GitHub Codespaces 开发环境

### 启动方式
1. 打开 https://github.com/LuengGa/Quantum-Field-Agent
2. 点击 **"Code"** → **"Create codespace on main"**
3. 等待环境自动配置（约2-3分钟）

### 手动设置（如果自动失败）
```bash
# 在 codespace 终端中
cd backend
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 添加 API keys
```

### 启动服务
```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 访问
- **API**: 点击 Ports 标签 → 点击 Globe 图标 (端口 8000)
- **文档**: http://localhost:8000/docs

---

## 2. Neon 数据库连接测试

### 在 Codespaces 中测试
```bash
cd backend
export DATABASE_URL="postgresql://neondb_owner:npg_3WIRYidN6HvZ@ep-old-bar-a1jkdluf-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
python3 -c "
from evolution.evolution_router_neon import get_neon_db
db = get_neon_db()
print('✅ Neon 连接成功:', db.get_statistics())
"
```

### 数据库表结构
- `patterns` - 模式存储
- `strategies` - 策略存储
- `hypotheses` - 假设存储
- `knowledge` - 知识存储
- `interactions` - 交互日志
- `evolution_events` - 进化事件

---

## 3. 免费服务器部署选项

### 选项 A: GitHub Codespaces（推荐用于开发）
- **免费额度**: 120 core-hours/月，2 CPU，4GB RAM
- **适合**: 开发、测试、小规模演示
- **限制**: 空闲 30 分钟后停止

### 选项 B: Oracle Cloud Always Free（推荐用于生产）
- **免费额度**: 永久免费，1GB RAM，1 CPU
- **适合**: 长期运行的生产服务
- **设置**:
  1. 注册 https://cloud.oracle.com
  2. 创建 Always Free VM
  3. SSH 登录并部署

### 选项 C: Railway（简单部署）
- **免费额度**: $5/月
- **适合**: 快速部署
- **设置**:
  ```bash
  npm install -g railway
  railway login
  railway init
  railway up
  ```

### 选项 D: Render
- **免费额度**: 750小时/月
- **适合**: Web 服务
- **设置**: 连接 GitHub 仓库自动部署

---

## 4. Oracle Cloud 部署步骤（永久免费）

### 步骤 1: 创建 VM
1. 登录 https://cloud.oracle.com
2. Compute → Instances → Create Instance
3. 选择: **Oracle Linux 8** (或 Ubuntu)
4. Shape: **VM.Standard.E2.1.Micro** (Always Free)
5. 添加 SSH 密钥

### 步骤 2: 连接到服务器
```bash
ssh -i your-key.pem opc@your-server-ip
```

### 步骤 3: 安装依赖
```bash
sudo yum install -y git python3 pip docker
sudo systemctl start docker
sudo usermod -aG docker opc
```

### 步骤 4: 部署项目
```bash
git clone https://github.com/LuengGa/Quantum-Field-Agent.git
cd Quantum-Field-GUIDE/backend
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 添加 API keys
```

### 步骤 5: 使用 Docker 启动
```bash
docker build -t quantum-agent .
docker run -d -p 8000:8000 --env-file .env quantum-agent
```

### 步骤 6: 配置反向代理（可选）
使用 Nginx 或 Caddy 提供 HTTPS 和域名访问。

---

## 5. 验证部署

### 测试 Neon 连接
```bash
python3 -c "
from evolution.evolution_router_neon import get_neon_db
db = get_neon_db()
print('Neon 统计:', db.get_statistics())
"
```

### 测试 API
```bash
curl http://localhost:8000/health
```

---

## 6. 常见问题

### Q: Python 3.14 兼容性问题？
A: 考虑使用 Python 3.11 或 3.12，或者更新 requirements.txt

### Q: Neon 连接失败？
A: 检查 .env 中的 DATABASE_URL 格式是否正确

### Q: API Keys 安全存储？
A: 使用 GitHub Secrets 或服务器环境变量，从不提交到代码库
