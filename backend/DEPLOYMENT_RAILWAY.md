# Railway 部署指南

## 快速部署（5分钟）

### 步骤 1: 注册 Railway

1. 访问 https://railway.app
2. 点击 **"Sign Up"**
3. 选择 **"Continue with GitHub"**
4. 授权 GitHub 账号

### 步骤 2: 创建项目

1. 点击 **"New Project"**
2. 选择 **"Deploy from GitHub repo"**
3. 搜索并选择 `LuengGa/Quantum-Field-Agent`
4. 点击 **"Deploy Now"**

### 步骤 3: 添加环境变量

项目创建后，在 **"Variables"** 标签添加：

```env
# Neon 数据库（必需）
DATABASE_URL=postgresql://neondb_owner:npg_3WIRYidN6HvZ@ep-old-bar-a1jkdluf-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

# OpenAI（可选，用于 AI 功能）
OPENAI_API_KEY=your_openai_api_key

# 日志级别
LOG_LEVEL=INFO

# 环境
ENVIRONMENT=production
```

**注意**: DATABASE_URL 已包含在上面，直接复制使用。

### 步骤 4: 部署完成

1. Railway 会自动构建和部署
2. 等待 **"Deployed"** 状态（绿色勾）
3. 点击生成的 **URL** 访问服务

---

## 手动部署（如果自动失败）

### 使用 Railway CLI

```bash
# 安装 Railway CLI
npm install -g railway

# 登录
railway login

# 连接项目
railway init
railway link

# 设置环境变量
railway variables set DATABASE_URL="postgresql://..."

# 部署
railway up
```

---

## 本地测试

```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000

---

## 故障排除

### 问题: 构建失败
```bash
# 检查 requirements.txt
# 确保所有包兼容 Python 3.11
```

### 问题: 无法连接 Neon
```bash
# 验证 DATABASE_URL 格式正确
# 确保 Neon 数据库没有被暂停
```

### 问题: 内存不足
```bash
# Railway 免费版限制 512MB RAM
# 考虑简化依赖或使用 Oracle Cloud
```

---

## 访问地址

部署成功后，访问格式：
```
https://your-project-name.up.railway.app
```

API 文档：
```
https://your-project-name.up.railway.app/docs
```

---

## 费用

- 免费额度: $5/月
- 超出后: 按使用付费
- 监控: 在 Railway Dashboard 查看使用情况
