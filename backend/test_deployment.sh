#!/bin/bash
# Local Deployment Test - 本地部署测试
# ===============================

set -e

echo "============================================"
echo "  Meta Quantum Field Agent - 本地部署测试"
echo "============================================"

# 检查Docker
echo ""
echo "📦 检查Docker环境..."
docker --version || { echo "❌ Docker未安装"; exit 1; }
docker-compose --version || { echo "❌ docker-compose未安装"; exit 1; }
echo "✅ Docker环境正常"

# 创建必要目录
echo ""
echo "📁 创建目录结构..."
mkdir -p data logs frontend nginx ssl monitoring/prometheus/rules

# 构建前端
echo ""
echo "🌐 构建前端..."
if [ -d "../frontend" ]; then
    cp ../frontend/index.html frontend/
    echo "✅ 前端已复制"
else
    echo "⚠️  前端目录不存在，跳过"
fi

# 创建默认配置
echo ""
echo "⚙️  生成配置..."

# .env文件
cat > .env << EOF
DATABASE_TYPE=sqlite
SECRET_KEY=$(openssl rand -base64 32 2>/dev/null || echo "dev-secret-key-change-in-production")
LOG_LEVEL=INFO
TAG=latest
EOF

# 简化的docker-compose
cat > docker-compose.local.yml << 'EOF'
version: '3.8'

services:
  backend:
    build: .
    container_name: quantum-field-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///data/evolution.db
      - SECRET_KEY=dev-secret-key
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  default:
    name: quantum-field-network
EOF

echo "✅ 配置已生成"

# 简化Dockerfile
cat > Dockerfile.local << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# 复制应用
COPY evolution/ ./evolution/
COPY main.py .

# 创建数据目录
RUN mkdir -p /app/data /app/logs

EXPOSE 8000

CMD ["python", "main.py"]
EOF

echo "✅ Dockerfile已生成"

# 运行本地测试
echo ""
echo "🧪 运行本地测试..."
cd ..
python3 -m pytest backend/tests/ -v --tb=short 2>&1 | tail -5
cd backend

echo ""
echo "✅ 本地测试完成"

# 准备部署包
echo ""
echo "📦 准备部署包..."
cd ..
tar -czvf quantum-field-deploy.tar.gz \
    backend/docker-compose.local.yml \
    backend/Dockerfile.local \
    backend/.env \
    backend/deploy_tencent.sh \
    backend/DEPLOYMENT_TENCENT_CLOUD.md \
    backend/nginx/nginx.conf \
    backend/monitoring/ \
    backend/evolution/ \
    backend/main.py \
    backend/requirements.txt \
    frontend/ 2>/dev/null || \
tar -czvf quantum-field-deploy.tar.gz \
    backend/docker-compose.local.yml \
    backend/Dockerfile.local \
    backend/.env \
    backend/deploy_tencent.sh \
    backend/DEPLOYMENT_TENCENT_CLOUD.md \
    backend/nginx/ \
    backend/monitoring/ \
    backend/evolution/ \
    backend/main.py \
    backend/requirements.txt \
    backend/tests/ 2>/dev/null
cd backend

echo "✅ 部署包已生成: quantum-field-deploy.tar.gz"

echo ""
echo "============================================"
echo "  本地测试完成！"
echo "============================================"
echo ""
echo "📋 下一步："
echo "   1. 购买腾讯云服务器"
echo "   2. 上传部署包: scp quantum-field-deploy.tar.gz root@IP:/opt/"
echo "   3. 解压并部署: ./deploy_tencent.sh local"
echo ""
echo "📄 部署文档: DEPLOYMENT_TENCENT_CLOUD.md"
