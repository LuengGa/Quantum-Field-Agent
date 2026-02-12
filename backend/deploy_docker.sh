#!/bin/bash
# Quantum Field Agent - 本地 Docker 部署脚本

set -e

echo "🚀 开始本地 Docker 部署..."
echo "================================"

cd "$(dirname \"$0\")\"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

echo "✅ Docker 已安装"

# 构建镜像
echo ""
echo "🔨 构建 Docker 镜像..."
docker build -t quantum-agent:latest ./backend

# 运行容器
echo ""
echo "🚀 启动容器..."
docker run -d \
    --name quantum-agent \
    -p 8000:8000 \
    --env-file ./backend/.env.docker \
    quantum-agent:latest

echo ""
echo "✅ 部署完成！"
echo ""
echo "📝 访问地址:"
echo "   - API: http://localhost:8000"
echo "   - 文档: http://localhost:8000/docs"
echo ""
echo "📝 管理命令:"
echo "   - 查看日志: docker logs -f quantum-agent"
echo "   - 停止: docker stop quantum-agent"
echo "   - 删除: docker rm quantum-agent"
