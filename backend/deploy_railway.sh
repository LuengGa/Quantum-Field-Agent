#!/bin/bash
# Quantum Field Agent - Railway 一键部署脚本

set -e

echo "🚀 开始部署到 Railway..."
echo "================================"

# 检查是否安装了 railway CLI
if ! command -v railway &> /dev/null; then
    echo "📦 安装 Railway CLI..."
    npm install -g railway
fi

# 检查是否已登录
echo ""
echo "🔐 请登录 Railway（需要浏览器）..."
railway login

# 初始化项目
echo ""
echo "🔗 链接到 Railway 项目..."
echo "请在浏览器中选择或创建新项目"

railway init

# 设置环境变量
echo ""
echo "⚙️ 设置环境变量..."
echo "请输入您的信息（直接回车跳过）"

read -p "Neon DATABASE_URL: " DATABASE_URL
if [ -n "$DATABASE_URL" ]; then
    railway variables set DATABASE_URL="$DATABASE_URL"
fi

read -p "OpenAI API Key (可选): " OPENAI_API_KEY
if [ -n "$OPENAI_API_KEY" ]; then
    railway variables set OPENAI_API_KEY="$OPENAI_API_KEY"
fi

railway variables set LOG_LEVEL="INFO"
railway variables set ENVIRONMENT="production"

# 部署
echo ""
echo "🚀 部署中..."
railway up

# 获取访问 URL
echo ""
echo "✅ 部署完成！"
echo "访问您的应用:"
railway open

echo ""
echo "================================"
echo "📝 提示: 如果部署失败，运行 'railway logs' 查看错误"
