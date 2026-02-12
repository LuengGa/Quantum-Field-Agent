#!/bin/bash
# Neon Deployment Script - Neon 数据库部署脚本
# ==============================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Meta Quantum Field Agent - Neon 部署             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"

# 配置
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${PROJECT_DIR}/backend"
DOCKER_COMPOSE_FILE="${BACKEND_DIR}/docker-compose.neon.yml"
ENV_FILE="${BACKEND_DIR}/.env"

# 检查依赖
check_dependencies() {
    echo -e "\n${YELLOW}📦 检查依赖...${NC}"
    
    command -v docker >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  Docker未安装${NC}"
    command -v psql >/dev/null 2>&1 || echo -e "${YELLOW}⚠️  psql未安装${NC}"
    
    echo -e "${GREEN}✅ 依赖检查完成${NC}"
}

# 配置 Neon
configure_neon() {
    echo -e "\n${YELLOW}🗄️ 配置 Neon 数据库...${NC}"
    
    # 检查连接字符串
    if [ -z "${NEON_CONNECTION_STRING}" ]; then
        echo -e "${YELLOW}⚠️  未设置 NEON_CONNECTION_STRING${NC}"
        echo "请从 Neon 控制台获取连接字符串:"
        echo "  1. 打开 https://neon.tech"
        echo "  2. 选择你的项目"
        echo "  3. Settings → Connection"
        echo "  4. 复制 Connection string"
        echo ""
        read -p "粘贴连接字符串: " NEON_CONNECTION_STRING
    fi
    
    # 验证连接
    echo -e "${YELLOW}🔗 验证 Neon 连接...${NC}"
    if command -v psql &> /dev/null; then
        if psql "${NEON_CONNECTION_STRING}" -c "SELECT 1;" &> /dev/null; then
            echo -e "${GREEN}✅ Neon 连接成功${NC}"
        else
            echo -e "${RED}❌ Neon 连接失败${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️  psql未安装，跳过验证${NC}"
    fi
    
    echo -e "${GREEN}✅ Neon 配置完成${NC}"
}

# 生成环境变量
generate_env() {
    echo -e "\n${YELLOW}⚙️  生成环境变量...${NC}"
    
    cat > "${ENV_FILE}" << EOF
# Neon 数据库配置
DATABASE_TYPE=postgresql
DATABASE_URL=${NEON_CONNECTION_STRING}

# Redis (可选)
REDIS_HOST=localhost
REDIS_PORT=6379

# 安全配置
SECRET_KEY=$(openssl rand -base64 32 2>/dev/null || echo "dev-key-$(date +%s)")
LOG_LEVEL=INFO

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000

# 环境
ENVIRONMENT=production
EOF
    
    echo -e "${GREEN}✅ 环境变量已生成: ${ENV_FILE}${NC}"
}

# 创建 Docker Compose 配置
create_docker_compose() {
    echo -e "\n${YELLOW}🐳 创建 Docker 配置...${NC}"
    
    cat > "${DOCKER_COMPOSE_FILE}" << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.neon
    image: quantum-field-agent:neon
    container_name: quantum-field-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_TYPE=postgresql
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - ENVIRONMENT=production
    volumes:
      - app_data:/app/data
      - app_logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: quantum-field-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  app_data:
    driver: local
  app_logs:
    driver: local
  redis_data:
    driver: local
EOF
    
    echo -e "${GREEN}✅ Docker 配置已创建${NC}"
}

# 创建 Dockerfile
create_dockerfile() {
    echo -e "\n${YELLOW}📦 创建 Dockerfile...${NC}"
    
    cat > "${BACKEND_DIR}/Dockerfile.neon" << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -r requirements.txt \
    psycopg2-binary \
    uvicorn \
    gunicorn

# 复制应用
COPY evolution/ ./evolution/
COPY main.py .
COPY prometheus_metrics.py .

# 创建目录
RUN mkdir -p /app/data /app/logs

EXPOSE 8000

# 使用 gunicorn 启动
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2"]
EOF
    
    echo -e "${GREEN}✅ Dockerfile 已创建${NC}"
}

# 本地测试
test_local() {
    echo -e "\n${YELLOW}🧪 本地测试...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 安装 PostgreSQL 驱动
    pip install -q psycopg2-binary 2>/dev/null || true
    
    # 运行测试
    python3 -m pytest tests/ -v --tb=short 2>&1 | grep -E "passed|failed" | tail -3
    
    echo -e "${GREEN}✅ 测试完成${NC}"
}

# 构建 Docker 镜像
build_docker() {
    echo -e "\n${YELLOW}🔨 构建 Docker 镜像...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 构建镜像
    docker build -f Dockerfile.neon -t quantum-field-agent:neon .
    
    echo -e "${GREEN}✅ Docker 镜像构建完成${NC}"
}

# 启动服务
start() {
    echo -e "\n${YELLOW}🚀 启动服务...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 停止现有服务
    docker-compose -f "${DOCKER_COMPOSE_FILE}" down 2>/dev/null || true
    
    # 启动服务
    docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d
    
    # 等待启动
    sleep 5
    
    # 检查状态
    docker-compose -f "${DOCKER_COMPOSE_FILE}" ps
    
    echo -e "${GREEN}✅ 服务已启动${NC}"
}

# 停止服务
stop() {
    echo -e "\n${YELLOW}🛑 停止服务...${NC}"
    
    cd "${BACKEND_DIR}"
    docker-compose -f "${DOCKER_COMPOSE_FILE}" down
    
    echo -e "${GREEN}✅ 服务已停止${NC}"
}

# 重启服务
restart() {
    echo -e "\n${YELLOW}🔄 重启服务...${NC}"
    stop
    sleep 2
    start
}

# 查看状态
status() {
    echo -e "\n${YELLOW}📊 服务状态...${NC}"
    
    cd "${BACKEND_DIR}"
    docker-compose -f "${DOCKER_COMPOSE_FILE}" ps
    
    echo ""
    echo -e "${YELLOW}健康检查:${NC}"
    curl -s http://localhost:8000/health 2>/dev/null || echo "服务未运行"
}

# 查看日志
logs() {
    echo -e "\n${YELLOW}📋 查看日志...${NC}"
    
    cd "${BACKEND_DIR}"
    docker-compose -f "${DOCKER_COMPOSE_FILE}" logs -f --tail=100
}

# 完整部署
full_deploy() {
    echo -e "\n${BLUE}🚀 开始完整部署...${NC}"
    
    check_dependencies
    configure_neon
    generate_env
    create_docker_compose
    create_dockerfile
    test_local
    build_docker
    start
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    部署完成！                             ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "访问地址:"
    echo -e "  🌐 API:   ${BLUE}http://localhost:8000${NC}"
    echo -e "  📖 Docs:  ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  💚 Health:${BLUE}http://localhost:8000/health${NC}"
}

# 创建服务器部署脚本
create_server_script() {
    echo -e "\n${YELLOW}📦 创建服务器部署脚本...${NC}"
    
    cat > "${BACKEND_DIR}/deploy-to-server.sh" << 'SERVERSCRIPT'
#!/bin/bash
# Deploy to Server - 服务器部署脚本
# =================================

set -e

SERVER_IP="$1"
PROJECT_DIR="/opt/quantum-field-agent"

if [ -z "$SERVER_IP" ]; then
    echo "用法: $0 <服务器IP>"
    exit 1
fi

echo "🚀 开始部署到服务器: $SERVER_IP"

# 1. 上传文件
echo "📤 上传文件..."
scp -r $(dirname "$0")/../* root@${SERVER_IP}:${PROJECT_DIR}/

# 2. SSH 连接并部署
ssh root@${SERVER_IP} << 'DEPLOY'
    set -e
    
    cd ${PROJECT_DIR}
    
    # 安装 Docker
    curl -fsSL https://get.docker.com | sh
    
    # 启动服务
    chmod +x deploy_neon.sh
    ./deploy_neon.sh full_deploy
    
    echo "✅ 部署完成"
DEPLOY

echo "✅ 服务器部署完成"
SERVERSCRIPT
    
    chmod +x "${BACKEND_DIR}/deploy-to-server.sh"
    echo -e "${GREEN}✅ 部署脚本已创建: deploy-to-server.sh${NC}"
}

# 帮助
help() {
    echo ""
    echo "用法: $0 <命令> [参数]"
    echo ""
    echo "命令:"
    echo "  configure     配置 Neon 数据库"
    echo "  generate-env  生成环境变量"
    echo "  docker        创建 Docker 配置"
    echo "  test          本地测试"
    echo "  build         构建 Docker 镜像"
    echo "  start         启动服务"
    echo "  stop          停止服务"
    echo "  restart       重启服务"
    echo "  status        查看状态"
    echo "  logs          查看日志"
    echo "  full          完整部署 (推荐)"
    echo "  server        创建服务器部署脚本"
    echo ""
    echo "环境变量:"
    echo "  NEON_CONNECTION_STRING  Neon 连接字符串"
    echo ""
    echo "示例:"
    echo "  NEON_CONNECTION_STRING='postgresql://...' $0 full"
    echo "  $0 start"
}

# 主函数
case "${1:-help}" in
    configure)
        configure_neon
        ;;
    generate-env)
        generate_env
        ;;
    docker)
        create_docker_compose
        create_dockerfile
        ;;
    test)
        test_local
        ;;
    build)
        build_docker
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    full|full_deploy)
        full_deploy
        ;;
    server)
        create_server_script
        ;;
    help|--help|-h)
        help
        ;;
    *)
        echo -e "${RED}未知命令: $1${NC}"
        help
        exit 1
        ;;
esac
