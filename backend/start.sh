#!/bin/bash
# Quantum Field Agent - 一键部署脚本
# ================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Meta Quantum Field Agent - 一键部署             ║${NC}"
echo -e "${BLUE}║                                                          ║${NC}"
echo -e "${BLUE}║                    过程即幻觉，I/O即实相                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"

# 配置
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${PROJECT_DIR}/backend"
DOCKER_IMAGE="quantum-field-agent"
CONTAINER_NAME="quantum-field-backend"
API_PORT=8000

# 检查Docker
check_docker() {
    echo -e "\n${YELLOW}📦 检查Docker环境...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker未安装${NC}"
        echo "   安装方法: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ docker-compose未安装${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Docker环境正常${NC}"
    return 0
}

# 准备环境
prepare() {
    echo -e "\n${YELLOW}📁 准备环境...${NC}"
    
    # 创建目录
    mkdir -p "${BACKEND_DIR}/data"
    mkdir -p "${BACKEND_DIR}/logs"
    mkdir -p "${BACKEND_DIR}/frontend"
    
    # 复制前端
    if [ -f "${BACKEND_DIR}/../frontend/index.html" ]; then
        cp "${BACKEND_DIR}/../frontend/index.html" "${BACKEND_DIR}/frontend/"
        echo -e "${GREEN}✅ 前端已准备${NC}"
    fi
    
    # 生成环境变量
    cat > "${BACKEND_DIR}/.env" << EOF
DATABASE_URL=sqlite:///data/evolution.db
SECRET_KEY=$(openssl rand -base64 32 2>/dev/null || echo "dev-key-$(date +%s)")
LOG_LEVEL=INFO
EOF
    
    echo -e "${GREEN}✅ 环境准备完成${NC}"
}

# 运行测试
run_tests() {
    echo -e "\n${YELLOW}🧪 运行测试...${NC}"
    
    cd "${BACKEND_DIR}"
    python3 -m pytest tests/ -v --tb=short 2>&1 | grep -E "passed|failed|ERROR" | tail -5
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✅ 测试通过${NC}"
    else
        echo -e "${YELLOW}⚠️  测试有问题，但继续部署${NC}"
    fi
}

# 本地启动
start_local() {
    echo -e "\n${YELLOW}🚀 启动本地服务...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 检查8000端口
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  端口8000已被占用，停止现有服务...${NC}"
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    fi
    
    # 直接用Python启动（不依赖Docker）
    echo -e "${YELLOW}🐍 使用Python启动...${NC}"
    
    # 安装依赖
    pip install -q fastapi uvicorn 2>/dev/null || true
    
    # 启动
    cd "${BACKEND_DIR}"
    nohup python3 main.py > logs/app.log 2>&1 &
    PID=$!
    
    sleep 3
    
    if kill -0 $PID 2>/dev/null; then
        echo -e "${GREEN}✅ 服务已启动 (PID: $PID)${NC}"
        echo ""
        echo "========================================"
        echo -e "  🌐 访问地址: ${GREEN}http://localhost:8000${NC}"
        echo -e "  📊 健康检查: ${GREEN}http://localhost:8000/health${NC}"
        echo -e "  📖 API文档: ${GREEN}http://localhost:8000/docs${NC}"
        echo "========================================"
    else
        echo -e "${RED}❌ 启动失败，查看日志: cat ${BACKEND_DIR}/logs/app.log${NC}"
    fi
}

# Docker启动
start_docker() {
    echo -e "\n${YELLOW}🐳 使用Docker启动...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 构建镜像
    echo "构建Docker镜像..."
    docker build -t "${DOCKER_IMAGE}:latest" . || {
        echo -e "${RED}❌ Docker构建失败${NC}"
        return 1
    }
    
    # 运行容器
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p "${API_PORT}:8000" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        -e DATABASE_URL="sqlite:///data/evolution.db" \
        "${DOCKER_IMAGE}:latest"
    
    echo -e "${GREEN}✅ Docker容器已启动${NC}"
    
    sleep 3
    
    echo ""
    echo "========================================"
    echo -e "  🌐 访问地址: ${GREEN}http://localhost:8000${NC}"
    echo -e "  📊 健康检查: ${GREEN}http://localhost:8000/health${NC}"
    echo "========================================"
}

# 停止服务
stop() {
    echo -e "\n${YELLOW}🛑 停止服务...${NC}"
    
    # 停止Python进程
    pkill -f "python3 main.py" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    
    # 停止Docker容器
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    
    echo -e "${GREEN}✅ 服务已停止${NC}"
}

# 查看状态
status() {
    echo -e "\n${YELLOW}📊 服务状态...${NC}"
    
    # 检查Python进程
    if pgrep -f "python3 main.py" > /dev/null; then
        echo -e "${GREEN}✅ Python服务: 运行中${NC}"
    else
        echo -e "${RED}❌ Python服务: 未运行${NC}"
    fi
    
    # 检查Docker容器
    if docker ps --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
        echo -e "${GREEN}✅ Docker容器: 运行中${NC}"
    else
        echo -e "${YELLOW}⚠️  Docker容器: 未运行${NC}"
    fi
    
    # 检查端口
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✅ 端口8000: 监听中${NC}"
    else
        echo -e "${RED}❌ 端口8000: 未监听${NC}"
    fi
    
    # 检查健康
    curl -s http://localhost:8000/health 2>/dev/null && echo -e "\n${GREEN}✅ API健康检查: 通过${NC}" || echo -e "\n${RED}❌ API健康检查: 失败${NC}"
}

# 查看日志
logs() {
    echo -e "\n${YELLOW}📋 查看日志...${NC}"
    tail -50 "${BACKEND_DIR}/logs/app.log" 2>/dev/null || echo "日志文件不存在"
}

# 一键完整部署
deploy() {
    echo -e "\n${BLUE}🚀 开始一键部署...${NC}"
    
    check_docker || return 1
    prepare
    run_tests
    start_local
}

# 生成部署包
package() {
    echo -e "\n${YELLOW}📦 生成部署包...${NC}"
    
    cd "${BACKEND_DIR}/.."
    
    tar -czvf quantum-field-deploy-$(date +%Y%m%d).tar.gz \
        backend/docker-compose.yml \
        backend/Dockerfile \
        backend/.env \
        backend/deploy_tencent.sh \
        backend/DEPLOYMENT_TENCENT_CLOUD.md \
        backend/main.py \
        backend/requirements.txt \
        backend/frontend/ \
        backend/evolution/ 2>/dev/null
    
    echo -e "${GREEN}✅ 部署包已生成: quantum-field-deploy-$(date +%Y%m%d).tar.gz${NC}"
}

# 帮助
help() {
    echo ""
    echo "用法: $0 <命令>"
    echo ""
    echo "命令:"
    echo "  prepare   准备环境"
    echo "  test      运行测试"
    echo "  start     启动服务 (本地Python)"
    echo "  docker    启动服务 (Docker)"
    echo "  deploy    一键部署"
    echo "  stop      停止服务"
    echo "  status    查看状态"
    echo "  logs      查看日志"
    echo "  package   生成部署包"
    echo "  help      显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 deploy     # 一键部署"
    echo "  $0 status     # 查看状态"
    echo "  $0 stop       # 停止服务"
}

# 主函数
case "${1:-help}" in
    prepare)
        prepare
        ;;
    test)
        run_tests
        ;;
    start)
        start_local
        ;;
    docker)
        start_docker
        ;;
    deploy)
        deploy
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    package)
        package
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
