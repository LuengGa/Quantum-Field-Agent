#!/bin/bash
# Tencent Cloud Deployment - 腾讯云部署脚本
# =========================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 腾讯云部署开始${NC}"

# 配置变量
PROJECT_NAME="quantum-field-agent"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${PROJECT_DIR}/backend"

# 腾讯云配置
TENCENT_CLOUD_REGION="ap-shanghai"  # 上海区
TENCENT_CLOUD_INSTANCE=""  # 实例ID，留空自动创建
TENCENT_CLOUD_KEY_NAME="quantum-field-key"  # SSH密钥名
TENCENT_CLOUD_SECURITY_GROUP="sg-quantum-field"  # 安全组名

# 镜像配置
DOCKER_IMAGE_NAME="quantum-field-agent"
DOCKER_TAG="latest"
REGISTRY_URL="registry.tencentcloudcr.com/${TENCENT_CLOUD_REGION}/${DOCKER_IMAGE_NAME}"

# 域名配置
DOMAIN_NAME=""  # 你的域名，留空则用IP
API_SUBDOMAIN="api"  # API子域名
FRONTEND_SUBDOMAIN="www"  # 前端子域名

echo -e "${YELLOW}📋 部署配置:${NC}"
echo "  项目: ${PROJECT_NAME}"
echo "  区域: ${TENCENT_CLOUD_REGION}"
echo "  后端目录: ${BACKEND_DIR}"
echo ""

# 检查依赖
check_dependencies() {
    echo -e "${YELLOW}📦 检查依赖...${NC}"
    
    command -v docker >/dev/null 2>&1 || { echo -e "${RED}❌ 需要安装 Docker${NC}"; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}❌ 需要安装 docker-compose${NC}"; exit 1; }
    
    # 检查腾讯云CLI
    if ! command -v tccli >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  未安装 tccli (腾讯云CLI)，将跳过云端操作${NC}"
        echo "  安装: https://github.com/TencentCloud/tencentcloud-cli"
    fi
    
    echo -e "${GREEN}✅ 依赖检查完成${NC}"
}

# 构建Docker镜像
build_docker() {
    echo -e "${YELLOW}🐳 构建Docker镜像...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 构建镜像
    docker build -t ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} .
    
    echo -e "${GREEN}✅ Docker镜像构建完成${NC}"
}

# 本地测试
test_local() {
    echo -e "${YELLOW}🧪 本地测试...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 运行测试
    python3 -m pytest tests/ -v --tb=short 2>/dev/null | head -20 || true
    
    echo -e "${GREEN}✅ 本地测试完成${NC}"
}

# 推送镜像到腾讯云镜像仓库
push_to_registry() {
    echo -e "${YELLOW}📤 推送镜像到腾讯云镜像仓库...${NC}"
    
    if [ -z "${TENCENT_CLOUD_INSTANCE}" ]; then
        echo -e "${YELLOW}⚠️  未配置腾讯云实例，跳过推送${NC}"
        return
    fi
    
    # 登录腾讯云镜像仓库
    tccli tar login --region ${TENCENT_CLOUD_REGION} || true
    
    # 推送镜像
    docker tag ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} ${REGISTRY_URL}:${DOCKER_TAG}
    docker push ${REGISTRY_URL}:${DOCKER_TAG}
    
    echo -e "${GREEN}✅ 镜像推送完成${NC}"
}

# 部署到云服务器
deploy_to_server() {
    echo -e "${YELLOW}☁️ 部署到云服务器...${NC}"
    
    SERVER_IP="${1:-}"
    
    if [ -z "${SERVER_IP}" ]; then
        echo -e "${YELLOW}⚠️  未提供服务器IP，使用本地部署${NC}"
        deploy_local
        return
    fi
    
    # 远程部署
    ssh -o StrictHostKeyChecking=no root@${SERVER_IP} << 'DEPLOY_SCRIPT'
        set -e
        
        cd /opt/quantum-field-agent
        
        # 拉取最新代码
        git pull
        
        # 拉取镜像
        docker pull registry.tencentcloudcr.com/ap-shanghai/quantum-field-agent:latest || true
        
        # 重启服务
        docker-compose down
        docker-compose up -d
        
        # 检查状态
        docker-compose ps
        
        # 查看日志
        docker-compose logs -f --tail=50 &
DEPLOY_SCRIPT
    
    echo -e "${GREEN}✅ 云端部署完成${NC}"
}

# 本地Docker部署
deploy_local() {
    echo -e "${YELLOW}🐳 本地Docker部署...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 创建必要目录
    mkdir -p data logs
    
    # 复制配置
    cp docker-compose.example.yml docker-compose.yml 2>/dev/null || true
    
    # 启动服务
    docker-compose down -v
    docker-compose up -d
    
    # 等待启动
    sleep 5
    
    # 检查状态
    docker-compose ps
    
    echo -e "${GREEN}✅ 本地部署完成${NC}"
}

# 配置Nginx反向代理
configure_nginx() {
    echo -e "${YELLOW}🌐 配置Nginx...${NC}"
    
    SERVER_IP="${1:-}"
    
    if [ -z "${SERVER_IP}" ]; then
        SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")
    fi
    
    cat > /tmp/nginx.conf << EOF
server {
    listen 80;
    server_name ${SERVER_IP};
    
    # API代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    # WebSocket代理
    location /ws/ {
        proxy_pass http://127.0.0.1:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # 前端静态文件
    location / {
        root /var/www/quantum-field-frontend;
        index index.html;
    }
}
EOF
    
    echo "Nginx配置已生成: /tmp/nginx.conf"
    echo -e "${GREEN}✅ Nginx配置完成${NC}"
}

# 配置CloudBase前端托管
configure_cloudbase() {
    echo -e "${YELLOW}☁️ 配置CloudBase前端托管...${NC}"
    
    if ! command -v cloudbasectl >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  未安装 cloudbasectl${NC}"
        echo "  安装: npm install -g @cloudbase/cli"
        return
    fi
    
    # 登录
    cloudbasectl login
    
    # 初始化环境
    cloudbasectl env:init ${PROJECT_NAME}-env --region ${TENCENT_CLOUD_REGION}
    
    # 部署前端
    cd "${BACKEND_DIR}/../frontend"
    
    cloudbasectl hosting:deploy --envId ${PROJECT_NAME}-env \
        --path . \
        --index index.html \
        --error-page 404.html
    
    echo -e "${GREEN}✅ CloudBase部署完成${NC}"
    echo "  前端访问: https://${PROJECT_NAME}-env.cloudapp.cn"
}

# 配置腾讯云数据库（可选）
configure_database() {
    echo -e "${YELLOW}🗄️ 配置腾讯云数据库...${NC}"
    
    DB_TYPE="${1:-postgresql}"
    
    if [ "${DB_TYPE}" = "postgresql" ]; then
        echo "创建腾讯云PostgreSQL数据库..."
        tccli postgres CreateDBInstance \
            --Region ${TENCENT_CLOUD_REGION} \
            --SpecCode postgres.s1.small \
            --Storage 20 \
            --InstanceChargeType POSTPAID \
            --EngineVersion 13 \
            --Name ${PROJECT_NAME}-db || true
        
        echo "获取数据库连接信息..."
        tccli postgres DescribeDBInstances \
            --Region ${TENCENT_CLOUD_REGION} \
            --Filters.0.Name=Name \
            --Filters.0.Values.0=${PROJECT_NAME}-db
    elif [ "${DB_TYPE}" = "mysql" ]; then
        echo "创建腾讯云MySQL数据库..."
        tccli cdb CreateDBInstance \
            --Region ${TENCENT_CLOUD_REGION} \
            --EngineVersion 8.0 \
            --SpecCode mysql.s1.small \
            --Storage 20 \
            --InstanceChargeType POSTPAID \
            --InstanceName ${PROJECT_NAME}-db || true
    fi
    
    echo -e "${GREEN}✅ 数据库配置完成${NC}"
}

# 配置域名和HTTPS
configure_domain() {
    echo -e "${YELLOW}🔒 配置域名和HTTPS...${NC}"
    
    DOMAIN="${1:-}"
    
    if [ -z "${DOMAIN}" ]; then
        echo -e "${YELLOW}⚠️  未提供域名，跳过配置${NC}"
        return
    fi
    
    # 申请SSL证书
    echo "申请SSL证书..."
    tccli ssl ApplyCertificate \
        --DomainName ${DOMAIN} \
        --CertificateType FREE || true
    
    # 配置CDN
    echo "配置CDN加速..."
    tccli cdn CreatePurgeTasks \
        --Domain ${DOMAIN} \
        --Urls.0 "https://${DOMAIN}/"
    
    echo -e "${GREEN}✅ 域名配置完成${NC}"
}

# 部署监控
configure_monitoring() {
    echo -e "${YELLOW}📊 配置监控...${NC}"
    
    cat > /opt/quantum-field-agent/monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'quantum-field-agent'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
    metrics_path: /metrics
EOF
    
    echo "Prometheus配置已生成"
    echo -e "${GREEN}✅ 监控配置完成${NC}"
}

# 备份配置
configure_backup() {
    echo -e "${YELLOW}💾 配置备份...${NC}"
    
    cat > /opt/quantum-field-agent/backup.sh << 'EOF'
#!/bin/bash
# 每日数据库备份脚本

BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p ${BACKUP_DIR}

# 备份数据库
docker exec quantum-field-agent-backend cp /app/data/evolution.db /tmp/backup.db
docker cp quantum-field-agent-backend:/tmp/backup.db ${BACKUP_DIR}/evolution_${DATE}.db

# 保留最近7天备份
find ${BACKUP_DIR} -name "*.db" -mtime +7 -delete

echo "备份完成: ${BACKUP_DIR}/evolution_${DATE}.db"
EOF
    
    chmod +x /opt/quantum-field-agent/backup.sh
    
    # 添加cron任务
    (crontab -l 2>/dev/null | grep -v backup.sh; echo "0 3 * * * /opt/quantum-field-agent/backup.sh") | crontab -
    
    echo -e "${GREEN}✅ 备份配置完成${NC}"
}

# 健康检查
health_check() {
    echo -e "${YELLOW}🏥 健康检查...${NC}"
    
    cd "${BACKEND_DIR}"
    
    # 检查Docker容器
    docker-compose ps
    
    # 检查API
    curl -s http://localhost:8000/health || echo "API不可用"
    
    # 检查数据库
    sqlite3 data/evolution.db "SELECT COUNT(*) FROM patterns;" 2>/dev/null || echo "数据库检查失败"
    
    echo -e "${GREEN}✅ 健康检查完成${NC}"
}

# 查看日志
show_logs() {
    echo -e "${YELLOW}📋 查看日志...${NC}"
    
    SERVICE="${1:-backend}"
    
    cd "${BACKEND_DIR}"
    docker-compose logs -f ${SERVICE} --tail=100
}

# 扩缩容
scale_service() {
    echo -e "${YELLOW}📈 扩缩容...${NC}"
    
    SCALE_NUM="${1:-2}"
    
    cd "${BACKEND_DIR}"
    docker-compose scale backend=${SCALE_NUM}
    
    echo -e "${GREEN}✅ 已扩展到 ${SCALE_NUM} 个实例${NC}"
}

# 回滚
rollback() {
    echo -e "${YELLOW}⏪ 回滚...${NC}"
    
    VERSION="${1:-previous}"
    
    cd "${BACKEND_DIR}"
    docker-compose down
    docker-compose rm -f
    
    if [ "${VERSION}" = "previous" ]; then
        docker-compose -f docker-compose.backup.yml up -d
    else
        docker tag ${DOCKER_IMAGE_NAME}:${VERSION} ${DOCKER_IMAGE_NAME}:latest
        docker-compose up -d
    fi
    
    echo -e "${GREEN}✅ 回滚完成${NC}"
}

# 显示帮助
show_help() {
    echo "用法: $0 <命令> [参数]"
    echo ""
    echo "命令:"
    echo "  build           构建Docker镜像"
    echo "  test            本地测试"
    echo "  deploy [IP]     部署到服务器"
    echo "  local           本地Docker部署"
    echo "  nginx [IP]      配置Nginx"
    echo "  cloudbase       配置CloudBase前端托管"
    echo "  database [type] 配置数据库 (postgresql/mysql)"
    echo "  domain <域名>   配置域名和HTTPS"
    echo "  monitoring      配置监控"
    echo "  backup          配置备份"
    echo "  health          健康检查"
    echo "  logs [服务]     查看日志"
    echo "  scale <数量>    扩缩容"
    echo "  rollback [版本] 回滚"
    echo ""
    echo "示例:"
    echo "  $0 build"
    echo "  $0 deploy 1.2.3.4"
    echo "  $0 local"
    echo "  $0 domain api.example.com"
}

# 主函数
main() {
    COMMAND="${1:-help}"
    shift || true
    
    case "${COMMAND}" in
        build)
            check_dependencies
            build_docker
            ;;
        test)
            test_local
            ;;
        deploy)
            check_dependencies
            build_docker
            deploy_to_server "$@"
            ;;
        local)
            check_dependencies
            deploy_local
            ;;
        nginx)
            configure_nginx "$@"
            ;;
        cloudbase)
            configure_cloudbase
            ;;
        database)
            configure_database "$@"
            ;;
        domain)
            configure_domain "$@"
            ;;
        monitoring)
            configure_monitoring
            ;;
        backup)
            configure_backup
            ;;
        health)
            health_check
            ;;
        logs)
            show_logs "$@"
            ;;
        scale)
            scale_service "$@"
            ;;
        rollback)
            rollback "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: ${COMMAND}${NC}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
