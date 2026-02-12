"""
分布式量子场 API Gateway (V1.5)
集成：场状态管理、分布式计算、健康检查
"""

import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from distributed_field import (
    DistributedQuantumField,
    ComputeFieldWorker,
    FieldState,
    field_manager,
)

load_dotenv()

# 启动时初始化Worker（可选，也可单独部署）
worker_task = None
worker_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_task, worker_instance
    # 启动时：启动后台Worker（如果是混合部署）
    if os.getenv("ENABLE_WORKER", "true").lower() == "true":
        worker_instance = ComputeFieldWorker(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        worker_task = asyncio.create_task(worker_instance.run())
        print("[系统] 计算场Worker已启动")

    yield

    # 关闭时：清理
    if worker_task and worker_instance:
        worker_instance.running = False
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Quantum Field Agent V1.5 (Distributed)",
    description="分布式量子场架构 - 支持场状态管理、自动负载均衡、流式响应",
    version="1.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== 数据模型 ==============


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的自然语言意图")
    user_id: str = Field(default="user_default", description="用户标识")
    session_id: str = Field(default="session_default", description="会话标识")


class FieldStatusResponse(BaseModel):
    user_id: str
    entropy: float
    activated_skills: list
    last_update: float
    in_local_cache: bool


# ============== API端点 ==============


@app.get("/")
async def root():
    """API根路径 - 返回基本信息"""
    return {
        "name": "Quantum Field Agent V1.5",
        "version": "1.5.0",
        "status": "running",
        "features": ["distributed_field", "entropy_based_routing", "stream_response"],
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    分布式场坍缩接口

    自动判断：本地处理 or 分布式 offload
    - 场熵 < 0.8：本地快速处理（gpt-4o-mini）
    - 场熵 > 0.8：分发到计算集群（gpt-4o）

    返回SSE流式响应
    """
    user_id = request.user_id

    async def generate():
        import json

        # 发送开始标记
        start_data = json.dumps({"type": "start", "user_id": user_id})
        yield f"data: {start_data}\n\n"

        async for token in field_manager.process_intent(
            user_id, request.message, request.session_id
        ):
            # SSE格式
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"

        # 发送结束标记
        end_data = json.dumps({"type": "end", "user_id": user_id})
        yield f"data: {end_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/field/status/{user_id}", response_model=FieldStatusResponse)
async def get_field_status(user_id: str):
    """
    查询用户场状态（调试接口）

    返回：
    - entropy: 场熵（0-1，越高越复杂）
    - activated_skills: 最近激活的技能
    - last_update: 最后更新时间
    - in_local_cache: 是否在本地缓存
    """
    state = await field_manager.locate_field(user_id)
    if not state:
        raise HTTPException(status_code=404, detail="场未找到")

    return {
        "user_id": state.user_id,
        "entropy": state.entropy,
        "activated_skills": state.activated_skills[-5:],  # 最近5个
        "last_update": state.last_update,
        "in_local_cache": user_id in field_manager.local_cache,
    }


@app.post("/field/reset/{user_id}")
async def reset_field(user_id: str):
    """
    重置用户场（回到基态）

    清除所有记忆和状态，回到初始基态
    """
    async with field_manager._get_lock(user_id):
        base_state = field_manager._create_base_field(user_id)
        await field_manager.save_field(base_state)
        if user_id in field_manager.local_cache:
            del field_manager.local_cache[user_id]

    return {
        "status": "reset",
        "user_id": user_id,
        "message": "场已重置为基态",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """
    健康检查

    检查Redis连接状态和系统健康
    """
    try:
        # 检查Redis连接
        await field_manager.redis.ping()
        redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)}"

    return {
        "status": "healthy" if redis_status == "connected" else "degraded",
        "version": "1.5.0-distributed",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "redis": redis_status,
            "field_manager": "active",
            "worker": "active" if worker_task else "disabled",
        },
    }


@app.get("/stats")
async def get_stats():
    """
    系统统计信息
    """
    try:
        # 获取Redis信息
        info = await field_manager.redis.info()

        # 统计场数量
        field_keys = await field_manager.redis.keys("qf:field:*")

        return {
            "fields_active": len(field_keys),
            "local_cache_size": len(field_manager.local_cache),
            "redis_used_memory": info.get("used_memory_human", "N/A"),
            "redis_connected_clients": info.get("connected_clients", 0),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"统计失败: {str(e)}")


# ============== 前端服务 ==============

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/frontend", response_class=HTMLResponse)
async def serve_frontend():
    """提供前端页面"""
    try:
        with open(FRONTEND_DIR / "index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1>Quantum Field Agent V1.5</h1>
                <p>前端文件未找到</p>
                <p>API端点: <code>/chat</code></p>
            </body>
            </html>
            """,
            status_code=404,
        )


# ============== 启动 ==============

if __name__ == "__main__":
    import uvicorn
    import json

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     Quantum Field Agent V1.5 (Distributed)              ║
    ║                                                          ║
    ║     分布式量子场架构                                      ║
    ║     - 场状态管理                                          ║
    ║     - 自动负载均衡                                        ║
    ║     - 流式响应                                            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    🚀 启动中...
    📡 监听地址: http://{host}:{port}
    🔌 Redis: {os.getenv("REDIS_URL", "redis://localhost:6379")}
    🤖 LLM模型: {os.getenv("MODEL_NAME", "gpt-4o-mini")}
    ⚙️  Worker: {"启用" if os.getenv("ENABLE_WORKER", "true").lower() == "true" else "禁用"}
    
    📚 API文档: http://{host}:{port}/docs
    🏠 前端页面: http://{host}:{port}/frontend
    
    """)

    uvicorn.run(app, host=host, port=port)
