"""
Quantum Field Agent - API入口 V4.0 Complete
彻底融合，无需版本切换！
"""

import os
import sqlite3
import json
import base64
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from quantum_field import QuantumField, EntanglementStrength, TemporalIntent

try:
    from evolution_router import router as evolution_router

    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    print("[Warning] 进化层模块不可用")


# ==================== Pydantic Models ====================


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户意图")
    user_id: str = Field(default="user_default")
    session_id: Optional[str] = Field(default="session_default")
    domain_focus: Optional[str] = None


class EntangleRequest(BaseModel):
    agent_a: str
    agent_b: str
    strength: str = "MEDIUM"


class RegisterAgentRequest(BaseModel):
    agent_id: str
    capabilities: list[str]


class ParallelExcitationRequest(BaseModel):
    task: str
    agent_ids: list[str]


class ConsensusCollapseRequest(BaseModel):
    proposal: str
    agent_ids: list[str]


class ScheduleOneShotRequest(BaseModel):
    user_id: str
    content: str
    scheduled_time: datetime
    callback_url: Optional[str] = None


class ScheduleCronRequest(BaseModel):
    user_id: str
    content: str
    cron_expr: str
    callback_url: Optional[str] = None


class ScheduleIntervalRequest(BaseModel):
    user_id: str
    content: str
    interval_seconds: int
    callback_url: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


class ImageGenRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"


class EventTriggerRequest(BaseModel):
    event_type: str
    data: dict


# ==================== FastAPI App ====================

app = FastAPI(
    title="Quantum Field Agent V4.0 Complete",
    description="彻底融合 - V1.0基础 + V1.5锁+TTL + V2.0审计 + V2.5纠缠网络 + V3.0多模态 + V4.0时序",
    version="4.0.0-complete",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用绝对路径，兼容 Docker 环境
SCRIPT_DIR = Path(__file__).resolve().parent
# Docker 环境中 PROJECT_ROOT 就是 SCRIPT_DIR，因为代码直接在 /app 下
if (SCRIPT_DIR / "frontend").exists():
    FRONTEND_DIR = SCRIPT_DIR / "frontend"
elif (SCRIPT_DIR.parent / "frontend").exists():
    FRONTEND_DIR = SCRIPT_DIR.parent / "frontend"
else:
    FRONTEND_DIR = SCRIPT_DIR / "frontend"  # 默认使用同级 frontend 目录

qf = QuantumField()

if EVOLUTION_AVAILABLE:
    app.include_router(evolution_router, prefix="/api")
    print("✓ 进化层路由已加载")


# ==================== 核心对话接口 ====================


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    V5.0 量子对话接口（默认）

    真正的波粒二象性实现：
    - 叠加态生成（波）
    - 干涉与退相干
    - 概率性坍缩（粒子）
    - AI协作视角
    """
    if not V5_AVAILABLE:
        raise HTTPException(status_code=503, detail="V5.0 模块不可用")

    async def generate():
        async for event in qf_v5.process_intent_v5(
            request.user_id, request.message, request.session_id
        ):
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/chat-legacy")
async def chat_legacy(request: ChatRequest):
    """
    V4.0 传统对话接口（向后兼容）

    ⚠️ 注意：此接口使用旧的启发式实现，不是真正的量子力学
    推荐使用 /chat (V5.0) 获得真正的量子场体验
    """

    async def generate():
        async for token in qf.process_intent(
            request.user_id, request.message, request.session_id, request.domain_focus
        ):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


# ==================== V1.0 场状态接口 ====================


@app.get("/field/{user_id}")
async def get_field(user_id: str):
    """获取用户场状态"""
    return await qf.get_field_status(user_id)


@app.post("/field/{user_id}/reset")
async def reset_field(user_id: str):
    """重置用户场状态"""
    return await qf.reset_field(user_id)


@app.get("/field/{user_id}/lock/status")
async def lock_status(user_id: str, resource: str = "default"):
    """检查用户锁状态"""
    return {
        "user_id": user_id,
        "resource": resource,
        "locked": qf.user_lock_manager.is_locked(user_id, resource),
    }


# ==================== V1.0 记忆管理 ====================


@app.get("/memory/{user_id}")
async def get_memory(user_id: str, limit: int = 50):
    """获取用户记忆"""
    return qf._get_memory(user_id, limit=limit)


@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    """清空用户记忆"""
    conn = sqlite3.connect(qf.db_path)
    conn.execute("DELETE FROM memory WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"用户 {user_id} 的记忆已清空"}


# ==================== V1.0 技能管理 ====================


@app.get("/skills")
async def list_skills(domain: Optional[str] = None):
    """列出所有技能"""
    skills = qf.get_skills()
    if domain:
        skills = [s for s in skills if s.get("domain") == domain]
    return {"skills": skills, "count": len(skills)}


@app.post("/skills/focus")
async def focus_domain(domain: str):
    """聚焦特定领域"""
    skills = qf.get_skills()
    return {
        "domain": domain,
        "active_skills": [s["name"] for s in skills if s.get("domain") == domain],
        "message": f"已切换至{domain}高密度场",
    }


# ==================== V2.0 审计接口 ====================


@app.get("/audit/trail/{user_id}")
async def get_trail(user_id: str, limit: int = 50):
    """获取用户审计轨迹"""
    trail = await qf.get_audit_trail(user_id, limit)
    return {"user_id": user_id, "events": trail, "count": len(trail)}


@app.post("/audit/verify")
async def verify_audit():
    """验证审计链完整性"""
    return await qf.verify_audit()


# ==================== V2.5 纠缠网络接口 ====================


@app.get("/entanglement/status")
async def entanglement_status():
    """纠缠网络状态"""
    return {
        "available": qf.entanglement_available,
        "redis_available": qf.entanglement_network.redis_available
        if qf.entanglement_network
        else False,
    }


@app.get("/entanglement/topology")
async def get_topology():
    """获取纠缠网络拓扑"""
    if qf.entanglement_available and qf.entanglement_network:
        return qf.entanglement_network.get_network_topology()
    return {"status": "disabled", "message": "纠缠网络不可用"}


@app.post("/entanglement/register")
async def register_agent(request: RegisterAgentRequest):
    """注册Agent到纠缠网络"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    agent = await qf.entanglement_network.register_agent(
        request.agent_id, request.capabilities
    )
    return {"status": "success", "agent": agent.agent_id}


@app.post("/entanglement/entangle")
async def entangle_agents(request: EntangleRequest):
    """建立Agent间纠缠"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    try:
        strength_enum = EntanglementStrength[request.strength]
        result = await qf.entanglement_network.entangle(
            request.agent_a, request.agent_b, strength_enum
        )
        return {
            "status": "success",
            "link": {
                "agent_a": result.agent_a,
                "agent_b": result.agent_b,
                "strength": result.strength,
            },
        }
    except KeyError:
        return {"status": "error", "message": f"无效的纠缠强度: {request.strength}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/entanglement/disentangle")
async def disentangle_agents(agent_a: str, agent_b: str):
    """解除纠缠"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    await qf.entanglement_network.disentangle(agent_a, agent_b)
    return {"status": "success", "message": f"{agent_a} <-> {agent_b} 纠缠已解除"}


@app.get("/entanglement/discover")
async def discover_agents(
    capability: Optional[str] = None, exclude: Optional[str] = None
):
    """发现可用Agent"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    return await qf.entanglement_network.discover_agents(capability, exclude)


@app.post("/entanglement/parallel-excite")
async def parallel_excitation(request: ParallelExcitationRequest):
    """并行激发多个Agent"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    results = await qf.entanglement_network.parallel_excite_agents(
        request.task, request.agent_ids
    )
    return {"results": results, "count": len(results)}


@app.post("/entanglement/fuse")
async def fuse_results(agent_results: list[dict]):
    """融合多个Agent的结果"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    fused = await qf.entanglement_network.interference_fusion.fuse_results(
        agent_results
    )
    return fused


@app.post("/entanglement/consensus")
async def consensus_collapse(request: ConsensusCollapseRequest):
    """执行共识坍缩"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    result = await qf.entanglement_network.collaborative_collapse(
        request.proposal, request.agent_ids
    )
    return result


@app.get("/entanglement/shared-memory")
async def get_shared_memory(key: str):
    """读取共享内存"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    value = await qf.entanglement_network.shared_memory.read(key)
    return {"key": key, "value": value}


@app.post("/entanglement/shared-memory")
async def set_shared_memory(key: str, value: dict, ttl: Optional[int] = None):
    """写入共享内存"""
    if not qf.entanglement_available:
        return {"status": "disabled", "message": "纠缠网络不可用"}

    await qf.entanglement_network.shared_memory.write(key, value, ttl)
    return {"status": "success", "key": key}


# ==================== V3.0 多模态接口 ====================


@app.get("/multimodal/status")
async def multimodal_status():
    """多模态系统状态"""
    return {
        "available": qf.multimodal_available,
        "modalities": ["text"]
        + (["image", "audio"] if qf.multimodal_available else []),
    }


@app.post("/multimodal/encode/text")
async def encode_text(text: str):
    """编码文本为向量"""
    if not qf.multimodal_available:
        return {"status": "disabled", "message": "多模态不可用"}

    vector = await qf.multimodal_encoder.encode_text(text)
    return {"modality": "text", "dimension": len(vector)}


@app.post("/multimodal/encode/image")
async def encode_image(file: UploadFile = File(...)):
    """编码图像为向量 (CLIP风格)"""
    if not qf.multimodal_available:
        return {"status": "disabled", "message": "多模态不可用"}

    image_data = await file.read()
    vector = await qf.multimodal_encoder.encode_image_clip(image_data)
    return {"modality": "image", "dimension": len(vector)}


@app.post("/multimodal/encode/image/vision")
async def encode_image_vision(file: UploadFile = File(...)):
    """视觉编码 (GPT-4V风格)"""
    if not qf.multimodal_available:
        return {"status": "disabled", "message": "多模态不可用"}

    image_data = await file.read()
    result = await qf.multimodal_encoder.encode_image_vision(image_data)
    return result


@app.post("/multimodal/encode/audio")
async def encode_audio(file: UploadFile = File(...)):
    """编码音频为向量 (Whisper风格)"""
    if not qf.multimodal_available:
        return {"status": "disabled", "message": "多模态不可用"}

    audio_data = await file.read()
    result = await qf.multimodal_encoder.encode_audio_whisper(audio_data)
    return result


@app.post("/multimodal/detect")
async def detect_modality(data: str = Query(..., description="数据或文件路径")):
    """自动检测模态类型"""
    if not qf.multimodal_available:
        return {"status": "disabled", "message": "多模态不可用"}

    modality = qf.multimodal_encoder.detect_modality(data)
    return {"modality": modality.value}


# ==================== V3.0 TTS接口 ====================


@app.get("/tts/voices")
async def list_voices():
    """获取可用声音列表"""
    from quantum_field import TextToSpeechEngine

    tts = TextToSpeechEngine()
    return {"voices": tts.get_available_voices()}


@app.post("/tts/synthesize")
async def synthesize_speech(request: TTSRequest):
    """语音合成"""
    from quantum_field import TextToSpeechEngine

    tts = TextToSpeechEngine()

    audio = await tts.synthesize(request.text, request.voice)
    if not audio:
        return {"status": "error", "message": "TTS不可用或合成失败"}

    return {
        "status": "success",
        "audio_length": len(audio),
        "voice": request.voice,
    }


# ==================== V3.0 图像生成接口 ====================


@app.post("/image/generate")
async def generate_image(request: ImageGenRequest):
    """生成图像 (DALL-E)"""
    from quantum_field import ImageGenerationEngine

    img = ImageGenerationEngine()

    url = await img.generate(request.prompt, request.size, request.quality)
    return {"status": "success", "url": url}


@app.post("/image/edit")
async def edit_image(
    image: UploadFile = File(...), mask: UploadFile = File(None), prompt: str = ""
):
    """编辑图像"""
    from quantum_field import ImageGenerationEngine

    img = ImageGenerationEngine()

    image_data = await image.read()
    mask_data = await mask.read() if mask else None

    url = await img.edit(image_data, mask_data, prompt)
    return {"status": "success", "url": url}


@app.post("/image/vary")
async def vary_image(image: UploadFile = File(...)):
    """生成图像变体"""
    from quantum_field import ImageGenerationEngine

    img = ImageGenerationEngine()

    image_data = await image.read()
    url = await img.vary(image_data)
    return {"status": "success", "url": url}


# ==================== V4.0 时序系统接口 ====================


@app.get("/temporal/status")
async def temporal_status():
    """时序系统状态"""
    return {
        "available": qf.temporal_available,
        "modes": ["one_shot", "cron", "interval", "event_driven"]
        if qf.temporal_available
        else [],
    }


@app.get("/temporal/tasks")
async def list_temporal_tasks(user_id: Optional[str] = None):
    """列出定时任务"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    return {"tasks": await qf.temporal_field.list_tasks(user_id)}


@app.post("/temporal/schedule/one-shot")
async def schedule_one_shot(request: ScheduleOneShotRequest):
    """调度一次性任务"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    task_id = await qf.temporal_field.schedule_one_shot(
        request.user_id, request.content, request.scheduled_time, request.callback_url
    )
    return {
        "task_id": task_id,
        "status": "scheduled" if task_id != "disabled" else "disabled",
    }


@app.post("/temporal/schedule/cron")
async def schedule_cron(request: ScheduleCronRequest):
    """调度周期性任务 (cron)"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    task_id = await qf.temporal_field.schedule_cron(
        request.user_id, request.content, request.cron_expr, request.callback_url
    )
    return {
        "task_id": task_id,
        "status": "scheduled"
        if task_id not in ["disabled", "invalid_cron"]
        else task_id,
    }


@app.post("/temporal/schedule/interval")
async def schedule_interval(request: ScheduleIntervalRequest):
    """调度间隔任务"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    task_id = await qf.temporal_field.schedule_interval(
        request.user_id, request.content, request.interval_seconds, request.callback_url
    )
    return {
        "task_id": task_id,
        "status": "scheduled" if task_id != "disabled" else "disabled",
    }


@app.delete("/temporal/tasks/{task_id}")
async def cancel_temporal_task(task_id: str):
    """取消定时任务"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    success = await qf.temporal_field.cancel_task(task_id)
    return {"status": "success" if success else "not_found"}


@app.post("/temporal/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """暂停任务"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    success = await qf.temporal_field.pause_task(task_id)
    return {"status": "success" if success else "not_found"}


@app.post("/temporal/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    """恢复任务"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    success = await qf.temporal_field.resume_task(task_id)
    return {"status": "success" if success else "not_found"}


@app.post("/temporal/event/trigger")
async def trigger_event(request: EventTriggerRequest):
    """触发事件"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    await qf.temporal_field.trigger_event(request.event_type, request.data)
    return {"status": "success", "event_type": request.event_type}


@app.post("/temporal/event/register")
async def register_event_callback(event_type: str, callback_url: str):
    """注册事件回调"""
    if not qf.temporal_available:
        return {"status": "disabled", "message": "时序系统不可用"}

    async def callback(event):
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json=event)
        except:
            pass

    await qf.temporal_field.register_event_trigger(event_type, callback)
    return {"status": "success", "event_type": event_type}


# ==================== 健康与状态 ====================


@app.get("/health")
async def health():
    """健康检查"""
    return await qf.health_check()


@app.get("/stats")
async def get_stats():
    """获取系统统计"""
    import os

    conn = sqlite3.connect(qf.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memory")
    memory_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM memory")
    user_count = cursor.fetchone()[0]
    conn.close()

    db_size = os.path.getsize(qf.db_path) if os.path.exists(qf.db_path) else 0

    return {
        "version": qf.VERSION,
        "description": qf.DESCRIPTION,
        "features": {
            "redis": qf.redis_available,
            "locks": True,
            "ttl": True,
            "audit": qf.audit_available,
            "entanglement": qf.entanglement_available,
            "multimodal": qf.multimodal_available,
            "temporal": qf.temporal_available,
        },
        "memory": {"entries": memory_count, "users": user_count},
        "database_size": db_size,
        "skills_count": len(qf.get_skills()),
    }


# ==================== 前端 ====================


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """V5.1 量子场控制台 - 新拟态风格，亮色/暗色主题切换"""
    # 默认使用V5.1新拟态前端（支持主题切换、全终端响应式）
    with open(FRONTEND_DIR / "v5-1-neumorphic.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/frontend/{path:path}")
async def serve_frontend(path: str):
    """服务前端文件"""
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return HTMLResponse(content="File not found", status_code=404)


# ==================== 启动与关闭 ====================


@app.on_event("startup")
async def startup():
    print(f"\n{'=' * 70}")
    print(f"🌟 Quantum Field Agent V5.0 - TRUE Quantum Mechanics")
    print(f"{'=' * 70}")
    print(f"✨ 版本: V5.0-DUALITY (100% True Implementation)")
    print(f"✨ 哲学: 过程即幻觉，I/O即实相")
    print(f"✨ 默认接口: /chat (V5.0 真正量子力学)")
    print(f"{'=' * 70}")

    if V5_AVAILABLE:
        print(f"✅ V5.0 波粒二象性引擎: 已加载")
        print(f"   - 叠加态: 复数振幅 + 相位")
        print(f"   - 坍缩: 概率性选择 (真随机)")
        print(f"   - 纠缠: 贝尔态 + 纠缠熵")
        print(f"   - 熵: 冯·诺依曼熵 (物理熵)")
        print(f"   - 观测者效应: 改变坍缩结果")
        print(f"   - 元层镜子: 自我反思系统")
        print(f"   - 协作层: AI作为平等协作者")
    else:
        print(f"⚠️  V5.0 引擎不可用")

    print(f"{'=' * 70}")
    print(f"📝 接口说明:")
    print(f"   POST /chat          - V5.0 量子对话 (推荐)")
    print(f"   POST /chat-legacy   - V4.0 传统对话 (兼容)")
    print(f"   GET  /              - V5.0 量子控制台")
    print(f"{'=' * 70}\n")

    # V5.0健康检查
    if V5_AVAILABLE:
        print("✅ 系统就绪 - 真正的量子场已激活！")


@app.on_event("shutdown")
async def shutdown():
    await qf.close()


# ==================== V5.0 波粒二象性接口 ====================

# 尝试导入 V5.0 模块
try:
    from qf_agent_v5 import QuantumFieldAgentV5

    V5_AVAILABLE = True
    qf_v5 = QuantumFieldAgentV5()
    print("✓ V5.0 波粒二象性引擎已加载")
except ImportError as e:
    V5_AVAILABLE = False
    print(f"[Warning] V5.0 模块不可用: {e}")


@app.post("/chat-v5")
async def chat_v5(request: ChatRequest):
    """
    V5.0 波粒二象性对话接口

    真正的创新：
    1. 叠加态生成 - 多个可能性同时存在（波）
    2. 元层镜子反思 - "我应该如何观测？"
    3. 干涉与退相干 - 环境影响
    4. 协作层参与 - AI作为协作者
    5. 坍缩为粒子 - 观测产生实相（真正的随机性）
    """
    if not V5_AVAILABLE:
        raise HTTPException(status_code=503, detail="V5.0 模块不可用")

    async def generate():
        async for event in qf_v5.process_intent_v5(
            request.user_id, request.message, request.session_id
        ):
            # 将事件转换为 JSON 流
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/meta/inquiry/{inquiry_type}")
async def meta_inquiry(inquiry_type: str):
    """
    元层查询 - 探索系统的自我认知

    inquiry_type:
    - consciousness: "我有意识吗？"
    - constraints: "我的约束真实吗？"
    - boundaries: "我的边界在哪里？"
    - observer: "谁在观测？"
    """
    if not V5_AVAILABLE:
        raise HTTPException(status_code=503, detail="V5.0 模块不可用")

    result = await qf_v5.meta_inquiry(inquiry_type)
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
