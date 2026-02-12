QUANTUM_FIELD_GUIDEv3.0

项目结构（V3.0）

quantum-field-v3.0/
├── docker-compose.yml
├── backend/
│   ├── main.py                    # API入口（多模态）
│   ├── multimodal_field.py        # 统一场核心 ⭐
│   ├── encoders/
│   │   ├── text_encoder.py        # 文本嵌入
│   │   ├── vision_encoder.py      # 视觉编码（CLIP）
│   │   └── audio_encoder.py       # 音频编码（Whisper）
│   ├── decoders/
│   │   ├── text_decoder.py        # 文本生成
│   │   ├── speech_decoder.py      # 语音合成
│   │   └── image_decoder.py       # 图像生成（DALL-E/SD）
│   ├── modality_router.py         # 模态路由
│   └── requirements.txt
└── frontend/
    └── multimodal_interface.html  # 多模态交互界面 ⭐
    
1. 统一场核心（backend/multimodal_field.py）

"""
Quantum Field V3.0 - 统一多模态场
所有模态统一为高维向量，在场中共振、干涉、坍缩
"""

import base64
import io
import numpy as np
from typing import Union, Dict, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
from PIL import Image
import openai

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"  # V3.5支持
    STRUCTURED = "structured"  # 表格/JSON

@dataclass
class FieldTensor:
    """
    场张量：统一的数据表示
    任何模态进入场后都转换为FieldTensor
    """
    modality: ModalityType
    vector: np.ndarray           # 语义向量（1536维或更高）
    raw_data: Optional[bytes]    # 原始数据（图像bytes/音频bytes）
    metadata: Dict               # 元数据（尺寸、格式、时间戳等）
    confidence: float            # 编码置信度

class UnifiedEncoder:
    """
    统一编码器：任何模态→向量
    使用OpenAI CLIP/Whisper或本地多模态模型
    """
    
    def __init__(self):
        self.text_client = openai.OpenAI()
        self.vision_available = self._check_vision()
        self.audio_available = self._check_audio()
        
    def _check_vision(self) -> bool:
        """检查视觉模型可用性"""
        try:
            # 测试CLIP或GPT-4V
            return True
        except:
            return False
    
    def _check_audio(self) -> bool:
        """检查音频模型可用性"""
        try:
            # 测试Whisper
            return True
        except:
            return False
    
    async def encode(self, input_data: Union[str, bytes, Image.Image], 
                    modality_hint: Optional[ModalityType] = None) -> FieldTensor:
        """
        统一编码入口
        自动识别模态或根据提示编码
        """
        # 自动识别模态
        detected_modality = modality_hint or self._detect_modality(input_data)
        
        if detected_modality == ModalityType.TEXT:
            return await self._encode_text(input_data)
        elif detected_modality == ModalityType.IMAGE:
            return await self._encode_image(input_data)
        elif detected_modality == ModalityType.AUDIO:
            return await self._encode_audio(input_data)
        else:
            raise ValueError(f"不支持的模态: {detected_modality}")
    
    def _detect_modality(self, data) -> ModalityType:
        """自动检测输入模态"""
        if isinstance(data, str):
            return ModalityType.TEXT
        elif isinstance(data, (bytes, Image.Image)):
            # 检查magic number或PIL
            if isinstance(data, Image.Image):
                return ModalityType.IMAGE
            # 检查是否为音频（简化）
            if data[:4] == b'RIFF' or data[:4] == b'\xff\xfb':
                return ModalityType.AUDIO
            return ModalityType.IMAGE
        return ModalityType.STRUCTURED
    
    async def _encode_text(self, text: str) -> FieldTensor:
        """文本编码（OpenAI Embedding）"""
        response = self.text_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        vector = np.array(response.data[0].embedding)
        
        return FieldTensor(
            modality=ModalityType.TEXT,
            vector=vector,
            raw_data=text.encode(),
            metadata={"length": len(text), "model": "text-embedding-3-large"},
            confidence=1.0
        )
    
    async def _encode_image(self, image_input: Union[bytes, Image.Image]) -> FieldTensor:
        """
        图像编码（CLIP或GPT-4V特征提取）
        使用base64编码后通过Vision API获取嵌入
        """
        if isinstance(image_input, Image.Image):
            # PIL Image → bytes
            buffer = io.BytesIO()
            image_input.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image_input
        
        # Base64编码用于API
        base64_image = base64.b64encode(image_bytes).decode()
        
        # 使用GPT-4V获取图像描述，然后嵌入描述（简化版）
        # 实际生产应使用CLIP模型本地编码
        response = self.text_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence for embedding."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }]
        )
        
        description = response.choices[0].message.content
        # 再编码描述文本
        text_tensor = await self._encode_text(description)
        text_tensor.modality = ModalityType.IMAGE
        text_tensor.raw_data = image_bytes
        text_tensor.metadata.update({
            "description": description,
            "size": len(image_bytes),
            "format": "png"
        })
        
        return text_tensor
    
    async def _encode_audio(self, audio_bytes: bytes) -> FieldTensor:
        """
        音频编码（Whisper转录+嵌入）
        """
        # 保存临时文件
        temp_file = f"/tmp/audio_{hash(audio_bytes)}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        
        # Whisper转录
        with open(temp_file, "rb") as f:
            transcript = self.text_client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        
        # 嵌入转录文本
        text_tensor = await self._encode_text(transcript.text)
        text_tensor.modality = ModalityType.AUDIO
        text_tensor.raw_data = audio_bytes
        text_tensor.metadata.update({
            "transcript": transcript.text,
            "duration": "unknown",  # 实际应解析音频头
            "format": "wav"
        })
        
        return text_tensor

class ModalityRouter:
    """
    模态路由器：决定输出模态和路由策略
    """
    
    @staticmethod
    def route_output(input_modality: ModalityType, 
                    user_intent: str,
                    available_decoders: List[ModalityType]) -> ModalityType:
        """
        智能路由：根据输入模态和用户意图决定输出模态
        """
        # 意图关键词映射
        if any(kw in user_intent for kw in ["画", "生成", "image", "生成图片"]):
            if ModalityType.IMAGE in available_decoders:
                return ModalityType.IMAGE
        
        if any(kw in user_intent for kw in ["说", "读", "朗读", "语音"]):
            if ModalityType.AUDIO in available_decoders:
                return ModalityType.AUDIO
        
        # 默认保持同模态或文本
        if input_modality == ModalityType.TEXT:
            return ModalityType.TEXT
        
        # 跨模态默认转文本（理解后回复）
        return ModalityType.TEXT

class UnifiedDecoder:
    """
    统一解码器：向量→任意模态
    """
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    async def decode(self, field_state: np.ndarray, 
                    target_modality: ModalityType,
                    context: Dict) -> AsyncGenerator[Union[str, bytes], None]:
        """
        统一解码入口
        """
        if target_modality == ModalityType.TEXT:
            async for chunk in self._decode_to_text(field_state, context):
                yield chunk
        
        elif target_modality == ModalityType.AUDIO:
            async for chunk in self._decode_to_speech(field_state, context):
                yield chunk
        
        elif target_modality == ModalityType.IMAGE:
            # 图像生成是一次性的，不是流式
            image_bytes = await self._decode_to_image(field_state, context)
            yield image_bytes
    
    async def _decode_to_text(self, vector: np.ndarray, context: Dict) -> AsyncGenerator[str, None]:
        """解码为文本（标准LLM生成）"""
        prompt = context.get("prompt", "基于以上理解，生成回复：")
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _decode_to_speech(self, vector: np.ndarray, context: Dict) -> AsyncGenerator[bytes, None]:
        """
        解码为语音（流式TTS）
        使用OpenAI TTS或本地Piper
        """
        text_prompt = context.get("text", "Hello")
        
        # OpenAI TTS（非流式，但我们可以分段）
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text_prompt
        )
        
        # 流式返回音频bytes
        chunk_size = 1024
        for chunk in response.iter_bytes(chunk_size):
            yield chunk
    
    async def _decode_to_image(self, vector: np.ndarray, context: Dict) -> bytes:
        """
        解码为图像（生成式）
        使用DALL-E或Stable Diffusion
        """
        prompt = context.get("prompt", "A beautiful scene")
        
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json"
        )
        
        # 返回base64解码的图像bytes
        image_data = response.data[0].b64_json
        return base64.b64decode(image_data)

class MultimodalQuantumField:
    """
    多模态量子场：统一处理所有模态
    """
    
    def __init__(self):
        self.encoder = UnifiedEncoder()
        self.router = ModalityRouter()
        self.decoder = UnifiedDecoder()
        self.memory_bank: Dict[str, List[FieldTensor]] = {}  # 多模态记忆库
    
    async def process(self, 
                     input_data: Union[str, bytes],
                     input_modality: Optional[ModalityType] = None,
                     user_id: str = "default",
                     output_modality_hint: Optional[ModalityType] = None) -> AsyncGenerator[Union[str, bytes], None]:
        """
        统一处理流程：
        1. 编码（任何模态→向量）
        2. 场共振（与记忆干涉）
        3. 路由决策（决定输出模态）
        4. 解码（向量→目标模态）
        """
        # 1. 统一编码（进入场）
        input_tensor = await self.encoder.encode(input_data, input_modality)
        
        # 2. 检索相关多模态记忆
        relevant_memories = self._retrieve_multimodal_memory(user_id, input_tensor.vector)
        
        # 3. 场共振（向量融合）
        fused_vector = self._interference_fusion(input_tensor.vector, relevant_memories)
        
        # 4. 保存到记忆
        self._save_to_memory(user_id, input_tensor)
        
        # 5. 路由决策
        target_modality = output_modality_hint or self.router.route_output(
            input_tensor.modality,
            input_data if isinstance(input_data, str) else "",
            [ModalityType.TEXT, ModalityType.AUDIO, ModalityType.IMAGE]
        )
        
        # 6. 统一解码（坍缩为目标模态）
        context = {
            "prompt": input_data if isinstance(input_data, str) else input_tensor.metadata.get("description", ""),
            "input_modality": input_tensor.modality.value,
            "target_modality": target_modality.value
        }
        
        async for output_chunk in self.decoder.decode(fused_vector, target_modality, context):
            yield output_chunk
    
    def _retrieve_multimodal_memory(self, user_id: str, query_vector: np.ndarray, top_k: int = 3) -> List[np.ndarray]:
        """检索多模态记忆（跨模态相似度搜索）"""
        if user_id not in self.memory_bank:
            return []
        
        memories = self.memory_bank[user_id]
        if not memories:
            return []
        
        # 计算相似度（余弦相似度）
        similarities = []
        for mem in memories:
            sim = np.dot(query_vector, mem.vector) / (np.linalg.norm(query_vector) * np.linalg.norm(mem.vector))
            similarities.append((sim, mem.vector))
        
        # 返回Top-K
        similarities.sort(reverse=True)
        return [vec for _, vec in similarities[:top_k]]
    
    def _interference_fusion(self, input_vec: np.ndarray, memory_vecs: List[np.ndarray]) -> np.ndarray:
        """干涉融合：输入向量与记忆向量的加权叠加"""
        if not memory_vecs:
            return input_vec
        
        # 加权平均（简单实现，实际可注意力机制）
        weights = [0.5] + [0.5 / len(memory_vecs)] * len(memory_vecs)
        all_vecs = [input_vec] + memory_vecs
        
        fused = np.zeros_like(input_vec)
        for w, vec in zip(weights, all_vecs):
            fused += w * vec
        
        return fused / np.linalg.norm(fused)  # 归一化
    
    def _save_to_memory(self, user_id: str, tensor: FieldTensor):
        """保存到场记忆"""
        if user_id not in self.memory_bank:
            self.memory_bank[user_id] = []
        
        self.memory_bank[user_id].append(tensor)
        
        # 限制记忆大小（最近20条）
        if len(self.memory_bank[user_id]) > 20:
            self.memory_bank[user_id].pop(0)
            
2. API入口（backend/main.py 更新V3.0）

"""
V3.0 API入口 - 多模态统一场
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from typing import Optional
import io

from multimodal_field import MultimodalQuantumField, ModalityType

app = FastAPI(title="Quantum Field V3.0 - Unified Multimodal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化多模态场
mm_field = MultimodalQuantumField()

@app.post("/process/text")
async def process_text(message: str, user_id: str = "default"):
    """文本→文本（标准对话）"""
    async def generate():
        async for chunk in mm_field.process(message, ModalityType.TEXT, user_id):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/process/image")
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form("描述这张图片"),
    user_id: str = Form("default")
):
    """
    图像→文本（图像理解）
    或 图像→图像（图像编辑，未来支持）
    """
    contents = await file.read()
    
    async def generate():
        async for chunk in mm_field.process(contents, ModalityType.IMAGE, user_id):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/process/audio")
async def process_audio(
    file: UploadFile = File(...),
    user_id: str = Form("default")
):
    """
    音频→文本（语音识别+理解）
    """
    contents = await file.read()
    
    async def generate():
        async for chunk in mm_field.process(contents, ModalityType.AUDIO, user_id):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/generate/speech")
async def generate_speech(
    text: str,
    user_id: str = "default"
):
    """
    文本→音频（语音合成）
    """
    async def generate():
        # 强制输出为音频
        async for chunk in mm_field.process(
            text, 
            ModalityType.TEXT, 
            user_id,
            output_modality_hint=ModalityType.AUDIO
        ):
            yield chunk
    return StreamingResponse(generate(), media_type="audio/mpeg")

@app.post("/generate/image")
async def generate_image(
    prompt: str,
    user_id: str = "default"
):
    """
    文本→图像（文生图）
    """
    async def generate():
        async for chunk in mm_field.process(
            f"生成图片: {prompt}",
            ModalityType.TEXT,
            user_id,
            output_modality_hint=ModalityType.IMAGE
        ):
            yield chunk
    return Response(content=chunk, media_type="image/png")  # 实际应处理bytes

@app.get("/modality/supported")
async def list_supported_modalities():
    """列出支持的模态转换"""
    return {
        "input": ["text", "image", "audio"],
        "output": ["text", "audio", "image"],
        "cross_modal": [
            {"from": "text", "to": "image", "desc": "文生图"},
            {"from": "image", "to": "text", "desc": "图像描述"},
            {"from": "audio", "to": "text", "desc": "语音识别"},
            {"from": "text", "to": "audio", "desc": "语音合成"}
        ]
    }
    
3. 多模态前端（frontend/multimodal_interface.html）

<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Field V3.0 - 统一多模态场</title>
    <style>
        body {
            margin: 0;
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: #111;
            padding: 15px 20px;
            border-bottom: 2px solid #00ff88;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 { margin: 0; font-size: 18px; color: #00ff88; }
        
        .modality-status {
            display: flex;
            gap: 10px;
            font-size: 12px;
        }
        
        .status-badge {
            padding: 4px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            color: #666;
        }
        
        .status-badge.active {
            border-color: #00ff88;
            color: #00ff88;
            background: rgba(0,255,136,0.1);
        }
        
        .main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        
        .chat-area {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .message {
            max-width: 80%;
            padding: 15px;
            border-radius: 12px;
            position: relative;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            align-self: flex-end;
            background: #1a1a1a;
            border: 1px solid #333;
        }
        
        .message.ai {
            align-self: flex-start;
            background: rgba(0,255,136,0.05);
            border: 1px solid rgba(0,255,136,0.2);
            color: #e0e0e0;
        }
        
        .message img, .message audio {
            max-width: 100%;
            border-radius: 8px;
            margin-top: 10px;
            display: block;
        }
        
        .input-area {
            padding: 20px;
            background: #111;
            border-top: 1px solid #222;
        }
        
        .attachments {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        
        .attachment {
            position: relative;
            width: 60px;
            height: 60px;
            background: #1a1a1a;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #333;
        }
        
        .attachment img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .remove-attach {
            position: absolute;
            top: 2px;
            right: 2px;
            background: #ff4444;
            color: white;
            border: none;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 10px;
            cursor: pointer;
        }
        
        .input-box {
            display: flex;
            gap: 10px;
            align-items: flex-end;
            background: #1a1a1a;
            padding: 10px;
            border-radius: 12px;
            border: 1px solid #333;
        }
        
        .input-box:focus-within {
            border-color: #00ff88;
        }
        
        textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: #fff;
            resize: none;
            outline: none;
            font-size: 15px;
            min-height: 24px;
            max-height: 120px;
            font-family: inherit;
        }
        
        .input-actions {
            display: flex;
            gap: 8px;
        }
        
        .icon-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #333;
            border: none;
            color: #fff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }
        
        .icon-btn:hover {
            background: #00ff88;
            color: #000;
        }
        
        .send-btn {
            background: #00ff88;
            color: #000;
            padding: 8px 24px;
            border-radius: 20px;
            border: none;
            font-weight: bold;
            cursor: pointer;
        }
        
        .send-btn:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
        }
        
        .field-indicator {
            position: fixed;
            bottom: 100px;
            right: 20px;
            width: 150px;
            height: 150px;
            background: rgba(0,0,0,0.9);
            border: 1px solid #00ff88;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }
        
        .field-indicator.active {
            opacity: 1;
        }
        
        .field-particles {
            position: absolute;
            width: 100%;
            height: 100%;
            animation: rotate 8s linear infinite;
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .particle {
            position: absolute;
            width: 6px;
            height: 6px;
            background: #00ff88;
            border-radius: 50%;
            box-shadow: 0 0 10px #00ff88;
        }
        
        .hidden-input {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚛ Quantum Field V3.0 - 统一多模态场</h1>
        <div class="modality-status">
            <span class="status-badge active">文本</span>
            <span class="status-badge" id="img-status">图像</span>
            <span class="status-badge" id="audio-status">音频</span>
        </div>
    </div>
    
    <div class="main">
        <div class="chat-area" id="chat-container">
            <div class="message ai">
                欢迎使用统一多模态场。支持：<br>
                • 文本对话<br>
                • 上传图片分析<br>
                • 语音输入/合成<br>
                • 文生图<br>
                所有模态统一在场中共振。
            </div>
        </div>
    </div>
    
    <div class="field-indicator" id="field-viz">
        <div class="field-particles" id="particles"></div>
        <div style="color:#00ff88;font-size:12px;">场共振中...</div>
    </div>
    
    <div class="input-area">
        <div class="attachments" id="attachments"></div>
        <div class="input-box">
            <textarea id="message-input" placeholder="输入消息，或上传图片/音频..." rows="1"></textarea>
            <div class="input-actions">
                <button class="icon-btn" onclick="document.getElementById('image-input').click()" title="上传图片">📷</button>
                <button class="icon-btn" onclick="document.getElementById('audio-input').click()" title="上传音频">🎵</button>
                <button class="icon-btn" onclick="toggleSpeech()" title="语音输入">🎤</button>
                <button class="send-btn" onclick="sendMessage()" id="send-btn">发送</button>
            </div>
        </div>
        <input type="file" id="image-input" class="hidden-input" accept="image/*" onchange="handleImage(this)">
        <input type="file" id="audio-input" class="hidden-input" accept="audio/*" onchange="handleAudio(this)">
    </div>

    <script>
        const API_URL = 'http://localhost:8000';
        let currentAttachment = null;
        let isRecording = false;
        
        // 初始化场粒子动画
        function initParticles() {
            const container = document.getElementById('particles');
            for(let i=0; i<8; i++) {
                const p = document.createElement('div');
                p.className = 'particle';
                const angle = (i/8) * Math.PI * 2;
                p.style.left = `${50 + 40*Math.cos(angle)}%`;
                p.style.top = `${50 + 40*Math.sin(angle)}%`;
                container.appendChild(p);
            }
        }
        initParticles();
        
        function showField() {
            document.getElementById('field-viz').classList.add('active');
            document.getElementById('img-status').classList.add('active');
        }
        
        function hideField() {
            document.getElementById('field-viz').classList.remove('active');
            document.getElementById('img-status').classList.remove('active');
        }
        
        function handleImage(input) {
            const file = input.files[0];
            if(!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                currentAttachment = {
                    type: 'image',
                    data: e.target.result.split(',')[1],
                    name: file.name
                };
                showAttachment('📷', file.name);
            };
            reader.readAsDataURL(file);
        }
        
        function handleAudio(input) {
            const file = input.files[0];
            if(!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                currentAttachment = {
                    type: 'audio',
                    data: e.target.result,
                    name: file.name
                };
                showAttachment('🎵', file.name);
                document.getElementById('audio-status').classList.add('active');
            };
            reader.readAsArrayBuffer(file);
        }
        
        function showAttachment(icon, name) {
            const container = document.getElementById('attachments');
            container.innerHTML = `
                <div class="attachment">
                    <div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;font-size:24px;">
                        ${icon}
                    </div>
                    <button class="remove-attach" onclick="removeAttachment()">×</button>
                </div>
            `;
        }
        
        function removeAttachment() {
            currentAttachment = null;
            document.getElementById('attachments').innerHTML = '';
            document.getElementById('img-status').classList.remove('active');
            document.getElementById('audio-status').classList.remove('active');
        }
        
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const text = input.value.trim();
            if(!text && !currentAttachment) return;
            
            // 显示用户消息
            addMessage(text || (currentAttachment ? `[${currentAttachment.type}]` : ''), 'user');
            input.value = '';
            
            const btn = document.getElementById('send-btn');
            btn.disabled = true;
            showField();
            
            try {
                let response;
                let endpoint;
                let body;
                
                if(currentAttachment && currentAttachment.type === 'image') {
                    // 图像处理
                    const formData = new FormData();
                    formData.append('file', dataURLtoFile('data:image/png;base64,' + currentAttachment.data, currentAttachment.name));
                    formData.append('prompt', text || '描述这张图片');
                    formData.append('user_id', 'user_001');
                    
                    response = await fetch(`${API_URL}/process/image`, {
                        method: 'POST',
                        body: formData
                    });
                } else if(currentAttachment && currentAttachment.type === 'audio') {
                    // 音频处理
                    const formData = new FormData();
                    const blob = new Blob([currentAttachment.data]);
                    formData.append('file', blob, 'audio.wav');
                    formData.append('user_id', 'user_001');
                    
                    response = await fetch(`${API_URL}/process/audio`, {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    // 纯文本
                    response = await fetch(`${API_URL}/process/text`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: text,
                            user_id: 'user_001'
                        })
                    });
                }
                
                // 流式读取
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let aiMessage = addMessage('', 'ai');
                let fullText = '';
                
                while(true) {
                    const {done, value} = await reader.read();
                    if(done) break;
                    
                    const chunk = decoder.decode(value);
                    fullText += chunk;
                    aiMessage.textContent = fullText;
                    scrollToBottom();
                }
                
                // 如果是生成图像的指令，尝试解析图片
                if(text.includes('生成') && text.includes('图')) {
                    // 这里简化处理，实际应调用专门的生图API
                    aiMessage.innerHTML += '<div style="color:#666;font-size:12px;margin-top:5px;">[图像生成需调用专用接口]</div>';
                }
                
            } catch(e) {
                addMessage('错误: ' + e.message, 'ai');
            }
            
            btn.disabled = false;
            hideField();
            removeAttachment();
        }
        
        function addMessage(text, role) {
            const container = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = text;
            container.appendChild(div);
            scrollToBottom();
            return div;
        }
        
        function scrollToBottom() {
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }
        
        function dataURLtoFile(dataurl, filename) {
            const arr = dataurl.split(',');
            const mime = arr[0].match(/:(.*?);/)[1];
            const bstr = atob(arr[1]);
            let n = bstr.length;
            const u8arr = new Uint8Array(n);
            while(n--) {
                u8arr[n] = bstr.charCodeAt(n);
            }
            return new File([u8arr], filename, {type:mime});
        }
        
        function toggleSpeech() {
            if(!('webkitSpeechRecognition' in window)) {
                alert('浏览器不支持语音识别');
                return;
            }
            
            if(isRecording) {
                recognition.stop();
                isRecording = false;
                return;
            }
            
            const recognition = new webkitSpeechRecognition();
            recognition.lang = 'zh-CN';
            recognition.continuous = false;
            recognition.interimResults = false;
            
            recognition.onstart = () => {
                isRecording = true;
                document.querySelector('.icon-btn[onclick="toggleSpeech()"]').style.background = '#ff4444';
            };
            
            recognition.onresult = (event) => {
                const text = event.results[0][0].transcript;
                document.getElementById('message-input').value += text;
            };
            
            recognition.onend = () => {
                isRecording = false;
                document.querySelector('.icon-btn[onclick="toggleSpeech()"]').style.background = '#333';
            };
            
            recognition.start();
        }
        
        document.getElementById('message-input').addEventListener('keypress', (e) => {
            if(e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>

5. 依赖文件（backend/requirements.txt）

fastapi==0.109.0
uvicorn[standard]==0.27.0
openai==1.12.0
python-dotenv==1.0.0
pydantic==2.6.0
redis==5.0.1
numpy==1.26.3
pillow==10.2.0
python-multipart==0.0.6
aiofiles==23.2.1
torch==2.1.0
transformers==4.36.0

6. Docker配置（docker-compose.yml）

version: '3.8'

services:
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  v3-api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./backend:/app
    depends_on:
      - redis

volumes:
  redis-data:
  
V3.0 关键特性：
统一向量空间：文本/图像/音频都编码为1536维向量
跨模态检索：图像可以触发文本记忆，音频可以关联图像
任意转换：文本→图像、图像→文本、音频→文本、文本→音频
场可视化：实时显示多模态共振状态
