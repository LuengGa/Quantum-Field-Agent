"""
API Documentation - API 文档
===========================

Meta Quantum Field Agent API 文档示例

## 快速开始

```bash
# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000

# 访问文档
# http://localhost:8000/docs
```

## 认证

所有API需要JWT认证：

```bash
# 获取令牌
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# 使用令牌
curl http://localhost:8000/api/v1/status \
  -H "Authorization: Bearer <your_token>"
```

## API 端点

### 健康检查

**GET /health**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-12T20:30:00",
  "version": "4.0.0"
}
```

### 处理交互

**POST /api/v1/evolution/interactions**

请求体：
```json
{
  "user_id": "user_001",
  "session_id": "session_123",
  "interaction_type": "question_answer",
  "input_summary": "用户询问量子纠缠",
  "output_summary": "解释量子纠缠原理",
  "outcome": "positive",
  "effectiveness": 0.85,
  "feedback": "解释清晰易懂"
}
```

响应：
```json
{
  "status": "recorded",
  "message": "交互已记录"
}
```

### 模式挖掘

**POST /api/v1/evolution/patterns/mine**

请求体：
```json
{
  "min_confidence": 0.6,
  "interaction_types": ["question_answer", "problem_solving"]
}
```

响应：
```json
{
  "total_patterns": 5,
  "time_patterns": [
    {
      "id": "pat_001",
      "name": "时间模式",
      "type": "time_pattern",
      "confidence": 0.85,
      "occurrences": 12
    }
  ],
  "causality_patterns": [...],
  "sequence_patterns": [...],
  "clustering_patterns": [...],
  "anomaly_patterns": [...]
}
```

### 策略选择

**POST /api/v1/evolution/strategies/select**

请求体：
```json
{
  "context": {
    "user_expertise": "beginner",
    "task_type": "explanation",
    "complexity": "high"
  },
  "available_strategies": ["str_001", "str_002"]
}
```

响应：
```json
{
  "selected_strategy": {
    "id": "str_002",
    "name": "类比说明",
    "type": "explanation",
    "actions": ["使用日常例子", "逐步解释"],
    "confidence": 0.82
  },
  "alternatives": [...]
}
```

### 假设验证

**POST /api/v1/evolution/hypotheses/validate**

请求体：
```json
{
  "hypothesis_id": "hyp_001",
  "validation_type": "automatic",
  "evidence": {
    "test_results": [...],
    "observations": [...]
  }
}
```

响应：
```json
{
  "hypothesis_id": "hyp_001",
  "validation_result": "confirmed",
  "confidence_score": 0.85,
  "supporting_evidence": 12,
  "contradicting_evidence": 1,
  "statistical_significance": 0.95
}
```

### 知识检索

**GET /api/v1/evolution/knowledge**

请求参数：
- domain: 知识领域（可选）
- context: 查询上下文（可选）
- limit: 返回数量限制

响应：
```json
{
  "knowledge": [
    {
      "id": "know_001",
      "title": "渐进式解释原则",
      "domain": "education",
      "content": "逐步增加复杂度...",
      "confidence": 0.85,
      "applicability": {
        "scope": "wide",
        "conditions": ["初学者", "复杂概念"]
      },
      "evidence": [...],
      "tags": ["教学", "解释", "认知"]
    }
  ],
  "total_count": 45
}
```

### A/B 测试

**POST /api/v1/evolution/experiments/ab**

请求体：
```json
{
  "name": "策略效果对比",
  "strategy_a": "str_001",
  "strategy_b": "str_002",
  "traffic_split": 0.5,
  "min_sample_size": 100
}
```

响应：
```json
{
  "experiment_id": "exp_001",
  "status": "running",
  "group_a": {
    "strategy": "str_001",
    "samples": 45,
    "avg_effectiveness": 0.72
  },
  "group_b": {
    "strategy": "str_002",
    "samples": 55,
    "avg_effectiveness": 0.78
  },
  "winner": "str_002",
  "confidence": 0.85,
  "p_value": 0.03
}
```

### 数据收集

**POST /api/v1/evolution/data/collect**

请求体：
```json
{
  "source": "interaction",
  "data_type": "user_feedback",
  "payload": {
    "rating": 5,
    "comment": "很有帮助",
    "context": "量子力学解释"
  },
  "user_id": "user_001",
  "session_id": "session_123"
}
```

响应：
```json
{
  "data_point_id": "dp_001",
  "quality_score": 0.85,
  "collected_at": "2026-02-12T20:30:00"
}
```

### 性能指标

**GET /metrics**

返回 Prometheus 格式的指标：
```
# HELP quantum_field_info Quantum Field Agent Info
# TYPE quantum_field_info gauge
quantum_field_info{version="4.0.0"} 1

# HELP patterns_total Total patterns discovered
# TYPE patterns_total gauge
patterns_total 5

# HELP strategies_total Total strategies
# TYPE strategies_total gauge
strategies_total 13

# HELP hypotheses_total Total hypotheses
# TYPE hypotheses_total gauge
hypotheses_total 271
```

## 错误处理

所有API遵循统一的错误格式：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "输入验证失败",
    "details": [...]
  }
}
```

常见错误码：
- `AUTH_REQUIRED`: 需要认证
- `INVALID_TOKEN`: 无效的令牌
- `PERMISSION_DENIED`: 权限不足
- `NOT_FOUND`: 资源不存在
- `VALIDATION_ERROR`: 输入验证失败
- `INTERNAL_ERROR`: 内部错误

## 速率限制

- 默认: 100次/分钟
- API端点: 1000次/分钟
- 认证端点: 10次/分钟

## WebSocket 实时更新

连接到 `ws://localhost:8000/ws/evolution` 接收实时更新：

```json
{
  "type": "pattern_discovered",
  "data": {
    "pattern_id": "pat_001",
    "pattern_type": "time_pattern",
    "confidence": 0.85
  }
}
```
"""

from fastapi import APIRouter

router = APIRouter(prefix="/docs", tags=["documentation"])


@router.get("/")
async def api_overview():
    """API概述"""
    return {
        "name": "Meta Quantum Field Agent API",
        "version": "4.0.0",
        "description": "AI协作系统 - 过程即幻觉，I/O即实相",
        "endpoints": {
            "auth": "/api/v1/auth",
            "evolution": "/api/v1/evolution",
            "monitoring": "/health",
            "metrics": "/metrics",
        },
        "documentation": {"openapi": "/docs", "redoc": "/redoc"},
    }


@router.get("/examples")
async def api_examples():
    """API使用示例"""
    return {
        "authentication": {
            "description": "获取访问令牌",
            "endpoint": "POST /api/v1/auth/token",
            "request": {"username": "admin", "password": "admin123"},
            "response": {
                "access_token": "eyJhbGciOiJIUzI1NiIs...",
                "token_type": "bearer",
                "expires_in": 1800,
            },
        },
        "process_interaction": {
            "description": "处理用户交互",
            "endpoint": "POST /api/v1/evolution/interactions",
            "request": {
                "user_id": "user_001",
                "session_id": "session_123",
                "interaction_type": "question_answer",
                "input_summary": "用户询问量子纠缠",
                "output_summary": "解释量子纠缠原理",
                "outcome": "positive",
            },
            "response": {"status": "recorded", "message": "交互已记录"},
        },
        "mine_patterns": {
            "description": "运行模式挖掘",
            "endpoint": "POST /api/v1/evolution/patterns/mine",
            "response": {
                "total_patterns": 5,
                "time_patterns": [...],
                "causality_patterns": [...],
                "sequence_patterns": [...],
                "clustering_patterns": [...],
                "anomaly_patterns": [...],
            },
        },
    }
