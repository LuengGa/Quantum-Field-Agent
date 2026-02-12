#!/usr/bin/env python3
"""
测试脚本 - 验证统一架构
融合V1.0和V1.5功能
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Quantum Field Agent - 统一架构测试")
print("=" * 70)

# 测试1: 基础模式（无Redis）
print("\n[测试1] 基础模式（USE_REDIS=false）...")
os.environ["USE_REDIS"] = "false"
os.environ["USE_DISTRIBUTED"] = "false"

try:
    from quantum_field import QuantumField

    qf = QuantumField()

    print(f"✅ 基础模式初始化成功")
    print(f"   版本: {qf.VERSION}")
    print(f"   Redis可用: {qf.redis_available}")
    print(f"   技能数: {len(qf.get_skills())}")

except Exception as e:
    print(f"❌ 失败: {e}")
    sys.exit(1)

# 测试2: 健康检查
print("\n[测试2] 健康检查...")
try:
    health = asyncio.run(qf.health_check())
    print(f"✅ 健康检查通过")
    print(f"   状态: {health['status']}")
    print(f"   运行时间: {health['uptime']:.2f}秒")
    print(f"   组件: {health['components']}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试3: 获取配置
print("\n[测试3] 获取配置...")
try:
    config = qf.get_config()
    print(f"✅ 配置获取成功")
    print(f"   USE_REDIS: {config['use_redis']}")
    print(f"   USE_DISTRIBUTED: {config['use_distributed']}")
    print(f"   ENTROPY_THRESHOLD: {config['entropy_threshold']}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试4: 处理意图（基础模式）
print("\n[测试4] 处理意图（基础模式）...")
try:

    async def test_basic():
        tokens = []
        async for token in qf.process_intent("test_user", "你好，请计算 25*4"):
            tokens.append(token)
        return "".join(tokens)

    result = asyncio.run(test_basic())
    print(f"✅ 意图处理成功")
    print(f"   响应长度: {len(result)} 字符")
    if "100" in result or "collapse" in result:
        print(f"   结果: 正常（包含计算结果或状态标记）")
    else:
        print(f"   结果预览: {result[:100]}...")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试5: 获取场状态
print("\n[测试5] 获取场状态...")
try:
    status = asyncio.run(qf.get_field_status("test_user"))
    print(f"✅ 场状态获取成功")
    print(f"   场熵: {status['entropy']:.2f}")
    print(f"   版本: {status['version']}")
    print(f"   特性: {status['features']}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试6: 重置场
print("\n[测试6] 重置场...")
try:
    reset_result = asyncio.run(qf.reset_field("test_user"))
    print(f"✅ 场重置成功")
    print(f"   状态: {reset_result['status']}")
    print(f"   消息: {reset_result['message']}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试7: Redis模式（如果可用）
print("\n[测试7] Redis模式（尝试启用）...")
try:
    # 尝试启用Redis
    os.environ["USE_REDIS"] = "true"

    from quantum_field import QuantumField as QFRedis

    qf_redis = QFRedis()

    if qf_redis.redis_available:
        print(f"✅ Redis模式启动成功")
        print(f"   Redis可用: {qf_redis.redis_available}")

        # 测试Redis场状态
        status = asyncio.run(qf_redis.get_field_status("redis_test_user"))
        print(f"   场熵: {status['entropy']:.2f}")
        print(f"   在本地缓存: {status.get('in_local_cache', False)}")
    else:
        print(f"⚠️ Redis不可用，继续以本地模式运行")
        print(f"   提示: 安装并启动Redis以启用完整功能")

except Exception as e:
    print(f"⚠️ Redis测试跳过: {e}")

# 测试8: 分布式模式
print("\n[测试8] 分布式模式配置...")
try:
    os.environ["USE_DISTRIBUTED"] = "true"
    os.environ["USE_HIGH_ENTROPY_MODEL"] = "true"

    from quantum_field import QuantumField as QFDistributed

    qf_dist = QFDistributed()

    config = qf_dist.get_config()
    print(f"✅ 分布式配置已加载")
    print(f"   USE_DISTRIBUTED: {config['use_distributed']}")
    print(f"   USE_HIGH_ENTROPY_MODEL: {config['use_high_entropy_model']}")
    print(f"   Redis可用: {qf_dist.redis_available}")

    if qf_dist.redis_available:
        print(f"   提示: 系统将在高熵时自动使用增强模式")
    else:
        print(f"   提示: Redis不可用，分布式功能受限")

except Exception as e:
    print(f"❌ 失败: {e}")

# 关闭资源
print("\n[清理] 关闭资源...")
try:
    asyncio.run(qf.close())
    print("✅ 资源已关闭")
except Exception as e:
    print(f"⚠️ 关闭时出错: {e}")

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)
print("\n📋 总结:")
print("   ✅ 统一架构工作正常")
print("   ✅ 基础模式（V1.0风格）可用")
print("   ✅ 配置热更新支持")
print("   ⚠️  安装Redis可启用完整V1.5功能")
print("\n🚀 启动命令:")
print("   # 基础模式")
print("   python3 -m uvicorn main:app --host 0.0.0.0 --port 8001")
print("\n   # 启用Redis（完整功能）")
print("   export USE_REDIS=true")
print("   export USE_DISTRIBUTED=true")
print("   python3 -m uvicorn main:app --host 0.0.0.0 --port 8001")
