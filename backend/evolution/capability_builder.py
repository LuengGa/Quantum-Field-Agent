"""
Capability Builder - 能力构建器
==============================

基于需求动态构建新能力：
1. 需求分析 - 分析当前能力缺口
2. 能力设计 - 设计新能力的实现
3. 快速原型 - 快速构建能力原型
4. 迭代优化 - 根据反馈优化能力
5. 能力注册 - 将能力注册到系统中

核心理念：
- 能力不是预定义的，而是在实践中涌现的
- 能力需要快速验证和迭代
- 能力是可组合的
"""

import json
import inspect
from typing import Dict, List, Any, Optional, Callable, Type
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
import uuid


@dataclass
class Capability:
    """能力定义"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = "general"
    version: str = "1.0.0"

    trigger_conditions: Dict = field(default_factory=dict)

    implementation_type: str = "function"
    implementation_code: str = ""
    implementation_function: Optional[Callable] = None

    dependencies: List[str] = field(default_factory=list)
    required_modules: List[str] = field(default_factory=list)

    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)

    performance_metrics: Dict = field(default_factory=dict)

    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    is_active: bool = True
    is_verified: bool = False
    verification_notes: str = ""

    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CapabilityRequest:
    """能力构建请求"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = "general"

    trigger_conditions: Dict = field(default_factory=dict)

    input_requirements: Dict = field(default_factory=dict)
    output_specifications: Dict = field(default_factory=dict)

    priority: str = "medium"
    deadline: str = ""

    required_skills: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"

    notes: List[str] = field(default_factory=list)


class CapabilityBuilder:
    """
    能力构建器

    动态构建和优化能力：
    - 分析能力需求
    - 构建能力原型
    - 优化和验证
    - 注册和管理
    """

    def __init__(self, db):
        self.db = db

        self._capabilities: Dict[str, Capability] = {}
        self._requests: Dict[str, CapabilityRequest] = {}
        self._capability_registry: Dict[str, Capability] = {}

        self._load_capabilities()

    def _load_capabilities(self):
        """从数据库加载现有能力"""
        pass

    async def analyze_capability_gap(
        self, task_requirements: Dict, current_capabilities: Optional[List[str]] = None
    ) -> List[CapabilityRequest]:
        """
        分析能力缺口

        Args:
            task_requirements: 任务需求
            current_capabilities: 当前能力列表

        Returns:
            能力需求列表
        """
        requests = []

        required_capabilities = task_requirements.get("required_capabilities", [])

        existing = set(current_capabilities or [])

        missing = [c for c in required_capabilities if c not in existing]

        for capability in missing:
            request = CapabilityRequest(
                name=f"需要-{capability}",
                description=f"需要构建能力: {capability}",
                category=task_requirements.get("category", "general"),
                trigger_conditions={"task_type": capability},
                input_requirements=task_requirements.get("input_schema", {}),
                output_specifications=task_requirements.get("output_schema", {}),
                priority="high",
            )

            requests.append(request)
            self._requests[request.id] = request

        return requests

    async def build_capability(
        self, request: CapabilityRequest
    ) -> Optional[Capability]:
        """
        根据需求构建能力

        Args:
            request: 能力需求

        Returns:
            构建的能力
        """
        capability = Capability(
            name=request.name,
            description=request.description,
            category=request.category,
            trigger_conditions=request.trigger_conditions,
            input_schema=request.input_requirements,
            output_schema=request.output_specifications,
            dependencies=request.dependencies,
            tags=[request.category],
        )

        if capability.category == "text_generation":
            capability.implementation_type = "function"
            capability.implementation_code = self._generate_text_generation_code(
                request
            )

        elif capability.category == "data_analysis":
            capability.implementation_type = "function"
            capability.implementation_code = self._generate_analysis_code(request)

        elif capability.category == "pattern_matching":
            capability.implementation_type = "function"
            capability.implementation_code = self._generate_matching_code(request)

        else:
            capability.implementation_type = "function"
            capability.implementation_code = self._generate_generic_code(request)

        capability.is_verified = False

        self._capabilities[capability.id] = capability
        self._capability_registry[capability.name] = capability

        return capability

    def _generate_text_generation_code(self, request: CapabilityRequest) -> str:
        """生成文本生成能力代码"""
        return f'''
def {request.name.replace("-", "_")}(input_data):
    """
    {request.description}
    
    Args:
        input_data: 输入数据
        
    Returns:
        生成的文本
    """
    prompt = input_data.get("prompt", "")
    
    response = generate_response(prompt)
    
    return {{
        "result": response,
        "confidence": 0.8
    }}
'''

    def _generate_analysis_code(self, request: CapabilityRequest) -> str:
        """生成数据分析能力代码"""
        return f'''
def {request.name.replace("-", "_")}(input_data):
    """
    {request.description}
    
    Args:
        input_data: 输入数据
        
    Returns:
        分析结果
    """
    data = input_data.get("data", [])
    
    stats = {{
        "count": len(data),
        "sum": sum(data),
        "avg": sum(data) / len(data) if data else 0
    }}
    
    return {{
        "statistics": stats,
        "insights": []
    }}
'''

    def _generate_matching_code(self, request: CapabilityRequest) -> str:
        """生成模式匹配能力代码"""
        return f'''
def {request.name.replace("-", "_")}(input_data):
    """
    {request.description}
    
    Args:
        input_data: 输入数据
        
    Returns:
        匹配结果
    """
    patterns = {request.trigger_conditions.get("patterns", [])}
    target = input_data.get("target", "")
    
    matches = [p for p in patterns if p in target]
    
    return {{
        "matches": matches,
        "count": len(matches)
    }}
'''

    def _generate_generic_code(self, request: CapabilityRequest) -> str:
        """生成通用能力代码"""
        return f'''
def {request.name.replace("-", "_")}(input_data):
    """
    {request.description}
    
    Args:
        input_data: 输入数据
        
    Returns:
        处理结果
    """
    # TODO: 实现具体逻辑
    
    return {{
        "status": "completed",
        "data": input_data
    }}
'''

    async def verify_capability(
        self, capability_id: str, test_cases: List[Dict]
    ) -> Dict:
        """
        验证能力

        Args:
            capability_id: 能力ID
            test_cases: 测试用例

        Returns:
            验证结果
        """
        if capability_id not in self._capabilities:
            return {"error": "能力不存在"}

        capability = self._capabilities[capability_id]

        results = []
        passed = 0

        for i, test_case in enumerate(test_cases):
            try:
                result = await self._execute_capability(capability, test_case["input"])

                expected = test_case.get("expected", {})

                if self._compare_results(result, expected):
                    passed += 1
                    results.append(
                        {"case": i + 1, "status": "passed", "result": result}
                    )
                else:
                    results.append(
                        {
                            "case": i + 1,
                            "status": "failed",
                            "expected": expected,
                            "actual": result,
                        }
                    )
            except Exception as e:
                results.append({"case": i + 1, "status": "error", "error": str(e)})

        success_rate = passed / len(test_cases) if test_cases else 0

        capability.is_verified = success_rate >= 0.8

        capability.verification_notes = json.dumps(
            {
                "test_date": datetime.now().isoformat(),
                "passed": passed,
                "total": len(test_cases),
                "success_rate": success_rate,
                "results": results,
            }
        )

        capability.last_updated = datetime.now().isoformat()

        return {
            "capability_id": capability_id,
            "is_verified": capability.is_verified,
            "success_rate": success_rate,
            "passed": passed,
            "total": len(test_cases),
            "results": results,
        }

    async def _execute_capability(
        self, capability: Capability, input_data: Dict
    ) -> Dict:
        """执行能力"""
        if capability.implementation_type == "function":
            try:
                exec_globals = {}
                exec(capability.implementation_code, exec_globals)

                func_name = capability.name.replace("-", "_")
                if func_name in exec_globals:
                    result = exec_globals[func_name](input_data)
                    return result

            except Exception as e:
                return {"error": str(e)}

        return {"status": "not_implemented"}

    def _compare_results(self, actual: Dict, expected: Dict) -> bool:
        """比较结果"""
        if not expected:
            return True

        for key, exp_value in expected.items():
            actual_value = actual.get(key)

            if isinstance(exp_value, dict):
                if not self._compare_results(actual_value, exp_value):
                    return False
            elif actual_value != exp_value:
                return False

        return True

    async def optimize_capability(
        self, capability_id: str, feedback: Dict
    ) -> Capability:
        """
        优化能力

        Args:
            capability_id: 能力ID
            feedback: 反馈信息

        Returns:
            优化后的能力
        """
        if capability_id not in self._capabilities:
            return None

        capability = self._capabilities[capability_id]

        improvements = feedback.get("improvements", [])

        for improvement in improvements:
            if improvement.get("type") == "parameter_tuning":
                self._tune_parameters(capability, improvement.get("parameters", {}))

            elif improvement.get("type") == "logic_refinement":
                self._refine_logic(capability, improvement.get("changes", []))

        capability.last_updated = datetime.now().isoformat()

        return capability

    def _tune_parameters(self, capability: Capability, parameters: Dict):
        """调整参数"""
        if "performance_metrics" not in capability.metadata:
            capability.metadata["performance_metrics"] = {}

        for param, value in parameters.items():
            capability.metadata["performance_metrics"][param] = value

    def _refine_logic(self, capability: Capability, changes: List[Dict]):
        """优化逻辑"""
        for change in changes:
            old_code = change.get("old_code", "")
            new_code = change.get("new_code", "")

            if old_code in capability.implementation_code:
                capability.implementation_code = capability.implementation_code.replace(
                    old_code, new_code
                )

    async def register_capability(
        self, capability_id: str, registry_name: Optional[str] = None
    ) -> bool:
        """
        注册能力

        Args:
            capability_id: 能力ID
            registry_name: 注册名称

        Returns:
            是否成功
        """
        if capability_id not in self._capabilities:
            return False

        capability = self._capabilities[capability_id]

        name = registry_name or capability.name

        self._capability_registry[name] = capability

        return True

    def get_capability(self, name: str) -> Optional[Capability]:
        """获取能力"""
        return self._capability_registry.get(name)

    def list_capabilities(
        self, category: Optional[str] = None, active_only: bool = True
    ) -> List[Capability]:
        """列出能力"""
        capabilities = list(self._capability_registry.values())

        if category:
            capabilities = [c for c in capabilities if c.category == category]

        if active_only:
            capabilities = [c for c in capabilities if c.is_active]

        return capabilities

    async def execute_capability(self, name: str, input_data: Dict) -> Optional[Dict]:
        """
        执行能力

        Args:
            name: 能力名称
            input_data: 输入数据

        Returns:
            执行结果
        """
        capability = self._capability_registry.get(name)

        if not capability:
            return None

        if not capability.is_active:
            return None

        capability.usage_count += 1
        capability.last_used = datetime.now().isoformat()

        try:
            result = await self._execute_capability(capability, input_data)

            if result.get("status") == "completed":
                capability.success_count += 1
            else:
                capability.failure_count += 1

            return result

        except Exception as e:
            capability.failure_count += 1
            return {"error": str(e)}

    def get_capability_statistics(self) -> Dict:
        """获取能力统计"""
        capabilities = list(self._capability_registry.values())

        active = sum(1 for c in capabilities if c.is_active)
        verified = sum(1 for c in capabilities if c.is_verified)

        total_usage = sum(c.usage_count for c in capabilities)

        if capabilities:
            avg_success_rate = sum(
                c.success_count / max(c.usage_count, 1) for c in capabilities
            ) / len(capabilities)
        else:
            avg_success_rate = 0

        categories = {}
        for c in capabilities:
            if c.category not in categories:
                categories[c.category] = 0
            categories[c.category] += 1

        return {
            "total_capabilities": len(capabilities),
            "active_capabilities": active,
            "verified_capabilities": verified,
            "total_usage": total_usage,
            "avg_success_rate": avg_success_rate,
            "by_category": categories,
        }

    def export_capability_registry(self) -> Dict:
        """导出能力注册表"""
        return {
            "capabilities": {
                name: {
                    "id": cap.id,
                    "name": cap.name,
                    "description": cap.description,
                    "category": cap.category,
                    "is_active": cap.is_active,
                    "is_verified": cap.is_verified,
                    "usage_count": cap.usage_count,
                }
                for name, cap in self._capability_registry.items()
            },
            "exported_at": datetime.now().isoformat(),
        }
