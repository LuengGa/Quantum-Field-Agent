"""
预定义协作策略库
================

基于协作实验发现的模式，定义有效的协作策略：
1. 问题解决策略
2. 知识解释策略
3. 代码开发策略
4. 学习辅导策略
5. 创意生成策略
"""

from dataclasses import asdict
from evolution.strategy_evolver import StrategyVariant, StrategyEvolver


def get_predefined_strategies() -> list:
    """获取预定义策略"""

    strategies = []

    # 策略1: 苏格拉底式提问
    strategies.append(
        StrategyVariant(
            id="strategy_socratic_questioning",
            name="苏格拉底式提问",
            strategy_type="questioning",
            conditions={
                "interaction_type": "question",
                "user_expertise": "beginner",
                "topic_complexity": "high",
            },
            actions=[
                {"action": "ask_clarifying_question", "priority": 1},
                {"action": "provide_analogy", "priority": 2},
                {"action": "check_understanding", "priority": 3},
            ],
            parameters={
                "question_depth": 3,
                "analogy_count": 2,
                "comprehension_checks": True,
            },
            avg_effectiveness=0.85,
            confidence=0.8,
            tags=["questioning", "beginner", "educational"],
        )
    )

    # 策略2: 专家式解答
    strategies.append(
        StrategyVariant(
            id="strategy_expert_answer",
            name="专家式解答",
            strategy_type="explanation",
            conditions={
                "interaction_type": "question",
                "user_expertise": "expert",
                "topic_complexity": "high",
            },
            actions=[
                {"action": "direct_answer", "priority": 1},
                {"action": "provide_technical_details", "priority": 2},
                {"action": "cite_references", "priority": 3},
            ],
            parameters={
                "technical_level": "advanced",
                "include_references": True,
                "depth": "comprehensive",
            },
            avg_effectiveness=0.88,
            confidence=0.85,
            tags=["explanation", "expert", "technical"],
        )
    )

    # 策略3: 代码生成
    strategies.append(
        StrategyVariant(
            id="strategy_code_generation",
            name="代码生成",
            strategy_type="code",
            conditions={"interaction_type": "code_request", "task_type": "generation"},
            actions=[
                {"action": "generate_skeleton", "priority": 1},
                {"action": "fill_implementation", "priority": 2},
                {"action": "add_comments", "priority": 3},
                {"action": "provide_usage_example", "priority": 4},
            ],
            parameters={
                "include_documentation": True,
                "error_handling": True,
                "test_cases": True,
                "coding_style": "pep8",
            },
            avg_effectiveness=0.82,
            confidence=0.75,
            tags=["code", "programming", "development"],
        )
    )

    # 策略4: 代码调试
    strategies.append(
        StrategyVariant(
            id="strategy_code_debugging",
            name="代码调试",
            strategy_type="code",
            conditions={"interaction_type": "debug_request", "task_type": "debugging"},
            actions=[
                {"action": "analyze_error", "priority": 1},
                {"action": "identify_root_cause", "priority": 2},
                {"action": "propose_fix", "priority": 3},
                {"action": "explain_fix", "priority": 4},
            ],
            parameters={
                "include_test": True,
                "explain_reason": True,
                "prevent_recurrence": True,
            },
            avg_effectiveness=0.80,
            confidence=0.78,
            tags=["code", "debugging", "problem_solving"],
        )
    )

    # 策略5: 渐进式学习
    strategies.append(
        StrategyVariant(
            id="strategy_progressive_learning",
            name="渐进式学习",
            strategy_type="learning",
            conditions={
                "interaction_type": "learning_path",
                "user_expertise": "beginner",
            },
            actions=[
                {"action": "assess_current_level", "priority": 1},
                {"action": "outline_prerequisites", "priority": 2},
                {"action": "suggest_learning_order", "priority": 3},
                {"action": "provide_milestones", "priority": 4},
            ],
            parameters={
                "pace": "self_paced",
                "include_practice": True,
                "milestone_count": 5,
                "resource_types": ["books", "courses", "practice"],
            },
            avg_effectiveness=0.90,
            confidence=0.82,
            tags=["learning", "education", "beginner"],
        )
    )

    # 策略6: 概念对比
    strategies.append(
        StrategyVariant(
            id="strategy_concept_comparison",
            name="概念对比",
            strategy_type="analysis",
            conditions={"interaction_type": "comparison", "topic_type": "concept"},
            actions=[
                {"action": "define_both_concepts", "priority": 1},
                {"action": "highlight_differences", "priority": 2},
                {"action": "highlight_similarities", "priority": 3},
                {"action": "use_examples", "priority": 4},
                {"action": "summarize_comparison", "priority": 5},
            ],
            parameters={
                "include_table": True,
                "example_count": 2,
                "summary_length": "concise",
            },
            avg_effectiveness=0.87,
            confidence=0.80,
            tags=["analysis", "comparison", "concepts"],
        )
    )

    # 策略7: 创意头脑风暴
    strategies.append(
        StrategyVariant(
            id="strategy_creative_brainstorm",
            name="创意头脑风暴",
            strategy_type="creative",
            conditions={"interaction_type": "brainstorm", "creativity_level": "high"},
            actions=[
                {"action": "generate_ideas", "priority": 1},
                {"action": "expand_on_ideas", "priority": 2},
                {"action": "combine_ideas", "priority": 3},
                {"action": "evaluate_feasibility", "priority": 4},
            ],
            parameters={
                "idea_count": 5,
                "divergence_level": "high",
                "convergence_after": True,
            },
            avg_effectiveness=0.85,
            confidence=0.75,
            tags=["creative", "brainstorm", "innovation"],
        )
    )

    # 策略8: 资源推荐
    strategies.append(
        StrategyVariant(
            id="strategy_resource_recommendation",
            name="资源推荐",
            strategy_type="learning",
            conditions={"interaction_type": "resource_request", "topic_type": "any"},
            actions=[
                {"action": "categorize_resources", "priority": 1},
                {"action": "rank_by_quality", "priority": 2},
                {"action": "match_to_level", "priority": 3},
                {"action": "provide_access_info", "priority": 4},
            ],
            parameters={
                "resource_count": 5,
                "include_free": True,
                "level_matching": True,
                "diversity": True,
            },
            avg_effectiveness=0.88,
            confidence=0.82,
            tags=["resources", "recommendation", "learning"],
        )
    )

    # 策略9: 异常分析
    strategies.append(
        StrategyVariant(
            id="strategy_anomaly_analysis",
            name="异常分析",
            strategy_type="analysis",
            conditions={
                "interaction_type": "anomaly",
                "analysis_type": "investigation",
            },
            actions=[
                {"action": "characterize_anomaly", "priority": 1},
                {"action": "search_patterns", "priority": 2},
                {"action": "hypothesize_causes", "priority": 3},
                {"action": "prioritize_causes", "priority": 4},
                {"action": "recommend_actions", "priority": 5},
            ],
            parameters={
                "hypothesis_count": 3,
                "evidence_weighting": True,
                "include_probabilities": True,
            },
            avg_effectiveness=0.78,
            confidence=0.70,
            tags=["analysis", "anomaly", "investigation"],
        )
    )

    # 策略10: 协作对话
    strategies.append(
        StrategyVariant(
            id="strategy_collaborative_dialogue",
            name="协作对话",
            strategy_type="dialogue",
            conditions={
                "interaction_type": "collaborative",
                "dialogue_type": "interactive",
            },
            actions=[
                {"action": "acknowledge_input", "priority": 1},
                {"action": "build_on_idea", "priority": 2},
                {"action": "offer_perspective", "priority": 3},
                {"action": "invite_expansion", "priority": 4},
            ],
            parameters={
                "response_length": "moderate",
                "question_frequency": "moderate",
                "affirmation_style": "explicit",
            },
            avg_effectiveness=0.86,
            confidence=0.78,
            tags=["dialogue", "collaboration", "interactive"],
        )
    )

    return strategies


def get_strategy_by_context(context: dict) -> StrategyVariant:
    """根据上下文获取最佳策略"""

    strategies = get_predefined_strategies()

    best_match = None
    best_score = 0

    for strategy in strategies:
        score = calculate_context_match(strategy, context)
        if score > best_score:
            best_score = score
            best_match = strategy

    return best_match or strategies[0]


def calculate_context_match(strategy: StrategyVariant, context: dict) -> float:
    """计算策略与上下文的匹配度"""

    if not strategy.conditions:
        return 0.5

    matches = 0
    total = 0

    for key, expected in strategy.conditions.items():
        total += 1
        actual = context.get(key)

        if isinstance(expected, list):
            if actual in expected:
                matches += 1
        elif actual == expected:
            matches += 1

    return matches / total if total > 0 else 0


def initialize_strategy_evolvers(engine) -> int:
    """初始化策略进化器中的预定义策略"""

    strategies = get_predefined_strategies()

    for strategy in strategies:
        if strategy.id not in engine.strategy_evolver._active_variants:
            engine.strategy_evolver._active_variants[strategy.id] = strategy
            engine.strategy_evolver._strategy_pool[strategy.strategy_type].append(
                strategy.id
            )
            engine.db.save_strategy(asdict(strategy))

    return len(strategies)
