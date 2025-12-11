"""
决策融合模块
融合规则决策和模型决策
"""
from typing import Dict, Optional
from dataclasses import dataclass
try:
    from .model_decision import ModelDecisionMaker, DecisionResult
    from .enhanced_rule_engine import EnhancedRuleEngine
except ImportError:
    from model_decision import ModelDecisionMaker, DecisionResult
    from enhanced_rule_engine import EnhancedRuleEngine


@dataclass
class FusedDecision:
    """融合后的决策"""
    final_decision: str  # 'approve', 'reject', 'manual_review'
    confidence: float
    model_decision: DecisionResult
    rule_adjustments: Dict
    fusion_reasoning: str
    default_probability: float
    expected_profit: float


class DecisionFusion:
    """决策融合器"""
    
    def __init__(self, model_decision_maker: ModelDecisionMaker,
                 rule_engine: EnhancedRuleEngine):
        """
        初始化决策融合器
        
        Args:
            model_decision_maker: 模型决策器
            rule_engine: 规则引擎
        """
        self.model_decision_maker = model_decision_maker
        self.rule_engine = rule_engine
        self.fusion_strategy = 'weighted'  # 'weighted', 'priority', 'consensus'
    
    def fuse_decisions(self, customer: Dict, loan: Dict, market: Dict,
                      model_weight: float = 0.7,
                      rule_weight: float = 0.3) -> FusedDecision:
        """
        融合模型决策和规则决策
        
        Args:
            customer: 客户数据
            loan: 贷款数据
            market: 市场数据
            model_weight: 模型权重
            rule_weight: 规则权重
        
        Returns:
            融合后的决策
        """
        # 1. 获取模型决策
        model_result = self.model_decision_maker.make_decision(
            customer, loan, market
        )
        
        # 2. 应用规则
        rule_result = self.rule_engine.apply_rules_to_customer(
            customer, loan, market
        )
        
        # 3. 融合决策
        final_decision = self._fuse_decision_logic(
            model_result, rule_result, model_weight, rule_weight
        )
        
        # 4. 调整违约概率和利润（基于规则惩罚）
        adjusted_default_prob = model_result.default_probability
        adjusted_profit = model_result.expected_profit
        
        if rule_result['total_penalty'] < 0:
            # 规则惩罚增加违约概率
            penalty_factor = abs(rule_result['total_penalty']) / 10
            adjusted_default_prob = min(1.0, adjusted_default_prob + penalty_factor)
            adjusted_profit += rule_result['total_penalty'] * 1000
        
        # 5. 应用规则调整
        if 'force_approve' in rule_result['adjustments']:
            final_decision = 'approve'
        elif 'force_reject' in rule_result['adjustments']:
            final_decision = 'reject'
        
        # 6. 生成融合理由
        reasoning = self._generate_fusion_reasoning(
            model_result, rule_result, final_decision
        )
        
        return FusedDecision(
            final_decision=final_decision,
            confidence=model_result.confidence,
            model_decision=model_result,
            rule_adjustments=rule_result['adjustments'],
            fusion_reasoning=reasoning,
            default_probability=adjusted_default_prob,
            expected_profit=adjusted_profit
        )
    
    def _fuse_decision_logic(self, model_result: DecisionResult,
                           rule_result: Dict,
                           model_weight: float,
                           rule_weight: float) -> str:
        """融合决策逻辑"""
        model_decision = model_result.decision
        triggered_rules = rule_result['triggered_rules']
        
        # 检查规则强制决策
        for rule in triggered_rules:
            adjustments = rule.get('adjustments', {})
            if adjustments.get('force_approve'):
                return 'approve'
            if adjustments.get('force_reject'):
                return 'reject'
        
        # 加权融合
        if self.fusion_strategy == 'weighted':
            # 模型决策权重
            if model_decision == 'approve':
                model_score = model_weight
            elif model_decision == 'manual_review':
                model_score = model_weight * 0.5
            else:
                model_score = -model_weight
            
            # 规则影响（触发规则越多，越倾向于拒绝）
            rule_score = rule_weight * (1 - len(triggered_rules) / 10)
            
            total_score = model_score + rule_score
            
            if total_score > 0.5:
                return 'approve'
            elif total_score > 0:
                return 'manual_review'
            else:
                return 'reject'
        
        # 优先级策略
        elif self.fusion_strategy == 'priority':
            # 规则优先级高于模型
            if len(triggered_rules) > 0:
                max_priority_rule = max(triggered_rules, key=lambda r: r.get('priority', 0))
                adjustments = max_priority_rule.get('adjustments', {})
                if adjustments.get('force_approve'):
                    return 'approve'
                if adjustments.get('force_reject'):
                    return 'reject'
            
            # 否则使用模型决策
            return model_decision
        
        # 共识策略
        else:  # consensus
            # 如果模型和规则都同意，采用该决策
            # 否则需要人工审核
            rule_suggests_approve = any(
                r.get('adjustments', {}).get('force_approve', False)
                for r in triggered_rules
            )
            rule_suggests_reject = any(
                r.get('adjustments', {}).get('force_reject', False)
                for r in triggered_rules
            )
            
            if model_decision == 'approve' and rule_suggests_approve:
                return 'approve'
            elif model_decision == 'reject' and rule_suggests_reject:
                return 'reject'
            else:
                return 'manual_review'
    
    def _generate_fusion_reasoning(self, model_result: DecisionResult,
                                  rule_result: Dict,
                                  final_decision: str) -> str:
        """生成融合理由"""
        reasoning_parts = []
        
        # 模型决策理由
        reasoning_parts.append(f"模型决策: {model_result.decision} ({model_result.reasoning})")
        
        # 规则影响
        if rule_result['triggered_count'] > 0:
            reasoning_parts.append(
                f"触发 {rule_result['triggered_count']} 条规则"
            )
            for rule in rule_result['triggered_rules'][:3]:  # 只显示前3条
                reasoning_parts.append(f"  - {rule.get('rule_name', '未知规则')}")
        
        # 最终决策
        if final_decision != model_result.decision:
            reasoning_parts.append(
                f"融合后决策调整为: {final_decision}"
            )
        
        return " | ".join(reasoning_parts)


def main():
    """主函数：测试决策融合"""
    print("=" * 80)
    print("决策融合模块测试")
    print("=" * 80)
    
    # 创建组件
    model_decision_maker = ModelDecisionMaker()
    rule_engine = EnhancedRuleEngine()
    rule_engine.load_rules_from_file(
        'data/historical/extracted_rules.json',
        'data/historical/quantified_rules.json'
    )
    
    # 创建融合器
    fusion = DecisionFusion(model_decision_maker, rule_engine)
    
    # 测试用例
    customer = {
        'customer_type': 'personal',
        'age': 32,
        'monthly_income': 10000,
        'credit_score': 680,
        'debt_ratio': 0.4,
        'years_in_job': 5
    }
    
    loan = {
        'loan_amount': 80000,
        'approved_rate': 0.09,
        'approved_term_months': 24
    }
    
    market = {
        'gdp_growth': 0.03,
        'base_interest_rate': 0.05,
        'unemployment_rate': 0.05,
        'inflation_rate': 0.02,
        'credit_spread': 0.02
    }
    
    # 融合决策
    fused_result = fusion.fuse_decisions(customer, loan, market)
    
    print(f"\n融合决策结果:")
    print(f"最终决策: {fused_result.final_decision}")
    print(f"置信度: {fused_result.confidence:.2%}")
    print(f"违约概率: {fused_result.default_probability:.2%}")
    print(f"预期利润: ¥{fused_result.expected_profit:,.2f}")
    print(f"\n融合理由:")
    print(f"  {fused_result.fusion_reasoning}")
    print(f"\n规则调整:")
    print(f"  {fused_result.rule_adjustments}")
    
    return fusion


if __name__ == '__main__':
    main()

