"""
增强版规则引擎
实现规则应用、参数调整、冲突处理
集成量化后的规则和业务规则
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class RuleApplicationResult:
    """规则应用结果"""
    rule_id: str
    rule_name: str
    triggered: bool
    adjustments: Dict
    penalty: float
    priority: int


class EnhancedRuleEngine:
    """增强版规则引擎"""
    
    def __init__(self, rules_config: Optional[List[Dict]] = None,
                 quantified_rules: Optional[List] = None):
        """
        初始化规则引擎
        
        Args:
            rules_config: 业务规则配置（来自rule_engine.py的格式）
            quantified_rules: 量化后的规则（来自rule_quantifier.py）
        """
        self.rules_config = rules_config or []
        self.quantified_rules = quantified_rules or []
        self.applied_rules = []
        self.conflict_resolution_strategy = 'priority'  # 'priority', 'weight', 'first'
    
    def load_rules_from_file(self, rules_path: str, quantified_path: Optional[str] = None):
        """从文件加载规则"""
        # 加载业务规则
        if os.path.exists(rules_path):
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules_config = json.load(f)
            print(f"✅ 已加载 {len(self.rules_config)} 条业务规则")
        
        # 加载量化规则
        if quantified_path and os.path.exists(quantified_path):
            with open(quantified_path, 'r', encoding='utf-8') as f:
                quantified_data = json.load(f)
                # 注意：量化规则中的函数无法从JSON恢复，需要重新量化
                self.quantified_rules = quantified_data
            print(f"✅ 已加载 {len(self.quantified_rules)} 条量化规则元数据")
    
    def check_rule_condition(self, rule: Dict, customer: Dict, loan: Dict, 
                            market: Dict) -> bool:
        """
        检查规则条件是否满足
        
        Args:
            rule: 规则配置
            customer: 客户数据
            loan: 贷款数据
            market: 市场数据
        
        Returns:
            是否满足条件
        """
        conditions = rule.get('conditions', [])
        if not conditions:
            return True
        
        # 合并所有数据
        state = {**customer, **loan, **market}
        
        # 检查每个条件
        for condition in conditions:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if field not in state:
                return False
            
            field_value = state[field]
            
            # 根据操作符判断
            if operator == '>=':
                if field_value < value:
                    return False
            elif operator == '<=':
                if field_value > value:
                    return False
            elif operator == '>':
                if field_value <= value:
                    return False
            elif operator == '<':
                if field_value >= value:
                    return False
            elif operator == '==':
                if field_value != value:
                    return False
            elif operator == 'in':
                if field_value not in value:
                    return False
            elif operator == 'between':
                min_val, max_val = value
                if not (min_val <= field_value <= max_val):
                    return False
        
        return True
    
    def apply_rule_action(self, rule: Dict, customer: Dict, loan: Dict) -> Dict:
        """
        应用规则动作
        
        Args:
            rule: 规则配置
            customer: 客户数据
            loan: 贷款数据
        
        Returns:
            调整后的数据
        """
        action = rule.get('action', {})
        adjustments = {}
        
        # 贷款金额调整
        if 'loan_amount_multiplier' in action:
            multiplier = action['loan_amount_multiplier']
            adjustments['loan_amount'] = loan.get('loan_amount', 0) * multiplier
        
        # 利率调整
        if 'interest_rate_premium' in action:
            premium = action['interest_rate_premium']
            adjustments['rate_spread'] = loan.get('rate_spread', 0) + premium
        
        # 期限调整
        if 'term_months_delta' in action:
            delta = action['term_months_delta']
            adjustments['term_months'] = loan.get('term_months', 0) + delta
        
        # 强制审批/拒绝
        if action.get('force_approve', False):
            adjustments['force_approve'] = True
        if action.get('force_reject', False):
            adjustments['force_reject'] = True
        
        # 附加条件
        if 'require_collateral' in action and action['require_collateral']:
            adjustments['require_collateral'] = True
        if 'require_guarantor' in action and action['require_guarantor']:
            adjustments['require_guarantor'] = True
        
        return adjustments
    
    def calculate_rule_penalty(self, rule: Dict, customer: Dict, loan: Dict) -> float:
        """
        计算规则惩罚
        
        Args:
            rule: 规则配置
            customer: 客户数据
            loan: 贷款数据
        
        Returns:
            惩罚值
        """
        penalty_config = rule.get('penalty', {})
        if not penalty_config:
            return 0.0
        
        # 计算各项惩罚
        total_penalty = 0.0
        
        # 评分惩罚
        total_penalty += penalty_config.get('profit_score_delta', 0)
        total_penalty += penalty_config.get('risk_score_delta', 0)
        total_penalty += penalty_config.get('compliance_score_delta', 0)
        
        # 违约概率调整
        if 'default_probability_multiplier' in penalty_config:
            multiplier = penalty_config['default_probability_multiplier']
            # 这里简化处理，实际应该影响违约概率
            total_penalty += (multiplier - 1) * 0.1
        
        return total_penalty
    
    def apply_rules_to_customer(self, customer: Dict, loan: Dict, 
                                market: Dict) -> Dict:
        """
        对单个客户应用所有规则
        
        Args:
            customer: 客户数据
            loan: 贷款数据
            market: 市场数据
        
        Returns:
            规则应用结果
        """
        triggered_rules = []
        all_adjustments = {}
        total_penalty = 0.0
        
        # 按优先级排序规则
        sorted_rules = sorted(
            self.rules_config,
            key=lambda r: r.get('priority', 0),
            reverse=True
        )
        
        # 应用每条规则
        for rule in sorted_rules:
            if not rule.get('enabled', True):
                continue
            
            # 检查客户类型
            customer_type = customer.get('customer_type', '')
            rule_customer_type = rule.get('customer_type', 'both')
            if rule_customer_type != 'both' and rule_customer_type != customer_type:
                continue
            
            # 检查条件
            if self.check_rule_condition(rule, customer, loan, market):
                # 应用动作
                adjustments = self.apply_rule_action(rule, customer, loan)
                
                # 计算惩罚
                penalty = self.calculate_rule_penalty(rule, customer, loan)
                
                # 处理冲突（如果规则冲突，使用优先级策略）
                if self._has_conflict(all_adjustments, adjustments):
                    if self.conflict_resolution_strategy == 'priority':
                        # 优先级高的规则覆盖优先级低的
                        rule_priority = rule.get('priority', 0)
                        existing_priority = max([
                            r.get('priority', 0) for r in triggered_rules
                        ], default=0)
                        
                        if rule_priority > existing_priority:
                            # 新规则优先级更高，应用新规则
                            all_adjustments.update(adjustments)
                            total_penalty += penalty
                            triggered_rules.append({
                                'rule_id': rule.get('rule_id', ''),
                                'rule_name': rule.get('name', ''),
                                'priority': rule_priority,
                                'adjustments': adjustments,
                                'penalty': penalty
                            })
                    elif self.conflict_resolution_strategy == 'weight':
                        # 根据权重合并
                        rule_weight = rule.get('weight', 1.0)
                        for key, value in adjustments.items():
                            if key in all_adjustments:
                                # 加权平均
                                all_adjustments[key] = (
                                    all_adjustments[key] * 0.5 + value * 0.5
                                )
                            else:
                                all_adjustments[key] = value
                        total_penalty += penalty * rule_weight
                        triggered_rules.append({
                            'rule_id': rule.get('rule_id', ''),
                            'rule_name': rule.get('name', ''),
                            'priority': rule.get('priority', 0),
                            'adjustments': adjustments,
                            'penalty': penalty
                        })
                else:
                    # 无冲突，直接应用
                    all_adjustments.update(adjustments)
                    total_penalty += penalty
                    triggered_rules.append({
                        'rule_id': rule.get('rule_id', ''),
                        'rule_name': rule.get('name', ''),
                        'priority': rule.get('priority', 0),
                        'adjustments': adjustments,
                        'penalty': penalty
                    })
        
        return {
            'triggered_rules': triggered_rules,
            'adjustments': all_adjustments,
            'total_penalty': total_penalty,
            'triggered_count': len(triggered_rules)
        }
    
    def _has_conflict(self, existing_adjustments: Dict, new_adjustments: Dict) -> bool:
        """检查是否有冲突"""
        # 检查关键字段冲突
        conflict_fields = ['force_approve', 'force_reject', 'loan_amount', 'rate_spread']
        
        for field in conflict_fields:
            if field in existing_adjustments and field in new_adjustments:
                if existing_adjustments[field] != new_adjustments[field]:
                    return True
        
        return False
    
    def adjust_rule_parameters(self, rule_id: str, new_params: Dict):
        """
        调整规则参数
        
        Args:
            rule_id: 规则ID
            new_params: 新参数
        """
        for rule in self.rules_config:
            if rule.get('rule_id') == rule_id or rule.get('name') == rule_id:
                # 更新参数
                if 'action' in new_params:
                    rule['action'].update(new_params['action'])
                if 'penalty' in new_params:
                    rule['penalty'].update(new_params['penalty'])
                if 'priority' in new_params:
                    rule['priority'] = new_params['priority']
                print(f"✅ 已更新规则: {rule_id}")
                return
        
        print(f"❌ 未找到规则: {rule_id}")
    
    def get_rule_statistics(self) -> Dict:
        """获取规则统计信息"""
        return {
            'total_rules': len(self.rules_config),
            'enabled_rules': sum(1 for r in self.rules_config if r.get('enabled', True)),
            'quantified_rules': len(self.quantified_rules),
            'conflict_resolution': self.conflict_resolution_strategy
        }


def main():
    """主函数：测试增强版规则引擎"""
    print("=" * 80)
    print("增强版规则引擎测试")
    print("=" * 80)
    
    # 创建规则引擎
    engine = EnhancedRuleEngine()
    
    # 加载规则
    rules_path = 'data/historical/extracted_rules.json'
    quantified_path = 'data/historical/quantified_rules.json'
    
    if os.path.exists(rules_path):
        engine.load_rules_from_file(rules_path, quantified_path)
    else:
        print("⚠️  规则文件不存在，使用空规则引擎")
    
    # 测试客户
    test_customer = {
        'customer_id': 'TEST001',
        'customer_type': 'personal',
        'age': 32,
        'monthly_income': 8000,
        'credit_score': 650,
        'debt_ratio': 0.4
    }
    
    test_loan = {
        'loan_amount': 50000,
        'rate_spread': 0.01,
        'term_months': 24
    }
    
    test_market = {
        'gdp_growth': 0.03,
        'base_interest_rate': 0.05,
        'unemployment_rate': 0.05
    }
    
    # 应用规则
    result = engine.apply_rules_to_customer(test_customer, test_loan, test_market)
    
    print(f"\n规则应用结果:")
    print(f"触发规则数: {result['triggered_count']}")
    print(f"总惩罚: {result['total_penalty']:.4f}")
    print(f"调整项: {result['adjustments']}")
    
    if result['triggered_rules']:
        print(f"\n触发的规则:")
        for rule in result['triggered_rules']:
            print(f"  - {rule['rule_name']} (优先级: {rule['priority']})")
    
    # 统计信息
    stats = engine.get_rule_statistics()
    print(f"\n规则统计:")
    print(f"  总规则数: {stats['total_rules']}")
    print(f"  启用规则数: {stats['enabled_rules']}")
    print(f"  量化规则数: {stats['quantified_rules']}")
    
    return engine


if __name__ == '__main__':
    main()

