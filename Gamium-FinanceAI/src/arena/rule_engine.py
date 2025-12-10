"""
规则引擎模块
支持条件-动作-惩罚配置，用于贷款审批决策
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConditionOperator(str, Enum):
    """条件操作符"""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NE = "!="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


@dataclass
class RuleCondition:
    """规则条件"""
    field: str  # 字段名，如 monthly_income, age, debt_ratio
    operator: ConditionOperator  # 操作符
    value: Any  # 比较值
    logical_op: Optional[str] = None  # 逻辑运算符，用于组合多个条件 (AND/OR)


@dataclass
class RuleAction:
    """规则动作"""
    approval_threshold_delta: float = 0.0  # 审批阈值调整
    rate_spread_delta: float = 0.0  # 利差调整
    loan_amount_multiplier: float = 1.0  # 贷款金额倍数
    term_months_delta: int = 0  # 期限调整（月）
    force_approve: bool = False  # 强制通过
    force_reject: bool = False  # 强制拒绝
    require_collateral: bool = False  # 要求抵押
    require_guarantor: bool = False  # 要求担保


@dataclass
class RulePenalty:
    """规则惩罚（用于评分）"""
    score_delta: float = 0.0  # 评分调整
    risk_multiplier: float = 1.0  # 风险倍数
    profit_discount: float = 1.0  # 利润折扣


@dataclass
class Rule:
    """完整规则定义"""
    name: str
    description: str
    priority: int = 0  # 优先级，数字越大优先级越高
    enabled: bool = True
    conditions: List[RuleCondition]
    action: RuleAction
    penalty: RulePenalty = None
    category: str = "default"  # 规则类别：risk_control, profit_optimization, compliance等


class RuleEngine:
    """规则引擎"""
    
    def __init__(self, rules: List[Dict[str, Any]] = None):
        """
        初始化规则引擎
        
        Args:
            rules: 规则列表，格式为字典列表，每个字典包含name, conditions, action, penalty等
        """
        self.rules: List[Rule] = []
        if rules:
            self.load_rules(rules)
    
    def load_rules(self, rules: List[Dict[str, Any]]):
        """从字典列表加载规则"""
        self.rules = []
        for rule_dict in rules:
            rule = self._parse_rule(rule_dict)
            if rule:
                self.rules.append(rule)
        # 按优先级排序
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def _parse_rule(self, rule_dict: Dict[str, Any]) -> Optional[Rule]:
        """解析单个规则字典"""
        try:
            name = rule_dict.get('name', '未命名规则')
            description = rule_dict.get('description', '')
            priority = rule_dict.get('priority', 0)
            enabled = rule_dict.get('enabled', True)
            category = rule_dict.get('category', 'default')
            
            # 解析条件
            conditions = []
            cond_list = rule_dict.get('conditions', [])
            if isinstance(cond_list, dict):
                cond_list = [cond_list]  # 单个条件转为列表
            
            for cond_dict in cond_list:
                field = cond_dict.get('field')
                op_str = cond_dict.get('op') or cond_dict.get('operator', '==')
                value = cond_dict.get('value')
                logical_op = cond_dict.get('logical_op')
                
                if not field:
                    continue
                
                try:
                    operator = ConditionOperator(op_str)
                except ValueError:
                    continue
                
                conditions.append(RuleCondition(
                    field=field,
                    operator=operator,
                    value=value,
                    logical_op=logical_op
                ))
            
            if not conditions:
                return None
            
            # 解析动作
            action_dict = rule_dict.get('action', {})
            action = RuleAction(
                approval_threshold_delta=float(action_dict.get('approval_threshold_delta', 0.0)),
                rate_spread_delta=float(action_dict.get('rate_spread_delta', 0.0)),
                loan_amount_multiplier=float(action_dict.get('loan_amount_multiplier', 1.0)),
                term_months_delta=int(action_dict.get('term_months_delta', 0)),
                force_approve=bool(action_dict.get('force_approve', False)),
                force_reject=bool(action_dict.get('force_reject', False)),
                require_collateral=bool(action_dict.get('require_collateral', False)),
                require_guarantor=bool(action_dict.get('require_guarantor', False)),
            )
            
            # 解析惩罚
            penalty_dict = rule_dict.get('penalty', {})
            penalty = RulePenalty(
                score_delta=float(penalty_dict.get('score_delta', 0.0)),
                risk_multiplier=float(penalty_dict.get('risk_multiplier', 1.0)),
                profit_discount=float(penalty_dict.get('profit_discount', 1.0)),
            )
            
            return Rule(
                name=name,
                description=description,
                priority=priority,
                enabled=enabled,
                conditions=conditions,
                action=action,
                penalty=penalty,
                category=category
            )
        except Exception as e:
            print(f"解析规则失败: {e}")
            return None
    
    def evaluate_condition(self, condition: RuleCondition, customer: Any) -> bool:
        """
        评估单个条件
        
        Args:
            condition: 规则条件
            customer: 客户对象（CustomerProfile）
        
        Returns:
            条件是否满足
        """
        try:
            # 获取字段值
            field_value = getattr(customer, condition.field, None)
            if field_value is None:
                # 尝试从字典获取
                if hasattr(customer, '__dict__'):
                    field_value = customer.__dict__.get(condition.field)
                if field_value is None:
                    return False
            
            op = condition.operator
            target_value = condition.value
            
            # 执行比较
            if op == ConditionOperator.GT:
                return field_value > target_value
            elif op == ConditionOperator.GTE:
                return field_value >= target_value
            elif op == ConditionOperator.LT:
                return field_value < target_value
            elif op == ConditionOperator.LTE:
                return field_value <= target_value
            elif op == ConditionOperator.EQ:
                return field_value == target_value
            elif op == ConditionOperator.NE:
                return field_value != target_value
            elif op == ConditionOperator.IN:
                return field_value in target_value if isinstance(target_value, (list, tuple, set)) else False
            elif op == ConditionOperator.NOT_IN:
                return field_value not in target_value if isinstance(target_value, (list, tuple, set)) else True
            elif op == ConditionOperator.CONTAINS:
                return target_value in str(field_value) if field_value else False
            
            return False
        except Exception as e:
            print(f"评估条件失败: {condition.field} {condition.operator} {condition.value}, 错误: {e}")
            return False
    
    def evaluate_conditions(self, conditions: List[RuleCondition], customer: Any) -> bool:
        """
        评估多个条件（支持AND/OR逻辑）
        
        Args:
            conditions: 条件列表
            customer: 客户对象
        
        Returns:
            所有条件是否满足
        """
        if not conditions:
            return True
        
        results = []
        for cond in conditions:
            results.append(self.evaluate_condition(cond, customer))
        
        # 默认使用AND逻辑
        # 如果条件中有logical_op，可以扩展支持OR
        return all(results)
    
    def apply_rule(self, rule: Rule, customer: Any, 
                   base_threshold: float, base_spread: float,
                   base_loan_amount: float, base_term_months: int) -> Tuple[Dict[str, Any], List[str]]:
        """
        应用规则，返回调整后的参数和触发的规则名称列表
        
        Args:
            rule: 规则对象
            customer: 客户对象
            base_threshold: 基础审批阈值
            base_spread: 基础利差
            base_loan_amount: 基础贷款金额
            base_term_months: 基础期限
        
        Returns:
            (调整后的参数字典, 触发的规则名称列表)
        """
        if not rule.enabled:
            return {}, []
        
        # 评估条件
        if not self.evaluate_conditions(rule.conditions, customer):
            return {}, []
        
        # 条件满足，应用动作
        adjustments = {
            'approval_threshold': base_threshold + rule.action.approval_threshold_delta,
            'rate_spread': base_spread + rule.action.rate_spread_delta,
            'loan_amount': base_loan_amount * rule.action.loan_amount_multiplier,
            'term_months': base_term_months + rule.action.term_months_delta,
            'force_approve': rule.action.force_approve,
            'force_reject': rule.action.force_reject,
            'require_collateral': rule.action.require_collateral,
            'require_guarantor': rule.action.require_guarantor,
            'penalty': rule.penalty,
            'rule_name': rule.name,
            'rule_category': rule.category
        }
        
        return adjustments, [rule.name]
    
    def process_customer(self, customer: Any,
                         base_threshold: float, base_spread: float,
                         base_loan_amount: float, base_term_months: int) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        """
        处理单个客户，应用所有匹配的规则
        
        Args:
            customer: 客户对象
            base_threshold: 基础审批阈值
            base_spread: 基础利差
            base_loan_amount: 基础贷款金额
            base_term_months: 基础期限
        
        Returns:
            (最终调整后的参数, 触发的规则名称列表, 评分调整信息)
        """
        final_adjustments = {
            'approval_threshold': base_threshold,
            'rate_spread': base_spread,
            'loan_amount': base_loan_amount,
            'term_months': base_term_months,
            'force_approve': False,
            'force_reject': False,
            'require_collateral': False,
            'require_guarantor': False,
        }
        
        triggered_rules = []
        score_adjustments = {
            'score_delta': 0.0,
            'risk_multiplier': 1.0,
            'profit_discount': 1.0
        }
        
        # 按优先级顺序应用规则
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            adjustments, rule_names = self.apply_rule(
                rule, customer,
                final_adjustments['approval_threshold'],
                final_adjustments['rate_spread'],
                final_adjustments['loan_amount'],
                final_adjustments['term_months']
            )
            
            if adjustments:
                triggered_rules.extend(rule_names)
                # 更新参数（累积调整）
                for key in ['approval_threshold', 'rate_spread', 'loan_amount', 'term_months']:
                    if key in adjustments:
                        final_adjustments[key] = adjustments[key]
                
                # 布尔标志（任一为True即为True）
                for key in ['force_approve', 'force_reject', 'require_collateral', 'require_guarantor']:
                    if key in adjustments and adjustments[key]:
                        final_adjustments[key] = True
                
                # 累积评分调整
                if 'penalty' in adjustments and adjustments['penalty']:
                    penalty = adjustments['penalty']
                    score_adjustments['score_delta'] += penalty.score_delta
                    score_adjustments['risk_multiplier'] *= penalty.risk_multiplier
                    score_adjustments['profit_discount'] *= penalty.profit_discount
        
        return final_adjustments, triggered_rules, score_adjustments

