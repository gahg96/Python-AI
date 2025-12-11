"""
规则量化模块
将提取的业务规则转化为可执行的约束函数
"""
import json
import os
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class QuantifiedRule:
    """量化后的规则"""
    rule_id: str
    rule_name: str
    rule_type: str
    customer_type: str
    description: str
    check_function: Callable  # 检查函数
    penalty_function: Callable  # 惩罚函数
    weight: float  # 规则权重
    priority: int  # 优先级
    
    def to_dict(self):
        """转换为字典（不包含函数）"""
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'rule_type': self.rule_type,
            'customer_type': self.customer_type,
            'description': self.description,
            'weight': self.weight,
            'priority': self.priority
        }


class RuleQuantifier:
    """规则量化器"""
    
    def __init__(self, rules: List[Dict]):
        """
        初始化规则量化器
        
        Args:
            rules: 提取的规则列表（字典格式）
        """
        self.rules = rules
        self.quantified_rules = []
    
    def quantify_rule(self, rule: Dict) -> QuantifiedRule:
        """
        量化单个规则
        
        Args:
            rule: 规则字典
        
        Returns:
            量化后的规则对象
        """
        rule_type = rule['rule_type']
        field = rule['field']
        operator = rule['operator']
        value = rule['value']
        customer_type = rule['customer_type']
        confidence = rule.get('confidence', 0.5)
        support = rule.get('support', 0.1)
        conditions = rule.get('conditions', [])
        
        # 根据规则类型创建检查函数和惩罚函数
        if rule_type == 'threshold':
            check_func, penalty_func = self._create_threshold_functions(
                field, operator, value, confidence
            )
        
        elif rule_type == 'range':
            check_func, penalty_func = self._create_range_functions(
                field, value, confidence
            )
        
        elif rule_type == 'ratio':
            check_func, penalty_func = self._create_ratio_functions(
                field, operator, value, confidence
            )
        
        elif rule_type == 'composite':
            check_func, penalty_func = self._create_composite_functions(
                conditions, confidence
            )
        
        else:
            # 默认函数
            def check_func(state):
                return True
            
            def penalty_func(state):
                return 0
            
            check_func, penalty_func = check_func, penalty_func
        
        # 计算权重
        weight = confidence * support
        
        return QuantifiedRule(
            rule_id=rule['rule_id'],
            rule_name=rule['rule_name'],
            rule_type=rule_type,
            customer_type=customer_type,
            description=rule.get('description', ''),
            check_function=check_func,
            penalty_function=penalty_func,
            weight=weight,
            priority=rule.get('priority', 1)
        )
    
    def _create_threshold_functions(self, field: str, operator: str, 
                                   value: float, confidence: float) -> Tuple[Callable, Callable]:
        """创建阈值规则的函数"""
        
        def check(state: Dict) -> bool:
            """检查函数"""
            field_value = state.get(field, None)
            if field_value is None:
                return False  # 字段不存在，视为不满足
            
            if operator == '>=':
                return field_value >= value
            elif operator == '<=':
                return field_value <= value
            elif operator == '>':
                return field_value > value
            elif operator == '<':
                return field_value < value
            elif operator == '==':
                return abs(field_value - value) < 1e-6
            else:
                return False
        
        def penalty(state: Dict) -> float:
            """惩罚函数"""
            if check(state):
                return 0.0  # 满足规则，无惩罚
            
            field_value = state.get(field, 0)
            
            if operator == '>=':
                diff = value - field_value
            elif operator == '<=':
                diff = field_value - value
            elif operator == '>':
                diff = value - field_value + 1e-6
            elif operator == '<':
                diff = field_value - value - 1e-6
            else:
                diff = abs(field_value - value)
            
            # 惩罚 = 置信度 × 差异 × 惩罚系数
            penalty_amount = confidence * abs(diff) * 10
            
            return -penalty_amount
        
        return check, penalty
    
    def _create_range_functions(self, field: str, value: tuple, 
                               confidence: float) -> Tuple[Callable, Callable]:
        """创建范围规则的函数"""
        min_val, max_val = value
        
        def check(state: Dict) -> bool:
            """检查函数"""
            field_value = state.get(field, None)
            if field_value is None:
                return False
            
            return min_val <= field_value <= max_val
        
        def penalty(state: Dict) -> float:
            """惩罚函数"""
            if check(state):
                return 0.0
            
            field_value = state.get(field, 0)
            
            if field_value < min_val:
                diff = min_val - field_value
            elif field_value > max_val:
                diff = field_value - max_val
            else:
                diff = 0
            
            penalty_amount = confidence * diff * 10
            return -penalty_amount
        
        return check, penalty
    
    def _create_ratio_functions(self, field: str, operator: str, 
                               value: float, confidence: float) -> Tuple[Callable, Callable]:
        """创建比例规则的函数"""
        # 解析字段（格式：field1/field2）
        if '/' in field:
            field1, field2 = field.split('/')
        else:
            field1, field2 = field, '1'
        
        def check(state: Dict) -> bool:
            """检查函数"""
            val1 = state.get(field1, 0)
            val2 = state.get(field2, 1)
            
            if val2 == 0:
                return False
            
            ratio = val1 / val2
            
            if operator == '<=':
                return ratio <= value
            elif operator == '>=':
                return ratio >= value
            elif operator == '<':
                return ratio < value
            elif operator == '>':
                return ratio > value
            else:
                return False
        
        def penalty(state: Dict) -> float:
            """惩罚函数"""
            if check(state):
                return 0.0
            
            val1 = state.get(field1, 0)
            val2 = state.get(field2, 1)
            
            if val2 == 0:
                return -confidence * 1000  # 严重惩罚
            
            ratio = val1 / val2
            
            if operator == '<=':
                diff = ratio - value
            elif operator == '>=':
                diff = value - ratio
            else:
                diff = abs(ratio - value)
            
            penalty_amount = confidence * diff * 1000
            return -penalty_amount
        
        return check, penalty
    
    def _create_composite_functions(self, conditions: List[Dict], 
                                    confidence: float) -> Tuple[Callable, Callable]:
        """创建复合规则的函数"""
        
        def check(state: Dict) -> bool:
            """检查函数"""
            for cond in conditions:
                field = cond[0]
                op = cond[1]
                val = cond[2]
                
                field_value = state.get(field, None)
                if field_value is None:
                    return False
                
                if op == '<=':
                    if field_value > val:
                        return False
                elif op == '>':
                    if field_value <= val:
                        return False
                elif op == '<':
                    if field_value >= val:
                        return False
                elif op == '>=':
                    if field_value < val:
                        return False
                elif op == '==':
                    if abs(field_value - val) > 1e-6:
                        return False
            
            return True
        
        def penalty(state: Dict) -> float:
            """惩罚函数"""
            if check(state):
                return 0.0
            
            # 计算违反条件的严重程度
            total_violation = 0
            for cond in conditions:
                field = cond[0]
                op = cond[1]
                val = cond[2]
                
                field_value = state.get(field, 0)
                
                if op == '<=' and field_value > val:
                    total_violation += (field_value - val) / (val + 1e-6)
                elif op == '>' and field_value <= val:
                    total_violation += (val - field_value) / (val + 1e-6)
                elif op == '<' and field_value >= val:
                    total_violation += (field_value - val) / (val + 1e-6)
                elif op == '>=' and field_value < val:
                    total_violation += (val - field_value) / (val + 1e-6)
            
            penalty_amount = confidence * total_violation * 100
            return -penalty_amount
        
        return check, penalty
    
    def quantify_all_rules(self) -> List[QuantifiedRule]:
        """
        量化所有规则
        
        Returns:
            量化后的规则列表
        """
        print("=" * 80)
        print("规则量化")
        print("=" * 80)
        
        quantified = []
        
        for rule in self.rules:
            try:
                quantified_rule = self.quantify_rule(rule)
                quantified.append(quantified_rule)
                print(f"✅ {quantified_rule.rule_name}: {quantified_rule.description}")
            except Exception as e:
                print(f"❌ 量化规则失败 {rule.get('rule_name', 'unknown')}: {e}")
        
        # 按优先级和权重排序
        quantified.sort(key=lambda r: (r.priority, -r.weight))
        
        print(f"\n✅ 成功量化 {len(quantified)} 条规则")
        
        self.quantified_rules = quantified
        return quantified
    
    def test_rule(self, rule: QuantifiedRule, test_cases: List[Dict]) -> Dict:
        """
        测试规则
        
        Args:
            rule: 量化后的规则
            test_cases: 测试用例列表
        
        Returns:
            测试结果
        """
        results = {
            'rule_id': rule.rule_id,
            'rule_name': rule.rule_name,
            'total_tests': len(test_cases),
            'passed': 0,
            'failed': 0,
            'total_penalty': 0.0
        }
        
        for test_case in test_cases:
            if rule.check_function(test_case):
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['total_penalty'] += abs(rule.penalty_function(test_case))
        
        results['pass_rate'] = results['passed'] / results['total_tests'] if results['total_tests'] > 0 else 0
        results['avg_penalty'] = results['total_penalty'] / results['failed'] if results['failed'] > 0 else 0
        
        return results
    
    def save_quantified_rules(self, output_path: str = 'data/historical/quantified_rules.json'):
        """
        保存量化后的规则（不包含函数，只保存元数据）
        
        注意：函数无法序列化，只保存规则元数据
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        rules_metadata = [rule.to_dict() for rule in self.quantified_rules]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 已保存规则元数据到: {output_path}")
        print(f"   注意：函数无法序列化，需要在运行时重新量化")


def main():
    """主函数：执行规则量化"""
    import sys
    import os
    
    # 加载提取的规则
    rules_path = 'data/historical/extracted_rules.json'
    if not os.path.exists(rules_path):
        print(f"❌ 规则文件不存在: {rules_path}")
        print("请先运行 rule_extractor.py 提取规则")
        sys.exit(1)
    
    print("正在加载规则...")
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    print(f"✅ 已加载 {len(rules)} 条规则")
    
    # 执行量化
    quantifier = RuleQuantifier(rules)
    quantified_rules = quantifier.quantify_all_rules()
    
    # 保存元数据
    quantifier.save_quantified_rules()
    
    # 测试规则（示例）
    print("\n" + "=" * 80)
    print("规则测试（示例）")
    print("=" * 80)
    
    if len(quantified_rules) > 0:
        test_cases = [
            {'age': 30, 'monthly_income': 10000, 'credit_score': 700},
            {'age': 25, 'monthly_income': 5000, 'credit_score': 550},
        ]
        
        for rule in quantified_rules[:3]:  # 测试前3条规则
            result = quantifier.test_rule(rule, test_cases)
            print(f"\n规则: {rule.rule_name}")
            print(f"  通过率: {result['pass_rate']:.2%}")
            print(f"  平均惩罚: {result['avg_penalty']:.2f}")
    
    return quantified_rules


if __name__ == '__main__':
    main()

