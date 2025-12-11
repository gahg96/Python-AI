"""
合规性检查模块
自动检查业务操作是否符合监管要求
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class ComplianceRule:
    """合规规则"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # approval_limit, rate_limit, concentration_limit, info_protection
    check_function: callable
    severity: str  # low, medium, high, critical
    enabled: bool = True


@dataclass
class ComplianceCheckResult:
    """合规检查结果"""
    rule_id: str
    rule_name: str
    passed: bool
    message: str
    severity: str
    timestamp: datetime
    details: Dict[str, Any]


class ComplianceChecker:
    """合规性检查器"""
    
    def __init__(self):
        self.rules: List[ComplianceRule] = []
        self.check_history: List[ComplianceCheckResult] = []
        self._init_default_rules()
        
    def _init_default_rules(self):
        """初始化默认合规规则"""
        
        # 规则1：单一客户授信限制
        def check_single_customer_limit(customer_data: Dict, loan_data: Dict, bank_metrics: Dict) -> tuple:
            """检查单一客户授信是否超过资本净额的10%"""
            customer_id = customer_data.get('customer_id', '')
            loan_amount = loan_data.get('loan_amount', 0)
            capital_net = bank_metrics.get('capital_net', 1)
            existing_exposure = bank_metrics.get('customer_exposures', {}).get(customer_id, 0)
            total_exposure = existing_exposure + loan_amount
            concentration = (total_exposure / (capital_net + 1e-6)) * 100
            
            if concentration > 10.0:
                return False, f"单一客户集中度{concentration:.2f}%超过监管要求10%"
            return True, "通过"
            
        self.rules.append(ComplianceRule(
            rule_id='single_customer_limit',
            name='单一客户授信限制',
            description='单一客户授信不得超过资本净额的10%',
            rule_type='concentration_limit',
            check_function=check_single_customer_limit,
            severity='critical',
            enabled=True
        ))
        
        # 规则2：关联交易授信限制
        def check_related_party_limit(customer_data: Dict, loan_data: Dict, bank_metrics: Dict) -> tuple:
            """检查关联交易授信是否超过资本净额的50%"""
            is_related_party = customer_data.get('is_related_party', False)
            if not is_related_party:
                return True, "非关联交易"
                
            loan_amount = loan_data.get('loan_amount', 0)
            capital_net = bank_metrics.get('capital_net', 1)
            existing_related_exposure = bank_metrics.get('related_party_exposure', 0)
            total_related_exposure = existing_related_exposure + loan_amount
            concentration = (total_related_exposure / (capital_net + 1e-6)) * 100
            
            if concentration > 50.0:
                return False, f"关联交易集中度{concentration:.2f}%超过监管要求50%"
            return True, "通过"
            
        self.rules.append(ComplianceRule(
            rule_id='related_party_limit',
            name='关联交易授信限制',
            description='关联交易授信不得超过资本净额的50%',
            rule_type='concentration_limit',
            check_function=check_related_party_limit,
            severity='critical',
            enabled=True
        ))
        
        # 规则3：利率上限检查
        def check_rate_limit(customer_data: Dict, loan_data: Dict, bank_metrics: Dict) -> tuple:
            """检查利率是否超过上限"""
            interest_rate = loan_data.get('interest_rate', 0)
            max_rate = bank_metrics.get('max_interest_rate', 0.24)  # 默认24%上限
            
            if interest_rate > max_rate:
                return False, f"利率{interest_rate*100:.2f}%超过上限{max_rate*100:.2f}%"
            return True, "通过"
            
        self.rules.append(ComplianceRule(
            rule_id='rate_limit',
            name='利率上限检查',
            description='贷款利率不得超过监管上限',
            rule_type='rate_limit',
            check_function=check_rate_limit,
            severity='high',
            enabled=True
        ))
        
        # 规则4：客户资质检查
        def check_customer_qualification(customer_data: Dict, loan_data: Dict, bank_metrics: Dict) -> tuple:
            """检查客户是否符合放贷条件"""
            # 检查黑名单
            if customer_data.get('is_blacklisted', False):
                return False, "客户在黑名单中，禁止放贷"
            
            # 检查年龄限制
            age = customer_data.get('age', 0)
            if age < 18 or age > 70:
                return False, f"客户年龄{age}不符合要求（18-70岁）"
            
            # 检查企业资质
            if customer_data.get('customer_type') == 'corporate':
                operating_years = customer_data.get('operating_years', 0)
                if operating_years < 1:
                    return False, "企业成立年限不足1年，不符合放贷条件"
            
            return True, "通过"
            
        self.rules.append(ComplianceRule(
            rule_id='customer_qualification',
            name='客户资质检查',
            description='检查客户是否符合放贷基本条件',
            rule_type='approval_limit',
            check_function=check_customer_qualification,
            severity='high',
            enabled=True
        ))
        
        # 规则5：资本充足率检查
        def check_capital_adequacy(customer_data: Dict, loan_data: Dict, bank_metrics: Dict) -> tuple:
            """检查放贷后资本充足率是否仍满足要求"""
            capital_adequacy_ratio = bank_metrics.get('capital_adequacy_ratio', 0)
            loan_amount = loan_data.get('loan_amount', 0)
            risk_weight = loan_data.get('risk_weight', 1.0)
            
            # 简单估算：假设放贷后风险加权资产增加
            estimated_new_rwa = loan_amount * risk_weight
            current_rwa = bank_metrics.get('risk_weighted_assets', 1)
            total_capital = bank_metrics.get('total_capital', 1)
            
            estimated_new_ratio = (total_capital / (current_rwa + estimated_new_rwa + 1e-6)) * 100
            
            if estimated_new_ratio < 8.0:
                return False, f"放贷后预计资本充足率{estimated_new_ratio:.2f}%低于监管要求8%"
            return True, "通过"
            
        self.rules.append(ComplianceRule(
            rule_id='capital_adequacy',
            name='资本充足率检查',
            description='检查放贷后资本充足率是否仍满足监管要求',
            rule_type='approval_limit',
            check_function=check_capital_adequacy,
            severity='critical',
            enabled=True
        ))
        
    def check_approval(self, customer_data: Dict, loan_data: Dict, bank_metrics: Dict) -> List[ComplianceCheckResult]:
        """检查审批决策是否符合监管要求"""
        results = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            try:
                passed, message = rule.check_function(customer_data, loan_data, bank_metrics)
                result = ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=passed,
                    message=message,
                    severity=rule.severity,
                    timestamp=datetime.now(),
                    details={
                        'rule_type': rule.rule_type,
                        'description': rule.description
                    }
                )
                results.append(result)
                self.check_history.append(result)
            except Exception as e:
                # 检查失败，记录错误
                result = ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=False,
                    message=f"检查异常: {str(e)}",
                    severity=rule.severity,
                    timestamp=datetime.now(),
                    details={'error': str(e)}
                )
                results.append(result)
                self.check_history.append(result)
        
        return results
        
    def get_check_summary(self) -> Dict[str, Any]:
        """获取检查汇总"""
        total_checks = len(self.check_history)
        passed_checks = sum(1 for r in self.check_history if r.passed)
        failed_checks = total_checks - passed_checks
        
        severity_counts = {
            'critical': sum(1 for r in self.check_history if not r.passed and r.severity == 'critical'),
            'high': sum(1 for r in self.check_history if not r.passed and r.severity == 'high'),
            'medium': sum(1 for r in self.check_history if not r.passed and r.severity == 'medium'),
            'low': sum(1 for r in self.check_history if not r.passed and r.severity == 'low')
        }
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'pass_rate': (passed_checks / (total_checks + 1e-6)) * 100,
            'severity_counts': severity_counts,
            'recent_failures': [
                {
                    'rule_name': r.rule_name,
                    'message': r.message,
                    'severity': r.severity,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.check_history[-10:] if not r.passed
            ]
        }

