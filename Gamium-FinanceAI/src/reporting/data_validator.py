"""
数据校验引擎
校验报表数据的完整性、准确性、一致性
"""
from typing import Dict, List, Optional, Any
from datetime import datetime


class ValidationResult:
    """校验结果"""
    def __init__(self, field_id: str, field_name: str, passed: bool, message: str = ""):
        self.field_id = field_id
        self.field_name = field_name
        self.passed = passed
        self.message = message
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_id': self.field_id,
            'field_name': self.field_name,
            'passed': self.passed,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }


class DataValidator:
    """数据校验器"""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        
    def validate_required(self, field_id: str, field_name: str, value: Any, required: bool) -> ValidationResult:
        """校验必填字段"""
        if required and (value is None or value == ""):
            return ValidationResult(
                field_id=field_id,
                field_name=field_name,
                passed=False,
                message=f"{field_name}为必填字段，不能为空"
            )
        return ValidationResult(
            field_id=field_id,
            field_name=field_name,
            passed=True
        )
        
    def validate_type(self, field_id: str, field_name: str, value: Any, field_type: str) -> ValidationResult:
        """校验字段类型"""
        if value is None:
            return ValidationResult(
                field_id=field_id,
                field_name=field_name,
                passed=True  # None值由required校验处理
            )
            
        if field_type == 'number':
            if not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    return ValidationResult(
                        field_id=field_id,
                        field_name=field_name,
                        passed=False,
                        message=f"{field_name}必须是数字类型"
                    )
        elif field_type == 'date':
            if not isinstance(value, str):
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}必须是日期格式字符串"
                )
        elif field_type == 'text':
            if not isinstance(value, str):
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}必须是文本类型"
                )
                
        return ValidationResult(
            field_id=field_id,
            field_name=field_name,
            passed=True
        )
        
    def validate_range(self, field_id: str, field_name: str, value: Any, min_value: Optional[float] = None, 
                     max_value: Optional[float] = None) -> ValidationResult:
        """校验数值范围"""
        if value is None:
            return ValidationResult(
                field_id=field_id,
                field_name=field_name,
                passed=True
            )
            
        try:
            num_value = float(value)
            if min_value is not None and num_value < min_value:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}不能小于{min_value}"
                )
            if max_value is not None and num_value > max_value:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}不能大于{max_value}"
                )
        except (ValueError, TypeError):
            return ValidationResult(
                field_id=field_id,
                field_name=field_name,
                passed=False,
                message=f"{field_name}无法转换为数值进行范围校验"
            )
            
        return ValidationResult(
            field_id=field_id,
            field_name=field_name,
            passed=True
        )
        
    def validate_business_rules(self, field_id: str, field_name: str, value: Any, 
                               all_data: Dict[str, Any]) -> ValidationResult:
        """校验业务规则"""
        # 业务规则1：资本充足率不能低于8%
        if field_id == 'capital_adequacy_ratio':
            if value is not None and float(value) < 8.0:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}不能低于8%（监管要求）"
                )
                
        # 业务规则2：核心一级资本充足率不能低于5%
        if field_id == 'core_tier1_ratio':
            if value is not None and float(value) < 5.0:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}不能低于5%（监管要求）"
                )
                
        # 业务规则3：不良贷款率不能超过5%
        if field_id == 'npl_ratio':
            if value is not None and float(value) > 5.0:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}不能超过5%（监管红线）"
                )
                
        # 业务规则4：流动性覆盖率不能低于100%
        if field_id == 'lcr':
            if value is not None and float(value) < 100.0:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}不能低于100%（监管要求）"
                )
                
        # 业务规则5：总资本 = 核心一级资本 + 其他一级资本 + 二级资本
        if field_id == 'total_capital':
            core = all_data.get('core_tier1_capital', 0)
            other = all_data.get('other_tier1_capital', 0)
            tier2 = all_data.get('tier2_capital', 0)
            expected = core + other + tier2
            if value is not None and abs(float(value) - expected) > 0.01:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}计算错误，应为{expected}，实际为{value}"
                )
                
        # 业务规则6：贷款总额 = 各类贷款之和
        if field_id == 'total_loans':
            normal = all_data.get('normal_loans', 0)
            special = all_data.get('special_mention_loans', 0)
            sub = all_data.get('substandard_loans', 0)
            doubt = all_data.get('doubtful_loans', 0)
            loss = all_data.get('loss_loans', 0)
            expected = normal + special + sub + doubt + loss
            if value is not None and abs(float(value) - expected) > 0.01:
                return ValidationResult(
                    field_id=field_id,
                    field_name=field_name,
                    passed=False,
                    message=f"{field_name}计算错误，应为{expected}，实际为{value}"
                )
                
        return ValidationResult(
            field_id=field_id,
            field_name=field_name,
            passed=True
        )
        
    def validate_fields(self, fields: List[Dict[str, Any]], data: Dict[str, Any]) -> List[ValidationResult]:
        """
        校验所有字段
        
        Args:
            fields: 字段定义列表
            data: 字段数据字典
        
        Returns:
            校验结果列表
        """
        results = []
        
        for field in fields:
            field_id = field.get('field_id')
            field_name = field.get('field_name', field_id)
            field_type = field.get('field_type', 'text')
            required = field.get('required', False)
            value = data.get(field_id)
            
            # 1. 必填校验
            result = self.validate_required(field_id, field_name, value, required)
            results.append(result)
            if not result.passed:
                continue
                
            # 2. 类型校验
            if value is not None:
                result = self.validate_type(field_id, field_name, value, field_type)
                results.append(result)
                if not result.passed:
                    continue
                    
                # 3. 范围校验（如果有配置）
                validation_rules = field.get('validation_rules', [])
                for rule in validation_rules:
                    if rule.get('type') == 'range':
                        min_val = rule.get('min')
                        max_val = rule.get('max')
                        result = self.validate_range(field_id, field_name, value, min_val, max_val)
                        results.append(result)
                        if not result.passed:
                            break
                            
                # 4. 业务规则校验
                result = self.validate_business_rules(field_id, field_name, value, data)
                results.append(result)
                
        self.validation_results = results
        return results
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取校验汇总"""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.passed)
        failed = total - passed
        
        failed_results = [r.to_dict() for r in self.validation_results if not r.passed]
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'failed_fields': failed_results
        }

