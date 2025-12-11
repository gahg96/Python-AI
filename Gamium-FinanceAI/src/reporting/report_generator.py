"""
报表生成引擎
根据模板自动生成监管报表
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from reporting.report_template import ReportTemplate, ReportTemplateManager, ReportStatus
from reporting.data_collector import DataCollector
from reporting.data_validator import DataValidator


class ReportInstance:
    """报表实例"""
    def __init__(self, instance_id: str, template_id: str, report_period: str, 
                 report_data: Dict[str, Any], status: ReportStatus = ReportStatus.DRAFT):
        self.instance_id = instance_id
        self.template_id = template_id
        self.report_period = report_period
        self.report_data = report_data
        self.status = status
        self.created_at = datetime.now()
        self.submitted_at: Optional[datetime] = None
        self.approved_at: Optional[datetime] = None
        self.sent_at: Optional[datetime] = None
        self.validation_results: List[Dict[str, Any]] = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'template_id': self.template_id,
            'report_period': self.report_period,
            'report_data': self.report_data,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'validation_results': self.validation_results
        }


class ReportGenerator:
    """报表生成器"""
    
    def __init__(self, template_manager: ReportTemplateManager, 
                 data_collector: DataCollector,
                 data_validator: DataValidator):
        self.template_manager = template_manager
        self.data_collector = data_collector
        self.data_validator = data_validator
        self.instances: Dict[str, ReportInstance] = {}
        
    def calculate_formula(self, formula: str, data: Dict[str, Any]) -> Optional[float]:
        """
        计算公式字段
        
        Args:
            formula: 公式字符串，如 'core_tier1_capital + other_tier1_capital'
            data: 数据字典
        
        Returns:
            计算结果
        """
        try:
            # 替换公式中的字段名为实际值
            formula_str = formula
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    formula_str = formula_str.replace(key, str(value))
                else:
                    # 如果字段值不是数字，尝试从data中获取
                    pass
                    
            # 简单的公式计算（支持 +, -, *, /, ( )）
            # 注意：这里使用eval，实际生产环境应该使用更安全的表达式解析器
            result = eval(formula_str)
            return float(result)
        except Exception as e:
            print(f"公式计算错误: {formula}, 错误: {e}")
            return None
            
    def generate_report(self, template_id: str, report_period: str) -> Optional[ReportInstance]:
        """
        生成报表
        
        Args:
            template_id: 模板ID
            report_period: 报表期间
        
        Returns:
            报表实例
        """
        # 1. 获取模板
        template = self.template_manager.get_template(template_id)
        if not template:
            return None
            
        # 2. 采集数据
        all_fields = []
        for section in template.sections:
            all_fields.extend(section.fields)
            
        # 转换为字典格式
        fields_dict = [
            {
                'field_id': f.field_id,
                'field_name': f.field_name,
                'field_type': f.field_type,
                'data_source': f.data_source,
                'formula': f.formula,
                'required': f.required,
                'default_value': f.default_value,
                'validation_rules': f.validation_rules
            }
            for f in all_fields
        ]
        
        # 采集非公式字段数据
        data = self.data_collector.collect_all_fields(fields_dict)
        
        # 3. 计算公式字段
        for field in all_fields:
            if field.formula:
                result = self.calculate_formula(field.formula, data)
                if result is not None:
                    data[field.field_id] = result
                    
        # 4. 数据校验
        validation_results = self.data_validator.validate_fields(fields_dict, data)
        validation_summary = self.data_validator.get_validation_summary()
        
        # 5. 创建报表实例
        instance_id = f"{template_id}_{report_period}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        instance = ReportInstance(
            instance_id=instance_id,
            template_id=template_id,
            report_period=report_period,
            report_data=data,
            status=ReportStatus.DRAFT
        )
        instance.validation_results = [r.to_dict() for r in validation_results]
        
        self.instances[instance_id] = instance
        return instance
        
    def get_instance(self, instance_id: str) -> Optional[ReportInstance]:
        """获取报表实例"""
        return self.instances.get(instance_id)
        
    def get_all_instances(self) -> List[Dict[str, Any]]:
        """获取所有报表实例"""
        return [inst.to_dict() for inst in self.instances.values()]
        
    def update_instance_data(self, instance_id: str, field_id: str, value: Any) -> bool:
        """
        更新报表实例数据
        
        Args:
            instance_id: 实例ID
            field_id: 字段ID
            value: 字段值
        
        Returns:
            是否成功
        """
        instance = self.instances.get(instance_id)
        if not instance:
            return False
            
        if instance.status != ReportStatus.DRAFT:
            return False  # 只能修改草稿状态的报表
            
        instance.report_data[field_id] = value
        return True
        
    def submit_instance(self, instance_id: str) -> bool:
        """
        提交报表实例
        
        Args:
            instance_id: 实例ID
        
        Returns:
            是否成功
        """
        instance = self.instances.get(instance_id)
        if not instance:
            return False
            
        if instance.status != ReportStatus.DRAFT:
            return False  # 只能提交草稿状态的报表
            
        # 再次校验
        template = self.template_manager.get_template(instance.template_id)
        if not template:
            return False
            
        all_fields = []
        for section in template.sections:
            all_fields.extend(section.fields)
            
        fields_dict = [
            {
                'field_id': f.field_id,
                'field_name': f.field_name,
                'field_type': f.field_type,
                'data_source': f.data_source,
                'formula': f.formula,
                'required': f.required,
                'default_value': f.default_value,
                'validation_rules': f.validation_rules
            }
            for f in all_fields
        ]
        
        validation_results = self.data_validator.validate_fields(fields_dict, instance.report_data)
        validation_summary = self.data_validator.get_validation_summary()
        
        # 如果有校验失败，不允许提交
        if validation_summary['failed'] > 0:
            return False
            
        instance.status = ReportStatus.SUBMITTED
        instance.submitted_at = datetime.now()
        instance.validation_results = [r.to_dict() for r in validation_results]
        return True

