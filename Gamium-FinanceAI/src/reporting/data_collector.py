"""
数据采集引擎
从业务系统自动采集报表所需数据
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class DataCollector:
    """数据采集器"""
    
    def __init__(self, regulatory_monitor=None):
        """
        初始化数据采集器
        
        Args:
            regulatory_monitor: 监管指标监控器实例
        """
        self.regulatory_monitor = regulatory_monitor
        self.collection_log: List[Dict[str, Any]] = []
        
    def collect_from_regulatory_monitor(self, field_path: str) -> Optional[Any]:
        """
        从监管指标监控器采集数据
        
        Args:
            field_path: 字段路径，如 'regulatory_monitor.core_tier1_capital'
        
        Returns:
            字段值
        """
        if not self.regulatory_monitor:
            return None
            
        try:
            parts = field_path.split('.')
            if len(parts) < 2:
                return None
                
            if parts[0] == 'regulatory_monitor':
                if parts[1] == 'core_tier1_capital':
                    return self.regulatory_monitor.metrics.core_tier1_capital if self.regulatory_monitor.metrics else None
                elif parts[1] == 'other_tier1_capital':
                    return self.regulatory_monitor.metrics.other_tier1_capital if self.regulatory_monitor.metrics else None
                elif parts[1] == 'tier2_capital':
                    return self.regulatory_monitor.metrics.tier2_capital if self.regulatory_monitor.metrics else None
                elif parts[1] == 'risk_weighted_assets':
                    return self.regulatory_monitor.metrics.risk_weighted_assets if self.regulatory_monitor.metrics else None
                elif parts[1] == 'total_loans':
                    return self.regulatory_monitor.metrics.total_loans if self.regulatory_monitor.metrics else None
                elif parts[1] == 'non_performing_loans':
                    return self.regulatory_monitor.metrics.non_performing_loans if self.regulatory_monitor.metrics else None
                elif parts[1] == 'loan_loss_provision':
                    return self.regulatory_monitor.metrics.loan_loss_provision if self.regulatory_monitor.metrics else None
                elif parts[1] == 'high_quality_liquid_assets':
                    return self.regulatory_monitor.metrics.high_quality_liquid_assets if self.regulatory_monitor.metrics else None
                elif parts[1] == 'net_cash_outflow_30d':
                    return self.regulatory_monitor.metrics.net_cash_outflow_30d if self.regulatory_monitor.metrics else None
                elif parts[1] == 'deposits':
                    return self.regulatory_monitor.metrics.deposits if self.regulatory_monitor.metrics else None
                elif parts[1] == 'single_customer_exposure':
                    return self.regulatory_monitor.metrics.single_customer_exposure if self.regulatory_monitor.metrics else None
                elif parts[1] == 'single_industry_exposure':
                    return self.regulatory_monitor.metrics.single_industry_exposure if self.regulatory_monitor.metrics else None
                elif parts[1] == 'related_party_exposure':
                    return self.regulatory_monitor.metrics.related_party_exposure if self.regulatory_monitor.metrics else None
                    
        except Exception as e:
            self.collection_log.append({
                'field_path': field_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
            
        return None
        
    def collect_from_system(self, field_path: str) -> Optional[Any]:
        """
        从系统采集数据（如报表期间、报表日期等）
        
        Args:
            field_path: 字段路径，如 'system.report_period'
        
        Returns:
            字段值
        """
        try:
            parts = field_path.split('.')
            if len(parts) < 2:
                return None
                
            if parts[0] == 'system':
                if parts[1] == 'report_period':
                    # 返回当前季度（示例）
                    now = datetime.now()
                    quarter = (now.month - 1) // 3 + 1
                    return f"{now.year}年第{quarter}季度"
                elif parts[1] == 'report_date':
                    return datetime.now().strftime('%Y-%m-%d')
                elif parts[1] == 'bank_name':
                    return '示例银行'
                    
        except Exception as e:
            self.collection_log.append({
                'field_path': field_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
            
        return None
        
    def collect_from_loan_system(self, field_path: str) -> Optional[Any]:
        """
        从贷款系统采集数据（模拟数据）
        
        Args:
            field_path: 字段路径，如 'loan_system.normal_loans'
        
        Returns:
            字段值
        """
        # 这里应该对接真实的贷款系统，目前返回模拟数据
        if not self.regulatory_monitor or not self.regulatory_monitor.metrics:
            return None
            
        m = self.regulatory_monitor.metrics
        
        try:
            parts = field_path.split('.')
            if len(parts) < 2:
                return None
                
            if parts[0] == 'loan_system':
                if parts[1] == 'normal_loans':
                    # 正常类贷款 = 总贷款 - 不良贷款
                    return m.total_loans - m.non_performing_loans
                elif parts[1] == 'special_mention_loans':
                    # 关注类贷款（模拟）
                    return m.total_loans * 0.05
                elif parts[1] == 'substandard_loans':
                    # 次级类贷款（模拟）
                    return m.non_performing_loans * 0.3
                elif parts[1] == 'doubtful_loans':
                    # 可疑类贷款（模拟）
                    return m.non_performing_loans * 0.5
                elif parts[1] == 'loss_loans':
                    # 损失类贷款（模拟）
                    return m.non_performing_loans * 0.2
                    
        except Exception as e:
            self.collection_log.append({
                'field_path': field_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
            
        return None
        
    def collect_field_data(self, field_path: str) -> Optional[Any]:
        """
        采集字段数据
        
        Args:
            field_path: 字段路径
        
        Returns:
            字段值
        """
        if field_path.startswith('regulatory_monitor.'):
            return self.collect_from_regulatory_monitor(field_path)
        elif field_path.startswith('system.'):
            return self.collect_from_system(field_path)
        elif field_path.startswith('loan_system.'):
            return self.collect_from_loan_system(field_path)
        elif field_path == 'formula':
            # 公式字段需要后续计算
            return None
        else:
            return None
            
    def collect_all_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        采集所有字段数据
        
        Args:
            fields: 字段列表
        
        Returns:
            字段数据字典
        """
        data = {}
        for field in fields:
            field_id = field.get('field_id')
            data_source = field.get('data_source')
            
            if data_source and data_source != 'formula':
                value = self.collect_field_data(data_source)
                if value is not None:
                    data[field_id] = value
            elif field.get('default_value') is not None:
                data[field_id] = field.get('default_value')
                
        return data
        
    def get_collection_log(self) -> List[Dict[str, Any]]:
        """获取采集日志"""
        return self.collection_log

