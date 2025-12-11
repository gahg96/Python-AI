"""
报表模板管理模块
管理各类监管报表模板，支持模板的创建、编辑、版本管理
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class ReportFrequency(Enum):
    """报表频率"""
    DAILY = "daily"  # 日报
    WEEKLY = "weekly"  # 周报
    MONTHLY = "monthly"  # 月报
    QUARTERLY = "quarterly"  # 季报
    YEARLY = "yearly"  # 年报
    EVENT_TRIGGERED = "event_triggered"  # 事件触发


class ReportStatus(Enum):
    """报表状态"""
    DRAFT = "draft"  # 草稿
    SUBMITTED = "submitted"  # 已提交
    APPROVED = "approved"  # 已审批
    REJECTED = "rejected"  # 已退回
    SENT = "sent"  # 已报送
    CONFIRMED = "confirmed"  # 已确认


@dataclass
class ReportField:
    """报表字段"""
    field_id: str
    field_name: str
    field_type: str  # text, number, date, formula
    data_source: str  # 数据来源字段
    formula: Optional[str] = None  # 计算公式
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    required: bool = True
    default_value: Any = None
    description: str = ""  # 字段详细说明
    unit: str = ""  # 单位
    regulatory_requirement: str = ""  # 监管要求


@dataclass
class ReportSection:
    """报表章节"""
    section_id: str
    section_name: str
    section_type: str  # header, body, footer, summary
    fields: List[ReportField] = field(default_factory=list)
    rows: int = 1  # 行数（用于表格类型）
    columns: int = 1  # 列数


@dataclass
class ReportTemplate:
    """报表模板"""
    template_id: str
    template_name: str
    template_code: str  # 报表编号（如G01）
    template_type: str  # capital_adequacy, asset_quality, liquidity, etc.
    frequency: ReportFrequency
    due_days: int  # 报送截止天数（报表期间结束后多少天内）
    description: str
    sections: List[ReportSection] = field(default_factory=list)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


class ReportTemplateManager:
    """报表模板管理器"""
    
    def __init__(self):
        self.templates: Dict[str, ReportTemplate] = {}
        self._init_default_templates()
        
    def _get_field_info(self, field_id: str) -> Dict[str, str]:
        """获取字段详细信息"""
        try:
            from reporting.field_descriptions import get_field_description
            return get_field_description(field_id)
        except:
            return {
                'description': '',
                'unit': '',
                'regulatory_requirement': '',
                'calculation_method': '',
                'data_source': ''
            }
        
    def _init_default_templates(self):
        """初始化默认报表模板"""
        
        # 模板1：资本充足率报表（G01）
        template1 = ReportTemplate(
            template_id='G01',
            template_name='资本充足率计算表',
            template_code='G01',
            template_type='capital_adequacy',
            frequency=ReportFrequency.QUARTERLY,
            due_days=15,
            description='根据《商业银行资本管理办法》要求，计算资本充足率相关指标',
            sections=[
                ReportSection(
                    section_id='header',
                    section_name='表头',
                    section_type='header',
                    fields=[
                        ReportField('report_period', '报表期间', 'text', 'system', required=True),
                        ReportField('report_date', '报表日期', 'date', 'system', required=True),
                        ReportField('bank_name', '银行名称', 'text', 'system', required=True)
                    ]
                ),
                ReportSection(
                    section_id='capital_composition',
                    section_name='资本构成',
                    section_type='body',
                    fields=[
                        ReportField('core_tier1_capital', '核心一级资本', 'number', 'regulatory_monitor.core_tier1_capital', required=True,
                                  description=self._get_field_info('core_tier1_capital')['description'],
                                  unit=self._get_field_info('core_tier1_capital')['unit'],
                                  regulatory_requirement=self._get_field_info('core_tier1_capital')['regulatory_requirement']),
                        ReportField('other_tier1_capital', '其他一级资本', 'number', 'regulatory_monitor.other_tier1_capital', required=True,
                                  description=self._get_field_info('other_tier1_capital')['description'],
                                  unit=self._get_field_info('other_tier1_capital')['unit'],
                                  regulatory_requirement=self._get_field_info('other_tier1_capital')['regulatory_requirement']),
                        ReportField('tier2_capital', '二级资本', 'number', 'regulatory_monitor.tier2_capital', required=True,
                                  description=self._get_field_info('tier2_capital')['description'],
                                  unit=self._get_field_info('tier2_capital')['unit'],
                                  regulatory_requirement=self._get_field_info('tier2_capital')['regulatory_requirement']),
                        ReportField('total_capital', '总资本', 'number', 'formula', formula='core_tier1_capital + other_tier1_capital + tier2_capital', required=True,
                                  description=self._get_field_info('total_capital')['description'],
                                  unit=self._get_field_info('total_capital')['unit'],
                                  regulatory_requirement=self._get_field_info('total_capital')['regulatory_requirement'])
                    ]
                ),
                ReportSection(
                    section_id='risk_weighted_assets',
                    section_name='风险加权资产',
                    section_type='body',
                    fields=[
                        ReportField('credit_risk_rwa', '信用风险加权资产', 'number', 'regulatory_monitor.risk_weighted_assets', required=True,
                                  description=self._get_field_info('credit_risk_rwa')['description'],
                                  unit=self._get_field_info('credit_risk_rwa')['unit'],
                                  regulatory_requirement=self._get_field_info('credit_risk_rwa')['regulatory_requirement']),
                        ReportField('market_risk_rwa', '市场风险加权资产', 'number', 'system', default_value=0, required=False,
                                  description=self._get_field_info('market_risk_rwa')['description'],
                                  unit=self._get_field_info('market_risk_rwa')['unit'],
                                  regulatory_requirement=self._get_field_info('market_risk_rwa')['regulatory_requirement']),
                        ReportField('operational_risk_rwa', '操作风险加权资产', 'number', 'system', default_value=0, required=False,
                                  description=self._get_field_info('operational_risk_rwa')['description'],
                                  unit=self._get_field_info('operational_risk_rwa')['unit'],
                                  regulatory_requirement=self._get_field_info('operational_risk_rwa')['regulatory_requirement']),
                        ReportField('total_rwa', '风险加权资产合计', 'number', 'formula', formula='credit_risk_rwa + market_risk_rwa + operational_risk_rwa', required=True,
                                  description=self._get_field_info('total_rwa')['description'],
                                  unit=self._get_field_info('total_rwa')['unit'],
                                  regulatory_requirement=self._get_field_info('total_rwa')['regulatory_requirement'])
                    ]
                ),
                ReportSection(
                    section_id='capital_adequacy_ratio',
                    section_name='资本充足率',
                    section_type='summary',
                    fields=[
                        ReportField('capital_adequacy_ratio', '资本充足率', 'number', 'formula', 
                                 formula='(total_capital / total_rwa) * 100', required=True,
                                 description=self._get_field_info('capital_adequacy_ratio')['description'],
                                 unit=self._get_field_info('capital_adequacy_ratio')['unit'],
                                 regulatory_requirement=self._get_field_info('capital_adequacy_ratio')['regulatory_requirement']),
                        ReportField('tier1_capital_ratio', '一级资本充足率', 'number', 'formula',
                                 formula='((core_tier1_capital + other_tier1_capital) / total_rwa) * 100', required=True,
                                 description=self._get_field_info('tier1_capital_ratio')['description'],
                                 unit=self._get_field_info('tier1_capital_ratio')['unit'],
                                 regulatory_requirement=self._get_field_info('tier1_capital_ratio')['regulatory_requirement']),
                        ReportField('core_tier1_ratio', '核心一级资本充足率', 'number', 'formula',
                                 formula='(core_tier1_capital / total_rwa) * 100', required=True,
                                 description=self._get_field_info('core_tier1_ratio')['description'],
                                 unit=self._get_field_info('core_tier1_ratio')['unit'],
                                 regulatory_requirement=self._get_field_info('core_tier1_ratio')['regulatory_requirement'])
                    ]
                )
            ]
        )
        self.templates[template1.template_id] = template1
        
        # 模板2：资产质量报表（G11）
        template2 = ReportTemplate(
            template_id='G11',
            template_name='贷款五级分类明细表',
            template_code='G11',
            template_type='asset_quality',
            frequency=ReportFrequency.MONTHLY,
            due_days=10,
            description='根据《贷款风险分类指引》要求，统计贷款五级分类情况',
            sections=[
                ReportSection(
                    section_id='header',
                    section_name='表头',
                    section_type='header',
                    fields=[
                        ReportField('report_period', '报表期间', 'text', 'system', required=True,
                                  description=self._get_field_info('report_period')['description'],
                                  unit=self._get_field_info('report_period')['unit'],
                                  regulatory_requirement=self._get_field_info('report_period')['regulatory_requirement']),
                        ReportField('report_date', '报表日期', 'date', 'system', required=True,
                                  description=self._get_field_info('report_date')['description'],
                                  unit=self._get_field_info('report_date')['unit'],
                                  regulatory_requirement=self._get_field_info('report_date')['regulatory_requirement'])
                    ]
                ),
                ReportSection(
                    section_id='loan_classification',
                    section_name='贷款分类',
                    section_type='body',
                    fields=[
                        ReportField('normal_loans', '正常类贷款', 'number', 'loan_system.normal_loans', required=True,
                                  description=self._get_field_info('normal_loans')['description'],
                                  unit=self._get_field_info('normal_loans')['unit'],
                                  regulatory_requirement=self._get_field_info('normal_loans')['regulatory_requirement']),
                        ReportField('special_mention_loans', '关注类贷款', 'number', 'loan_system.special_mention_loans', required=True,
                                  description=self._get_field_info('special_mention_loans')['description'],
                                  unit=self._get_field_info('special_mention_loans')['unit'],
                                  regulatory_requirement=self._get_field_info('special_mention_loans')['regulatory_requirement']),
                        ReportField('substandard_loans', '次级类贷款', 'number', 'loan_system.substandard_loans', required=True,
                                  description=self._get_field_info('substandard_loans')['description'],
                                  unit=self._get_field_info('substandard_loans')['unit'],
                                  regulatory_requirement=self._get_field_info('substandard_loans')['regulatory_requirement']),
                        ReportField('doubtful_loans', '可疑类贷款', 'number', 'loan_system.doubtful_loans', required=True,
                                  description=self._get_field_info('doubtful_loans')['description'],
                                  unit=self._get_field_info('doubtful_loans')['unit'],
                                  regulatory_requirement=self._get_field_info('doubtful_loans')['regulatory_requirement']),
                        ReportField('loss_loans', '损失类贷款', 'number', 'loan_system.loss_loans', required=True,
                                  description=self._get_field_info('loss_loans')['description'],
                                  unit=self._get_field_info('loss_loans')['unit'],
                                  regulatory_requirement=self._get_field_info('loss_loans')['regulatory_requirement']),
                        ReportField('total_loans', '贷款总额', 'number', 'formula',
                                 formula='normal_loans + special_mention_loans + substandard_loans + doubtful_loans + loss_loans', required=True,
                                 description=self._get_field_info('total_loans')['description'],
                                 unit=self._get_field_info('total_loans')['unit'],
                                 regulatory_requirement=self._get_field_info('total_loans')['regulatory_requirement']),
                        ReportField('npl_loans', '不良贷款', 'number', 'formula',
                                 formula='substandard_loans + doubtful_loans + loss_loans', required=True,
                                 description=self._get_field_info('npl_loans')['description'],
                                 unit=self._get_field_info('npl_loans')['unit'],
                                 regulatory_requirement=self._get_field_info('npl_loans')['regulatory_requirement']),
                        ReportField('npl_ratio', '不良贷款率', 'number', 'formula',
                                 formula='(npl_loans / total_loans) * 100', required=True,
                                 description=self._get_field_info('npl_ratio')['description'],
                                 unit=self._get_field_info('npl_ratio')['unit'],
                                 regulatory_requirement=self._get_field_info('npl_ratio')['regulatory_requirement'])
                    ]
                )
            ]
        )
        self.templates[template2.template_id] = template2
        
        # 模板3：流动性报表（G21）
        template3 = ReportTemplate(
            template_id='G21',
            template_name='流动性覆盖率计算表',
            template_code='G21',
            template_type='liquidity',
            frequency=ReportFrequency.MONTHLY,
            due_days=10,
            description='根据《商业银行流动性风险管理办法》要求，计算流动性覆盖率',
            sections=[
                ReportSection(
                    section_id='header',
                    section_name='表头',
                    section_type='header',
                    fields=[
                        ReportField('report_period', '报表期间', 'text', 'system', required=True,
                                  description=self._get_field_info('report_period')['description'],
                                  unit=self._get_field_info('report_period')['unit'],
                                  regulatory_requirement=self._get_field_info('report_period')['regulatory_requirement']),
                        ReportField('report_date', '报表日期', 'date', 'system', required=True,
                                  description=self._get_field_info('report_date')['description'],
                                  unit=self._get_field_info('report_date')['unit'],
                                  regulatory_requirement=self._get_field_info('report_date')['regulatory_requirement'])
                    ]
                ),
                ReportSection(
                    section_id='liquidity_assets',
                    section_name='优质流动性资产',
                    section_type='body',
                    fields=[
                        ReportField('high_quality_liquid_assets', '优质流动性资产', 'number', 
                                 'regulatory_monitor.high_quality_liquid_assets', required=True,
                                 description=self._get_field_info('high_quality_liquid_assets')['description'],
                                 unit=self._get_field_info('high_quality_liquid_assets')['unit'],
                                 regulatory_requirement=self._get_field_info('high_quality_liquid_assets')['regulatory_requirement'])
                    ]
                ),
                ReportSection(
                    section_id='cash_outflow',
                    section_name='未来30天现金净流出',
                    section_type='body',
                    fields=[
                        ReportField('net_cash_outflow_30d', '未来30天现金净流出', 'number',
                                 'regulatory_monitor.net_cash_outflow_30d', required=True,
                                 description=self._get_field_info('net_cash_outflow_30d')['description'],
                                 unit=self._get_field_info('net_cash_outflow_30d')['unit'],
                                 regulatory_requirement=self._get_field_info('net_cash_outflow_30d')['regulatory_requirement'])
                    ]
                ),
                ReportSection(
                    section_id='lcr',
                    section_name='流动性覆盖率',
                    section_type='summary',
                    fields=[
                        ReportField('lcr', '流动性覆盖率', 'number', 'formula',
                                 formula='(high_quality_liquid_assets / net_cash_outflow_30d) * 100', required=True,
                                 description=self._get_field_info('lcr')['description'],
                                 unit=self._get_field_info('lcr')['unit'],
                                 regulatory_requirement=self._get_field_info('lcr')['regulatory_requirement'])
                    ]
                )
            ]
        )
        self.templates[template3.template_id] = template3
        
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """获取模板"""
        return self.templates.get(template_id)
        
    def get_all_templates(self) -> List[Dict[str, Any]]:
        """获取所有模板"""
        result = []
        for template in self.templates.values():
            result.append({
                'template_id': template.template_id,
                'template_name': template.template_name,
                'template_code': template.template_code,
                'template_type': template.template_type,
                'frequency': template.frequency.value,
                'due_days': template.due_days,
                'description': template.description,
                'version': template.version,
                'enabled': template.enabled,
                'sections_count': len(template.sections),
                'fields_count': sum(len(s.fields) for s in template.sections)
            })
        return result
        
    def get_template_detail(self, template_id: str) -> Optional[Dict[str, Any]]:
        """获取模板详情"""
        template = self.templates.get(template_id)
        if not template:
            return None
            
        return {
            'template_id': template.template_id,
            'template_name': template.template_name,
            'template_code': template.template_code,
            'template_type': template.template_type,
            'frequency': template.frequency.value,
            'due_days': template.due_days,
            'description': template.description,
            'version': template.version,
            'enabled': template.enabled,
            'sections': [
                {
                    'section_id': s.section_id,
                    'section_name': s.section_name,
                    'section_type': s.section_type,
                    'fields': [
                        {
                            'field_id': f.field_id,
                            'field_name': f.field_name,
                            'field_type': f.field_type,
                            'data_source': f.data_source,
                            'formula': f.formula,
                            'required': f.required,
                            'default_value': f.default_value,
                            'description': f.description,
                            'unit': f.unit,
                            'regulatory_requirement': f.regulatory_requirement
                        }
                        for f in s.fields
                    ]
                }
                for s in template.sections
            ],
            'created_at': template.created_at.isoformat(),
            'updated_at': template.updated_at.isoformat()
        }

