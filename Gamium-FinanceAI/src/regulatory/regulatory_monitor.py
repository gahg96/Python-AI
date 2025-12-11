"""
监管指标监控核心模块
实时计算和监控核心监管指标
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json


@dataclass
class RegulatoryIndicator:
    """监管指标数据类"""
    name: str  # 指标名称
    value: float  # 指标值
    threshold_red: float  # 监管红线
    threshold_warning: float  # 预警线
    threshold_target_min: float  # 目标范围最小值
    threshold_target_max: float  # 目标范围最大值
    unit: str  # 单位
    status: str  # 状态：normal/warning/critical/emergency
    trend: str  # 趋势：up/down/stable
    last_update: datetime  # 最后更新时间


@dataclass
class BankMetrics:
    """银行核心指标数据"""
    # 资本数据
    core_tier1_capital: float  # 核心一级资本
    other_tier1_capital: float  # 其他一级资本
    tier2_capital: float  # 二级资本
    total_capital: float  # 总资本
    
    # 资产数据
    risk_weighted_assets: float  # 风险加权资产
    total_assets: float  # 总资产
    adjusted_assets: float  # 调整后表内外资产余额
    
    # 贷款数据
    total_loans: float  # 贷款总额
    non_performing_loans: float  # 不良贷款余额
    loan_loss_provision: float  # 贷款损失准备
    required_provision: float  # 应计提贷款损失准备
    
    # 流动性数据
    high_quality_liquid_assets: float  # 优质流动性资产
    net_cash_outflow_30d: float  # 未来30天现金净流出
    available_stable_funding: float  # 可用稳定资金
    required_stable_funding: float  # 所需稳定资金
    deposits: float  # 存款余额
    
    # 集中度数据
    single_customer_exposure: float  # 单一客户授信总额
    single_industry_exposure: float  # 单一行业授信总额
    related_party_exposure: float  # 关联交易总额
    
    # 合规数据
    regulatory_reports_on_time: int  # 按时报送次数
    regulatory_reports_total: int  # 应报送次数
    violations_count: int  # 违规事件数量
    compliant_customers: int  # 合规客户数
    total_customers: int  # 总客户数


class RegulatoryMonitor:
    """监管指标监控器"""
    
    def __init__(self):
        self.metrics: Optional[BankMetrics] = None
        self.indicators: Dict[str, RegulatoryIndicator] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
    def update_metrics(self, metrics: BankMetrics):
        """更新银行指标数据"""
        self.metrics = metrics
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        """计算所有监管指标"""
        if not self.metrics:
            return
            
        # 1. 资本充足性指标
        self._calculate_capital_indicators()
        
        # 2. 资产质量指标
        self._calculate_asset_quality_indicators()
        
        # 3. 流动性指标
        self._calculate_liquidity_indicators()
        
        # 4. 集中度风险指标
        self._calculate_concentration_indicators()
        
        # 5. 合规性指标
        self._calculate_compliance_indicators()
        
    def _calculate_capital_indicators(self):
        """计算资本充足性指标"""
        m = self.metrics
        
        # 资本充足率 = (核心一级资本 + 其他一级资本 + 二级资本) / 风险加权资产
        capital_adequacy_ratio = (
            (m.core_tier1_capital + m.other_tier1_capital + m.tier2_capital) 
            / (m.risk_weighted_assets + 1e-6)
        ) * 100
        
        # 一级资本充足率 = (核心一级资本 + 其他一级资本) / 风险加权资产
        tier1_capital_ratio = (
            (m.core_tier1_capital + m.other_tier1_capital) 
            / (m.risk_weighted_assets + 1e-6)
        ) * 100
        
        # 核心一级资本充足率 = 核心一级资本 / 风险加权资产
        core_tier1_ratio = (
            m.core_tier1_capital / (m.risk_weighted_assets + 1e-6)
        ) * 100
        
        # 杠杆率 = 一级资本 / 调整后表内外资产余额
        leverage_ratio = (
            (m.core_tier1_capital + m.other_tier1_capital) 
            / (m.adjusted_assets + 1e-6)
        ) * 100
        
        # 计算资本净额（用于集中度计算）
        capital_net = m.core_tier1_capital + m.other_tier1_capital + m.tier2_capital
        
        self.indicators['capital_adequacy_ratio'] = RegulatoryIndicator(
            name='资本充足率',
            value=capital_adequacy_ratio,
            threshold_red=8.0,
            threshold_warning=10.0,
            threshold_target_min=10.0,
            threshold_target_max=15.0,
            unit='%',
            status=self._get_status(capital_adequacy_ratio, 8.0, 10.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['tier1_capital_ratio'] = RegulatoryIndicator(
            name='一级资本充足率',
            value=tier1_capital_ratio,
            threshold_red=6.0,
            threshold_warning=8.0,
            threshold_target_min=8.0,
            threshold_target_max=12.0,
            unit='%',
            status=self._get_status(tier1_capital_ratio, 6.0, 8.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['core_tier1_ratio'] = RegulatoryIndicator(
            name='核心一级资本充足率',
            value=core_tier1_ratio,
            threshold_red=5.0,
            threshold_warning=7.0,
            threshold_target_min=7.0,
            threshold_target_max=10.0,
            unit='%',
            status=self._get_status(core_tier1_ratio, 5.0, 7.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['leverage_ratio'] = RegulatoryIndicator(
            name='杠杆率',
            value=leverage_ratio,
            threshold_red=4.0,
            threshold_warning=5.0,
            threshold_target_min=5.0,
            threshold_target_max=8.0,
            unit='%',
            status=self._get_status(leverage_ratio, 4.0, 5.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        # 保存资本净额用于集中度计算
        self.capital_net = capital_net
        
    def _calculate_asset_quality_indicators(self):
        """计算资产质量指标"""
        m = self.metrics
        
        # 不良贷款率 = 不良贷款余额 / 贷款总额
        npl_ratio = (m.non_performing_loans / (m.total_loans + 1e-6)) * 100
        
        # 拨备覆盖率 = 贷款损失准备 / 不良贷款余额
        provision_coverage_ratio = (
            m.loan_loss_provision / (m.non_performing_loans + 1e-6)
        ) * 100
        
        # 贷款损失准备充足率 = 贷款损失准备 / 应计提贷款损失准备
        provision_adequacy_ratio = (
            m.loan_loss_provision / (m.required_provision + 1e-6)
        ) * 100
        
        self.indicators['npl_ratio'] = RegulatoryIndicator(
            name='不良贷款率',
            value=npl_ratio,
            threshold_red=5.0,
            threshold_warning=4.0,
            threshold_target_min=0.0,
            threshold_target_max=2.0,
            unit='%',
            status=self._get_status_reverse(npl_ratio, 5.0, 4.0),  # 越高越差
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['provision_coverage_ratio'] = RegulatoryIndicator(
            name='拨备覆盖率',
            value=provision_coverage_ratio,
            threshold_red=150.0,
            threshold_warning=180.0,
            threshold_target_min=200.0,
            threshold_target_max=300.0,
            unit='%',
            status=self._get_status(provision_coverage_ratio, 150.0, 180.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['provision_adequacy_ratio'] = RegulatoryIndicator(
            name='贷款损失准备充足率',
            value=provision_adequacy_ratio,
            threshold_red=100.0,
            threshold_warning=110.0,
            threshold_target_min=120.0,
            threshold_target_max=200.0,
            unit='%',
            status=self._get_status(provision_adequacy_ratio, 100.0, 110.0),
            trend='stable',
            last_update=datetime.now()
        )
        
    def _calculate_liquidity_indicators(self):
        """计算流动性指标"""
        m = self.metrics
        
        # 流动性覆盖率 = 优质流动性资产 / 未来30天现金净流出
        lcr = (
            m.high_quality_liquid_assets / (m.net_cash_outflow_30d + 1e-6)
        ) * 100
        
        # 净稳定资金比例 = 可用稳定资金 / 所需稳定资金
        nsfr = (
            m.available_stable_funding / (m.required_stable_funding + 1e-6)
        ) * 100
        
        # 存贷比 = 贷款余额 / 存款余额
        loan_to_deposit_ratio = (m.total_loans / (m.deposits + 1e-6)) * 100
        
        self.indicators['lcr'] = RegulatoryIndicator(
            name='流动性覆盖率',
            value=lcr,
            threshold_red=100.0,
            threshold_warning=110.0,
            threshold_target_min=120.0,
            threshold_target_max=200.0,
            unit='%',
            status=self._get_status(lcr, 100.0, 110.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['nsfr'] = RegulatoryIndicator(
            name='净稳定资金比例',
            value=nsfr,
            threshold_red=100.0,
            threshold_warning=110.0,
            threshold_target_min=120.0,
            threshold_target_max=200.0,
            unit='%',
            status=self._get_status(nsfr, 100.0, 110.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['loan_to_deposit_ratio'] = RegulatoryIndicator(
            name='存贷比',
            value=loan_to_deposit_ratio,
            threshold_red=75.0,
            threshold_warning=70.0,
            threshold_target_min=0.0,
            threshold_target_max=70.0,
            unit='%',
            status=self._get_status_reverse(loan_to_deposit_ratio, 75.0, 70.0),  # 越高越差
            trend='stable',
            last_update=datetime.now()
        )
        
    def _calculate_concentration_indicators(self):
        """计算集中度风险指标"""
        m = self.metrics
        capital_net = getattr(self, 'capital_net', m.total_capital)
        
        # 单一客户集中度 = 对单一客户授信总额 / 资本净额
        single_customer_concentration = (
            m.single_customer_exposure / (capital_net + 1e-6)
        ) * 100
        
        # 单一行业集中度 = 对单一行业授信总额 / 资本净额
        single_industry_concentration = (
            m.single_industry_exposure / (capital_net + 1e-6)
        ) * 100
        
        # 关联交易集中度 = 关联交易总额 / 资本净额
        related_party_concentration = (
            m.related_party_exposure / (capital_net + 1e-6)
        ) * 100
        
        self.indicators['single_customer_concentration'] = RegulatoryIndicator(
            name='单一客户集中度',
            value=single_customer_concentration,
            threshold_red=10.0,
            threshold_warning=8.0,
            threshold_target_min=0.0,
            threshold_target_max=8.0,
            unit='%',
            status=self._get_status_reverse(single_customer_concentration, 10.0, 8.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['single_industry_concentration'] = RegulatoryIndicator(
            name='单一行业集中度',
            value=single_industry_concentration,
            threshold_red=25.0,
            threshold_warning=20.0,
            threshold_target_min=0.0,
            threshold_target_max=20.0,
            unit='%',
            status=self._get_status_reverse(single_industry_concentration, 25.0, 20.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['related_party_concentration'] = RegulatoryIndicator(
            name='关联交易集中度',
            value=related_party_concentration,
            threshold_red=50.0,
            threshold_warning=40.0,
            threshold_target_min=0.0,
            threshold_target_max=40.0,
            unit='%',
            status=self._get_status_reverse(related_party_concentration, 50.0, 40.0),
            trend='stable',
            last_update=datetime.now()
        )
        
    def _calculate_compliance_indicators(self):
        """计算合规性指标"""
        m = self.metrics
        
        # 监管报送及时率 = 按时报送次数 / 应报送次数
        report_timeliness = (
            m.regulatory_reports_on_time / (m.regulatory_reports_total + 1e-6)
        ) * 100
        
        # 客户信息保护合规率 = 合规客户数 / 总客户数
        customer_info_compliance = (
            m.compliant_customers / (m.total_customers + 1e-6)
        ) * 100
        
        self.indicators['report_timeliness'] = RegulatoryIndicator(
            name='监管报送及时率',
            value=report_timeliness,
            threshold_red=95.0,
            threshold_warning=98.0,
            threshold_target_min=100.0,
            threshold_target_max=100.0,
            unit='%',
            status=self._get_status(report_timeliness, 95.0, 98.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['violations_count'] = RegulatoryIndicator(
            name='违规事件数量',
            value=float(m.violations_count),
            threshold_red=10.0,
            threshold_warning=5.0,
            threshold_target_min=0.0,
            threshold_target_max=0.0,
            unit='件',
            status=self._get_status_reverse(float(m.violations_count), 10.0, 5.0),
            trend='stable',
            last_update=datetime.now()
        )
        
        self.indicators['customer_info_compliance'] = RegulatoryIndicator(
            name='客户信息保护合规率',
            value=customer_info_compliance,
            threshold_red=95.0,
            threshold_warning=98.0,
            threshold_target_min=100.0,
            threshold_target_max=100.0,
            unit='%',
            status=self._get_status(customer_info_compliance, 95.0, 98.0),
            trend='stable',
            last_update=datetime.now()
        )
        
    def _get_status(self, value: float, red_threshold: float, warning_threshold: float) -> str:
        """获取指标状态（正常值越大越好）"""
        if value < red_threshold:
            return 'emergency'
        elif value < warning_threshold:
            return 'critical'
        elif value >= warning_threshold:
            return 'normal'
        else:
            return 'warning'
            
    def _get_status_reverse(self, value: float, red_threshold: float, warning_threshold: float) -> str:
        """获取指标状态（正常值越小越好）"""
        if value > red_threshold:
            return 'emergency'
        elif value > warning_threshold:
            return 'critical'
        elif value <= warning_threshold:
            return 'normal'
        else:
            return 'warning'
            
    def get_all_indicators(self) -> Dict[str, Dict[str, Any]]:
        """获取所有指标"""
        result = {}
        for key, indicator in self.indicators.items():
            result[key] = {
                'name': indicator.name,
                'value': round(indicator.value, 2),
                'threshold_red': indicator.threshold_red,
                'threshold_warning': indicator.threshold_warning,
                'threshold_target_min': indicator.threshold_target_min,
                'threshold_target_max': indicator.threshold_target_max,
                'unit': indicator.unit,
                'status': indicator.status,
                'trend': indicator.trend,
                'last_update': indicator.last_update.isoformat()
            }
        return result
        
    def check_alerts(self) -> List[Dict[str, Any]]:
        """检查告警"""
        alerts = []
        for key, indicator in self.indicators.items():
            if indicator.status in ['critical', 'emergency']:
                alert = {
                    'indicator_name': indicator.name,
                    'indicator_key': key,
                    'value': round(indicator.value, 2),
                    'threshold': indicator.threshold_red if indicator.status == 'emergency' else indicator.threshold_warning,
                    'status': indicator.status,
                    'level': 'emergency' if indicator.status == 'emergency' else 'critical',
                    'message': self._generate_alert_message(indicator),
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
                self.alert_history.append(alert)
        return alerts
        
    def _generate_alert_message(self, indicator: RegulatoryIndicator) -> str:
        """生成告警消息"""
        if indicator.status == 'emergency':
            return f"{indicator.name}已突破监管红线（{indicator.value:.2f}{indicator.unit} > {indicator.threshold_red}{indicator.unit}），请立即采取行动！"
        elif indicator.status == 'critical':
            return f"{indicator.name}接近监管红线（{indicator.value:.2f}{indicator.unit} > {indicator.threshold_warning}{indicator.unit}），请密切关注！"
        else:
            return f"{indicator.name}异常（{indicator.value:.2f}{indicator.unit}）"

