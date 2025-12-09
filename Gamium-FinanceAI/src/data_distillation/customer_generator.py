"""
客户生成器 - 生成虚拟但真实的客户画像

用于模拟和测试，生成符合统计分布的客户数据
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json


class Industry(Enum):
    """行业分类"""
    MANUFACTURING = "制造业"
    SERVICE = "服务业"
    RETAIL = "零售业"
    CATERING = "餐饮业"
    CONSTRUCTION = "建筑业"
    IT = "信息技术"
    FINANCE = "金融业"
    EDUCATION = "教育"
    HEALTHCARE = "医疗健康"
    AGRICULTURE = "农业"
    OTHER = "其他"


class CityTier(Enum):
    """城市等级"""
    TIER_1 = "一线城市"
    TIER_2 = "二线城市"
    TIER_3 = "三线城市"
    TIER_4 = "四线及以下"


class CustomerType(Enum):
    """客户类型"""
    SALARIED = "工薪阶层"
    SMALL_BUSINESS = "小微企业主"
    FREELANCER = "自由职业"
    FARMER = "农户"
    # 企业客户类型
    MICRO_ENTERPRISE = "微型企业"      # 年营收 < 1000万
    SMALL_ENTERPRISE = "小型企业"      # 年营收 1000万 - 5000万
    MEDIUM_ENTERPRISE = "中型企业"     # 年营收 5000万 - 2亿
    LARGE_ENTERPRISE = "大型企业"      # 年营收 > 2亿


@dataclass
class CustomerProfile:
    """客户画像 - 支持个人和企业客户"""
    customer_id: str
    
    # 基础信息
    age: int = 35  # 个人客户年龄，企业客户为成立年限
    city_tier: CityTier = None
    customer_type: CustomerType = None
    industry: Industry = None
    years_in_business: float = 5.0  # 从业/经营年限
    
    # 个人客户字段
    monthly_income: float = 0.0       # 月收入 (元) - 个人客户
    income_volatility: float = 0.2    # 收入波动率 (0-1) - 个人客户
    total_assets: float = 0.0         # 总资产 (元)
    total_liabilities: float = 0.0    # 总负债 (元)
    
    # 银行关系
    deposit_balance: float = 0.0      # 存款余额 (元)
    deposit_stability: float = 0.5    # 存款稳定性 (0-1)
    months_as_customer: int = 24     # 成为客户月数
    
    # 信贷历史
    previous_loans: int = 0         # 历史贷款次数
    max_historical_dpd: int = 0     # 历史最大逾期天数 (Days Past Due)
    months_since_last_loan: int = 12 # 距上次贷款月数
    
    # ========== 企业客户专用字段 ==========
    # 财务指标
    annual_revenue: float = 0.0          # 年营业收入 (元)
    net_profit: float = 0.0              # 净利润 (元)
    operating_cash_flow: float = 0.0     # 经营现金流 (元)
    ebitda: float = 0.0                  # EBITDA (元)
    revenue_growth_rate: float = 0.0     # 营收增长率
    profit_margin: float = 0.0           # 净利润率
    current_ratio: float = 0.0           # 流动比率
    quick_ratio: float = 0.0             # 速动比率
    asset_turnover: float = 0.0          # 资产周转率
    
    # 资产情况
    fixed_assets: float = 0.0            # 固定资产 (元)
    current_assets: float = 0.0         # 流动资产 (元)
    intangible_assets: float = 0.0       # 无形资产 (元)
    inventory: float = 0.0               # 存货 (元)
    accounts_receivable: float = 0.0     # 应收账款 (元)
    accounts_payable: float = 0.0       # 应付账款 (元)
    
    # 市场公开信息
    market_cap: float = 0.0             # 市值 (元) - 上市公司
    stock_price: float = 0.0             # 股价 (元) - 上市公司
    is_listed: bool = False              # 是否上市
    industry_ranking: int = 0            # 行业排名 (0表示未排名)
    market_share: float = 0.0            # 市场份额 (%)
    credit_rating: str = "BBB"           # 信用评级 (AAA, AA, A, BBB, BB, B, CCC等)
    
    # 专利情况
    total_patents: int = 0               # 专利总数
    invention_patents: int = 0           # 发明专利数量
    utility_patents: int = 0             # 实用新型专利数量
    design_patents: int = 0              # 外观设计专利数量
    patent_quality_score: float = 0.0   # 专利质量评分 (0-1)
    patent_citations: int = 0            # 专利被引用次数
    
    # 研发投入
    rnd_expense: float = 0.0             # 研发费用 (元)
    rnd_expense_ratio: float = 0.0      # 研发费用率 (%)
    rnd_personnel: int = 0               # 研发人员数量
    rnd_personnel_ratio: float = 0.0    # 研发人员占比 (%)
    rnd_projects: int = 0                # 在研项目数量
    
    # 其他企业指标
    employee_count: int = 0              # 员工数量
    registered_capital: float = 0.0      # 注册资本 (元)
    paid_in_capital: float = 0.0        # 实缴资本 (元)
    legal_representative_age: int = 45   # 法人代表年龄
    has_legal_disputes: bool = False    # 是否有法律纠纷
    legal_dispute_count: int = 0        # 法律纠纷数量
    tax_compliance_score: float = 1.0   # 税务合规评分 (0-1)
    environmental_compliance: bool = True # 环保合规
    
    # 衍生特征
    @property
    def is_enterprise(self) -> bool:
        """判断是否为企业客户"""
        return self.customer_type in [
            CustomerType.MICRO_ENTERPRISE,
            CustomerType.SMALL_ENTERPRISE,
            CustomerType.MEDIUM_ENTERPRISE,
            CustomerType.LARGE_ENTERPRISE
        ]
    
    @property
    def age_group(self) -> str:
        """年龄组（个人客户）或成立年限组（企业客户）"""
        if self.is_enterprise:
            if self.years_in_business < 3:
                return "0-3年"
            elif self.years_in_business < 5:
                return "3-5年"
            elif self.years_in_business < 10:
                return "5-10年"
            else:
                return "10年以上"
        else:
            if self.age < 25:
                return "18-25"
            elif self.age < 35:
                return "25-35"
            elif self.age < 45:
                return "35-45"
            elif self.age < 55:
                return "45-55"
            else:
                return "55+"
    
    @property
    def debt_ratio(self) -> float:
        """负债率（资产负债率）"""
        if self.total_assets <= 0:
            return 1.0
        return min(1.0, self.total_liabilities / self.total_assets)
    
    @property
    def debt_to_income(self) -> float:
        """债务收入比"""
        if self.is_enterprise:
            annual_income = self.annual_revenue
        else:
            annual_income = self.monthly_income * 12
        if annual_income <= 0:
            return float('inf')
        return self.total_liabilities / annual_income
    
    @property
    def enterprise_size_score(self) -> float:
        """企业规模评分 (0-1)"""
        if not self.is_enterprise:
            return 0.0
        if self.customer_type == CustomerType.MICRO_ENTERPRISE:
            return 0.2
        elif self.customer_type == CustomerType.SMALL_ENTERPRISE:
            return 0.4
        elif self.customer_type == CustomerType.MEDIUM_ENTERPRISE:
            return 0.7
        elif self.customer_type == CustomerType.LARGE_ENTERPRISE:
            return 1.0
        return 0.5
    
    @property
    def financial_health_score(self) -> float:
        """财务健康度评分 (0-1, 越高越健康) - 企业客户专用"""
        if not self.is_enterprise:
            return 0.5
        
        score = 0.5  # 基础分
        
        # 盈利能力
        if self.profit_margin > 0.1:
            score += 0.15
        elif self.profit_margin > 0.05:
            score += 0.10
        elif self.profit_margin > 0:
            score += 0.05
        
        # 流动性
        if self.current_ratio > 2.0:
            score += 0.10
        elif self.current_ratio > 1.5:
            score += 0.05
        
        # 现金流
        if self.operating_cash_flow > 0:
            score += 0.10
        
        # 增长性
        if self.revenue_growth_rate > 0.2:
            score += 0.10
        elif self.revenue_growth_rate > 0.1:
            score += 0.05
        
        # 负债率
        if self.debt_ratio < 0.3:
            score += 0.10
        elif self.debt_ratio < 0.5:
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    @property
    def innovation_score(self) -> float:
        """创新能力评分 (0-1) - 企业客户专用"""
        if not self.is_enterprise:
            return 0.0
        
        score = 0.0
        
        # 研发投入
        if self.rnd_expense_ratio > 0.10:
            score += 0.3
        elif self.rnd_expense_ratio > 0.05:
            score += 0.2
        elif self.rnd_expense_ratio > 0.02:
            score += 0.1
        
        # 专利数量
        if self.total_patents > 100:
            score += 0.3
        elif self.total_patents > 50:
            score += 0.2
        elif self.total_patents > 10:
            score += 0.1
        
        # 专利质量
        score += self.patent_quality_score * 0.2
        
        # 研发人员
        if self.rnd_personnel_ratio > 0.30:
            score += 0.2
        elif self.rnd_personnel_ratio > 0.15:
            score += 0.1
        
        return min(1.0, score)
    
    @property
    def risk_score(self) -> float:
        """风险评分 (0-1, 越高风险越大)"""
        if self.is_enterprise:
            return self._enterprise_risk_score()
        else:
            return self._personal_risk_score()
    
    def _personal_risk_score(self) -> float:
        """个人客户风险评分"""
        score = 0.0
        
        # 年龄因素
        if self.age < 25 or self.age > 60:
            score += 0.1
        
        # 收入稳定性
        score += self.income_volatility * 0.2
        
        # 负债率
        score += self.debt_ratio * 0.25
        
        # 历史信用
        if self.max_historical_dpd > 0:
            score += min(0.3, self.max_historical_dpd / 90 * 0.3)
        
        # 从业年限
        if self.years_in_business < 2:
            score += 0.1
        
        return min(1.0, score)
    
    def _enterprise_risk_score(self) -> float:
        """企业客户风险评分"""
        score = 0.0
        
        # 财务健康度（反向）
        score += (1 - self.financial_health_score) * 0.3
        
        # 负债率
        score += self.debt_ratio * 0.2
        
        # 流动性风险
        if self.current_ratio < 1.0:
            score += 0.15
        elif self.current_ratio < 1.2:
            score += 0.10
        
        # 现金流风险
        if self.operating_cash_flow < 0:
            score += 0.15
        
        # 历史信用
        if self.max_historical_dpd > 0:
            score += min(0.2, self.max_historical_dpd / 90 * 0.2)
        
        # 经营年限
        if self.years_in_business < 3:
            score += 0.10
        elif self.years_in_business < 5:
            score += 0.05
        
        # 法律纠纷
        if self.has_legal_disputes:
            score += 0.10
        score += min(0.05, self.legal_dispute_count * 0.01)
        
        # 税务合规
        score += (1 - self.tax_compliance_score) * 0.05
        
        return min(1.0, score)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        base_dict = {
            'customer_id': self.customer_id,
            'age': self.age,
            'age_group': self.age_group,
            'city_tier': self.city_tier.value if self.city_tier else None,
            'customer_type': self.customer_type.value if self.customer_type else None,
            'industry': self.industry.value if self.industry else None,
            'years_in_business': self.years_in_business,
            'is_enterprise': self.is_enterprise,
            'total_assets': self.total_assets,
            'total_liabilities': self.total_liabilities,
            'debt_ratio': self.debt_ratio,
            'debt_to_income': self.debt_to_income,
            'deposit_balance': self.deposit_balance,
            'deposit_stability': self.deposit_stability,
            'months_as_customer': self.months_as_customer,
            'previous_loans': self.previous_loans,
            'max_historical_dpd': self.max_historical_dpd,
            'months_since_last_loan': self.months_since_last_loan,
            'risk_score': self.risk_score,
        }
        
        # 个人客户字段
        if not self.is_enterprise:
            base_dict.update({
                'monthly_income': self.monthly_income,
                'income_volatility': self.income_volatility,
            })
        else:
            # 企业客户字段
            base_dict.update({
                # 财务指标
                'annual_revenue': self.annual_revenue,
                'net_profit': self.net_profit,
                'operating_cash_flow': self.operating_cash_flow,
                'ebitda': self.ebitda,
                'revenue_growth_rate': self.revenue_growth_rate,
                'profit_margin': self.profit_margin,
                'current_ratio': self.current_ratio,
                'quick_ratio': self.quick_ratio,
                'asset_turnover': self.asset_turnover,
                # 资产情况
                'fixed_assets': self.fixed_assets,
                'current_assets': self.current_assets,
                'intangible_assets': self.intangible_assets,
                'inventory': self.inventory,
                'accounts_receivable': self.accounts_receivable,
                'accounts_payable': self.accounts_payable,
                # 市场信息
                'market_cap': self.market_cap,
                'stock_price': self.stock_price,
                'is_listed': self.is_listed,
                'industry_ranking': self.industry_ranking,
                'market_share': self.market_share,
                'credit_rating': self.credit_rating,
                # 专利情况
                'total_patents': self.total_patents,
                'invention_patents': self.invention_patents,
                'utility_patents': self.utility_patents,
                'design_patents': self.design_patents,
                'patent_quality_score': self.patent_quality_score,
                'patent_citations': self.patent_citations,
                # 研发投入
                'rnd_expense': self.rnd_expense,
                'rnd_expense_ratio': self.rnd_expense_ratio,
                'rnd_personnel': self.rnd_personnel,
                'rnd_personnel_ratio': self.rnd_personnel_ratio,
                'rnd_projects': self.rnd_projects,
                # 其他指标
                'employee_count': self.employee_count,
                'registered_capital': self.registered_capital,
                'paid_in_capital': self.paid_in_capital,
                'legal_representative_age': self.legal_representative_age,
                'has_legal_disputes': self.has_legal_disputes,
                'legal_dispute_count': self.legal_dispute_count,
                'tax_compliance_score': self.tax_compliance_score,
                'environmental_compliance': self.environmental_compliance,
                # 衍生指标
                'financial_health_score': self.financial_health_score,
                'innovation_score': self.innovation_score,
                'enterprise_size_score': self.enterprise_size_score,
            })
        
        return base_dict
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量 (用于模型输入) - 支持个人和企业客户"""
        if not self.is_enterprise:
            # 个人客户特征向量（原有16维）
            features = [
                self.age / 100,
                list(CityTier).index(self.city_tier) / 3 if self.city_tier else 0,
                list(CustomerType).index(self.customer_type) / 7 if self.customer_type else 0,
                list(Industry).index(self.industry) / 10 if self.industry else 0,
                min(self.years_in_business / 30, 1.0),
                min(self.monthly_income / 100000, 1.0),
                self.income_volatility,
                min(self.total_assets / 5000000, 1.0),
                self.debt_ratio,
                self.debt_to_income / 10 if self.debt_to_income != float('inf') else 1.0,
                min(self.deposit_balance / 500000, 1.0),
                self.deposit_stability,
                min(self.months_as_customer / 120, 1.0),
                min(self.previous_loans / 10, 1.0),
                min(self.max_historical_dpd / 180, 1.0),
                min(self.months_since_last_loan / 60, 1.0) if self.months_since_last_loan > 0 else 0,
            ]
        else:
            # 企业客户特征向量（扩展至60+维）
            features = [
                # 基础信息 (5维)
                min(self.years_in_business / 30, 1.0),
                list(CityTier).index(self.city_tier) / 3 if self.city_tier else 0,
                list(CustomerType).index(self.customer_type) / 7 if self.customer_type else 0,
                list(Industry).index(self.industry) / 10 if self.industry else 0,
                self.enterprise_size_score,
                
                # 财务指标 (10维)
                min(self.annual_revenue / 1e10, 1.0),  # 归一化到100亿
                min(self.net_profit / 1e9, 1.0),  # 归一化到10亿
                min(self.operating_cash_flow / 1e9, 1.0),
                min(self.ebitda / 1e9, 1.0),
                np.clip(self.revenue_growth_rate, -0.5, 1.0) + 0.5,  # 归一化到0-1
                np.clip(self.profit_margin, -0.2, 0.5) + 0.2,  # 归一化到0-1
                min(self.current_ratio / 5.0, 1.0),
                min(self.quick_ratio / 5.0, 1.0),
                min(self.asset_turnover / 5.0, 1.0),
                self.debt_ratio,
                
                # 资产情况 (6维)
                min(self.fixed_assets / 1e9, 1.0),
                min(self.current_assets / 1e9, 1.0),
                min(self.intangible_assets / 1e8, 1.0),
                min(self.inventory / 1e8, 1.0),
                min(self.accounts_receivable / 1e8, 1.0),
                min(self.accounts_payable / 1e8, 1.0),
                
                # 市场信息 (6维)
                min(self.market_cap / 1e11, 1.0) if self.is_listed else 0.0,  # 归一化到1000亿
                min(self.stock_price / 100, 1.0) if self.is_listed else 0.0,
                1.0 if self.is_listed else 0.0,
                min(self.industry_ranking / 100, 1.0) if self.industry_ranking > 0 else 0.0,
                min(self.market_share / 50.0, 1.0),  # 归一化到50%
                # 信用评级编码 (简化)
                0.9 if self.credit_rating.startswith('AAA') else
                0.8 if self.credit_rating.startswith('AA') else
                0.7 if self.credit_rating.startswith('A') else
                0.6 if self.credit_rating.startswith('BBB') else
                0.4 if self.credit_rating.startswith('BB') else
                0.2 if self.credit_rating.startswith('B') else 0.1,
                
                # 专利情况 (6维)
                min(self.total_patents / 1000, 1.0),
                min(self.invention_patents / 500, 1.0),
                min(self.utility_patents / 500, 1.0),
                min(self.design_patents / 200, 1.0),
                self.patent_quality_score,
                min(self.patent_citations / 1000, 1.0),
                
                # 研发投入 (5维)
                min(self.rnd_expense / 1e9, 1.0),  # 归一化到10亿
                min(self.rnd_expense_ratio / 0.3, 1.0),  # 归一化到30%
                min(self.rnd_personnel / 10000, 1.0),
                min(self.rnd_personnel_ratio, 1.0),
                min(self.rnd_projects / 100, 1.0),
                
                # 其他指标 (6维)
                min(self.employee_count / 50000, 1.0),  # 归一化到5万人
                min(self.registered_capital / 1e9, 1.0),
                min(self.paid_in_capital / 1e9, 1.0),
                self.legal_representative_age / 100,
                1.0 if self.has_legal_disputes else 0.0,
                min(self.legal_dispute_count / 10, 1.0),
                self.tax_compliance_score,
                1.0 if self.environmental_compliance else 0.0,
                
                # 银行关系 (3维)
                min(self.deposit_balance / 1e9, 1.0),
                self.deposit_stability,
                min(self.months_as_customer / 120, 1.0),
                
                # 信贷历史 (3维)
                min(self.previous_loans / 20, 1.0),
                min(self.max_historical_dpd / 180, 1.0),
                min(self.months_since_last_loan / 60, 1.0) if self.months_since_last_loan > 0 else 0,
                
                # 衍生指标 (3维)
                self.financial_health_score,
                self.innovation_score,
                self.risk_score,
            ]
        
        return np.array(features, dtype=np.float32)


class CustomerGenerator:
    """
    客户生成器
    
    根据统计分布生成虚拟但符合真实规律的客户数据
    """
    
    # 各客户类型的分布参数
    DISTRIBUTION_PARAMS = {
        # 个人客户
        CustomerType.SALARIED: {
            'income_mean': 12000,
            'income_std': 8000,
            'assets_mean': 300000,
            'assets_std': 200000,
            'default_base_rate': 0.02,
        },
        CustomerType.SMALL_BUSINESS: {
            'income_mean': 25000,
            'income_std': 20000,
            'assets_mean': 800000,
            'assets_std': 600000,
            'default_base_rate': 0.04,
        },
        CustomerType.FREELANCER: {
            'income_mean': 15000,
            'income_std': 12000,
            'assets_mean': 200000,
            'assets_std': 150000,
            'default_base_rate': 0.05,
        },
        CustomerType.FARMER: {
            'income_mean': 6000,
            'income_std': 4000,
            'assets_mean': 400000,
            'assets_std': 300000,
            'default_base_rate': 0.03,
        },
        # 企业客户
        CustomerType.MICRO_ENTERPRISE: {
            'revenue_mean': 5e6,      # 500万
            'revenue_std': 3e6,
            'assets_mean': 10e6,      # 1000万
            'assets_std': 5e6,
            'default_base_rate': 0.06,
        },
        CustomerType.SMALL_ENTERPRISE: {
            'revenue_mean': 30e6,     # 3000万
            'revenue_std': 15e6,
            'assets_mean': 50e6,      # 5000万
            'assets_std': 25e6,
            'default_base_rate': 0.05,
        },
        CustomerType.MEDIUM_ENTERPRISE: {
            'revenue_mean': 100e6,    # 1亿
            'revenue_std': 50e6,
            'assets_mean': 200e6,     # 2亿
            'assets_std': 100e6,
            'default_base_rate': 0.04,
        },
        CustomerType.LARGE_ENTERPRISE: {
            'revenue_mean': 500e6,    # 5亿
            'revenue_std': 300e6,
            'assets_mean': 1000e6,    # 10亿
            'assets_std': 500e6,
            'default_base_rate': 0.03,
        },
    }
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self._id_counter = 0
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        self._id_counter += 1
        return f"CUST_{self._id_counter:08d}"
    
    def generate_one(
        self,
        customer_type: CustomerType = None,
        city_tier: CityTier = None,
        industry: Industry = None,
        risk_profile: str = "mixed"  # "low", "medium", "high", "mixed"
    ) -> CustomerProfile:
        """
        生成单个客户
        
        Args:
            customer_type: 指定客户类型，None则随机
            city_tier: 指定城市等级，None则随机
            risk_profile: 风险画像偏好
        """
        # 随机选择客户类型
        if customer_type is None:
            customer_type = self.rng.choice(list(CustomerType))
        
        # 随机选择城市
        if city_tier is None:
            city_tier = self.rng.choice(
                list(CityTier),
                p=[0.15, 0.30, 0.35, 0.20]
            )
        
        # 判断是否为企业客户
        is_enterprise = customer_type in [
            CustomerType.MICRO_ENTERPRISE,
            CustomerType.SMALL_ENTERPRISE,
            CustomerType.MEDIUM_ENTERPRISE,
            CustomerType.LARGE_ENTERPRISE
        ]
        
        params = self.DISTRIBUTION_PARAMS.get(customer_type, self.DISTRIBUTION_PARAMS[CustomerType.SALARIED])
        
        # 行业选择（如果未指定）
        if industry is None:
            if customer_type == CustomerType.FARMER:
                industry = Industry.AGRICULTURE
            elif customer_type == CustomerType.SALARIED:
                industry = self.rng.choice([
                    Industry.MANUFACTURING, Industry.SERVICE, Industry.IT,
                    Industry.FINANCE, Industry.EDUCATION, Industry.HEALTHCARE
                ])
            else:
                industry = self.rng.choice([
                    Industry.RETAIL, Industry.CATERING, Industry.SERVICE,
                    Industry.MANUFACTURING, Industry.CONSTRUCTION
                ])
        
        if is_enterprise:
            # 企业客户生成逻辑（简化版，后续可扩展）
            return self._generate_enterprise_customer(customer_type, city_tier, industry, risk_profile)
        else:
            # 个人客户生成逻辑（原有逻辑）
            # 年龄
            if customer_type == CustomerType.FARMER:
                age = int(self.rng.normal(45, 12))
            elif customer_type == CustomerType.SMALL_BUSINESS:
                age = int(self.rng.normal(40, 10))
            else:
                age = int(self.rng.normal(35, 10))
            age = max(22, min(65, age))
            
            # 行业已在上面定义
            
            # 收入 (根据风险画像调整)
            income_factor = 1.0
            if risk_profile == "low":
                income_factor = 1.3
            elif risk_profile == "high":
                income_factor = 0.7
            
            monthly_income = max(3000, self.rng.normal(
                params['income_mean'] * income_factor,
                params['income_std']
            ))
            
            # 收入波动性
            if customer_type in [CustomerType.SMALL_BUSINESS, CustomerType.FREELANCER]:
                income_volatility = self.rng.beta(2, 5)  # 偏高
            else:
                income_volatility = self.rng.beta(2, 10)  # 偏低
            
            # 资产
            total_assets = max(10000, self.rng.normal(
                params['assets_mean'],
                params['assets_std']
            ))
            
            # 负债 (根据风险画像调整)
            if risk_profile == "low":
                debt_ratio_mean = 0.25
            elif risk_profile == "high":
                debt_ratio_mean = 0.60
            else:
                debt_ratio_mean = 0.40
            
            debt_ratio = max(0, min(0.9, self.rng.normal(debt_ratio_mean, 0.15)))
            total_liabilities = total_assets * debt_ratio
            
            # 从业年限
            max_years = (age - 22) * 0.8
            years_in_business = max(0.5, self.rng.exponential(max_years / 3))
            years_in_business = min(years_in_business, max_years)
            
            # 存款
            deposit_balance = max(0, self.rng.exponential(monthly_income * 3))
            deposit_stability = self.rng.beta(3, 2)
            
            # 银行关系
            months_as_customer = int(self.rng.exponential(24))
            
            # 信贷历史
            previous_loans = int(self.rng.poisson(2))
            
            # 历史逾期 (根据风险画像)
            if risk_profile == "low":
                max_dpd = 0 if self.rng.random() > 0.1 else int(self.rng.exponential(7))
            elif risk_profile == "high":
                max_dpd = 0 if self.rng.random() > 0.4 else int(self.rng.exponential(30))
            else:
                max_dpd = 0 if self.rng.random() > 0.2 else int(self.rng.exponential(15))
            
            months_since_last_loan = int(self.rng.exponential(12)) if previous_loans > 0 else 0
            
            return CustomerProfile(
                customer_id=self._generate_id(),
                age=age,
                city_tier=city_tier,
                customer_type=customer_type,
                industry=industry,
                years_in_business=round(years_in_business, 1),
                monthly_income=round(monthly_income, 2),
                income_volatility=round(income_volatility, 3),
                total_assets=round(total_assets, 2),
                total_liabilities=round(total_liabilities, 2),
                deposit_balance=round(deposit_balance, 2),
                deposit_stability=round(deposit_stability, 3),
                months_as_customer=months_as_customer,
                previous_loans=previous_loans,
                max_historical_dpd=max_dpd,
                months_since_last_loan=months_since_last_loan,
            )
    
    def _generate_enterprise_customer(
        self,
        customer_type: CustomerType,
        city_tier: CityTier,
        industry: Industry,
        risk_profile: str
    ) -> CustomerProfile:
        """生成企业客户（简化版）"""
        params = self.DISTRIBUTION_PARAMS.get(customer_type, self.DISTRIBUTION_PARAMS[CustomerType.MICRO_ENTERPRISE])
        
        # 成立年限（代替年龄）
        years_in_business = max(1, self.rng.exponential(5))
        age = int(30 + years_in_business)  # 法人代表年龄
        
        # 行业（如果没有指定）
        if industry is None:
            industry = self.rng.choice(list(Industry))
        
        # 年营收
        revenue_factor = 1.0
        if risk_profile == "low":
            revenue_factor = 1.3
        elif risk_profile == "high":
            revenue_factor = 0.7
        
        annual_revenue = max(1e6, self.rng.normal(
            params['revenue_mean'] * revenue_factor,
            params['revenue_std']
        ))
        
        # 资产
        total_assets = max(1e6, self.rng.normal(
            params['assets_mean'],
            params['assets_std']
        ))
        
        # 负债率
        if risk_profile == "low":
            debt_ratio_mean = 0.3
        elif risk_profile == "high":
            debt_ratio_mean = 0.65
        else:
            debt_ratio_mean = 0.5
        
        debt_ratio = max(0, min(0.85, self.rng.normal(debt_ratio_mean, 0.15)))
        total_liabilities = total_assets * debt_ratio
        
        # 净利润（简化计算）
        profit_margin = self.rng.normal(0.08, 0.05) if risk_profile != "high" else self.rng.normal(0.02, 0.03)
        profit_margin = max(0.01, min(0.20, profit_margin))
        net_profit = annual_revenue * profit_margin
        
        # EBITDA（息税折旧摊销前利润）
        ebitda = net_profit * self.rng.uniform(1.3, 1.8)
        
        # 营收增长率
        if risk_profile == "low":
            revenue_growth_rate = self.rng.normal(0.15, 0.10)
        elif risk_profile == "high":
            revenue_growth_rate = self.rng.normal(-0.05, 0.10)
        else:
            revenue_growth_rate = self.rng.normal(0.05, 0.10)
        revenue_growth_rate = max(-0.3, min(0.5, revenue_growth_rate))
        
        # 现金流
        operating_cash_flow = net_profit * self.rng.uniform(0.8, 1.2)
        
        # 财务比率
        current_ratio = self.rng.normal(1.5, 0.5) if risk_profile != "high" else self.rng.normal(1.0, 0.3)
        current_ratio = max(0.5, current_ratio)
        quick_ratio = current_ratio * self.rng.uniform(0.7, 0.9)
        
        # 资产周转率
        asset_turnover = annual_revenue / total_assets if total_assets > 0 else 0
        asset_turnover = max(0.3, min(3.0, asset_turnover))
        
        # 资产结构
        fixed_assets = total_assets * self.rng.uniform(0.3, 0.6)
        current_assets = total_assets * self.rng.uniform(0.3, 0.5)
        intangible_assets = total_assets * self.rng.uniform(0.01, 0.1)
        
        # 流动资产明细
        inventory = current_assets * self.rng.uniform(0.2, 0.4)  # 存货
        accounts_receivable = current_assets * self.rng.uniform(0.3, 0.5)  # 应收账款
        accounts_payable = total_liabilities * self.rng.uniform(0.2, 0.4)  # 应付账款
        
        # 员工数（根据营收）- 需要先计算，因为后面要用
        employee_count = int(annual_revenue / self.rng.uniform(50000, 200000))
        employee_count = max(5, employee_count)  # 至少5人
        
        # 研发投入（根据企业规模）
        rnd_ratio = self.rng.uniform(0.02, 0.15) if customer_type in [CustomerType.MEDIUM_ENTERPRISE, CustomerType.LARGE_ENTERPRISE] else self.rng.uniform(0.01, 0.05)
        rnd_expense = annual_revenue * rnd_ratio
        
        # 研发人员
        rnd_personnel = int(employee_count * rnd_ratio * self.rng.uniform(0.8, 1.2))
        rnd_personnel = max(0, min(employee_count, rnd_personnel))
        rnd_personnel_ratio = rnd_personnel / max(employee_count, 1)
        rnd_projects = int(rnd_personnel / self.rng.uniform(3, 8)) if rnd_personnel > 0 else 0
        
        # 专利（根据研发投入）
        total_patents = int(rnd_expense / 1e5 * self.rng.uniform(0.5, 2.0))
        total_patents = max(0, total_patents)
        invention_patents = int(total_patents * self.rng.uniform(0.2, 0.4)) if total_patents > 0 else 0
        utility_patents = int(total_patents * self.rng.uniform(0.4, 0.6)) if total_patents > 0 else 0
        design_patents = max(0, total_patents - invention_patents - utility_patents)
        
        # 专利质量评分
        patent_quality_score = min(1.0, (invention_patents / max(total_patents, 1)) * 1.5 + (total_patents / 100) * 0.3) if total_patents > 0 else 0.0
        patent_citations = int(total_patents * self.rng.uniform(2, 10)) if total_patents > 0 else 0
        
        # 市场信息
        is_listed = self.rng.random() < 0.1 if customer_type == CustomerType.LARGE_ENTERPRISE else (
            self.rng.random() < 0.05 if customer_type == CustomerType.MEDIUM_ENTERPRISE else False
        )
        market_cap = annual_revenue * self.rng.uniform(1.5, 5.0) if is_listed else 0
        stock_price = self.rng.uniform(10, 100) if is_listed else 0
        
        # 行业排名和市场份额
        industry_ranking = 0
        market_share = 0.0
        if customer_type == CustomerType.LARGE_ENTERPRISE:
            industry_ranking = self.rng.integers(1, 50)
            market_share = self.rng.uniform(1.0, 10.0)
        elif customer_type == CustomerType.MEDIUM_ENTERPRISE:
            industry_ranking = self.rng.integers(50, 200) if self.rng.random() < 0.3 else 0
            market_share = self.rng.uniform(0.1, 2.0) if industry_ranking > 0 else 0.0
        
        # 计算财务健康度（用于信用评级）
        financial_health_score = 0.5  # 基础分
        if profit_margin > 0.1:
            financial_health_score += 0.15
        elif profit_margin > 0.05:
            financial_health_score += 0.10
        elif profit_margin > 0:
            financial_health_score += 0.05
        if current_ratio > 2.0:
            financial_health_score += 0.10
        elif current_ratio > 1.5:
            financial_health_score += 0.05
        if operating_cash_flow > 0:
            financial_health_score += 0.10
        if revenue_growth_rate > 0.2:
            financial_health_score += 0.10
        elif revenue_growth_rate > 0.1:
            financial_health_score += 0.05
        if debt_ratio < 0.3:
            financial_health_score += 0.10
        elif debt_ratio < 0.5:
            financial_health_score += 0.05
        financial_health_score = min(1.0, max(0.0, financial_health_score))
        
        # 信用评级（基于财务健康度）
        credit_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        if financial_health_score > 0.8:
            credit_rating = self.rng.choice(credit_ratings[:3], p=[0.1, 0.3, 0.6])
        elif financial_health_score > 0.6:
            credit_rating = self.rng.choice(credit_ratings[2:5], p=[0.2, 0.5, 0.3])
        else:
            credit_rating = self.rng.choice(credit_ratings[4:], p=[0.4, 0.4, 0.2])
        
        # 注册资本和实缴资本
        registered_capital = total_assets * self.rng.uniform(0.3, 0.8)
        paid_in_capital = registered_capital * self.rng.uniform(0.7, 1.0)
        
        # 法人代表年龄
        legal_representative_age = int(30 + years_in_business * 0.8)
        legal_representative_age = max(25, min(70, legal_representative_age))
        
        # 法律纠纷
        has_legal_disputes = self.rng.random() < 0.15
        legal_dispute_count = int(self.rng.exponential(2)) if has_legal_disputes else 0
        
        # 税务合规评分
        tax_compliance_score = self.rng.beta(8, 2) if risk_profile != "high" else self.rng.beta(5, 3)
        
        # 环保合规
        environmental_compliance = self.rng.random() > 0.1
        
        # 存款
        deposit_balance = max(0, annual_revenue * self.rng.uniform(0.05, 0.15))
        deposit_stability = self.rng.beta(3, 2)
        
        # 银行关系
        months_as_customer = int(self.rng.exponential(36))
        
        # 信贷历史
        previous_loans = int(self.rng.poisson(3))
        max_dpd = 0 if self.rng.random() > 0.15 else int(self.rng.exponential(20))
        months_since_last_loan = int(self.rng.exponential(12)) if previous_loans > 0 else 0
        
        return CustomerProfile(
            customer_id=self._generate_id(),
            age=age,
            city_tier=city_tier,
            customer_type=customer_type,
            industry=industry,
            years_in_business=round(years_in_business, 1),
            # 企业财务指标
            annual_revenue=round(annual_revenue, 2),
            net_profit=round(net_profit, 2),
            operating_cash_flow=round(operating_cash_flow, 2),
            ebitda=round(ebitda, 2),
            revenue_growth_rate=round(revenue_growth_rate, 3),
            profit_margin=round(profit_margin, 3),
            current_ratio=round(current_ratio, 2),
            quick_ratio=round(quick_ratio, 2),
            asset_turnover=round(asset_turnover, 2),
            total_assets=round(total_assets, 2),
            total_liabilities=round(total_liabilities, 2),
            fixed_assets=round(fixed_assets, 2),
            current_assets=round(current_assets, 2),
            intangible_assets=round(intangible_assets, 2),
            inventory=round(inventory, 2),
            accounts_receivable=round(accounts_receivable, 2),
            accounts_payable=round(accounts_payable, 2),
            # 市场信息
            market_cap=round(market_cap, 2),
            stock_price=round(stock_price, 2),
            is_listed=is_listed,
            industry_ranking=industry_ranking,
            market_share=round(market_share, 2),
            credit_rating=credit_rating,
            # 专利情况
            total_patents=total_patents,
            invention_patents=invention_patents,
            utility_patents=utility_patents,
            design_patents=design_patents,
            patent_quality_score=round(patent_quality_score, 3),
            patent_citations=patent_citations,
            # 研发投入
            rnd_expense=round(rnd_expense, 2),
            rnd_expense_ratio=round(rnd_ratio, 3),
            rnd_personnel=rnd_personnel,
            rnd_personnel_ratio=round(rnd_personnel_ratio, 3),
            rnd_projects=rnd_projects,
            # 其他指标
            employee_count=employee_count,
            registered_capital=round(registered_capital, 2),
            paid_in_capital=round(paid_in_capital, 2),
            legal_representative_age=legal_representative_age,
            has_legal_disputes=has_legal_disputes,
            legal_dispute_count=legal_dispute_count,
            tax_compliance_score=round(tax_compliance_score, 3),
            environmental_compliance=environmental_compliance,
            # 银行关系
            deposit_balance=round(deposit_balance, 2),
            deposit_stability=round(deposit_stability, 3),
            months_as_customer=months_as_customer,
            previous_loans=previous_loans,
            max_historical_dpd=max_dpd,
            months_since_last_loan=months_since_last_loan,
        )
    
    def generate_batch(
        self,
        n: int,
        risk_distribution: Dict[str, float] = None
    ) -> List[CustomerProfile]:
        """
        批量生成客户
        
        Args:
            n: 生成数量
            risk_distribution: 风险分布 {"low": 0.3, "medium": 0.5, "high": 0.2}
        """
        if risk_distribution is None:
            risk_distribution = {"low": 0.30, "medium": 0.50, "high": 0.20}
        
        customers = []
        for _ in range(n):
            risk = self.rng.choice(
                list(risk_distribution.keys()),
                p=list(risk_distribution.values())
            )
            customers.append(self.generate_one(risk_profile=risk))
        
        return customers
    
    def generate_historical_dataset(
        self,
        n_customers: int,
        years: int = 5
    ) -> List[Dict]:
        """
        生成历史数据集 (模拟真实的贷款历史记录)
        
        Returns:
            包含客户画像、贷款条件和实际结果的历史记录列表
        """
        from .world_model import WorldModel, LoanOffer, MarketConditions
        
        world_model = WorldModel(seed=self.rng.integers(0, 10000))
        records = []
        
        for year in range(years):
            # 模拟每年的宏观环境
            if year % 5 == 0:
                gdp_growth = self.rng.uniform(0.02, 0.06)  # 繁荣
            elif year % 5 == 2:
                gdp_growth = self.rng.uniform(-0.02, 0.02)  # 衰退
            else:
                gdp_growth = self.rng.uniform(0.01, 0.04)  # 正常
            
            base_rate = self.rng.uniform(0.03, 0.06)
            unemployment = max(0.03, 0.05 - gdp_growth * 0.5 + self.rng.normal(0, 0.01))
            
            market = MarketConditions(
                gdp_growth=gdp_growth,
                base_interest_rate=base_rate,
                unemployment_rate=unemployment,
                inflation_rate=self.rng.uniform(0.01, 0.04),
                credit_spread=self.rng.uniform(0.01, 0.03),
            )
            
            # 生成该年的客户
            n_year = n_customers // years
            customers = self.generate_batch(n_year)
            
            for customer in customers:
                # 生成贷款条件
                loan_amount = min(
                    customer.monthly_income * self.rng.uniform(6, 24),
                    customer.total_assets * 0.5
                )
                loan_rate = base_rate + 0.02 + customer.risk_score * 0.08
                
                offer = LoanOffer(
                    amount=loan_amount,
                    interest_rate=loan_rate,
                    term_months=int(self.rng.choice([12, 24, 36, 48, 60])),
                    approved=self.rng.random() > customer.risk_score * 0.5,
                )
                
                if offer.approved:
                    # 预测客户未来表现
                    future = world_model.predict_customer_future(
                        customer, offer, market
                    )
                    
                    # 模拟实际结果 (加入随机性)
                    actual_default = self.rng.random() < future.default_probability
                    actual_dpd = int(self.rng.exponential(30)) if actual_default else 0
                    
                    records.append({
                        'year': 2019 + year,
                        'customer': customer.to_dict(),
                        'loan_offer': {
                            'amount': offer.amount,
                            'interest_rate': offer.interest_rate,
                            'term_months': offer.term_months,
                        },
                        'market_conditions': {
                            'gdp_growth': market.gdp_growth,
                            'base_interest_rate': market.base_interest_rate,
                            'unemployment_rate': market.unemployment_rate,
                        },
                        'predicted': {
                            'default_probability': future.default_probability,
                            'expected_ltv': future.expected_ltv,
                        },
                        'actual': {
                            'defaulted': actual_default,
                            'max_dpd': actual_dpd,
                        },
                    })
        
        return records


if __name__ == "__main__":
    # 测试客户生成器
    generator = CustomerGenerator(seed=42)
    
    print("=" * 60)
    print("客户生成器测试")
    print("=" * 60)
    
    # 生成单个客户
    customer = generator.generate_one(risk_profile="medium")
    print("\n单个客户示例:")
    for key, value in customer.to_dict().items():
        print(f"  {key}: {value}")
    
    # 批量生成
    customers = generator.generate_batch(100)
    print(f"\n批量生成 {len(customers)} 个客户")
    
    # 统计
    risk_scores = [c.risk_score for c in customers]
    print(f"  风险评分分布: 均值={np.mean(risk_scores):.3f}, 标准差={np.std(risk_scores):.3f}")
    
    incomes = [c.monthly_income for c in customers]
    print(f"  月收入分布: 均值={np.mean(incomes):,.0f}, 标准差={np.std(incomes):,.0f}")

