"""
世界模型 - 从历史数据蒸馏出的"商业物理定律"

这是数据蒸馏的核心输出：一个能预测客户未来行为的函数
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import pickle
from pathlib import Path

from .customer_generator import CustomerProfile, CustomerType, CityTier, Industry


@dataclass
class LoanOffer:
    """贷款条件"""
    amount: float              # 贷款金额 (元)
    interest_rate: float       # 年利率
    term_months: int           # 贷款期限 (月)
    approved: bool = True      # 是否批准
    
    @property
    def monthly_payment(self) -> float:
        """月供"""
        if not self.approved or self.amount <= 0:
            return 0.0
        monthly_rate = self.interest_rate / 12
        n = self.term_months
        if monthly_rate <= 0:
            return self.amount / n
        return self.amount * monthly_rate * (1 + monthly_rate)**n / ((1 + monthly_rate)**n - 1)


@dataclass
class MarketConditions:
    """宏观经济环境 - 扩展版"""
    # 基础指标
    gdp_growth: float          # GDP 增长率 (-0.05 ~ 0.10)
    base_interest_rate: float  # 基准利率 (0.02 ~ 0.08)
    unemployment_rate: float   # 失业率 (0.03 ~ 0.15)
    inflation_rate: float      # 通胀率 (0.00 ~ 0.08)
    credit_spread: float       # 信用利差 (0.01 ~ 0.05)
    
    # 扩展指标 - 宏观经济
    consumer_confidence: float = 0.5    # 消费者信心指数 (0-1)
    manufacturing_pmi: float = 50.0     # 制造业PMI (0-100)
    housing_price_index: float = 100.0   # 房地产价格指数 (基准100)
    stock_index: float = 3000.0          # 股市指数 (基准3000)
    m2_growth: float = 0.10             # 货币供应量M2增长率
    exchange_rate: float = 7.0           # 汇率 (人民币/美元)
    trade_balance: float = 0.0           # 贸易顺差/逆差 (亿美元)
    
    # 政策指标
    fiscal_policy_stance: float = 0.5   # 财政政策立场 (0=紧缩, 1=扩张)
    monetary_policy_stance: float = 0.5  # 货币政策立场 (0=紧缩, 1=宽松)
    
    # 市场情绪指标
    risk_appetite: float = 0.5          # 风险偏好指数 (0-1)
    liquidity_index: float = 0.5          # 流动性指标 (0-1)
    market_volatility: float = 0.15      # 市场波动率 (0-1)
    
    # 行业指标 (字典，key为行业名称)
    industry_health: Dict[str, float] = None  # 行业健康度 (0-1)
    
    def __post_init__(self):
        if self.industry_health is None:
            self.industry_health = {
                '制造业': 0.6,
                '服务业': 0.7,
                '房地产': 0.5,
                '金融': 0.7,
                '零售': 0.6,
                '餐饮': 0.5,
                '科技': 0.8,
            }
    
    @property
    def economic_stress(self) -> float:
        """经济压力指数 (0-1)"""
        stress = 0.0
        stress += max(0, -self.gdp_growth) * 5  # 负增长增加压力
        stress += (self.unemployment_rate - 0.05) * 3  # 高失业增加压力
        stress += max(0, self.inflation_rate - 0.03) * 2  # 高通胀增加压力
        stress += (1 - self.consumer_confidence) * 2  # 低信心增加压力
        stress += max(0, (50 - self.manufacturing_pmi) / 50) * 1.5  # PMI低于50增加压力
        return min(1.0, max(0.0, stress))
    
    @property
    def economic_health_score(self) -> float:
        """经济健康度评分 (0-100)"""
        score = 50.0
        score += self.gdp_growth * 200  # GDP增长贡献
        score -= (self.unemployment_rate - 0.05) * 100  # 失业率影响
        score -= abs(self.inflation_rate - 0.02) * 100  # 通胀偏离目标
        score += (self.consumer_confidence - 0.5) * 40  # 消费者信心
        score += (self.manufacturing_pmi - 50) * 0.5  # PMI影响
        return max(0, min(100, score))
    
    @property
    def policy_stimulus_level(self) -> float:
        """政策刺激水平 (0-1)"""
        return (self.fiscal_policy_stance + self.monetary_policy_stance) / 2
    
    def to_array(self) -> np.ndarray:
        """转换为数组，用于模型输入"""
        return np.array([
            self.gdp_growth,
            self.base_interest_rate,
            self.unemployment_rate,
            self.inflation_rate,
            self.credit_spread,
            self.economic_stress,
            self.consumer_confidence,
            self.manufacturing_pmi / 100.0,  # 归一化
            (self.housing_price_index - 100) / 100.0,  # 归一化
            (self.stock_index - 3000) / 3000.0,  # 归一化
            self.m2_growth,
            self.exchange_rate / 10.0,  # 归一化
            self.trade_balance / 1000.0,  # 归一化
            self.fiscal_policy_stance,
            self.monetary_policy_stance,
            self.risk_appetite,
            self.liquidity_index,
            self.market_volatility,
        ], dtype=np.float32)
    
    def to_dict(self) -> dict:
        """转换为字典，用于API返回"""
        return {
            'gdp_growth': self.gdp_growth,
            'base_interest_rate': self.base_interest_rate,
            'unemployment_rate': self.unemployment_rate,
            'inflation_rate': self.inflation_rate,
            'credit_spread': self.credit_spread,
            'consumer_confidence': self.consumer_confidence,
            'manufacturing_pmi': self.manufacturing_pmi,
            'housing_price_index': self.housing_price_index,
            'stock_index': self.stock_index,
            'm2_growth': self.m2_growth,
            'exchange_rate': self.exchange_rate,
            'trade_balance': self.trade_balance,
            'fiscal_policy_stance': self.fiscal_policy_stance,
            'monetary_policy_stance': self.monetary_policy_stance,
            'risk_appetite': self.risk_appetite,
            'liquidity_index': self.liquidity_index,
            'market_volatility': self.market_volatility,
            'economic_stress': self.economic_stress,
            'economic_health_score': self.economic_health_score,
            'policy_stimulus_level': self.policy_stimulus_level,
            'industry_health': self.industry_health,
        }


@dataclass
class CustomerFuture:
    """客户未来预测结果"""
    default_probability: float      # 违约概率 (0-1)
    expected_ltv: float             # 预期生命周期价值 (元)
    churn_probability: float        # 流失/提前还款概率 (0-1)
    expected_dpd: float             # 预期逾期天数
    confidence: float               # 预测置信度 (0-1)
    
    # 分解因素
    risk_factors: Dict[str, float] = None
    
    def to_dict(self) -> dict:
        return {
            'default_probability': round(self.default_probability, 4),
            'expected_ltv': round(self.expected_ltv, 2),
            'churn_probability': round(self.churn_probability, 4),
            'expected_dpd': round(self.expected_dpd, 1),
            'confidence': round(self.confidence, 3),
            'risk_factors': self.risk_factors,
        }


class WorldModel:
    """
    世界模型 - 蒸馏后的商业物理定律
    
    这是一个从历史数据中学习到的模型，能够：
    1. 预测客户违约概率
    2. 预测客户生命周期价值
    3. 预测客户流失风险
    
    函数签名:
        predict_customer_future(
            customer: CustomerProfile,   # 客户画像
            loan_offer: LoanOffer,       # 贷款条件
            market: MarketConditions     # 宏观环境
        ) -> CustomerFuture
    """
    
    # 内置的"物理规律"参数 (实际项目中从数据中学习)
    DEFAULT_RULES = {
        # 基础违约率 (按客户类型)
        'base_default_rate': {
            CustomerType.SALARIED: 0.015,
            CustomerType.SMALL_BUSINESS: 0.035,
            CustomerType.FREELANCER: 0.045,
            CustomerType.FARMER: 0.025,
        },
        
        # 行业风险系数
        'industry_risk': {
            Industry.FINANCE: 0.8,
            Industry.IT: 0.9,
            Industry.HEALTHCARE: 0.9,
            Industry.EDUCATION: 0.85,
            Industry.MANUFACTURING: 1.0,
            Industry.SERVICE: 1.1,
            Industry.RETAIL: 1.2,
            Industry.CATERING: 1.4,  # 餐饮风险较高
            Industry.CONSTRUCTION: 1.3,
            Industry.AGRICULTURE: 1.1,
            Industry.OTHER: 1.2,
        },
        
        # 城市等级风险系数
        'city_risk': {
            CityTier.TIER_1: 0.9,
            CityTier.TIER_2: 1.0,
            CityTier.TIER_3: 1.1,
            CityTier.TIER_4: 1.2,
        },
    }
    
    def __init__(self, rules: Dict = None, seed: int = None):
        """
        初始化世界模型
        
        Args:
            rules: 规则参数 (从数据蒸馏得到)
            seed: 随机种子
        """
        self.rules = rules or self.DEFAULT_RULES
        self.rng = np.random.default_rng(seed)
        self.trained = False
        self.model = None  # 预留给 XGBoost 等模型
    
    def predict_customer_future(
        self,
        customer: CustomerProfile,
        loan_offer: LoanOffer,
        market: MarketConditions,
        add_noise: bool = True
    ) -> CustomerFuture:
        """
        预测客户未来 - 核心蒸馏函数
        
        这是从TB级历史数据中蒸馏出来的"物理定律"
        支持个人客户和企业客户
        """
        risk_factors = {}
        
        # 判断是否为企业客户
        is_enterprise = customer.is_enterprise
        
        if is_enterprise:
            return self._predict_enterprise_future(customer, loan_offer, market, add_noise)
        else:
            return self._predict_personal_future(customer, loan_offer, market, add_noise)
    
    def _predict_personal_future(
        self,
        customer: CustomerProfile,
        loan_offer: LoanOffer,
        market: MarketConditions,
        add_noise: bool = True
    ) -> CustomerFuture:
        """个人客户预测（原有逻辑）"""
        risk_factors = {}
        
        # === 1. 计算违约概率 ===
        
        # 1.1 基础违约率
        base_rate = self.rules['base_default_rate'].get(
            customer.customer_type, 0.03
        )
        risk_factors['base_rate'] = base_rate
        
        # 1.2 行业风险调整
        industry_factor = self.rules['industry_risk'].get(customer.industry, 1.0)
        risk_factors['industry_factor'] = industry_factor
        
        # 1.3 城市风险调整
        city_factor = self.rules['city_risk'].get(customer.city_tier, 1.0)
        risk_factors['city_factor'] = city_factor
        
        # 1.4 负债率影响 (关键！)
        # 当负债率超过50%，违约风险急剧上升
        debt_factor = 1.0
        if customer.debt_ratio > 0.7:
            debt_factor = 2.5
        elif customer.debt_ratio > 0.5:
            debt_factor = 1.5
        elif customer.debt_ratio > 0.3:
            debt_factor = 1.1
        risk_factors['debt_factor'] = debt_factor
        
        # 1.5 收入覆盖能力
        # 月供占收入比例
        monthly_income = customer.monthly_income if hasattr(customer, 'monthly_income') and customer.monthly_income > 0 else 0
        if monthly_income > 0:
            payment_ratio = loan_offer.monthly_payment / monthly_income
            payment_factor = 1.0
            if payment_ratio > 0.5:
                payment_factor = 3.0
            elif payment_ratio > 0.35:
                payment_factor = 1.8
            elif payment_ratio > 0.25:
                payment_factor = 1.2
        else:
            payment_factor = 1.0
        risk_factors['payment_factor'] = payment_factor
        
        # 1.6 历史信用表现
        history_factor = 1.0
        if customer.max_historical_dpd > 90:
            history_factor = 3.0
        elif customer.max_historical_dpd > 30:
            history_factor = 1.8
        elif customer.max_historical_dpd > 0:
            history_factor = 1.3
        elif customer.previous_loans > 3:
            history_factor = 0.9  # 多次良好记录是加分项
        risk_factors['history_factor'] = history_factor
        
        # 1.7 收入稳定性
        volatility_factor = 1.0 + customer.income_volatility * 1.5
        risk_factors['volatility_factor'] = volatility_factor
        
        # 1.8 宏观经济影响 (关键！经济周期调整)
        # 这是"物理定律"中最重要的部分
        economic_factor = 1.0
        if market.gdp_growth < 0:
            # 经济负增长时，小微企业主风险急剧上升
            if customer.customer_type == CustomerType.SMALL_BUSINESS:
                economic_factor = 2.0 + abs(market.gdp_growth) * 10
            else:
                economic_factor = 1.5 + abs(market.gdp_growth) * 5
        elif market.gdp_growth < 0.02:
            economic_factor = 1.3
        
        # 失业率影响
        if market.unemployment_rate > 0.08:
            economic_factor *= 1.3 + (market.unemployment_rate - 0.08) * 5
        
        risk_factors['economic_factor'] = economic_factor
        
        # 1.9 综合计算违约概率
        default_prob = (
            base_rate *
            industry_factor *
            city_factor *
            debt_factor *
            payment_factor *
            history_factor *
            volatility_factor *
            economic_factor
        )
        
        # 添加随机噪声使模拟更真实
        if add_noise:
            noise = self.rng.normal(0, 0.02)
            default_prob = default_prob * (1 + noise)
        
        default_prob = min(0.95, max(0.001, default_prob))
        
        # === 2. 计算预期生命周期价值 (LTV) ===
        
        # 利息收入
        total_interest = loan_offer.monthly_payment * loan_offer.term_months - loan_offer.amount
        
        # 预期损失
        expected_loss = loan_offer.amount * default_prob * 0.6  # 60% 损失率
        
        # 运营成本 (约贷款额的 2%)
        operating_cost = loan_offer.amount * 0.02
        
        # LTV = 利息收入 - 预期损失 - 运营成本
        expected_ltv = total_interest * (1 - default_prob) - expected_loss - operating_cost
        
        # === 3. 计算流失概率 ===
        
        # 保存流失概率计算因子
        churn_factors = {}
        
        # 基础流失率
        base_churn = 0.05
        churn_prob = base_churn
        churn_factors['base_churn'] = base_churn
        
        # 利率敏感性 (利率越高越可能提前还款)
        rate_sensitivity = 0.0
        if loan_offer.interest_rate > market.base_interest_rate + 0.04:
            rate_sensitivity = 0.1
            churn_prob += rate_sensitivity
        churn_factors['rate_sensitivity'] = rate_sensitivity
        churn_factors['rate_impact'] = f"贷款利率({loan_offer.interest_rate*100:.2f}%) vs 基准利率({market.base_interest_rate*100:.2f}%)"
        
        # 优质客户更可能有更好的选择
        quality_bonus = 0.0
        if customer.risk_score < 0.2:
            quality_bonus = 0.05
            churn_prob += quality_bonus
        churn_factors['quality_bonus'] = quality_bonus
        churn_factors['quality_impact'] = f"风险评分({customer.risk_score:.3f}) < 0.2，优质客户更可能找到更好的融资渠道"
        
        churn_prob = min(0.5, churn_prob)
        churn_factors['final_churn'] = churn_prob
        
        # 将流失因子添加到risk_factors中
        risk_factors['churn_factors'] = churn_factors
        
        # === 4. 计算预期逾期天数 ===
        
        if default_prob > 0.3:
            expected_dpd = 90 + self.rng.exponential(60)
        elif default_prob > 0.1:
            expected_dpd = 30 + self.rng.exponential(30)
        elif default_prob > 0.05:
            expected_dpd = self.rng.exponential(15)
        else:
            expected_dpd = 0
        
        # === 5. 置信度 ===
        
        # 数据越完整，置信度越高
        confidence = 0.7
        if customer.previous_loans > 0:
            confidence += 0.1
        if customer.months_as_customer > 12:
            confidence += 0.1
        monthly_income = customer.monthly_income if hasattr(customer, 'monthly_income') and customer.monthly_income > 0 else 0
        if monthly_income > 0 and customer.deposit_balance > monthly_income * 3:
            confidence += 0.1
        
        return CustomerFuture(
            default_probability=default_prob,
            expected_ltv=expected_ltv,
            churn_probability=churn_prob,
            expected_dpd=expected_dpd,
            confidence=min(1.0, confidence),
            risk_factors=risk_factors,
        )
    
    def _predict_enterprise_future(
        self,
        customer: CustomerProfile,
        loan_offer: LoanOffer,
        market: MarketConditions,
        add_noise: bool = True
    ) -> CustomerFuture:
        """企业客户预测 - 使用企业专用风险评估模型"""
        risk_factors = {}
        
        # === 1. 计算违约概率 ===
        
        # 1.1 基础违约率（根据企业规模）
        base_rates = {
            CustomerType.MICRO_ENTERPRISE: 0.08,
            CustomerType.SMALL_ENTERPRISE: 0.06,
            CustomerType.MEDIUM_ENTERPRISE: 0.04,
            CustomerType.LARGE_ENTERPRISE: 0.03,
        }
        base_rate = base_rates.get(customer.customer_type, 0.05)
        risk_factors['base_rate'] = base_rate
        
        # 1.2 财务健康度影响（关键！）
        financial_health = customer.financial_health_score
        financial_factor = 1.0 + (1 - financial_health) * 2.0  # 财务健康度越低，风险越高
        risk_factors['financial_health_factor'] = financial_factor
        
        # 1.3 盈利能力
        profit_factor = 1.0
        if customer.profit_margin < 0:
            profit_factor = 3.0  # 亏损企业风险极高
        elif customer.profit_margin < 0.02:
            profit_factor = 2.0
        elif customer.profit_margin < 0.05:
            profit_factor = 1.5
        elif customer.profit_margin > 0.15:
            profit_factor = 0.7  # 高利润企业风险较低
        risk_factors['profit_factor'] = profit_factor
        
        # 1.4 现金流风险
        cashflow_factor = 1.0
        if customer.operating_cash_flow < 0:
            cashflow_factor = 2.5  # 负现金流风险极高
        elif customer.operating_cash_flow < loan_offer.monthly_payment * 3:
            cashflow_factor = 1.8  # 现金流不足以覆盖3个月还款
        elif customer.operating_cash_flow < loan_offer.monthly_payment * 6:
            cashflow_factor = 1.3
        risk_factors['cashflow_factor'] = cashflow_factor
        
        # 1.5 流动性风险
        liquidity_factor = 1.0
        if customer.current_ratio < 1.0:
            liquidity_factor = 2.0  # 流动比率<1，短期偿债能力不足
        elif customer.current_ratio < 1.2:
            liquidity_factor = 1.5
        elif customer.current_ratio > 2.5:
            liquidity_factor = 0.9  # 流动性充足
        risk_factors['liquidity_factor'] = liquidity_factor
        
        # 1.6 负债率影响
        debt_factor = 1.0
        if customer.debt_ratio > 0.75:
            debt_factor = 2.5
        elif customer.debt_ratio > 0.65:
            debt_factor = 2.0
        elif customer.debt_ratio > 0.50:
            debt_factor = 1.5
        elif customer.debt_ratio < 0.30:
            debt_factor = 0.8  # 低负债率是加分项
        risk_factors['debt_factor'] = debt_factor
        
        # 1.7 还款能力（年营收覆盖能力）
        annual_payment = loan_offer.monthly_payment * 12
        payment_coverage = customer.annual_revenue / (annual_payment + 1) if customer.annual_revenue > 0 else 0
        payment_factor = 1.0
        if payment_coverage < 1.5:
            payment_factor = 3.0  # 营收不足以覆盖1.5倍年还款
        elif payment_coverage < 2.0:
            payment_factor = 2.0
        elif payment_coverage < 3.0:
            payment_factor = 1.3
        elif payment_coverage > 5.0:
            payment_factor = 0.7  # 覆盖能力强
        risk_factors['payment_coverage_factor'] = payment_factor
        
        # 1.8 经营年限
        years_factor = 1.0
        if customer.years_in_business < 2:
            years_factor = 1.8  # 新企业风险高
        elif customer.years_in_business < 5:
            years_factor = 1.3
        elif customer.years_in_business > 10:
            years_factor = 0.9  # 老企业更稳定
        risk_factors['years_factor'] = years_factor
        
        # 1.9 行业风险
        industry_factor = self.rules['industry_risk'].get(customer.industry, 1.0)
        risk_factors['industry_factor'] = industry_factor
        
        # 1.10 创新能力（专利和研发）
        innovation_factor = 1.0
        if customer.innovation_score > 0.7:
            innovation_factor = 0.8  # 创新能力强，风险较低
        elif customer.innovation_score < 0.2:
            innovation_factor = 1.3  # 创新能力弱，风险较高
        risk_factors['innovation_factor'] = innovation_factor
        
        # 1.11 法律纠纷
        legal_factor = 1.0
        if customer.has_legal_disputes:
            legal_factor = 1.5 + customer.legal_dispute_count * 0.1
        risk_factors['legal_factor'] = legal_factor
        
        # 1.12 税务合规
        tax_factor = 1.0 + (1 - customer.tax_compliance_score) * 0.5
        risk_factors['tax_factor'] = tax_factor
        
        # 1.13 历史信用
        history_factor = 1.0
        if customer.max_historical_dpd > 90:
            history_factor = 2.5
        elif customer.max_historical_dpd > 30:
            history_factor = 1.8
        elif customer.max_historical_dpd > 0:
            history_factor = 1.3
        elif customer.previous_loans > 5:
            history_factor = 0.85  # 多次良好记录
        risk_factors['history_factor'] = history_factor
        
        # 1.14 宏观经济影响（企业更敏感）
        economic_factor = 1.0
        if market.gdp_growth < 0:
            # 经济负增长时，企业风险急剧上升
            if customer.customer_type in [CustomerType.MICRO_ENTERPRISE, CustomerType.SMALL_ENTERPRISE]:
                economic_factor = 2.5 + abs(market.gdp_growth) * 12
            else:
                economic_factor = 2.0 + abs(market.gdp_growth) * 8
        elif market.gdp_growth < 0.02:
            economic_factor = 1.5
        
        # 失业率影响（影响消费需求）
        if market.unemployment_rate > 0.08:
            economic_factor *= 1.5 + (market.unemployment_rate - 0.08) * 6
        
        # PMI影响（制造业景气度）
        if hasattr(market, 'manufacturing_pmi') and market.manufacturing_pmi < 45:
            if customer.industry in [Industry.MANUFACTURING, Industry.CONSTRUCTION]:
                economic_factor *= 1.4
        
        risk_factors['economic_factor'] = economic_factor
        
        # 1.15 信用评级影响
        credit_rating_factor = 1.0
        if customer.credit_rating.startswith('AAA'):
            credit_rating_factor = 0.6
        elif customer.credit_rating.startswith('AA'):
            credit_rating_factor = 0.7
        elif customer.credit_rating.startswith('A'):
            credit_rating_factor = 0.85
        elif customer.credit_rating.startswith('BBB'):
            credit_rating_factor = 1.0
        elif customer.credit_rating.startswith('BB'):
            credit_rating_factor = 1.5
        elif customer.credit_rating.startswith('B'):
            credit_rating_factor = 2.0
        else:
            credit_rating_factor = 2.5
        risk_factors['credit_rating_factor'] = credit_rating_factor
        
        # 1.16 综合计算违约概率
        default_prob = (
            base_rate *
            financial_factor *
            profit_factor *
            cashflow_factor *
            liquidity_factor *
            debt_factor *
            payment_factor *
            years_factor *
            industry_factor *
            innovation_factor *
            legal_factor *
            tax_factor *
            history_factor *
            economic_factor *
            credit_rating_factor
        )
        
        # 添加随机噪声
        if add_noise:
            noise = self.rng.normal(0, 0.02)
            default_prob = default_prob * (1 + noise)
        
        default_prob = min(0.95, max(0.001, default_prob))
        
        # === 2. 计算预期生命周期价值 (LTV) ===
        
        # 利息收入
        total_interest = loan_offer.monthly_payment * loan_offer.term_months - loan_offer.amount
        
        # 预期损失（企业贷款损失率可能更高）
        loss_rate = 0.7 if customer.customer_type in [CustomerType.MICRO_ENTERPRISE, CustomerType.SMALL_ENTERPRISE] else 0.6
        expected_loss = loan_offer.amount * default_prob * loss_rate
        
        # 运营成本（企业贷款成本更高，约3%）
        operating_cost = loan_offer.amount * 0.03
        
        # LTV = 利息收入 - 预期损失 - 运营成本
        expected_ltv = total_interest * (1 - default_prob) - expected_loss - operating_cost
        
        # === 3. 计算流失概率 ===
        
        # 保存流失概率计算因子
        churn_factors = {}
        
        base_churn = 0.08  # 企业客户基础流失率稍高
        churn_prob = base_churn
        churn_factors['base_churn'] = base_churn
        
        # 利率敏感性
        rate_sensitivity = 0.0
        if loan_offer.interest_rate > market.base_interest_rate + 0.03:
            rate_sensitivity = 0.15
            churn_prob += rate_sensitivity
        churn_factors['rate_sensitivity'] = rate_sensitivity
        churn_factors['rate_impact'] = f"贷款利率({loan_offer.interest_rate*100:.2f}%) vs 基准利率({market.base_interest_rate*100:.2f}%)"
        
        # 优质企业更可能找到更好的融资渠道
        quality_bonus = 0.0
        if financial_health > 0.8:
            quality_bonus = 0.1
            churn_prob += quality_bonus
        churn_factors['quality_bonus'] = quality_bonus
        churn_factors['quality_impact'] = f"财务健康度({financial_health:.3f}) > 0.8，优质企业更可能找到更好的融资渠道"
        
        churn_prob = min(0.6, churn_prob)
        churn_factors['final_churn'] = churn_prob
        
        # 将流失因子添加到risk_factors中
        risk_factors['churn_factors'] = churn_factors
        
        # === 4. 计算预期逾期天数 ===
        
        if default_prob > 0.4:
            expected_dpd = 120 + self.rng.exponential(90)
        elif default_prob > 0.2:
            expected_dpd = 60 + self.rng.exponential(60)
        elif default_prob > 0.1:
            expected_dpd = 30 + self.rng.exponential(30)
        else:
            expected_dpd = self.rng.exponential(10)
        
        # === 5. 置信度 ===
        
        confidence = 0.65  # 企业客户基础置信度
        if customer.previous_loans > 0:
            confidence += 0.1
        if customer.months_as_customer > 24:
            confidence += 0.1
        if customer.is_listed:
            confidence += 0.15  # 上市公司信息更透明
        if customer.total_patents > 10:
            confidence += 0.05  # 有专利说明有技术实力
        if customer.rnd_expense_ratio > 0.05:
            confidence += 0.05  # 研发投入说明有创新能力
        
        return CustomerFuture(
            default_probability=default_prob,
            expected_ltv=expected_ltv,
            churn_probability=churn_prob,
            expected_dpd=expected_dpd,
            confidence=min(1.0, confidence),
            risk_factors=risk_factors,
        )
    
    def batch_predict(
        self,
        customers: list,
        loan_offers: list,
        market: MarketConditions
    ) -> list:
        """批量预测"""
        results = []
        for customer, offer in zip(customers, loan_offers):
            result = self.predict_customer_future(customer, offer, market)
            results.append(result)
        return results
    
    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'trained': self.trained,
            }, f)
        print(f"世界模型已保存: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'WorldModel':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(rules=data['rules'])
        model.trained = data['trained']
        return model
    
    def explain_prediction(self, future: CustomerFuture) -> str:
        """解释预测结果"""
        explanation = []
        
        if future.risk_factors:
            factors = future.risk_factors
            
            explanation.append(f"违约概率: {future.default_probability:.1%}")
            explanation.append("\n风险因素分解:")
            
            if factors.get('economic_factor', 1) > 1.3:
                explanation.append(f"  ⚠️  宏观经济压力: {factors['economic_factor']:.1f}x")
            
            if factors.get('debt_factor', 1) > 1.3:
                explanation.append(f"  ⚠️  负债率过高: {factors['debt_factor']:.1f}x")
            
            if factors.get('payment_factor', 1) > 1.5:
                explanation.append(f"  ⚠️  月供压力大: {factors['payment_factor']:.1f}x")
            
            if factors.get('history_factor', 1) > 1.3:
                explanation.append(f"  ⚠️  信用历史不佳: {factors['history_factor']:.1f}x")
            
            if factors.get('industry_factor', 1) > 1.2:
                explanation.append(f"  ⚠️  行业风险: {factors['industry_factor']:.1f}x")
        
        explanation.append(f"\n预期LTV: ¥{future.expected_ltv:,.0f}")
        explanation.append(f"置信度: {future.confidence:.0%}")
        
        return "\n".join(explanation)


if __name__ == "__main__":
    from customer_generator import CustomerGenerator
    
    print("=" * 60)
    print("世界模型测试")
    print("=" * 60)
    
    # 创建模型
    model = WorldModel(seed=42)
    generator = CustomerGenerator(seed=42)
    
    # 创建测试场景
    customer = generator.generate_one(risk_profile="medium")
    
    loan = LoanOffer(
        amount=100000,
        interest_rate=0.08,
        term_months=24,
    )
    
    # 场景1: 经济繁荣期
    market_boom = MarketConditions(
        gdp_growth=0.06,
        base_interest_rate=0.04,
        unemployment_rate=0.04,
        inflation_rate=0.02,
        credit_spread=0.02,
    )
    
    # 场景2: 经济萧条期
    market_recession = MarketConditions(
        gdp_growth=-0.02,
        base_interest_rate=0.02,
        unemployment_rate=0.10,
        inflation_rate=0.01,
        credit_spread=0.04,
    )
    
    print(f"\n客户画像:")
    print(f"  类型: {customer.customer_type.value}")
    print(f"  行业: {customer.industry.value}")
    print(f"  月收入: ¥{customer.monthly_income:,.0f}")
    print(f"  负债率: {customer.debt_ratio:.1%}")
    print(f"  风险评分: {customer.risk_score:.2f}")
    
    print(f"\n贷款条件:")
    print(f"  金额: ¥{loan.amount:,.0f}")
    print(f"  利率: {loan.interest_rate:.1%}")
    print(f"  月供: ¥{loan.monthly_payment:,.0f}")
    
    # 预测
    print("\n" + "=" * 60)
    print("📊 繁荣期预测")
    print("=" * 60)
    result_boom = model.predict_customer_future(customer, loan, market_boom)
    print(model.explain_prediction(result_boom))
    
    print("\n" + "=" * 60)
    print("📊 萧条期预测")
    print("=" * 60)
    result_recession = model.predict_customer_future(customer, loan, market_recession)
    print(model.explain_prediction(result_recession))
    
    print(f"\n⚡ 经济周期影响: 违约概率从 {result_boom.default_probability:.1%} "
          f"上升到 {result_recession.default_probability:.1%}")

