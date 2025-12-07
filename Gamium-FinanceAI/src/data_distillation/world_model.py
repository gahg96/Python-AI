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
    """宏观经济环境"""
    gdp_growth: float          # GDP 增长率 (-0.05 ~ 0.10)
    base_interest_rate: float  # 基准利率 (0.02 ~ 0.08)
    unemployment_rate: float   # 失业率 (0.03 ~ 0.15)
    inflation_rate: float      # 通胀率 (0.00 ~ 0.08)
    credit_spread: float       # 信用利差 (0.01 ~ 0.05)
    
    @property
    def economic_stress(self) -> float:
        """经济压力指数 (0-1)"""
        stress = 0.0
        stress += max(0, -self.gdp_growth) * 5  # 负增长增加压力
        stress += (self.unemployment_rate - 0.05) * 3  # 高失业增加压力
        stress += max(0, self.inflation_rate - 0.03) * 2  # 高通胀增加压力
        return min(1.0, max(0.0, stress))
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.gdp_growth,
            self.base_interest_rate,
            self.unemployment_rate,
            self.inflation_rate,
            self.credit_spread,
            self.economic_stress,
        ], dtype=np.float32)


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
        """
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
        payment_ratio = loan_offer.monthly_payment / (customer.monthly_income + 1)
        payment_factor = 1.0
        if payment_ratio > 0.5:
            payment_factor = 3.0
        elif payment_ratio > 0.35:
            payment_factor = 1.8
        elif payment_ratio > 0.25:
            payment_factor = 1.2
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
        
        # 基础流失率
        churn_prob = 0.05
        
        # 利率敏感性 (利率越高越可能提前还款)
        if loan_offer.interest_rate > market.base_interest_rate + 0.04:
            churn_prob += 0.1
        
        # 优质客户更可能有更好的选择
        if customer.risk_score < 0.2:
            churn_prob += 0.05
        
        churn_prob = min(0.5, churn_prob)
        
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
        if customer.deposit_balance > customer.monthly_income * 3:
            confidence += 0.1
        
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

