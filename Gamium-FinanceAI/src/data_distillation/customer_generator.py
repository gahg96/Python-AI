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


@dataclass
class CustomerProfile:
    """客户画像"""
    customer_id: str
    
    # 基础信息
    age: int
    city_tier: CityTier
    customer_type: CustomerType
    industry: Industry
    years_in_business: float  # 从业/经营年限
    
    # 收入与资产
    monthly_income: float       # 月收入 (元)
    income_volatility: float    # 收入波动率 (0-1)
    total_assets: float         # 总资产 (元)
    total_liabilities: float    # 总负债 (元)
    
    # 银行关系
    deposit_balance: float      # 存款余额 (元)
    deposit_stability: float    # 存款稳定性 (0-1)
    months_as_customer: int     # 成为客户月数
    
    # 信贷历史
    previous_loans: int         # 历史贷款次数
    max_historical_dpd: int     # 历史最大逾期天数 (Days Past Due)
    months_since_last_loan: int # 距上次贷款月数
    
    # 衍生特征
    @property
    def age_group(self) -> str:
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
        """负债率"""
        if self.total_assets <= 0:
            return 1.0
        return min(1.0, self.total_liabilities / self.total_assets)
    
    @property
    def debt_to_income(self) -> float:
        """债务收入比"""
        annual_income = self.monthly_income * 12
        if annual_income <= 0:
            return float('inf')
        return self.total_liabilities / annual_income
    
    @property
    def risk_score(self) -> float:
        """简单风险评分 (0-1, 越高风险越大)"""
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
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'customer_id': self.customer_id,
            'age': self.age,
            'age_group': self.age_group,
            'city_tier': self.city_tier.value,
            'customer_type': self.customer_type.value,
            'industry': self.industry.value,
            'years_in_business': self.years_in_business,
            'monthly_income': self.monthly_income,
            'income_volatility': self.income_volatility,
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
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量 (用于模型输入)"""
        # 数值特征
        features = [
            self.age / 100,
            list(CityTier).index(self.city_tier) / 3,
            list(CustomerType).index(self.customer_type) / 3,
            list(Industry).index(self.industry) / 10,
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
        return np.array(features, dtype=np.float32)


class CustomerGenerator:
    """
    客户生成器
    
    根据统计分布生成虚拟但符合真实规律的客户数据
    """
    
    # 各客户类型的分布参数
    DISTRIBUTION_PARAMS = {
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
        
        params = self.DISTRIBUTION_PARAMS[customer_type]
        
        # 年龄
        if customer_type == CustomerType.FARMER:
            age = int(self.rng.normal(45, 12))
        elif customer_type == CustomerType.SMALL_BUSINESS:
            age = int(self.rng.normal(40, 10))
        else:
            age = int(self.rng.normal(35, 10))
        age = max(22, min(65, age))
        
        # 行业
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

