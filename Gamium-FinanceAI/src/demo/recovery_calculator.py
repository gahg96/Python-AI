"""
回收率计算器
计算实际回收金额和回收率
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
try:
    from .repayment_simulator import RepaymentResult
except ImportError:
    from repayment_simulator import RepaymentResult


@dataclass
class RecoveryResult:
    """回收结果"""
    default_amount: float  # 违约金额
    recovery_amount: float  # 回收金额
    recovery_rate: float  # 回收率
    recovery_time_months: int  # 回收时间（月）
    recovery_method: str  # 回收方式
    recovery_cost: float  # 回收成本


class RecoveryCalculator:
    """回收率计算器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化计算器
        
        Args:
            seed: 随机种子
        """
        self.rng = np.random.default_rng(seed)
    
    def calculate_recovery(self, repayment_result: RepaymentResult,
                          loan_amount: float,
                          customer_data: Optional[Dict] = None,
                          market_data: Optional[Dict] = None) -> RecoveryResult:
        """
        计算回收金额和回收率
        
        Args:
            repayment_result: 还款结果
            loan_amount: 原始贷款金额
            customer_data: 客户数据
            market_data: 市场数据
        
        Returns:
            回收结果
        """
        if not repayment_result.defaulted:
            # 未违约，无回收
            return RecoveryResult(
                default_amount=0.0,
                recovery_amount=0.0,
                recovery_rate=0.0,
                recovery_time_months=0,
                recovery_method='none',
                recovery_cost=0.0
            )
        
        # 计算违约金额
        default_amount = loan_amount - repayment_result.total_principal_paid
        
        # 确定回收方式
        recovery_method = self._determine_recovery_method(customer_data)
        
        # 计算回收率（基于回收方式）
        base_recovery_rate = self._get_base_recovery_rate(recovery_method, customer_data)
        
        # 市场环境影响
        if market_data:
            market_factor = self._calculate_market_factor(market_data)
            recovery_rate = base_recovery_rate * market_factor
        else:
            recovery_rate = base_recovery_rate
        
        # 限制回收率范围
        recovery_rate = max(0.05, min(recovery_rate, 0.5))  # 5%-50%
        
        # 计算回收金额
        recovery_amount = default_amount * recovery_rate
        
        # 计算回收时间
        recovery_time = self._estimate_recovery_time(recovery_method, customer_data)
        
        # 计算回收成本
        recovery_cost = self._calculate_recovery_cost(
            default_amount, recovery_method, recovery_time
        )
        
        # 净回收金额（扣除成本）
        net_recovery = recovery_amount - recovery_cost
        
        return RecoveryResult(
            default_amount=default_amount,
            recovery_amount=max(0, net_recovery),  # 净回收不能为负
            recovery_rate=recovery_rate,
            recovery_time_months=recovery_time,
            recovery_method=recovery_method,
            recovery_cost=recovery_cost
        )
    
    def _determine_recovery_method(self, customer_data: Optional[Dict]) -> str:
        """确定回收方式"""
        if not customer_data:
            # 默认：法律诉讼
            return 'legal_action'
        
        # 根据客户特征选择回收方式
        has_collateral = customer_data.get('has_collateral', False)
        credit_score = customer_data.get('credit_score', 650)
        
        if has_collateral:
            return 'collateral_liquidation'  # 抵押物变现
        elif credit_score < 500:
            return 'debt_collection'  # 催收
        else:
            return 'legal_action'  # 法律诉讼
    
    def _get_base_recovery_rate(self, method: str, customer_data: Optional[Dict]) -> float:
        """获取基础回收率"""
        base_rates = {
            'collateral_liquidation': 0.3,  # 抵押物变现：30%
            'debt_collection': 0.15,  # 催收：15%
            'legal_action': 0.25,  # 法律诉讼：25%
            'negotiation': 0.20,  # 协商还款：20%
            'none': 0.0
        }
        
        base_rate = base_rates.get(method, 0.2)
        
        # 根据客户特征调整
        if customer_data:
            credit_score = customer_data.get('credit_score', 650)
            # 信用分越高，回收率可能越高（有还款意愿）
            if credit_score > 700:
                base_rate *= 1.2
            elif credit_score < 500:
                base_rate *= 0.8
        
        return base_rate
    
    def _calculate_market_factor(self, market_data: Dict) -> float:
        """计算市场环境影响因子"""
        # GDP增长、失业率等影响回收率
        gdp_growth = market_data.get('gdp_growth', 0.03)
        unemployment = market_data.get('unemployment_rate', 0.05)
        
        # GDP增长越高，经济越好，回收率越高
        gdp_factor = 1 + (gdp_growth - 0.03) * 2
        
        # 失业率越高，回收率越低
        unemployment_factor = 1 - (unemployment - 0.05) * 2
        
        market_factor = (gdp_factor + unemployment_factor) / 2
        return max(0.8, min(market_factor, 1.2))  # 限制在0.8-1.2
    
    def _estimate_recovery_time(self, method: str, customer_data: Optional[Dict]) -> int:
        """估算回收时间（月）"""
        base_times = {
            'collateral_liquidation': 3,  # 抵押物变现：3个月
            'debt_collection': 6,  # 催收：6个月
            'legal_action': 12,  # 法律诉讼：12个月
            'negotiation': 2,  # 协商还款：2个月
            'none': 0
        }
        
        base_time = base_times.get(method, 6)
        
        # 添加随机性
        time_variation = self.rng.integers(-2, 3)
        recovery_time = max(1, base_time + time_variation)
        
        return recovery_time
    
    def _calculate_recovery_cost(self, default_amount: float, method: str,
                                recovery_time: int) -> float:
        """计算回收成本"""
        # 基础成本（占违约金额的比例）
        cost_rates = {
            'collateral_liquidation': 0.05,  # 5%
            'debt_collection': 0.10,  # 10%
            'legal_action': 0.15,  # 15%
            'negotiation': 0.03,  # 3%
            'none': 0.0
        }
        
        cost_rate = cost_rates.get(method, 0.1)
        base_cost = default_amount * cost_rate
        
        # 时间成本（时间越长，成本越高）
        time_cost = default_amount * 0.01 * (recovery_time / 12)  # 每月1%
        
        total_cost = base_cost + time_cost
        return total_cost


def main():
    """主函数：测试回收率计算器"""
    print("=" * 80)
    print("回收率计算器测试")
    print("=" * 80)
    
    from repayment_simulator import RepaymentSimulator
    
    # 创建模拟器
    repayment_sim = RepaymentSimulator(seed=42)
    recovery_calc = RecoveryCalculator(seed=42)
    
    # 模拟一个违约案例
    print("\n模拟违约案例:")
    loan_amount = 100000
    repayment_result = repayment_sim.simulate_repayment(
        loan_amount=loan_amount,
        interest_rate=0.10,
        term_months=24,
        default_probability=0.25,  # 高违约概率
        customer_data={'credit_score': 550, 'has_collateral': False}
    )
    
    if repayment_result.defaulted:
        print(f"违约月份: {repayment_result.default_month}")
        print(f"已还本金: ¥{repayment_result.total_principal_paid:,.2f}")
        
        # 计算回收
        market_data = {
            'gdp_growth': 0.03,
            'unemployment_rate': 0.05
        }
        
        recovery_result = recovery_calc.calculate_recovery(
            repayment_result,
            loan_amount,
            customer_data={'credit_score': 550, 'has_collateral': False},
            market_data=market_data
        )
        
        print(f"\n回收结果:")
        print(f"违约金额: ¥{recovery_result.default_amount:,.2f}")
        print(f"回收金额: ¥{recovery_result.recovery_amount:,.2f}")
        print(f"回收率: {recovery_result.recovery_rate:.2%}")
        print(f"回收时间: {recovery_result.recovery_time_months} 个月")
        print(f"回收方式: {recovery_result.recovery_method}")
        print(f"回收成本: ¥{recovery_result.recovery_cost:,.2f}")
        
        # 计算净损失
        net_loss = recovery_result.default_amount - recovery_result.recovery_amount
        print(f"净损失: ¥{net_loss:,.2f}")
    
    return recovery_calc


if __name__ == '__main__':
    main()

