"""
还款行为模拟器
实现正常还款、提前还款、违约模拟
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PaymentRecord:
    """还款记录"""
    month: int
    status: str  # 'paid', 'defaulted', 'prepaid', 'partial'
    principal_paid: float
    interest_paid: float
    total_paid: float
    remaining_principal: float


@dataclass
class RepaymentResult:
    """还款结果"""
    total_principal_paid: float
    total_interest_paid: float
    total_paid: float
    defaulted: bool
    default_month: Optional[int]
    prepaid: bool
    prepaid_month: Optional[int]
    payment_history: List[PaymentRecord]
    actual_term_months: int


class RepaymentSimulator:
    """还款行为模拟器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化模拟器
        
        Args:
            seed: 随机种子
        """
        self.rng = np.random.default_rng(seed)
    
    def simulate_repayment(self, loan_amount: float, interest_rate: float,
                          term_months: int, default_probability: float,
                          customer_data: Optional[Dict] = None) -> RepaymentResult:
        """
        模拟还款过程
        
        Args:
            loan_amount: 贷款金额
            interest_rate: 年利率
            term_months: 贷款期限（月）
            default_probability: 违约概率
            customer_data: 客户数据（可选）
        
        Returns:
            还款结果
        """
        # 计算月还款额（等额本息）
        monthly_rate = interest_rate / 12
        monthly_payment = loan_amount * monthly_rate * (1 + monthly_rate) ** term_months / \
                         ((1 + monthly_rate) ** term_months - 1)
        
        # 月本金
        monthly_principal = loan_amount / term_months
        
        # 模拟还款过程
        payment_history = []
        remaining_principal = loan_amount
        total_principal_paid = 0.0
        total_interest_paid = 0.0
        defaulted = False
        default_month = None
        prepaid = False
        prepaid_month = None
        
        for month in range(1, term_months + 1):
            # 检查是否违约
            if not defaulted:
                # 违约概率随时间累积
                cumulative_default_prob = default_probability * (month / term_months)
                if self.rng.random() < cumulative_default_prob:
                    defaulted = True
                    default_month = month
                    
                    # 违约时可能还了部分本金
                    paid_ratio = self.rng.uniform(0.1, 0.3)  # 还了10%-30%
                    principal_paid = remaining_principal * paid_ratio
                    interest_paid = monthly_payment * 0.5  # 假设还了一半利息
                    
                    payment_history.append(PaymentRecord(
                        month=month,
                        status='defaulted',
                        principal_paid=principal_paid,
                        interest_paid=interest_paid,
                        total_paid=principal_paid + interest_paid,
                        remaining_principal=remaining_principal - principal_paid
                    ))
                    
                    total_principal_paid += principal_paid
                    total_interest_paid += interest_paid
                    break
            
            # 检查是否提前还款（在后期可能发生）
            if month > term_months * 0.5 and not prepaid and not defaulted:
                # 提前还款概率（与客户特征相关）
                prepay_prob = 0.05  # 基础概率5%
                if customer_data:
                    credit_score = customer_data.get('credit_score', 650)
                    if credit_score > 700:
                        prepay_prob = 0.1  # 高信用客户更可能提前还款
                
                if self.rng.random() < prepay_prob:
                    prepaid = True
                    prepaid_month = month
                    
                    # 提前还款：还清剩余本金
                    principal_paid = remaining_principal
                    interest_paid = monthly_payment - monthly_principal
                    
                    payment_history.append(PaymentRecord(
                        month=month,
                        status='prepaid',
                        principal_paid=principal_paid,
                        interest_paid=interest_paid,
                        total_paid=principal_paid + interest_paid,
                        remaining_principal=0.0
                    ))
                    
                    total_principal_paid += principal_paid
                    total_interest_paid += interest_paid
                    break
            
            # 正常还款
            principal_paid = monthly_principal
            interest_paid = monthly_payment - monthly_principal
            remaining_principal -= principal_paid
            
            # 可能部分还款（模拟客户偶尔延迟）
            if self.rng.random() < 0.05:  # 5%概率部分还款
                paid_ratio = self.rng.uniform(0.7, 0.95)
                principal_paid *= paid_ratio
                interest_paid *= paid_ratio
                remaining_principal += monthly_principal * (1 - paid_ratio)
                status = 'partial'
            else:
                status = 'paid'
            
            payment_history.append(PaymentRecord(
                month=month,
                status=status,
                principal_paid=principal_paid,
                interest_paid=interest_paid,
                total_paid=principal_paid + interest_paid,
                remaining_principal=remaining_principal
            ))
            
            total_principal_paid += principal_paid
            total_interest_paid += interest_paid
        
        actual_term_months = default_month or prepaid_month or term_months
        
        return RepaymentResult(
            total_principal_paid=total_principal_paid,
            total_interest_paid=total_interest_paid,
            total_paid=total_principal_paid + total_interest_paid,
            defaulted=defaulted,
            default_month=default_month,
            prepaid=prepaid,
            prepaid_month=prepaid_month,
            payment_history=payment_history,
            actual_term_months=actual_term_months
        )


def main():
    """主函数：测试还款模拟器"""
    print("=" * 80)
    print("还款行为模拟器测试")
    print("=" * 80)
    
    simulator = RepaymentSimulator(seed=42)
    
    # 测试用例1：正常还款
    print("\n测试用例1：正常还款（低违约概率）")
    result1 = simulator.simulate_repayment(
        loan_amount=100000,
        interest_rate=0.08,
        term_months=24,
        default_probability=0.05,
        customer_data={'credit_score': 750}
    )
    
    print(f"是否违约: {result1.defaulted}")
    print(f"是否提前还款: {result1.prepaid}")
    print(f"实际还款月数: {result1.actual_term_months}")
    print(f"总本金: ¥{result1.total_principal_paid:,.2f}")
    print(f"总利息: ¥{result1.total_interest_paid:,.2f}")
    print(f"总还款: ¥{result1.total_paid:,.2f}")
    
    # 测试用例2：高风险客户（可能违约）
    print("\n测试用例2：高风险客户（高违约概率）")
    result2 = simulator.simulate_repayment(
        loan_amount=50000,
        interest_rate=0.12,
        term_months=36,
        default_probability=0.20,
        customer_data={'credit_score': 550}
    )
    
    print(f"是否违约: {result2.defaulted}")
    if result2.defaulted:
        print(f"违约月份: {result2.default_month}")
    print(f"是否提前还款: {result2.prepaid}")
    print(f"实际还款月数: {result2.actual_term_months}")
    print(f"总本金: ¥{result2.total_principal_paid:,.2f}")
    print(f"总利息: ¥{result2.total_interest_paid:,.2f}")
    print(f"总还款: ¥{result2.total_paid:,.2f}")
    
    # 显示还款历史（前6个月）
    if result2.payment_history:
        print("\n前6个月还款历史:")
        for record in result2.payment_history[:6]:
            print(f"  第{record.month}月: {record.status}, "
                  f"本金¥{record.principal_paid:,.2f}, "
                  f"利息¥{record.interest_paid:,.2f}, "
                  f"剩余¥{record.remaining_principal:,.2f}")
    
    return simulator


if __name__ == '__main__':
    main()

