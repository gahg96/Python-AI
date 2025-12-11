"""
历史贷款数据生成器
用于生成模拟的历史贷款数据，包括申请、审批和结果数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os


class HistoricalLoanDataGenerator:
    """历史贷款数据生成器"""
    
    def __init__(self, seed: int = 42, start_date: str = '2020-01-01', 
                 end_date: str = '2024-12-31'):
        """
        初始化数据生成器
        
        Args:
            seed: 随机种子
            start_date: 开始日期
            end_date: 结束日期
        """
        self.rng = np.random.default_rng(seed)
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 行业列表
        self.industries = [
            '制造业', '零售业', '服务业', '建筑业', '房地产业',
            '金融业', 'IT/互联网', '教育', '医疗', '餐饮'
        ]
        
        # 就业状态
        self.employment_statuses = ['在职', '自由职业', '个体经营', '退休', '待业']
        
        # 教育水平
        self.education_levels = ['初中及以下', '高中', '大专', '本科', '硕士', '博士']
        
        # 婚姻状况
        self.marital_statuses = ['未婚', '已婚', '离异', '丧偶']
    
    def generate_personal_customer(self, customer_id: str) -> Dict:
        """生成对私客户特征"""
        age = int(self.rng.integers(22, 65))
        
        # 收入分布（对数正态）
        monthly_income = np.exp(self.rng.normal(np.log(8000), 0.8))
        monthly_income = max(3000, min(monthly_income, 50000))
        
        # 信用分（正态分布，均值650，标准差100）
        credit_score = int(self.rng.normal(650, 100))
        credit_score = max(300, min(credit_score, 850))
        
        # 负债率（Beta分布）
        debt_ratio = self.rng.beta(2, 5)
        debt_ratio = min(debt_ratio, 0.95)
        
        # 工作年限
        years_in_job = min(int(self.rng.integers(0, max(1, age - 18))), 30)
        
        return {
            'customer_id': customer_id,
            'customer_type': 'personal',
            'age': age,
            'monthly_income': round(monthly_income, 2),
            'credit_score': credit_score,
            'debt_ratio': round(debt_ratio, 4),
            'employment_status': self.rng.choice(self.employment_statuses, 
                                                 p=[0.6, 0.15, 0.15, 0.05, 0.05]),
            'education_level': self.rng.choice(self.education_levels,
                                              p=[0.1, 0.2, 0.25, 0.3, 0.1, 0.05]),
            'marital_status': self.rng.choice(self.marital_statuses,
                                             p=[0.3, 0.5, 0.15, 0.05]),
            'years_in_job': years_in_job,
            'has_collateral': self.rng.random() < 0.3,
            'collateral_value': round(self.rng.exponential(50000), 2) if self.rng.random() < 0.3 else 0,
        }
    
    def generate_corporate_customer(self, customer_id: str) -> Dict:
        """生成对公客户特征"""
        # 注册资本（对数正态）
        registered_capital = np.exp(self.rng.normal(np.log(1000000), 1.5))
        registered_capital = max(100000, min(registered_capital, 100000000))
        
        # 经营年限
        operating_years = int(self.rng.integers(1, 20))
        
        # 年营收（与注册资本相关）
        annual_revenue = registered_capital * self.rng.lognormal(0.5, 1.0)
        annual_revenue = max(500000, annual_revenue)
        
        # 资产负债率（Beta分布）
        debt_to_asset_ratio = self.rng.beta(3, 4)
        debt_to_asset_ratio = min(debt_to_asset_ratio, 0.9)
        
        # 流动比率（正态分布）
        current_ratio = self.rng.normal(1.5, 0.5)
        current_ratio = max(0.5, current_ratio)
        
        # 公司规模
        if annual_revenue < 5000000:
            company_size = 'small'
        elif annual_revenue < 50000000:
            company_size = 'medium'
        else:
            company_size = 'large'
        
        return {
            'customer_id': customer_id,
            'customer_type': 'corporate',
            'registered_capital': round(registered_capital, 2),
            'operating_years': operating_years,
            'annual_revenue': round(annual_revenue, 2),
            'debt_to_asset_ratio': round(debt_to_asset_ratio, 4),
            'current_ratio': round(current_ratio, 2),
            'industry': self.rng.choice(self.industries),
            'company_size': company_size,
        }
    
    def generate_loan_application(self, customer: Dict, application_date: datetime) -> Dict:
        """生成贷款申请"""
        customer_type = customer['customer_type']
        
        if customer_type == 'personal':
            # 对私贷款金额（与收入相关）
            max_loan = customer['monthly_income'] * 12 * 0.5  # 最多年收入的50%
            loan_amount = self.rng.uniform(10000, max_loan)
            loan_amount = round(loan_amount / 1000) * 1000  # 取整到千位
            
            # 贷款用途
            purposes = ['消费', '装修', '购车', '教育', '医疗', '其他']
            loan_purpose = self.rng.choice(purposes, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
            
            # 申请期限
            requested_term = self.rng.choice([6, 12, 24, 36, 48, 60], 
                                            p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        else:
            # 对公贷款金额（与年营收相关）
            max_loan = customer['annual_revenue'] * 0.3  # 最多年营收的30%
            loan_amount = self.rng.uniform(100000, max_loan)
            loan_amount = round(loan_amount / 10000) * 10000  # 取整到万位
            
            # 贷款用途
            purposes = ['流动资金', '设备采购', '项目投资', '债务重组', '其他']
            loan_purpose = self.rng.choice(purposes, p=[0.4, 0.2, 0.2, 0.1, 0.1])
            
            # 申请期限
            requested_term = self.rng.choice([12, 24, 36, 48, 60], 
                                            p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        return {
            'application_date': application_date.strftime('%Y-%m-%d'),
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'requested_term_months': requested_term,
        }
    
    def calculate_default_probability(self, customer: Dict, loan: Dict, 
                                     market: Dict) -> float:
        """计算违约概率"""
        base_prob = 0.1
        
        if customer['customer_type'] == 'personal':
            # 信用分影响
            credit_score = customer.get('credit_score', 650)
            credit_factor = 1 - (credit_score - 300) / 550 * 0.5  # 信用分越高，违约概率越低
            
            # 负债率影响
            debt_ratio = customer.get('debt_ratio', 0.5)
            debt_factor = 1 + debt_ratio * 0.5
            
            # 收入稳定性
            years_in_job = customer.get('years_in_job', 5)
            stability_factor = 1 - min(years_in_job / 10, 0.3)
            
            # 贷款金额/收入比
            loan_to_income = loan['loan_amount'] / (customer['monthly_income'] * 12)
            ratio_factor = 1 + min(loan_to_income / 0.5, 1.0) * 0.3
            
            default_prob = base_prob * credit_factor * debt_factor * stability_factor * ratio_factor
        
        else:
            # 对公客户
            # 经营年限影响
            operating_years = customer.get('operating_years', 5)
            years_factor = 1 - min(operating_years / 10, 0.3)
            
            # 资产负债率影响
            debt_to_asset = customer.get('debt_to_asset_ratio', 0.6)
            debt_factor = 1 + (debt_to_asset - 0.5) * 0.5
            
            # 流动比率影响
            current_ratio = customer.get('current_ratio', 1.5)
            liquidity_factor = 1 - min((current_ratio - 1.0) / 1.0, 0.3)
            
            # 贷款金额/年营收比
            loan_to_revenue = loan['loan_amount'] / customer['annual_revenue']
            ratio_factor = 1 + min(loan_to_revenue / 0.3, 1.0) * 0.3
            
            default_prob = base_prob * years_factor * debt_factor * liquidity_factor * ratio_factor
        
        # 市场环境影响
        market_factor = 1 + market.get('unemployment_rate', 0.05) * 2
        market_factor += (market.get('gdp_growth', 0.03) - 0.03) * -2  # GDP增长降低违约率
        
        default_prob *= market_factor
        
        # 限制在合理范围
        default_prob = max(0.01, min(default_prob, 0.5))
        
        return round(default_prob, 4)
    
    def expert_decision(self, customer: Dict, loan: Dict, default_prob: float) -> Dict:
        """模拟专家审批决策"""
        # 基础审批阈值（更严格，使审批率更合理）
        if customer['customer_type'] == 'personal':
            base_threshold = 0.15  # 降低阈值，更严格
            # 额外考虑：信用分、收入稳定性等
            credit_score = customer.get('credit_score', 650)
            if credit_score < 600:
                base_threshold = 0.12  # 信用分低，更严格
            elif credit_score < 550:
                base_threshold = 0.10  # 信用分很低，非常严格
        else:
            base_threshold = 0.12  # 对公更严格
            # 额外考虑：经营年限、资产负债率
            operating_years = customer.get('operating_years', 5)
            if operating_years < 2:
                base_threshold = 0.10  # 新企业更严格
        
        # 专家决策逻辑（添加一些随机性，模拟专家判断的不确定性）
        # 在阈值附近有10%的随机性
        adjusted_threshold = base_threshold + self.rng.normal(0, base_threshold * 0.1)
        
        if default_prob <= adjusted_threshold:
            decision = 'approve'
            
            # 确定贷款条件
            if customer['customer_type'] == 'personal':
                # 对私：可能调整金额和利率
                approved_amount = loan['loan_amount']
                if default_prob > 0.12:
                    approved_amount = loan['loan_amount'] * 0.9  # 降低10%
                
                # 利率（基准利率 + 利差）
                base_rate = 0.08
                credit_score = customer.get('credit_score', 650)
                if credit_score >= 750:
                    spread = 0.01
                elif credit_score >= 650:
                    spread = 0.015
                else:
                    spread = 0.02
                
                approved_rate = base_rate + spread
                approved_term = loan['requested_term_months']
                
            else:
                # 对公
                approved_amount = loan['loan_amount']
                base_rate = 0.06
                
                # 根据企业评级确定利差
                debt_ratio = customer.get('debt_to_asset_ratio', 0.6)
                if debt_ratio < 0.4:
                    spread = 0.005
                elif debt_ratio < 0.6:
                    spread = 0.01
                else:
                    spread = 0.015
                
                approved_rate = base_rate + spread
                approved_term = loan['requested_term_months']
            
            # 附加条件
            conditions = {}
            if default_prob > 0.12:
                if customer['customer_type'] == 'personal':
                    if not customer.get('has_collateral', False):
                        conditions['require_collateral'] = True
                else:
                    conditions['require_collateral'] = True
                    conditions['min_collateral_ratio'] = 0.5
            
        else:
            decision = 'reject'
            approved_amount = 0
            approved_rate = 0
            approved_term = 0
            conditions = {}
        
        return {
            'expert_decision': decision,
            'approved_amount': round(approved_amount, 2) if decision == 'approve' else 0,
            'approved_rate': round(approved_rate, 4) if decision == 'approve' else 0,
            'approved_term_months': approved_term if decision == 'approve' else 0,
            'conditions': conditions,
            'approval_date': None,  # 将在后续设置
        }
    
    def simulate_loan_outcome(self, customer: Dict, loan: Dict, 
                              approval: Dict, default_prob: float,
                              market: Dict) -> Dict:
        """模拟贷款结果"""
        if approval['expert_decision'] == 'reject':
            return {
                'actual_defaulted': False,
                'default_date': None,
                'default_amount': 0,
                'recovery_amount': 0,
                'recovery_rate': 0,
                'total_interest_paid': 0,
                'total_principal_paid': 0,
                'actual_profit': 0,
                'actual_roi': 0,
                'payment_history': [],
            }
        
        # 模拟是否违约
        actual_defaulted = self.rng.random() < default_prob
        
        if actual_defaulted:
            # 违约情况
            # 违约时间（在贷款期限内）
            default_month = int(self.rng.integers(1, approval['approved_term_months'] + 1))
            default_date = datetime.strptime(approval['approval_date'], '%Y-%m-%d') + \
                          timedelta(days=30 * default_month)
            
            # 已还本金（假设还了部分）
            paid_ratio = default_month / approval['approved_term_months']
            total_principal_paid = approval['approved_amount'] * paid_ratio * 0.3  # 假设还了30%
            total_interest_paid = approval['approved_amount'] * approval['approved_rate'] / 12 * default_month * 0.5
            
            # 回收金额（假设回收一部分）
            recovery_rate = self.rng.uniform(0.05, 0.3)  # 回收5%-30%
            recovery_amount = approval['approved_amount'] * recovery_rate
            default_amount = approval['approved_amount'] - total_principal_paid - recovery_amount
            
            # 利润
            total_interest_paid = min(total_interest_paid, 
                                     approval['approved_amount'] * approval['approved_rate'] * 
                                     (default_month / 12))
            actual_profit = total_interest_paid + recovery_amount - default_amount
            
            # 还款历史
            payment_history = []
            for month in range(1, default_month + 1):
                monthly_payment = approval['approved_amount'] * approval['approved_rate'] / 12 + \
                                approval['approved_amount'] / approval['approved_term_months']
                payment_history.append({
                    'month': month,
                    'status': 'paid' if month < default_month else 'defaulted',
                    'principal_paid': approval['approved_amount'] / approval['approved_term_months'],
                    'interest_paid': approval['approved_amount'] * approval['approved_rate'] / 12,
                    'total_paid': monthly_payment
                })
            
        else:
            # 正常还款
            default_date = None
            default_amount = 0
            
            # 计算总利息和本金
            total_interest_paid = approval['approved_amount'] * approval['approved_rate'] * \
                                (approval['approved_term_months'] / 12)
            total_principal_paid = approval['approved_amount']
            
            # 可能提前还款
            if self.rng.random() < 0.2:  # 20%概率提前还款
                early_month = int(self.rng.integers(approval['approved_term_months'] // 2, 
                                               approval['approved_term_months']))
                total_interest_paid = approval['approved_amount'] * approval['approved_rate'] * \
                                    (early_month / 12)
                total_principal_paid = approval['approved_amount']
            
            recovery_amount = 0
            recovery_rate = 0
            
            # 利润
            actual_profit = total_interest_paid
            
            # 还款历史
            payment_history = []
            for month in range(1, approval['approved_term_months'] + 1):
                monthly_payment = approval['approved_amount'] * approval['approved_rate'] / 12 + \
                                approval['approved_amount'] / approval['approved_term_months']
                status = 'paid'
                if month == approval['approved_term_months'] and self.rng.random() < 0.2:
                    status = 'prepaid'
                
                payment_history.append({
                    'month': month,
                    'status': status,
                    'principal_paid': approval['approved_amount'] / approval['approved_term_months'],
                    'interest_paid': approval['approved_amount'] * approval['approved_rate'] / 12,
                    'total_paid': monthly_payment
                })
        
        # ROI
        actual_roi = actual_profit / approval['approved_amount'] if approval['approved_amount'] > 0 else 0
        
        return {
            'actual_defaulted': actual_defaulted,
            'default_date': default_date.strftime('%Y-%m-%d') if default_date else None,
            'default_amount': round(default_amount, 2),
            'recovery_amount': round(recovery_amount, 2),
            'recovery_rate': round(recovery_rate, 4),
            'total_interest_paid': round(total_interest_paid, 2),
            'total_principal_paid': round(total_principal_paid, 2),
            'actual_profit': round(actual_profit, 2),
            'actual_roi': round(actual_roi, 4),
            'payment_history': payment_history,
        }
    
    def generate_market_conditions(self, date: datetime) -> Dict:
        """生成市场环境"""
        # 模拟市场环境随时间变化
        days_from_start = (date - self.start_date).days
        years_from_start = days_from_start / 365.25
        
        # GDP增长率（周期性变化）
        gdp_growth = 0.03 + 0.02 * np.sin(years_from_start * 2 * np.pi / 7) + \
                    self.rng.normal(0, 0.005)
        gdp_growth = max(0.01, min(gdp_growth, 0.08))
        
        # 基准利率（与GDP相关）
        base_interest_rate = 0.05 + (gdp_growth - 0.03) * 0.5 + \
                           self.rng.normal(0, 0.002)
        base_interest_rate = max(0.02, min(base_interest_rate, 0.1))
        
        # 失业率（与GDP负相关）
        unemployment_rate = 0.05 - (gdp_growth - 0.03) * 0.5 + \
                          self.rng.normal(0, 0.002)
        unemployment_rate = max(0.02, min(unemployment_rate, 0.1))
        
        # 通胀率
        inflation_rate = 0.02 + (gdp_growth - 0.03) * 0.3 + \
                        self.rng.normal(0, 0.003)
        inflation_rate = max(0.0, min(inflation_rate, 0.1))
        
        # 信用利差
        credit_spread = 0.02 + (unemployment_rate - 0.05) * 0.5 + \
                       self.rng.normal(0, 0.002)
        credit_spread = max(0.01, min(credit_spread, 0.05))
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'gdp_growth': round(gdp_growth, 4),
            'base_interest_rate': round(base_interest_rate, 4),
            'unemployment_rate': round(unemployment_rate, 4),
            'inflation_rate': round(inflation_rate, 4),
            'credit_spread': round(credit_spread, 4),
        }
    
    def generate_historical_loans(self, num_loans: int = 10000, 
                                  personal_ratio: float = 0.7) -> pd.DataFrame:
        """
        生成历史贷款数据
        
        Args:
            num_loans: 贷款数量
            personal_ratio: 对私贷款比例
        
        Returns:
            包含完整历史贷款数据的DataFrame
        """
        num_personal = int(num_loans * personal_ratio)
        num_corporate = num_loans - num_personal
        
        loans = []
        
        # 生成对私贷款
        for i in range(num_personal):
            customer_id = f"P{i+1:06d}"
            customer = self.generate_personal_customer(customer_id)
            
            # 随机申请日期
            days_offset = int(self.rng.integers(0, (self.end_date - self.start_date).days))
            application_date = self.start_date + timedelta(days=days_offset)
            
            # 生成申请
            loan = self.generate_loan_application(customer, application_date)
            
            # 市场环境
            market = self.generate_market_conditions(application_date)
            
            # 计算违约概率
            default_prob = self.calculate_default_probability(customer, loan, market)
            
            # 专家决策
            approval = self.expert_decision(customer, loan, default_prob)
            approval['approval_date'] = (application_date + timedelta(days=int(self.rng.integers(1, 7)))).strftime('%Y-%m-%d')
            
            # 模拟结果（只对批准的贷款）
            if approval['expert_decision'] == 'approve':
                outcome = self.simulate_loan_outcome(customer, loan, approval, 
                                                     default_prob, market)
            else:
                outcome = self.simulate_loan_outcome(customer, loan, approval, 0, market)
            
            # 合并所有数据
            loan_record = {
                **customer,
                **loan,
                **approval,
                **outcome,
                **market,
                'default_probability': default_prob,
            }
            
            loans.append(loan_record)
        
        # 生成对公贷款
        for i in range(num_corporate):
            customer_id = f"C{i+1:06d}"
            customer = self.generate_corporate_customer(customer_id)
            
            # 随机申请日期
            days_offset = int(self.rng.integers(0, (self.end_date - self.start_date).days))
            application_date = self.start_date + timedelta(days=days_offset)
            
            # 生成申请
            loan = self.generate_loan_application(customer, application_date)
            
            # 市场环境
            market = self.generate_market_conditions(application_date)
            
            # 计算违约概率
            default_prob = self.calculate_default_probability(customer, loan, market)
            
            # 专家决策
            approval = self.expert_decision(customer, loan, default_prob)
            approval['approval_date'] = (application_date + timedelta(days=int(self.rng.integers(1, 14)))).strftime('%Y-%m-%d')
            
            # 模拟结果
            if approval['expert_decision'] == 'approve':
                outcome = self.simulate_loan_outcome(customer, loan, approval, 
                                                     default_prob, market)
            else:
                outcome = self.simulate_loan_outcome(customer, loan, approval, 0, market)
            
            # 合并所有数据
            loan_record = {
                **customer,
                **loan,
                **approval,
                **outcome,
                **market,
                'default_probability': default_prob,
            }
            
            loans.append(loan_record)
        
        # 转换为DataFrame
        df = pd.DataFrame(loans)
        
        # 按申请日期排序
        df['application_date'] = pd.to_datetime(df['application_date'])
        df = df.sort_values('application_date').reset_index(drop=True)
        
        return df
    
    def save_to_files(self, df: pd.DataFrame, output_dir: str = 'data/historical'):
        """保存数据到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整数据（CSV）
        csv_path = os.path.join(output_dir, 'historical_loans.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 已保存完整数据到: {csv_path} ({len(df)} 条记录)")
        
        # 保存Parquet格式（更高效）
        parquet_path = os.path.join(output_dir, 'historical_loans.parquet')
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"✅ 已保存Parquet格式到: {parquet_path}")
        
        # 保存统计信息
        stats = {
            'total_loans': len(df),
            'personal_loans': len(df[df['customer_type'] == 'personal']),
            'corporate_loans': len(df[df['customer_type'] == 'corporate']),
            'approved_loans': len(df[df['expert_decision'] == 'approve']),
            'rejected_loans': len(df[df['expert_decision'] == 'reject']),
            'defaulted_loans': len(df[df['actual_defaulted'] == True]),
            'approval_rate': (df['expert_decision'] == 'approve').mean(),
            'default_rate': df[df['expert_decision'] == 'approve']['actual_defaulted'].mean() if len(df[df['expert_decision'] == 'approve']) > 0 else 0,
            'avg_profit': df[df['expert_decision'] == 'approve']['actual_profit'].mean() if len(df[df['expert_decision'] == 'approve']) > 0 else 0,
            'date_range': {
                'start': df['application_date'].min().strftime('%Y-%m-%d'),
                'end': df['application_date'].max().strftime('%Y-%m-%d'),
            }
        }
        
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存统计信息到: {stats_path}")
        
        return stats


def main():
    """主函数：生成历史数据"""
    print("=" * 80)
    print("历史贷款数据生成器")
    print("=" * 80)
    print()
    
    # 创建生成器
    generator = HistoricalLoanDataGenerator(
        seed=42,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # 生成数据
    print("正在生成历史贷款数据...")
    print(f"  对私贷款比例: 70%")
    print(f"  对公贷款比例: 30%")
    print()
    
    df = generator.generate_historical_loans(
        num_loans=10000,
        personal_ratio=0.7
    )
    
    # 保存数据
    print("正在保存数据...")
    stats = generator.save_to_files(df, output_dir='data/historical')
    
    print()
    print("=" * 80)
    print("数据生成完成！")
    print("=" * 80)
    print()
    print("统计信息:")
    print(f"  总贷款数: {stats['total_loans']}")
    print(f"  对私贷款: {stats['personal_loans']}")
    print(f"  对公贷款: {stats['corporate_loans']}")
    print(f"  审批通过: {stats['approved_loans']} ({stats['approval_rate']:.2%})")
    print(f"  审批拒绝: {stats['rejected_loans']} ({1-stats['approval_rate']:.2%})")
    print(f"  违约数量: {stats['defaulted_loans']}")
    print(f"  违约率: {stats['default_rate']:.2%}")
    print(f"  平均利润: ¥{stats['avg_profit']:,.2f}")
    print(f"  时间范围: {stats['date_range']['start']} 至 {stats['date_range']['end']}")
    print()


if __name__ == '__main__':
    main()

