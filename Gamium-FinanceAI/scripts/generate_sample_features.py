#!/usr/bin/env python3
"""
生成示例特征文件

用于演示特征文件的格式和结构

使用方法:
    python3 generate_sample_features.py --output data/sample_features.parquet --count 1000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_sample_features(n_samples: int = 1000) -> pd.DataFrame:
    """生成示例特征数据"""
    
    np.random.seed(42)
    
    # 生成客户ID
    customer_ids = [f"C{i:06d}" for i in range(n_samples)]
    
    # 基础特征
    ages = np.random.randint(22, 65, n_samples)
    genders = np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45])
    educations = np.random.choice(
        ['high_school', 'college', 'bachelor', 'master', 'phd'],
        n_samples,
        p=[0.2, 0.25, 0.35, 0.15, 0.05]
    )
    industries = np.random.choice(
        ['it', 'finance', 'manufacturing', 'service', 'retail', 'construction', 'education', 'healthcare'],
        n_samples
    )
    city_tiers = np.random.choice(
        ['tier_1', 'tier_2', 'tier_3', 'tier_4'],
        n_samples,
        p=[0.15, 0.30, 0.35, 0.20]
    )
    customer_types = np.random.choice(
        ['salaried', 'small_business', 'freelancer', 'farmer'],
        n_samples,
        p=[0.45, 0.25, 0.15, 0.15]
    )
    
    # 财务特征
    monthly_incomes = np.random.lognormal(9.5, 0.5, n_samples)  # 约5000-50000
    monthly_incomes = np.clip(monthly_incomes, 3000, 100000)
    
    total_assets = monthly_incomes * np.random.uniform(20, 60, n_samples)
    total_liabilities = total_assets * np.random.uniform(0.2, 0.7, n_samples)
    debt_ratios = total_liabilities / (total_assets + 1)
    debt_to_incomes = total_liabilities / (monthly_incomes * 12 + 1)
    
    total_deposit_balances = monthly_incomes * np.random.uniform(3, 15, n_samples)
    avg_account_balances = total_deposit_balances / np.random.uniform(1, 3, n_samples)
    
    # 交易特征
    total_incomes = monthly_incomes * 12 * np.random.uniform(0.8, 1.2, n_samples)
    total_expenses = total_incomes * np.random.uniform(0.6, 0.9, n_samples)
    savings_rates = (total_incomes - total_expenses) / (total_incomes + 1)
    income_volatilities = np.random.uniform(0.05, 0.35, n_samples)
    transaction_counts = np.random.poisson(150, n_samples)
    avg_transaction_amounts = total_expenses / (transaction_counts + 1)
    
    # 贷款历史特征
    total_loans = np.random.poisson(3, n_samples)
    total_loans = np.clip(total_loans, 0, 10)
    
    total_loan_amounts = monthly_incomes * np.random.uniform(5, 20, n_samples) * total_loans
    avg_loan_amounts = total_loan_amounts / (total_loans + 1)
    
    # 违约相关（基于风险特征）
    default_probs = (
        debt_ratios * 0.3 +
        (income_volatilities > 0.25).astype(int) * 0.2 +
        (total_loans > 5).astype(int) * 0.1 +
        np.random.uniform(0, 0.4, n_samples)
    )
    default_probs = np.clip(default_probs, 0, 1)
    
    defaulted = (default_probs > 0.5).astype(int)
    default_count = np.random.binomial(total_loans, default_probs / 2)
    default_count = np.minimum(default_count, total_loans)
    
    max_overdue_days = np.where(
        defaulted == 1,
        np.random.exponential(30, n_samples),
        np.random.exponential(5, n_samples)
    )
    max_overdue_days = np.clip(max_overdue_days, 0, 180).astype(int)
    
    avg_interest_rates = np.random.uniform(0.05, 0.08, n_samples)
    months_since_last_loan = np.random.exponential(12, n_samples)
    months_since_last_loan = np.clip(months_since_last_loan, 0, 60).astype(int)
    
    # 时间特征
    registration_dates = [
        (datetime.now() - timedelta(days=np.random.exponential(1000))).strftime('%Y-%m-%d')
        for _ in range(n_samples)
    ]
    months_as_customer = np.array([
        (datetime.now() - datetime.strptime(d, '%Y-%m-%d')).days / 30
        for d in registration_dates
    ]).astype(int)
    
    # 贷款状态
    loan_statuses = np.where(
        defaulted == 1,
        'defaulted',
        np.where(
            max_overdue_days > 0,
            'overdue',
            'normal'
        )
    )
    
    # 构建DataFrame
    features = pd.DataFrame({
        # 客户标识
        'customer_id': customer_ids,
        
        # 基础特征
        'age': ages,
        'gender': genders,
        'education': educations,
        'industry': industries,
        'city_tier': city_tiers,
        'customer_type': customer_types,
        
        # 财务特征
        'monthly_income': monthly_incomes.round(2),
        'total_assets': total_assets.round(2),
        'total_liabilities': total_liabilities.round(2),
        'debt_ratio': debt_ratios.round(4),
        'debt_to_income': debt_to_incomes.round(4),
        'total_deposit_balance': total_deposit_balances.round(2),
        'avg_account_balance': avg_account_balances.round(2),
        
        # 交易特征
        'total_income': total_incomes.round(2),
        'total_expense': total_expenses.round(2),
        'savings_rate': savings_rates.round(4),
        'income_volatility': income_volatilities.round(4),
        'transaction_count': transaction_counts,
        'avg_transaction_amount': avg_transaction_amounts.round(2),
        
        # 贷款历史特征
        'total_loans': total_loans,
        'total_loan_amount': total_loan_amounts.round(2),
        'avg_loan_amount': avg_loan_amounts.round(2),
        'default_count': default_count,
        'max_overdue_days': max_overdue_days,
        'avg_interest_rate': avg_interest_rates.round(4),
        'months_since_last_loan': months_since_last_loan,
        
        # 时间特征
        'months_as_customer': months_as_customer,
        'registration_date': registration_dates,
        
        # 标签
        'defaulted': defaulted,
        'default_probability': default_probs.round(4),
        'loan_status': loan_statuses
    })
    
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成示例特征文件')
    parser.add_argument('--output', type=str, default='data/sample_features.parquet',
                       help='输出文件路径')
    parser.add_argument('--count', type=int, default=1000,
                       help='生成样本数量')
    parser.add_argument('--format', type=str, choices=['parquet', 'csv'], default='parquet',
                       help='输出格式')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 生成示例特征文件")
    print("=" * 60)
    print(f"样本数量: {args.count:,}")
    print(f"输出格式: {args.format}")
    print(f"输出路径: {args.output}")
    print()
    
    # 生成特征
    features = generate_sample_features(args.count)
    
    # 保存文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'parquet':
        features.to_parquet(output_path, index=False)
    else:
        features.to_csv(output_path, index=False)
    
    print("=" * 60)
    print("✅ 生成完成！")
    print("=" * 60)
    print(f"\n文件信息:")
    print(f"  路径: {output_path}")
    print(f"  记录数: {len(features):,}")
    print(f"  特征数: {len(features.columns)}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n特征列表:")
    for i, col in enumerate(features.columns, 1):
        dtype = features[col].dtype
        print(f"  {i:2d}. {col:30s} ({dtype})")
    
    print(f"\n前5行数据:")
    print(features.head().to_string())
    
    print(f"\n统计信息:")
    print(features.describe().to_string())
    
    print(f"\n目标变量分布:")
    print(features['defaulted'].value_counts())
    print(f"违约率: {features['defaulted'].mean():.2%}")


