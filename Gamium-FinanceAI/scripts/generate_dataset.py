#!/usr/bin/env python3
"""
大规模历史数据集生成器

生成约 10G 的模拟银行贷款历史数据，包括：
1. 客户画像 (customers.parquet)
2. 贷款申请记录 (loan_applications.parquet)
3. 还款历史 (repayment_history.parquet)
4. 宏观经济数据 (macro_economics.parquet)

数据量估算：
- 500万客户 × 2KB ≈ 10G
- 1000万贷款申请 × 500B ≈ 5G
- 1亿条还款记录 × 100B ≈ 10G
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 配置
RANDOM_SEED = 42
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2024, 12, 31)

# 客户类型分布
CUSTOMER_TYPE_DIST = {
    'salaried': 0.45,      # 工薪阶层
    'small_business': 0.25, # 小微企业主
    'freelancer': 0.15,     # 自由职业
    'farmer': 0.15,         # 农户
}

# 行业分布
INDUSTRY_DIST = {
    'manufacturing': 0.15,
    'service': 0.18,
    'retail': 0.12,
    'catering': 0.08,
    'construction': 0.10,
    'it': 0.08,
    'finance': 0.05,
    'education': 0.08,
    'healthcare': 0.06,
    'agriculture': 0.10,
}

# 城市等级分布
CITY_TIER_DIST = {
    'tier_1': 0.15,
    'tier_2': 0.30,
    'tier_3': 0.35,
    'tier_4': 0.20,
}

# 经济周期参数
ECONOMIC_CYCLES = [
    # (start_year, end_year, phase, gdp_range, unemployment_range)
    (2015, 2016, 'boom', (0.06, 0.08), (0.04, 0.05)),
    (2017, 2018, 'normal', (0.04, 0.06), (0.05, 0.06)),
    (2019, 2019, 'recession', (0.02, 0.04), (0.06, 0.07)),
    (2020, 2020, 'depression', (-0.02, 0.02), (0.08, 0.12)),  # COVID
    (2021, 2022, 'recovery', (0.03, 0.05), (0.05, 0.07)),
    (2023, 2024, 'normal', (0.03, 0.05), (0.05, 0.06)),
]


class DataGenerator:
    """大规模数据生成器"""
    
    def __init__(self, output_dir: str, seed: int = RANDOM_SEED):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        
    def generate_customer_batch(self, batch_id: int, batch_size: int, seed: int) -> pd.DataFrame:
        """生成一批客户数据"""
        rng = np.random.default_rng(seed)
        
        customers = []
        for i in range(batch_size):
            customer_id = f"C{batch_id:04d}{i:06d}"
            
            # 客户类型
            customer_type = rng.choice(
                list(CUSTOMER_TYPE_DIST.keys()),
                p=list(CUSTOMER_TYPE_DIST.values())
            )
            
            # 根据客户类型生成特征
            if customer_type == 'salaried':
                age = int(rng.normal(35, 10))
                income_mean, income_std = 12000, 8000
                asset_mean, asset_std = 300000, 200000
                base_default_rate = 0.02
            elif customer_type == 'small_business':
                age = int(rng.normal(40, 10))
                income_mean, income_std = 25000, 20000
                asset_mean, asset_std = 800000, 600000
                base_default_rate = 0.04
            elif customer_type == 'freelancer':
                age = int(rng.normal(32, 8))
                income_mean, income_std = 15000, 12000
                asset_mean, asset_std = 200000, 150000
                base_default_rate = 0.05
            else:  # farmer
                age = int(rng.normal(45, 12))
                income_mean, income_std = 6000, 4000
                asset_mean, asset_std = 400000, 300000
                base_default_rate = 0.03
            
            age = max(22, min(65, age))
            
            # 城市等级
            city_tier = rng.choice(
                list(CITY_TIER_DIST.keys()),
                p=list(CITY_TIER_DIST.values())
            )
            
            # 行业
            industry = rng.choice(
                list(INDUSTRY_DIST.keys()),
                p=list(INDUSTRY_DIST.values())
            )
            
            # 收入
            monthly_income = max(3000, rng.normal(income_mean, income_std))
            income_volatility = rng.beta(2, 8 if customer_type == 'salaried' else 5)
            
            # 资产和负债
            total_assets = max(10000, rng.normal(asset_mean, asset_std))
            debt_ratio = max(0, min(0.9, rng.normal(0.4, 0.15)))
            total_liabilities = total_assets * debt_ratio
            
            # 存款
            deposit_balance = max(0, rng.exponential(monthly_income * 3))
            deposit_stability = rng.beta(3, 2)
            
            # 成为客户的时间
            customer_since = START_DATE + timedelta(
                days=int(rng.uniform(0, (END_DATE - START_DATE).days * 0.8))
            )
            
            # 教育程度
            education = rng.choice(
                ['high_school', 'college', 'bachelor', 'master', 'phd'],
                p=[0.30, 0.25, 0.30, 0.12, 0.03]
            )
            
            # 婚姻状态
            marital_status = rng.choice(
                ['single', 'married', 'divorced'],
                p=[0.30, 0.60, 0.10]
            )
            
            # 房产情况
            has_house = rng.random() < (0.6 if age > 35 else 0.3)
            has_car = rng.random() < (0.5 if monthly_income > 10000 else 0.2)
            
            # 信用评分 (模拟央行征信)
            credit_score = int(rng.normal(680, 80))
            credit_score = max(350, min(950, credit_score))
            
            customers.append({
                'customer_id': customer_id,
                'customer_type': customer_type,
                'age': age,
                'gender': rng.choice(['M', 'F']),
                'city_tier': city_tier,
                'province': rng.choice([
                    '广东', '浙江', '江苏', '山东', '河南', '四川', 
                    '湖北', '湖南', '河北', '福建', '上海', '北京',
                    '安徽', '辽宁', '陕西', '江西', '重庆', '云南',
                    '广西', '山西', '贵州', '新疆', '天津', '黑龙江',
                ]),
                'industry': industry,
                'education': education,
                'marital_status': marital_status,
                'years_employed': max(0.5, min(age - 22, rng.exponential(8))),
                'monthly_income': round(monthly_income, 2),
                'income_volatility': round(income_volatility, 4),
                'total_assets': round(total_assets, 2),
                'total_liabilities': round(total_liabilities, 2),
                'debt_ratio': round(debt_ratio, 4),
                'deposit_balance': round(deposit_balance, 2),
                'deposit_stability': round(deposit_stability, 4),
                'has_house': has_house,
                'has_car': has_car,
                'credit_score': credit_score,
                'customer_since': customer_since.strftime('%Y-%m-%d'),
                'base_default_rate': base_default_rate,
            })
        
        return pd.DataFrame(customers)
    
    def generate_loan_applications(
        self, 
        customers: pd.DataFrame, 
        batch_id: int,
        seed: int
    ) -> pd.DataFrame:
        """生成贷款申请记录"""
        rng = np.random.default_rng(seed)
        
        applications = []
        
        for _, customer in customers.iterrows():
            # 每个客户平均申请 2-5 次贷款
            n_applications = rng.poisson(3) + 1
            
            customer_since = datetime.strptime(customer['customer_since'], '%Y-%m-%d')
            available_days = (END_DATE - customer_since).days
            
            if available_days < 30:
                continue
            
            for app_idx in range(n_applications):
                # 申请日期
                days_offset = int(rng.uniform(30, available_days))
                apply_date = customer_since + timedelta(days=days_offset)
                
                if apply_date > END_DATE:
                    continue
                
                # 获取当时的经济环境
                year = apply_date.year
                eco_phase = 'normal'
                gdp_growth = 0.04
                unemployment = 0.05
                
                for cycle in ECONOMIC_CYCLES:
                    if cycle[0] <= year <= cycle[1]:
                        eco_phase = cycle[2]
                        gdp_growth = rng.uniform(*cycle[3])
                        unemployment = rng.uniform(*cycle[4])
                        break
                
                # 贷款金额
                max_amount = min(
                    customer['monthly_income'] * 36,
                    customer['total_assets'] * 0.5
                )
                loan_amount = rng.uniform(10000, max(10000, max_amount))
                
                # 贷款期限
                term_months = rng.choice([6, 12, 18, 24, 36, 48, 60])
                
                # 贷款用途
                purpose = rng.choice([
                    'consumption', 'business', 'house', 'car', 
                    'education', 'medical', 'other'
                ], p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.10])
                
                # 利率 (基于风险)
                base_rate = 0.04 + (0.02 if eco_phase == 'depression' else 0)
                risk_premium = (1 - customer['credit_score'] / 950) * 0.08
                interest_rate = base_rate + risk_premium + rng.uniform(-0.005, 0.01)
                interest_rate = max(0.04, min(0.24, interest_rate))
                
                # 审批结果
                approval_prob = 0.7
                approval_prob *= (customer['credit_score'] / 700)
                approval_prob *= (1 - customer['debt_ratio'])
                if eco_phase == 'depression':
                    approval_prob *= 0.7
                
                approved = rng.random() < approval_prob
                
                # 如果批准，生成还款信息
                if approved:
                    # 计算月供
                    monthly_rate = interest_rate / 12
                    n = term_months
                    monthly_payment = loan_amount * monthly_rate * (1 + monthly_rate)**n / ((1 + monthly_rate)**n - 1)
                    
                    # 违约概率计算
                    default_prob = customer['base_default_rate']
                    
                    # 经济周期影响
                    if eco_phase == 'depression':
                        default_prob *= 2.5
                    elif eco_phase == 'recession':
                        default_prob *= 1.5
                    elif eco_phase == 'boom':
                        default_prob *= 0.7
                    
                    # 负债率影响
                    if customer['debt_ratio'] > 0.6:
                        default_prob *= 2.0
                    elif customer['debt_ratio'] > 0.4:
                        default_prob *= 1.3
                    
                    # 月供占收入比
                    payment_ratio = monthly_payment / customer['monthly_income']
                    if payment_ratio > 0.5:
                        default_prob *= 2.5
                    elif payment_ratio > 0.3:
                        default_prob *= 1.5
                    
                    # 信用评分影响
                    default_prob *= (1.5 - customer['credit_score'] / 950)
                    
                    default_prob = min(0.8, max(0.001, default_prob))
                    
                    # 是否违约
                    defaulted = rng.random() < default_prob
                    
                    if defaulted:
                        # 违约时间（在哪个月违约）
                        default_month = int(rng.exponential(6)) + 1
                        default_month = min(default_month, term_months)
                        max_dpd = int(rng.exponential(60)) + 30
                        loan_status = 'defaulted'
                    else:
                        default_month = None
                        max_dpd = 0
                        if rng.random() < 0.1:  # 10%提前还款
                            loan_status = 'prepaid'
                        else:
                            loan_status = 'completed'
                else:
                    monthly_payment = 0
                    default_month = None
                    max_dpd = 0
                    loan_status = 'rejected'
                
                app_id = f"L{batch_id:04d}{len(applications):08d}"
                
                applications.append({
                    'application_id': app_id,
                    'customer_id': customer['customer_id'],
                    'apply_date': apply_date.strftime('%Y-%m-%d'),
                    'loan_amount': round(loan_amount, 2),
                    'term_months': term_months,
                    'interest_rate': round(interest_rate, 4),
                    'purpose': purpose,
                    'approved': approved,
                    'monthly_payment': round(monthly_payment, 2),
                    'loan_status': loan_status,
                    'default_month': default_month,
                    'max_dpd': max_dpd,
                    'eco_phase': eco_phase,
                    'gdp_growth': round(gdp_growth, 4),
                    'unemployment_rate': round(unemployment, 4),
                })
        
        return pd.DataFrame(applications)
    
    def generate_repayment_history(
        self,
        loans: pd.DataFrame,
        seed: int
    ) -> pd.DataFrame:
        """生成还款历史"""
        rng = np.random.default_rng(seed)
        
        repayments = []
        
        approved_loans = loans[loans['approved'] == True]
        
        for _, loan in approved_loans.iterrows():
            apply_date = datetime.strptime(loan['apply_date'], '%Y-%m-%d')
            
            # 生成每月还款记录
            for month in range(1, loan['term_months'] + 1):
                due_date = apply_date + timedelta(days=30 * month)
                
                if due_date > END_DATE:
                    break
                
                # 判断是否违约
                if loan['loan_status'] == 'defaulted' and loan['default_month'] and month >= loan['default_month']:
                    # 违约后不再还款
                    payment_status = 'defaulted'
                    actual_payment = 0
                    dpd = loan['max_dpd']
                elif loan['loan_status'] == 'prepaid' and month > loan['term_months'] * 0.6:
                    # 提前还款
                    payment_status = 'prepaid'
                    actual_payment = loan['monthly_payment'] * (loan['term_months'] - month + 1)
                    dpd = 0
                    # 提前还款后结束
                    repayments.append({
                        'repayment_id': f"R{loan['application_id']}_{month:03d}",
                        'application_id': loan['application_id'],
                        'customer_id': loan['customer_id'],
                        'due_date': due_date.strftime('%Y-%m-%d'),
                        'due_amount': round(loan['monthly_payment'], 2),
                        'actual_payment': round(actual_payment, 2),
                        'payment_date': due_date.strftime('%Y-%m-%d'),
                        'dpd': dpd,
                        'payment_status': payment_status,
                    })
                    break
                else:
                    # 正常还款
                    if rng.random() < 0.02:  # 2%概率逾期
                        dpd = int(rng.exponential(7))
                        payment_date = due_date + timedelta(days=dpd)
                        payment_status = 'late' if dpd > 0 else 'on_time'
                    else:
                        dpd = 0
                        payment_date = due_date - timedelta(days=int(rng.uniform(0, 5)))
                        payment_status = 'on_time'
                    
                    actual_payment = loan['monthly_payment']
                
                repayments.append({
                    'repayment_id': f"R{loan['application_id']}_{month:03d}",
                    'application_id': loan['application_id'],
                    'customer_id': loan['customer_id'],
                    'due_date': due_date.strftime('%Y-%m-%d'),
                    'due_amount': round(loan['monthly_payment'], 2),
                    'actual_payment': round(actual_payment, 2),
                    'payment_date': payment_date.strftime('%Y-%m-%d') if payment_status != 'defaulted' else None,
                    'dpd': dpd,
                    'payment_status': payment_status,
                })
        
        return pd.DataFrame(repayments)
    
    def generate_macro_economics(self) -> pd.DataFrame:
        """生成宏观经济数据"""
        records = []
        
        current_date = START_DATE
        while current_date <= END_DATE:
            year = current_date.year
            month = current_date.month
            
            # 找到对应的经济周期
            eco_phase = 'normal'
            gdp_range = (0.03, 0.05)
            unemp_range = (0.05, 0.06)
            
            for cycle in ECONOMIC_CYCLES:
                if cycle[0] <= year <= cycle[1]:
                    eco_phase = cycle[2]
                    gdp_range = cycle[3]
                    unemp_range = cycle[4]
                    break
            
            rng = np.random.default_rng(year * 100 + month)
            
            records.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'year': year,
                'month': month,
                'quarter': (month - 1) // 3 + 1,
                'eco_phase': eco_phase,
                'gdp_growth': round(rng.uniform(*gdp_range), 4),
                'cpi': round(rng.uniform(0.01, 0.04), 4),
                'ppi': round(rng.uniform(-0.02, 0.05), 4),
                'unemployment_rate': round(rng.uniform(*unemp_range), 4),
                'base_interest_rate': round(0.0435 if year < 2020 else 0.0385, 4),
                'lpr_1y': round(0.0435 if year < 2019 else (0.0385 if year < 2022 else 0.0345), 4),
                'lpr_5y': round(0.049 if year < 2019 else (0.0465 if year < 2022 else 0.042), 4),
                'm2_growth': round(rng.uniform(0.08, 0.12), 4),
                'credit_growth': round(rng.uniform(0.10, 0.15), 4),
                'house_price_index': round(100 + (year - 2015) * 5 + rng.normal(0, 3), 2),
                'stock_index': round(3000 + (year - 2015) * 100 + rng.normal(0, 200), 2),
            })
            
            current_date += timedelta(days=30)
        
        return pd.DataFrame(records)


def process_batch(args):
    """处理一个批次的数据生成"""
    batch_id, batch_size, output_dir, seed = args
    
    generator = DataGenerator(output_dir, seed)
    
    # 生成客户
    customers = generator.generate_customer_batch(batch_id, batch_size, seed)
    
    # 生成贷款申请
    loans = generator.generate_loan_applications(customers, batch_id, seed + 1)
    
    # 生成还款历史
    repayments = generator.generate_repayment_history(loans, seed + 2)
    
    # 保存到临时文件
    customers.to_parquet(f"{output_dir}/temp/customers_{batch_id:04d}.parquet", index=False)
    loans.to_parquet(f"{output_dir}/temp/loans_{batch_id:04d}.parquet", index=False)
    repayments.to_parquet(f"{output_dir}/temp/repayments_{batch_id:04d}.parquet", index=False)
    
    return {
        'batch_id': batch_id,
        'customers': len(customers),
        'loans': len(loans),
        'repayments': len(repayments),
    }


def main():
    parser = argparse.ArgumentParser(description='生成大规模历史数据集')
    parser.add_argument('--output', type=str, default='data/historical',
                        help='输出目录')
    parser.add_argument('--customers', type=int, default=5000000,
                        help='客户数量 (默认500万)')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='每批处理的客户数')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行工作进程数 (默认: CPU核心数)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式 (生成少量数据用于测试)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.customers = 10000
        args.batch_size = 1000
        print("🚀 快速模式: 生成 10,000 客户数据用于测试")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'temp').mkdir(exist_ok=True)
    
    n_batches = args.customers // args.batch_size
    n_workers = args.workers or mp.cpu_count()
    
    print("=" * 60)
    print("🏭 Gamium 大规模数据集生成器")
    print("=" * 60)
    print(f"  目标客户数: {args.customers:,}")
    print(f"  批次大小: {args.batch_size:,}")
    print(f"  总批次数: {n_batches}")
    print(f"  并行进程: {n_workers}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)
    
    # 准备批次任务
    tasks = [
        (i, args.batch_size, str(output_dir), RANDOM_SEED + i * 100)
        for i in range(n_batches)
    ]
    
    start_time = time.time()
    total_customers = 0
    total_loans = 0
    total_repayments = 0
    
    print("\n📊 开始生成数据...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_batch, task): task[0] for task in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            total_customers += result['customers']
            total_loans += result['loans']
            total_repayments += result['repayments']
            
            if (i + 1) % 10 == 0 or i == n_batches - 1:
                elapsed = time.time() - start_time
                progress = (i + 1) / n_batches * 100
                eta = elapsed / (i + 1) * (n_batches - i - 1)
                
                print(f"  进度: {progress:.1f}% | "
                      f"客户: {total_customers:,} | "
                      f"贷款: {total_loans:,} | "
                      f"还款: {total_repayments:,} | "
                      f"耗时: {elapsed:.0f}s | "
                      f"预计剩余: {eta:.0f}s")
    
    print("\n🔄 合并临时文件...")
    
    # 合并所有临时文件
    customer_files = sorted(output_dir.glob('temp/customers_*.parquet'))
    loan_files = sorted(output_dir.glob('temp/loans_*.parquet'))
    repayment_files = sorted(output_dir.glob('temp/repayments_*.parquet'))
    
    print("  合并客户数据...")
    customers = pd.concat([pd.read_parquet(f) for f in customer_files], ignore_index=True)
    customers.to_parquet(output_dir / 'customers.parquet', index=False)
    
    print("  合并贷款数据...")
    loans = pd.concat([pd.read_parquet(f) for f in loan_files], ignore_index=True)
    loans.to_parquet(output_dir / 'loan_applications.parquet', index=False)
    
    print("  合并还款数据...")
    # 分块合并还款数据 (数据量大)
    repayment_chunks = []
    for f in repayment_files:
        repayment_chunks.append(pd.read_parquet(f))
    repayments = pd.concat(repayment_chunks, ignore_index=True)
    repayments.to_parquet(output_dir / 'repayment_history.parquet', index=False)
    
    # 生成宏观经济数据
    print("  生成宏观经济数据...")
    generator = DataGenerator(str(output_dir))
    macro = generator.generate_macro_economics()
    macro.to_parquet(output_dir / 'macro_economics.parquet', index=False)
    
    # 清理临时文件
    print("  清理临时文件...")
    for f in output_dir.glob('temp/*.parquet'):
        f.unlink()
    (output_dir / 'temp').rmdir()
    
    # 统计最终数据
    total_time = time.time() - start_time
    
    # 计算文件大小
    total_size = 0
    for f in output_dir.glob('*.parquet'):
        total_size += f.stat().st_size
    
    print("\n" + "=" * 60)
    print("✅ 数据生成完成!")
    print("=" * 60)
    print(f"  总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  客户数据: {len(customers):,} 条")
    print(f"  贷款申请: {len(loans):,} 条")
    print(f"  还款记录: {len(repayments):,} 条")
    print(f"  宏观数据: {len(macro):,} 条")
    print(f"  总数据量: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print("\n📁 输出文件:")
    for f in sorted(output_dir.glob('*.parquet')):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name}: {size_mb:.1f} MB")
    print("=" * 60)
    
    # 生成数据摘要
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_customers': len(customers),
        'total_loans': len(loans),
        'total_repayments': len(repayments),
        'date_range': f"{START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}",
        'total_size_gb': round(total_size / 1024 / 1024 / 1024, 2),
        'files': [f.name for f in output_dir.glob('*.parquet')],
    }
    
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💡 使用方式:")
    print(f"   from gamium.data import load_historical_data")
    print(f"   data = load_historical_data('{output_dir}')")


if __name__ == '__main__':
    main()

