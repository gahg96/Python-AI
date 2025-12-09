#!/usr/bin/env python3
"""
银行数据提取和特征工程脚本

从银行现有系统中提取数据并构建特征

使用方法:
    python3 extract_banking_data.py --config config.yaml --output data/extracted/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
import hashlib

class BankingDataExtractor:
    """银行数据提取器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/extracted'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """从数据库提取数据"""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(connection_string)
            return pd.read_sql(query, engine)
        except ImportError:
            print("⚠️  需要安装 sqlalchemy: pip install sqlalchemy")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ 数据库提取失败: {e}")
            return pd.DataFrame()
    
    def extract_from_csv(self, file_path: str) -> pd.DataFrame:
        """从CSV文件提取数据"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ CSV提取失败: {e}")
            return pd.DataFrame()
    
    def extract_from_parquet(self, file_path: str) -> pd.DataFrame:
        """从Parquet文件提取数据"""
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"❌ Parquet提取失败: {e}")
            return pd.DataFrame()
    
    def extract_customer_data(self, source: Dict) -> pd.DataFrame:
        """提取客户基础数据"""
        print("📊 提取客户数据...")
        
        source_type = source.get('type', 'database')
        
        if source_type == 'database':
            query = source.get('query', '')
            conn_str = source.get('connection_string', '')
            df = self.extract_from_database(conn_str, query)
        elif source_type == 'csv':
            df = self.extract_from_csv(source.get('file_path', ''))
        elif source_type == 'parquet':
            df = self.extract_from_parquet(source.get('file_path', ''))
        else:
            print(f"❌ 不支持的数据源类型: {source_type}")
            return pd.DataFrame()
        
        print(f"   ✅ 提取 {len(df):,} 条客户记录")
        return df
    
    def calculate_customer_features(self, customers: pd.DataFrame) -> pd.DataFrame:
        """计算客户特征"""
        print("🔧 计算客户特征...")
        
        features = customers.copy()
        
        # 基础特征（如果不存在则创建）
        if 'age' in features.columns:
            features['age_group'] = pd.cut(
                features['age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['18-25', '25-35', '35-45', '45-55', '55+']
            )
        
        if 'registration_date' in features.columns:
            features['registration_date'] = pd.to_datetime(features['registration_date'])
            features['months_as_customer'] = (
                datetime.now() - features['registration_date']
            ).dt.days / 30
        
        # 财务比率特征
        if 'total_assets' in features.columns and 'total_liabilities' in features.columns:
            features['debt_ratio'] = features['total_liabilities'] / (
                features['total_assets'] + 1
            )
        
        if 'monthly_income' in features.columns and 'total_liabilities' in features.columns:
            features['debt_to_income'] = features['total_liabilities'] / (
                features['monthly_income'] * 12 + 1
            )
        
        # 处理缺失值
        features = self.handle_missing_values(features)
        
        print(f"   ✅ 特征计算完成，共 {len(features.columns)} 个特征")
        return features
    
    def calculate_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """计算交易特征"""
        if transactions.empty:
            return pd.DataFrame()
        
        print("🔧 计算交易特征...")
        
        # 确保日期列是datetime类型
        if 'transaction_date' in transactions.columns:
            transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        # 按客户聚合
        features = transactions.groupby('customer_id').agg({
            'transaction_amount': ['mean', 'std', 'sum', 'count'],
        }).reset_index()
        
        # 展平列名
        features.columns = ['customer_id', 'avg_transaction', 'transaction_std', 
                           'transaction_sum', 'transaction_count']
        
        # 计算收入支出
        if 'transaction_type' in transactions.columns:
            income = transactions[transactions['transaction_type'] == 'income'].groupby('customer_id')[
                'transaction_amount'
            ].sum().reset_index()
            income.columns = ['customer_id', 'total_income']
            
            expense = transactions[transactions['transaction_type'] == 'expense'].groupby('customer_id')[
                'transaction_amount'
            ].sum().reset_index()
            expense.columns = ['customer_id', 'total_expense']
            
            features = features.merge(income, on='customer_id', how='left')
            features = features.merge(expense, on='customer_id', how='left')
            
            # 计算储蓄率
            features['savings_rate'] = (
                features['total_income'] - features['total_expense']
            ) / (features['total_income'] + 1)
            
            # 计算收入波动率
            features['income_volatility'] = features['transaction_std'] / (
                features['avg_transaction'] + 1
            )
        
        print(f"   ✅ 交易特征计算完成")
        return features
    
    def calculate_loan_features(self, loans: pd.DataFrame) -> pd.DataFrame:
        """计算贷款特征"""
        if loans.empty:
            return pd.DataFrame()
        
        print("🔧 计算贷款特征...")
        
        features = loans.groupby('customer_id').agg({
            'loan_amount': ['count', 'sum', 'mean', 'max'],
            'interest_rate': 'mean',
            'overdue_days': 'max',
        }).reset_index()
        
        features.columns = ['customer_id', 'total_loans', 'total_loan_amount',
                           'avg_loan_amount', 'max_loan_amount', 'avg_interest_rate',
                           'max_overdue_days']
        
        # 计算违约次数
        if 'loan_status' in loans.columns:
            defaults = loans[loans['loan_status'] == 'defaulted'].groupby('customer_id').size().reset_index()
            defaults.columns = ['customer_id', 'default_count']
            features = features.merge(defaults, on='customer_id', how='left')
            features['default_count'] = features['default_count'].fillna(0)
        
        # 计算距上次贷款时间
        if 'apply_date' in loans.columns:
            loans['apply_date'] = pd.to_datetime(loans['apply_date'])
            last_loan = loans.groupby('customer_id')['apply_date'].max().reset_index()
            last_loan.columns = ['customer_id', 'last_loan_date']
            last_loan['months_since_last_loan'] = (
                datetime.now() - last_loan['last_loan_date']
            ).dt.days / 30
            features = features.merge(last_loan[['customer_id', 'months_since_last_loan']], 
                                    on='customer_id', how='left')
        
        print(f"   ✅ 贷款特征计算完成")
        return features
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 数值型用中位数
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # 分类型用众数
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def anonymize_data(self, df: pd.DataFrame, anonymize_cols: List[str] = None) -> pd.DataFrame:
        """数据脱敏"""
        if anonymize_cols is None:
            anonymize_cols = ['customer_id', 'id_card', 'phone', 'email']
        
        df_anon = df.copy()
        
        for col in anonymize_cols:
            if col in df_anon.columns:
                if col == 'customer_id':
                    # 使用哈希ID
                    df_anon[col] = df_anon[col].apply(
                        lambda x: hashlib.md5(str(x).encode()).hexdigest()[:16]
                    )
                else:
                    # 删除敏感列
                    df_anon = df_anon.drop(columns=[col])
        
        return df_anon
    
    def run(self):
        """运行数据提取流程"""
        print("=" * 60)
        print("🏦 银行数据提取和特征工程")
        print("=" * 60)
        
        # 提取客户数据
        customer_source = self.config.get('customer_source', {})
        customers = self.extract_customer_data(customer_source)
        
        if customers.empty:
            print("❌ 未提取到客户数据，请检查配置")
            return
        
        # 计算客户特征
        customer_features = self.calculate_customer_features(customers)
        
        # 提取并计算交易特征
        transaction_source = self.config.get('transaction_source', {})
        if transaction_source:
            transactions = self.extract_customer_data(transaction_source)
            transaction_features = self.calculate_transaction_features(transactions)
            
            if not transaction_features.empty:
                customer_features = customer_features.merge(
                    transaction_features, on='customer_id', how='left'
                )
        
        # 提取并计算贷款特征
        loan_source = self.config.get('loan_source', {})
        if loan_source:
            loans = self.extract_customer_data(loan_source)
            loan_features = self.calculate_loan_features(loans)
            
            if not loan_features.empty:
                customer_features = customer_features.merge(
                    loan_features, on='customer_id', how='left'
                )
        
        # 数据脱敏（如果配置了）
        if self.config.get('anonymize', False):
            customer_features = self.anonymize_data(customer_features)
        
        # 保存结果
        output_file = self.output_dir / 'customer_features.parquet'
        customer_features.to_parquet(output_file, index=False)
        
        print("=" * 60)
        print(f"✅ 数据提取完成")
        print(f"   输出文件: {output_file}")
        print(f"   记录数: {len(customer_features):,}")
        print(f"   特征数: {len(customer_features.columns)}")
        print("=" * 60)

def create_sample_config(output_path: str):
    """创建示例配置文件"""
    config = {
        'output_dir': 'data/extracted',
        'anonymize': True,
        
        'customer_source': {
            'type': 'parquet',  # 或 'database', 'csv'
            'file_path': 'data/historical/customers.parquet',
            # 如果是数据库:
            # 'type': 'database',
            # 'connection_string': 'postgresql://user:password@host:port/database',
            # 'query': 'SELECT * FROM customers WHERE registration_date >= %s'
        },
        
        'transaction_source': {
            'type': 'parquet',
            'file_path': 'data/historical/repayment_history.parquet',
        },
        
        'loan_source': {
            'type': 'parquet',
            'file_path': 'data/historical/loan_applications.parquet',
        },
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 示例配置文件已创建: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='银行数据提取和特征工程')
    parser.add_argument('--config', type=str, default='config/extract_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default='data/extracted',
                       help='输出目录')
    parser.add_argument('--create-config', action='store_true',
                       help='创建示例配置文件')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config(args.config)
    else:
        # 加载配置
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            print("   使用 --create-config 创建示例配置文件")
            sys.exit(1)
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # 覆盖输出目录
        if args.output:
            config['output_dir'] = args.output
        
        # 运行提取
        extractor = BankingDataExtractor(config)
        extractor.run()

