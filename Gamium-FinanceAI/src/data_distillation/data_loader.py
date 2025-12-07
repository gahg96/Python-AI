"""
数据加载器 - 加载真实的历史数据集

支持从 Parquet 文件加载大规模历史数据
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd


class HistoricalDataLoader:
    """
    历史数据加载器
    
    加载并预处理大规模历史贷款数据
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.customers: Optional[pd.DataFrame] = None
        self.loans: Optional[pd.DataFrame] = None
        self.repayments: Optional[pd.DataFrame] = None
        self.macro: Optional[pd.DataFrame] = None
        self._loaded = False
    
    def load(self, sample_size: Optional[int] = None) -> 'HistoricalDataLoader':
        """
        加载所有数据
        
        Args:
            sample_size: 如果指定，只加载采样数据
        """
        print(f"📂 从 {self.data_dir} 加载数据...")
        
        # 加载客户数据
        customers_path = self.data_dir / 'customers.parquet'
        if customers_path.exists():
            self.customers = pd.read_parquet(customers_path)
            if sample_size and len(self.customers) > sample_size:
                self.customers = self.customers.sample(n=sample_size, random_state=42)
            print(f"  ✅ 客户数据: {len(self.customers):,} 条")
        
        # 加载贷款数据
        loans_path = self.data_dir / 'loan_applications.parquet'
        if loans_path.exists():
            self.loans = pd.read_parquet(loans_path)
            if sample_size and self.customers is not None:
                customer_ids = set(self.customers['customer_id'])
                self.loans = self.loans[self.loans['customer_id'].isin(customer_ids)]
            print(f"  ✅ 贷款申请: {len(self.loans):,} 条")
        
        # 加载还款数据
        repayments_path = self.data_dir / 'repayment_history.parquet'
        if repayments_path.exists():
            self.repayments = pd.read_parquet(repayments_path)
            if sample_size and self.loans is not None:
                loan_ids = set(self.loans['application_id'])
                self.repayments = self.repayments[self.repayments['application_id'].isin(loan_ids)]
            print(f"  ✅ 还款记录: {len(self.repayments):,} 条")
        
        # 加载宏观数据
        macro_path = self.data_dir / 'macro_economics.parquet'
        if macro_path.exists():
            self.macro = pd.read_parquet(macro_path)
            print(f"  ✅ 宏观数据: {len(self.macro):,} 条")
        
        self._loaded = True
        return self
    
    def get_training_data(
        self,
        train_years: List[int] = None,
        test_years: List[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取训练和测试数据
        
        Args:
            train_years: 训练年份
            test_years: 测试年份
            
        Returns:
            (train_data, test_data)
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load() 加载数据")
        
        if train_years is None:
            train_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        if test_years is None:
            test_years = [2023, 2024]
        
        # 合并客户和贷款数据
        merged = self._merge_data()
        
        # 按年份分割
        merged['apply_year'] = pd.to_datetime(merged['apply_date']).dt.year
        
        train_data = merged[merged['apply_year'].isin(train_years)]
        test_data = merged[merged['apply_year'].isin(test_years)]
        
        print(f"  训练集: {len(train_data):,} 条 ({train_years})")
        print(f"  测试集: {len(test_data):,} 条 ({test_years})")
        
        return train_data, test_data
    
    def _merge_data(self) -> pd.DataFrame:
        """合并客户、贷款和宏观数据"""
        # 贷款数据作为基础
        merged = self.loans.copy()
        
        # 合并客户特征
        customer_cols = [
            'customer_id', 'customer_type', 'age', 'city_tier', 'industry',
            'education', 'monthly_income', 'income_volatility', 'total_assets',
            'total_liabilities', 'debt_ratio', 'deposit_balance', 'deposit_stability',
            'has_house', 'has_car', 'credit_score', 'base_default_rate'
        ]
        if self.customers is not None:
            available_cols = [c for c in customer_cols if c in self.customers.columns]
            merged = merged.merge(
                self.customers[available_cols],
                on='customer_id',
                how='left'
            )
        
        # 添加违约标签
        merged['defaulted'] = (merged['loan_status'] == 'defaulted').astype(int)
        
        return merged
    
    def build_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建特征矩阵和标签
        
        Args:
            data: 合并后的数据
            
        Returns:
            (features, labels)
        """
        feature_cols = [
            # 客户特征
            'age', 'monthly_income', 'income_volatility', 'total_assets',
            'debt_ratio', 'deposit_balance', 'deposit_stability',
            'credit_score',
            # 贷款特征
            'loan_amount', 'term_months', 'interest_rate',
            # 环境特征
            'gdp_growth', 'unemployment_rate',
        ]
        
        # 确保所有列都存在
        available_cols = [c for c in feature_cols if c in data.columns]
        
        features = data[available_cols].fillna(0).values.astype(np.float32)
        labels = data['defaulted'].values.astype(np.int32)
        
        # 归一化
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std
        
        return features, labels
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self._loaded:
            return {}
        
        stats = {
            'total_customers': len(self.customers) if self.customers is not None else 0,
            'total_loans': len(self.loans) if self.loans is not None else 0,
            'total_repayments': len(self.repayments) if self.repayments is not None else 0,
        }
        
        if self.loans is not None:
            stats['approval_rate'] = self.loans['approved'].mean()
            stats['default_rate'] = (self.loans['loan_status'] == 'defaulted').mean()
            stats['avg_loan_amount'] = self.loans['loan_amount'].mean()
            
            # 按年份统计
            self.loans['year'] = pd.to_datetime(self.loans['apply_date']).dt.year
            yearly = self.loans.groupby('year').agg({
                'application_id': 'count',
                'approved': 'mean',
                'loan_status': lambda x: (x == 'defaulted').mean()
            }).to_dict()
            stats['by_year'] = yearly
        
        return stats


def load_historical_data(data_dir: str, sample_size: Optional[int] = None) -> HistoricalDataLoader:
    """
    加载历史数据的快捷函数
    
    Args:
        data_dir: 数据目录
        sample_size: 采样大小
        
    Returns:
        HistoricalDataLoader 实例
    """
    loader = HistoricalDataLoader(data_dir)
    loader.load(sample_size=sample_size)
    return loader


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test_data"
    
    print("=" * 60)
    print("历史数据加载测试")
    print("=" * 60)
    
    loader = load_historical_data(data_dir)
    
    stats = loader.get_statistics()
    print(f"\n数据统计:")
    print(f"  客户数: {stats.get('total_customers', 0):,}")
    print(f"  贷款数: {stats.get('total_loans', 0):,}")
    print(f"  还款记录: {stats.get('total_repayments', 0):,}")
    print(f"  审批率: {stats.get('approval_rate', 0):.2%}")
    print(f"  违约率: {stats.get('default_rate', 0):.2%}")
    
    print("\n获取训练/测试数据...")
    train_data, test_data = loader.get_training_data()
    
    print("\n构建特征...")
    X_train, y_train = loader.build_features(train_data)
    X_test, y_test = loader.build_features(test_data)
    
    print(f"  训练特征: {X_train.shape}")
    print(f"  测试特征: {X_test.shape}")
    print(f"  训练集违约率: {y_train.mean():.2%}")
    print(f"  测试集违约率: {y_test.mean():.2%}")

