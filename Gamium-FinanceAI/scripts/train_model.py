#!/usr/bin/env python3
"""
模型训练脚本

使用方法:
    python3 train_model.py --features data/extracted/customer_features.parquet --output models/
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve
)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, features_path: str, output_dir: str):
        self.features_path = Path(features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_cols = None
        
    def load_data(self):
        """加载特征数据"""
        print(f"📊 加载特征文件: {self.features_path}")
        self.features = pd.read_parquet(self.features_path)
        print(f"   ✅ 加载完成: {len(self.features):,} 条记录, {len(self.features.columns)} 个特征")
        return self.features
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n🔧 数据预处理...")
        
        # 处理缺失值
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        self.features[numeric_cols] = self.features[numeric_cols].fillna(
            self.features[numeric_cols].median()
        )
        
        # 处理分类特征
        categorical_cols = ['gender', 'education', 'industry', 'city_tier', 'customer_type']
        for col in categorical_cols:
            if col in self.features.columns:
                le = LabelEncoder()
                self.features[col] = le.fit_transform(self.features[col].astype(str))
                self.label_encoders[col] = le
        
        print("   ✅ 预处理完成")
    
    def prepare_features(self):
        """准备特征和目标变量"""
        print("\n🎯 准备特征和目标变量...")
        
        # 选择特征列
        self.feature_cols = [
            'age', 'gender', 'education', 'industry', 'city_tier',
            'monthly_income', 'total_assets', 'total_liabilities',
            'debt_ratio', 'debt_to_income', 'total_deposit_balance',
            'savings_rate', 'income_volatility',
            'total_loans', 'default_count', 'max_overdue_days',
            'months_as_customer', 'months_since_last_loan'
        ]
        
        # 只选择存在的列
        self.feature_cols = [col for col in self.feature_cols if col in self.features.columns]
        
        X = self.features[self.feature_cols]
        y = self.features['defaulted']
        
        print(f"   特征数: {len(self.feature_cols)}")
        print(f"   目标变量分布:\n{y.value_counts()}")
        
        return X, y
    
    def split_data(self, X, y):
        """分割数据集"""
        print("\n📊 分割数据集...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"   训练集: {X_train.shape[0]:,} 条")
        print(f"   验证集: {X_val.shape[0]:,} 条")
        print(f"   测试集: {X_test.shape[0]:,} 条")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """特征标准化"""
        print("\n📏 特征标准化...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("   ✅ 标准化完成")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        print("\n🚀 训练模型...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        # 验证集评估
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        print(f"   ✅ 训练完成")
        print(f"   验证集 AUC: {val_auc:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        print("\n📈 模型评估...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"AUC Score: {auc:.4f}")
        print(f"\n分类报告:\n{report}")
        print(f"\n混淆矩阵:\n{cm}")
        
        return {
            'auc': float(auc),
            'report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, metrics):
        """保存模型"""
        print("\n💾 保存模型...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = self.output_dir / f'model_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存scaler
        scaler_path = self.output_dir / f'scaler_{timestamp}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存label encoders
        encoders_path = self.output_dir / f'encoders_{timestamp}.pkl'
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # 保存特征列表
        feature_info = {
            'feature_cols': self.feature_cols,
            'metrics': metrics,
            'timestamp': timestamp,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'encoders_path': str(encoders_path)
        }
        
        info_path = self.output_dir / f'model_info_{timestamp}.json'
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 模型已保存:")
        print(f"     模型: {model_path}")
        print(f"     Scaler: {scaler_path}")
        print(f"     Encoders: {encoders_path}")
        print(f"     信息: {info_path}")
        
        return model_path
    
    def run(self):
        """运行完整训练流程"""
        print("=" * 60)
        print("🎯 模型训练流程")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 预处理
        self.preprocess_data()
        
        # 3. 准备特征
        X, y = self.prepare_features()
        
        # 4. 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 5. 标准化
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # 6. 训练模型
        self.train_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # 7. 评估模型
        metrics = self.evaluate_model(X_test_scaled, y_test)
        
        # 8. 保存模型
        model_path = self.save_model(metrics)
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        
        return model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型训练脚本')
    parser.add_argument('--features', type=str, required=True,
                       help='特征文件路径 (Parquet格式)')
    parser.add_argument('--output', type=str, default='models',
                       help='模型输出目录')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.features, args.output)
    trainer.run()


