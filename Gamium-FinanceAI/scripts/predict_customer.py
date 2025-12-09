#!/usr/bin/env python3
"""
客户信用评分预测脚本

使用训练好的模型对特定贷款申请用户进行打分

使用方法:
    python3 predict_customer.py --model models/model_20241209_143022.pkl --customer customer_data.json
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

class CustomerPredictor:
    """客户信用评分预测器"""
    
    def __init__(self, model_path: str, scaler_path: str = None, encoders_path: str = None):
        self.model_path = Path(model_path)
        self.model_dir = self.model_path.parent
        
        # 加载模型
        print(f"📦 加载模型: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # 加载scaler
        if scaler_path:
            scaler_file = Path(scaler_path)
        else:
            # 自动查找scaler文件
            timestamp = self.model_path.stem.split('_')[-1]
            scaler_file = self.model_dir / f'scaler_{timestamp}.pkl'
        
        if scaler_file.exists():
            print(f"📦 加载Scaler: {scaler_file}")
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print("⚠️  未找到Scaler文件，将跳过标准化")
            self.scaler = None
        
        # 加载encoders
        if encoders_path:
            encoders_file = Path(encoders_path)
        else:
            # 自动查找encoders文件
            timestamp = self.model_path.stem.split('_')[-1]
            encoders_file = self.model_dir / f'encoders_{timestamp}.pkl'
        
        if encoders_file.exists():
            print(f"📦 加载Encoders: {encoders_file}")
            with open(encoders_file, 'rb') as f:
                self.encoders = pickle.load(f)
        else:
            print("⚠️  未找到Encoders文件")
            self.encoders = {}
        
        # 加载模型信息
        info_file = self.model_dir / f"model_info_{self.model_path.stem.split('_')[-1]}.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.model_info = json.load(f)
                self.feature_cols = self.model_info.get('feature_cols', [])
        else:
            print("⚠️  未找到模型信息文件，使用默认特征列表")
            self.feature_cols = [
                'age', 'gender', 'education', 'industry', 'city_tier',
                'monthly_income', 'total_assets', 'total_liabilities',
                'debt_ratio', 'debt_to_income', 'total_deposit_balance',
                'savings_rate', 'income_volatility',
                'total_loans', 'default_count', 'max_overdue_days',
                'months_as_customer', 'months_since_last_loan'
            ]
        
        print("✅ 模型加载完成\n")
    
    def prepare_customer_features(self, customer_data: Dict[str, Any]) -> pd.DataFrame:
        """准备客户特征"""
        # 创建特征字典
        features = {}
        
        # 基础特征
        features['age'] = customer_data.get('age', 35)
        features['gender'] = customer_data.get('gender', 'M')
        features['education'] = customer_data.get('education', 'bachelor')
        features['industry'] = customer_data.get('industry', 'service')
        features['city_tier'] = customer_data.get('city_tier', 'tier_2')
        features['customer_type'] = customer_data.get('customer_type', 'salaried')
        
        # 财务特征
        features['monthly_income'] = customer_data.get('monthly_income', 10000.0)
        features['total_assets'] = customer_data.get('total_assets', 300000.0)
        features['total_liabilities'] = customer_data.get('total_liabilities', 150000.0)
        
        # 计算衍生特征
        if 'debt_ratio' not in customer_data:
            features['debt_ratio'] = features['total_liabilities'] / (features['total_assets'] + 1)
        else:
            features['debt_ratio'] = customer_data['debt_ratio']
        
        if 'debt_to_income' not in customer_data:
            features['debt_to_income'] = features['total_liabilities'] / (features['monthly_income'] * 12 + 1)
        else:
            features['debt_to_income'] = customer_data['debt_to_income']
        
        features['total_deposit_balance'] = customer_data.get('total_deposit_balance', 50000.0)
        features['avg_account_balance'] = customer_data.get('avg_account_balance', 25000.0)
        
        # 交易特征
        total_income = customer_data.get('total_income', features['monthly_income'] * 12)
        total_expense = customer_data.get('total_expense', total_income * 0.8)
        
        features['total_income'] = total_income
        features['total_expense'] = total_expense
        features['savings_rate'] = (total_income - total_expense) / (total_income + 1)
        features['income_volatility'] = customer_data.get('income_volatility', 0.2)
        features['transaction_count'] = customer_data.get('transaction_count', 120)
        features['avg_transaction_amount'] = customer_data.get('avg_transaction_amount', 1000.0)
        
        # 贷款历史特征
        features['total_loans'] = customer_data.get('total_loans', 0)
        features['total_loan_amount'] = customer_data.get('total_loan_amount', 0.0)
        features['avg_loan_amount'] = customer_data.get('avg_loan_amount', 0.0)
        features['default_count'] = customer_data.get('default_count', 0)
        features['max_overdue_days'] = customer_data.get('max_overdue_days', 0)
        features['avg_interest_rate'] = customer_data.get('avg_interest_rate', 0.06)
        features['months_since_last_loan'] = customer_data.get('months_since_last_loan', 12)
        
        # 时间特征
        features['months_as_customer'] = customer_data.get('months_as_customer', 24)
        
        # 转换为DataFrame
        df = pd.DataFrame([features])
        
        # 编码分类特征
        for col in ['gender', 'education', 'industry', 'city_tier', 'customer_type']:
            if col in df.columns and col in self.encoders:
                try:
                    df[col] = self.encoders[col].transform([df[col].iloc[0]])[0]
                except ValueError:
                    # 如果值不在训练集中，使用最常见的值
                    df[col] = 0
        
        return df
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """预测客户信用评分"""
        print("🔍 准备客户特征...")
        
        # 准备特征
        customer_df = self.prepare_customer_features(customer_data)
        
        # 选择特征列
        X = customer_df[self.feature_cols]
        
        print(f"   特征数: {len(self.feature_cols)}")
        print(f"   特征值:\n{X.to_dict('records')[0]}")
        
        # 标准化
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # 预测
        print("\n🎯 进行预测...")
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        # 计算信用评分（0-1000分）
        default_prob = probability[1] if len(probability) > 1 else probability[0]
        credit_score = int((1 - default_prob) * 1000)
        
        # 风险等级
        if credit_score >= 800:
            risk_level = "低风险"
        elif credit_score >= 650:
            risk_level = "中低风险"
        elif credit_score >= 500:
            risk_level = "中风险"
        elif credit_score >= 350:
            risk_level = "中高风险"
        else:
            risk_level = "高风险"
        
        # 审批建议
        if credit_score >= 650:
            approval_suggestion = "建议通过"
        elif credit_score >= 500:
            approval_suggestion = "条件通过（需额外审核）"
        else:
            approval_suggestion = "建议拒绝"
        
        result = {
            'credit_score': credit_score,
            'default_probability': float(default_prob),
            'risk_level': risk_level,
            'approval_suggestion': approval_suggestion,
            'prediction': int(prediction),
            'probability_distribution': {
                'normal': float(probability[0]) if len(probability) > 0 else 0.0,
                'defaulted': float(probability[1]) if len(probability) > 1 else 0.0
            },
            'model_info': {
                'model_path': str(self.model_path),
                'model_auc': self.model_info.get('metrics', {}).get('auc', 'N/A'),
                'feature_count': len(self.feature_cols)
            }
        }
        
        return result
    
    def explain_prediction(self, customer_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """解释预测结果"""
        explanation = {
            'credit_score': result['credit_score'],
            'risk_level': result['risk_level'],
            'approval_suggestion': result['approval_suggestion'],
            'key_factors': [],
            'risk_factors': [],
            'positive_factors': []
        }
        
        # 分析关键因素
        customer_df = self.prepare_customer_features(customer_data)
        
        # 风险因素
        if customer_df['debt_ratio'].iloc[0] > 0.6:
            explanation['risk_factors'].append({
                'factor': '负债率过高',
                'value': f"{customer_df['debt_ratio'].iloc[0]:.2%}",
                'impact': '高'
            })
        
        if customer_df['default_count'].iloc[0] > 0:
            explanation['risk_factors'].append({
                'factor': '历史违约记录',
                'value': f"{customer_df['default_count'].iloc[0]}次",
                'impact': '高'
            })
        
        if customer_df['max_overdue_days'].iloc[0] > 30:
            explanation['risk_factors'].append({
                'factor': '历史逾期天数',
                'value': f"{customer_df['max_overdue_days'].iloc[0]}天",
                'impact': '中'
            })
        
        if customer_df['income_volatility'].iloc[0] > 0.3:
            explanation['risk_factors'].append({
                'factor': '收入波动较大',
                'value': f"{customer_df['income_volatility'].iloc[0]:.2%}",
                'impact': '中'
            })
        
        # 正面因素
        if customer_df['savings_rate'].iloc[0] > 0.2:
            explanation['positive_factors'].append({
                'factor': '储蓄率较高',
                'value': f"{customer_df['savings_rate'].iloc[0]:.2%}",
                'impact': '中'
            })
        
        if customer_df['months_as_customer'].iloc[0] > 36:
            explanation['positive_factors'].append({
                'factor': '客户关系稳定',
                'value': f"{customer_df['months_as_customer'].iloc[0]}个月",
                'impact': '中'
            })
        
        if customer_df['total_deposit_balance'].iloc[0] > customer_df['monthly_income'].iloc[0] * 6:
            explanation['positive_factors'].append({
                'factor': '存款余额充足',
                'value': f"{customer_df['total_deposit_balance'].iloc[0]:,.0f}元",
                'impact': '中'
            })
        
        return explanation

def main():
    parser = argparse.ArgumentParser(description='客户信用评分预测')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径 (.pkl)')
    parser.add_argument('--customer', type=str, required=True,
                       help='客户数据文件 (JSON格式)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果文件 (JSON格式)')
    parser.add_argument('--explain', action='store_true',
                       help='显示预测解释')
    
    args = parser.parse_args()
    
    # 加载客户数据
    with open(args.customer, 'r') as f:
        customer_data = json.load(f)
    
    # 创建预测器
    predictor = CustomerPredictor(args.model)
    
    # 预测
    result = predictor.predict(customer_data)
    
    # 解释（如果需要）
    if args.explain:
        explanation = predictor.explain_prediction(customer_data, result)
        result['explanation'] = explanation
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📊 预测结果")
    print("=" * 60)
    print(f"信用评分: {result['credit_score']} 分")
    print(f"违约概率: {result['default_probability']:.2%}")
    print(f"风险等级: {result['risk_level']}")
    print(f"审批建议: {result['approval_suggestion']}")
    
    if args.explain:
        print(f"\n风险因素 ({len(result['explanation']['risk_factors'])}个):")
        for factor in result['explanation']['risk_factors']:
            print(f"  - {factor['factor']}: {factor['value']} (影响: {factor['impact']})")
        
        print(f"\n正面因素 ({len(result['explanation']['positive_factors'])}个):")
        for factor in result['explanation']['positive_factors']:
            print(f"  - {factor['factor']}: {factor['value']} (影响: {factor['impact']})")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存到: {args.output}")
    
    return result

if __name__ == '__main__':
    main()


