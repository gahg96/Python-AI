"""
世界模型训练模块
训练违约预测模型和还款行为模型
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')


class WorldModelTrainer:
    """世界模型训练器"""
    
    def __init__(self, data: pd.DataFrame, seed: int = 42):
        """
        初始化训练器
        
        Args:
            data: 历史贷款数据（已进行特征工程）
            seed: 随机种子
        """
        self.data = data.copy()
        self.seed = seed
        self.models = {}
        self.feature_importance = {}
        self.model_metrics = {}
    
    def prepare_default_prediction_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备违约预测数据
        
        Returns:
            (特征DataFrame, 目标Series)
        """
        print("=" * 80)
        print("准备违约预测数据")
        print("=" * 80)
        
        # 只使用已批准的贷款
        approved_data = self.data[self.data['expert_decision'] == 'approve'].copy()
        
        if len(approved_data) == 0:
            raise ValueError("没有已批准的贷款数据")
        
        # 选择特征
        feature_cols = [
            # 对私特征
            'age', 'monthly_income', 'credit_score', 'debt_ratio',
            'years_in_job', 'loan_to_annual_income_ratio',
            'comprehensive_risk_score', 'job_stability',
            # 对公特征
            'registered_capital', 'operating_years', 'annual_revenue',
            'debt_to_asset_ratio', 'current_ratio', 'loan_to_revenue_ratio',
            # 贷款特征
            'loan_amount', 'requested_term_months',
            # 市场环境特征
            'gdp_growth', 'base_interest_rate', 'unemployment_rate',
            'inflation_rate', 'credit_spread', 'market_sentiment',
            # 时间特征
            'application_year', 'application_month', 'application_quarter'
        ]
        
        # 只保留存在的特征
        available_features = [f for f in feature_cols if f in approved_data.columns]
        
        # 处理缺失值
        X = approved_data[available_features].copy()
        X = X.fillna(X.median())
        
        # 目标变量
        y = approved_data['is_defaulted'].copy()
        
        print(f"特征数量: {len(available_features)}")
        print(f"样本数量: {len(X)}")
        print(f"违约率: {y.mean():.2%}")
        print(f"特征列表: {available_features[:10]}...")
        
        return X, y
    
    def train_default_prediction_model(self, model_type: str = 'random_forest') -> Dict:
        """
        训练违约预测模型
        
        Args:
            model_type: 模型类型 ('random_forest', 'gradient_boosting', 'logistic')
        
        Returns:
            训练结果字典
        """
        print("\n" + "=" * 80)
        print(f"训练违约预测模型 ({model_type})")
        print("=" * 80)
        
        # 准备数据
        X, y = self.prepare_default_prediction_data()
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 选择模型
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.seed,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.seed
            )
        elif model_type == 'logistic':
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.seed,
                n_jobs=-1
            )
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        # 训练模型
        print("\n正在训练模型...")
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }
        
        print("\n模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 重要特征:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            self.feature_importance['default_prediction'] = feature_importance
        
        # 保存模型
        self.models['default_prediction'] = model
        self.model_metrics['default_prediction'] = metrics
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
        }
    
    def prepare_profit_prediction_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备利润预测数据
        
        Returns:
            (特征DataFrame, 目标Series)
        """
        print("\n" + "=" * 80)
        print("准备利润预测数据")
        print("=" * 80)
        
        # 只使用已批准的贷款
        approved_data = self.data[self.data['expert_decision'] == 'approve'].copy()
        
        if len(approved_data) == 0:
            raise ValueError("没有已批准的贷款数据")
        
        # 选择特征
        feature_cols = [
            # 客户特征
            'age', 'monthly_income', 'credit_score', 'debt_ratio',
            'years_in_job', 'comprehensive_risk_score',
            # 贷款特征
            'loan_amount', 'approved_rate', 'approved_term_months',
            'loan_to_annual_income_ratio',
            # 市场环境
            'gdp_growth', 'base_interest_rate', 'unemployment_rate',
            'credit_spread', 'market_sentiment'
        ]
        
        # 只保留存在的特征
        available_features = [f for f in feature_cols if f in approved_data.columns]
        
        # 处理缺失值
        X = approved_data[available_features].copy()
        X = X.fillna(X.median())
        
        # 目标变量（实际利润）
        y = approved_data['actual_profit'].copy()
        
        print(f"特征数量: {len(available_features)}")
        print(f"样本数量: {len(X)}")
        print(f"平均利润: ¥{y.mean():,.2f}")
        print(f"利润范围: ¥{y.min():,.2f} ~ ¥{y.max():,.2f}")
        
        return X, y
    
    def train_profit_prediction_model(self) -> Dict:
        """
        训练利润预测模型
        
        Returns:
            训练结果字典
        """
        print("\n" + "=" * 80)
        print("训练利润预测模型")
        print("=" * 80)
        
        # 准备数据
        X, y = self.prepare_profit_prediction_data()
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 使用随机森林回归
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.seed,
            n_jobs=-1
        )
        
        # 训练模型
        print("\n正在训练模型...")
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        print("\n模型性能:")
        print(f"  MSE: {mse:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R²: {r2:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 重要特征:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 保存模型
        self.models['profit_prediction'] = model
        self.model_metrics['profit_prediction'] = metrics
        self.feature_importance['profit_prediction'] = feature_importance
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def train_repayment_behavior_model(self) -> Dict:
        """
        训练还款行为模型（预测是否提前还款）
        
        Returns:
            训练结果字典
        """
        print("\n" + "=" * 80)
        print("训练还款行为模型")
        print("=" * 80)
        
        # 只使用已批准且未违约的贷款
        approved_data = self.data[
            (self.data['expert_decision'] == 'approve') & 
            (self.data['actual_defaulted'] == False)
        ].copy()
        
        if len(approved_data) == 0:
            print("⚠️  没有可用的还款行为数据")
            return {}
        
        # 检查是否有提前还款信息（从payment_history中提取）
        # 简化处理：假设有prepaid状态
        if 'payment_history' not in approved_data.columns:
            print("⚠️  缺少还款历史数据，跳过还款行为模型训练")
            return {}
        
        # 这里简化处理，实际应该从payment_history中提取提前还款信息
        # 暂时使用一个简单的规则：如果实际还款期数 < 申请期数，视为提前还款
        approved_data['is_prepaid'] = (
            approved_data['approved_term_months'] > 
            approved_data.get('actual_term_months', approved_data['approved_term_months'])
        ).astype(int)
        
        # 选择特征
        feature_cols = [
            'age', 'monthly_income', 'credit_score', 'debt_ratio',
            'loan_amount', 'approved_rate', 'approved_term_months',
            'gdp_growth', 'base_interest_rate', 'market_sentiment'
        ]
        
        available_features = [f for f in feature_cols if f in approved_data.columns]
        
        X = approved_data[available_features].copy()
        X = X.fillna(X.median())
        y = approved_data['is_prepaid'].copy()
        
        if y.sum() == 0:
            print("⚠️  没有提前还款样本，跳过训练")
            return {}
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )
        
        # 训练模型
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.seed,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"提前还款率: {y.mean():.2%}")
        print(f"模型准确率: {accuracy:.4f}")
        
        self.models['repayment_behavior'] = model
        self.model_metrics['repayment_behavior'] = {'accuracy': accuracy}
        
        return {
            'model': model,
            'metrics': {'accuracy': accuracy}
        }
    
    def train_all_models(self) -> Dict:
        """
        训练所有模型
        
        Returns:
            所有模型的训练结果
        """
        print("\n" + "=" * 80)
        print("训练所有世界模型")
        print("=" * 80)
        
        results = {}
        
        # 1. 违约预测模型
        try:
            results['default_prediction'] = self.train_default_prediction_model('random_forest')
        except Exception as e:
            print(f"❌ 违约预测模型训练失败: {e}")
        
        # 2. 利润预测模型
        try:
            results['profit_prediction'] = self.train_profit_prediction_model()
        except Exception as e:
            print(f"❌ 利润预测模型训练失败: {e}")
        
        # 3. 还款行为模型
        try:
            results['repayment_behavior'] = self.train_repayment_behavior_model()
        except Exception as e:
            print(f"❌ 还款行为模型训练失败: {e}")
        
        print("\n" + "=" * 80)
        print("模型训练完成")
        print("=" * 80)
        print(f"成功训练 {len(results)} 个模型")
        
        return results
    
    def predict_default_probability(self, customer_data: Dict, loan_data: Dict, 
                                    market_data: Dict) -> float:
        """
        预测违约概率
        
        Args:
            customer_data: 客户数据
            loan_data: 贷款数据
            market_data: 市场数据
        
        Returns:
            违约概率
        """
        if 'default_prediction' not in self.models:
            raise ValueError("违约预测模型未训练")
        
        model = self.models['default_prediction']
        
        # 构建特征向量（需要与训练时一致）
        # 这里简化处理，实际应该使用训练时的特征列表
        return 0.1  # 占位符
    
    def predict_profit(self, customer_data: Dict, loan_data: Dict, 
                      market_data: Dict) -> float:
        """
        预测利润
        
        Args:
            customer_data: 客户数据
            loan_data: 贷款数据
            market_data: 市场数据
        
        Returns:
            预测利润
        """
        if 'profit_prediction' not in self.models:
            raise ValueError("利润预测模型未训练")
        
        model = self.models['profit_prediction']
        
        # 构建特征向量
        # 这里简化处理
        return 0.0  # 占位符
    
    def save_models(self, output_dir: str = 'data/historical/models'):
        """保存训练好的模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ 已保存模型: {model_path}")
        
        # 保存指标
        metrics_path = os.path.join(output_dir, 'model_metrics.json')
        import json
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_metrics, f, indent=2, default=str)
        print(f"✅ 已保存模型指标: {metrics_path}")
        
        # 保存特征重要性
        if self.feature_importance:
            importance_path = os.path.join(output_dir, 'feature_importance.json')
            importance_dict = {}
            for model_name, df in self.feature_importance.items():
                importance_dict[model_name] = df.to_dict('records')
            with open(importance_path, 'w', encoding='utf-8') as f:
                json.dump(importance_dict, f, indent=2, default=str)
            print(f"✅ 已保存特征重要性: {importance_path}")


def main():
    """主函数：训练世界模型"""
    import sys
    
    # 加载数据
    data_path = 'data/historical/historical_loans_engineered.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行 historical_data_generator.py 和 feature_engineer.py")
        sys.exit(1)
    
    print("正在加载数据...")
    data = pd.read_csv(data_path)
    print(f"✅ 已加载 {len(data)} 条记录，{len(data.columns)} 个特征")
    
    # 训练模型
    trainer = WorldModelTrainer(data, seed=42)
    results = trainer.train_all_models()
    
    # 保存模型
    trainer.save_models()
    
    return trainer, results


if __name__ == '__main__':
    main()

