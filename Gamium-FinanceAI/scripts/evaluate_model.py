#!/usr/bin/env python3
"""
模型评估脚本

评估模型在测试集上的表现，并生成详细的评估报告

使用方法:
    python3 evaluate_model.py --model models/model_20241209_143022.pkl --test data/test_features.parquet
"""

import argparse
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """模型评估器"""
    
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
            timestamp = self.model_path.stem.split('_')[-1]
            scaler_file = self.model_dir / f'scaler_{timestamp}.pkl'
        
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
        
        # 加载encoders
        if encoders_path:
            encoders_file = Path(encoders_path)
        else:
            timestamp = self.model_path.stem.split('_')[-1]
            encoders_file = self.model_dir / f'encoders_{timestamp}.pkl'
        
        if encoders_file.exists():
            with open(encoders_file, 'rb') as f:
                self.encoders = pickle.load(f)
        else:
            self.encoders = {}
        
        # 加载模型信息
        info_file = self.model_dir / f"model_info_{self.model_path.stem.split('_')[-1]}.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.model_info = json.load(f)
                self.feature_cols = self.model_info.get('feature_cols', [])
        else:
            self.feature_cols = []
        
        print("✅ 模型加载完成\n")
    
    def load_test_data(self, test_path: str) -> tuple:
        """加载测试数据"""
        print(f"📊 加载测试数据: {test_path}")
        test_data = pd.read_parquet(test_path)
        
        # 预处理（与训练时一致）
        numeric_cols = test_data.select_dtypes(include=[np.number]).columns
        test_data[numeric_cols] = test_data[numeric_cols].fillna(
            test_data[numeric_cols].median()
        )
        
        # 编码分类特征
        for col in ['gender', 'education', 'industry', 'city_tier', 'customer_type']:
            if col in test_data.columns and col in self.encoders:
                try:
                    test_data[col] = self.encoders[col].transform(test_data[col].astype(str))
                except ValueError:
                    # 处理未见过的值
                    test_data[col] = 0
        
        # 准备特征和目标
        X = test_data[self.feature_cols]
        y = test_data['defaulted']
        
        print(f"   ✅ 加载完成: {len(test_data):,} 条记录")
        print(f"   特征数: {len(self.feature_cols)}")
        print(f"   违约率: {y.mean():.2%}")
        
        return X, y
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """评估模型"""
        print("\n🎯 进行预测...")
        
        # 标准化
        if self.scaler:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test.values
        
        # 预测
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("   ✅ 预测完成\n")
        
        # 计算指标
        print("📈 计算评估指标...")
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        }
        
        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC曲线数据
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        # PR曲线数据
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # 计算最佳阈值
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = pr_thresholds[best_threshold_idx]
        
        results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'best_threshold': float(best_threshold),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            },
            'test_size': len(y_test),
            'positive_samples': int(y_test.sum()),
            'negative_samples': int((y_test == 0).sum()),
            'positive_rate': float(y_test.mean())
        }
        
        return results
    
    def print_report(self, results: Dict[str, Any]):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("📊 模型评估报告")
        print("=" * 60)
        
        metrics = results['metrics']
        print(f"\n核心指标:")
        print(f"  AUC Score:     {metrics['roc_auc']:.4f}")
        print(f"  准确率:        {metrics['accuracy']:.4f}")
        print(f"  精确率:        {metrics['precision']:.4f}")
        print(f"  召回率:        {metrics['recall']:.4f}")
        print(f"  F1分数:        {metrics['f1_score']:.4f}")
        
        print(f"\n测试集信息:")
        print(f"  总样本数:      {results['test_size']:,}")
        print(f"  正样本数:      {results['positive_samples']:,}")
        print(f"  负样本数:      {results['negative_samples']:,}")
        print(f"  正样本率:      {results['positive_rate']:.2%}")
        
        print(f"\n混淆矩阵:")
        cm = np.array(results['confusion_matrix'])
        print(f"              预测")
        print(f"           正常  违约")
        print(f"实际 正常  {cm[0,0]:6d} {cm[0,1]:6d}")
        print(f"     违约  {cm[1,0]:6d} {cm[1,1]:6d}")
        
        print(f"\n分类报告:")
        report = results['classification_report']
        print(f"  类别 0 (正常):")
        print(f"    精确率: {report['0']['precision']:.4f}")
        print(f"    召回率: {report['0']['recall']:.4f}")
        print(f"    F1分数: {report['0']['f1-score']:.4f}")
        print(f"  类别 1 (违约):")
        print(f"    精确率: {report['1']['precision']:.4f}")
        print(f"    召回率: {report['1']['recall']:.4f}")
        print(f"    F1分数: {report['1']['f1-score']:.4f}")
        
        print(f"\n最佳阈值: {results['best_threshold']:.4f}")
        print("=" * 60)
    
    def save_report(self, results: Dict[str, Any], output_path: str):
        """保存评估报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 评估报告已保存: {output_file}")
    
    def plot_curves(self, results: Dict[str, Any], output_dir: str):
        """绘制评估曲线"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC曲线
        plt.figure(figsize=(10, 6))
        fpr = results['roc_curve']['fpr']
        tpr = results['roc_curve']['tpr']
        auc = results['metrics']['roc_auc']
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PR曲线
        plt.figure(figsize=(10, 6))
        precision = results['pr_curve']['precision']
        recall = results['pr_curve']['recall']
        
        plt.plot(recall, precision, label='PR Curve', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 评估曲线已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='模型评估脚本')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径 (.pkl)')
    parser.add_argument('--test', type=str, required=True,
                       help='测试数据文件路径 (.parquet)')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                       help='评估报告输出路径')
    parser.add_argument('--plot', action='store_true',
                       help='生成评估曲线图')
    parser.add_argument('--plot-dir', type=str, default='evaluation_plots',
                       help='曲线图输出目录')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model)
    
    # 加载测试数据
    X_test, y_test = evaluator.load_test_data(args.test)
    
    # 评估
    results = evaluator.evaluate(X_test, y_test)
    
    # 打印报告
    evaluator.print_report(results)
    
    # 保存报告
    evaluator.save_report(results, args.output)
    
    # 绘制曲线
    if args.plot:
        evaluator.plot_curves(results, args.plot_dir)
    
    return results

if __name__ == '__main__':
    main()

