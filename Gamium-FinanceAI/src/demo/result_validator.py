"""
结果验证模块
实现违约率、利润分布、回收率对比分析
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import os
from scipy import stats


class ResultValidator:
    """结果验证器"""
    
    def __init__(self, historical_data: pd.DataFrame, simulated_data: pd.DataFrame):
        """
        初始化验证器
        
        Args:
            historical_data: 历史数据
            simulated_data: 模拟数据
        """
        self.historical = historical_data
        self.simulated = simulated_data
    
    def validate_default_rate(self) -> Dict:
        """验证违约率"""
        print("=" * 80)
        print("违约率验证")
        print("=" * 80)
        
        # 历史违约率
        if 'expert_decision' in self.historical.columns:
            hist_approved = self.historical[self.historical['expert_decision'] == 'approve']
        else:
            # 如果没有expert_decision字段，假设所有记录都是已批准的
            hist_approved = self.historical
        hist_default_rate = hist_approved['actual_defaulted'].mean() if len(hist_approved) > 0 else 0
        
        # 模拟违约率
        if 'expert_decision' in self.simulated.columns:
            sim_approved = self.simulated[self.simulated['expert_decision'] == 'approve']
        elif 'decision' in self.simulated.columns:
            # 使用decision字段作为fallback
            sim_approved = self.simulated[self.simulated['decision'] == 'approve']
        else:
            # 如果没有决策字段，假设所有记录都是已批准的
            sim_approved = self.simulated
        sim_default_rate = sim_approved['actual_defaulted'].mean() if len(sim_approved) > 0 else 0
        
        # 差异
        diff = abs(hist_default_rate - sim_default_rate)
        relative_diff = diff / hist_default_rate if hist_default_rate > 0 else 0
        
        # 统计检验
        if len(hist_approved) > 0 and len(sim_approved) > 0:
            # 卡方检验
            hist_default_count = hist_approved['actual_defaulted'].sum()
            hist_total = len(hist_approved)
            sim_default_count = sim_approved['actual_defaulted'].sum()
            sim_total = len(sim_approved)
            
            contingency = np.array([
                [hist_default_count, hist_total - hist_default_count],
                [sim_default_count, sim_total - sim_default_count]
            ])
            
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
        else:
            chi2, p_value = 0, 1
        
        result = {
            'historical_default_rate': hist_default_rate,
            'simulated_default_rate': sim_default_rate,
            'absolute_difference': diff,
            'relative_difference': relative_diff,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'is_acceptable': relative_diff < 0.2 and p_value > 0.05
        }
        
        print(f"历史违约率: {hist_default_rate:.2%}")
        print(f"模拟违约率: {sim_default_rate:.2%}")
        print(f"绝对差异: {diff:.2%}")
        print(f"相对差异: {relative_diff:.2%}")
        print(f"卡方统计量: {chi2:.4f}")
        print(f"P值: {p_value:.4f}")
        print(f"验证结果: {'✅ 通过' if result['is_acceptable'] else '❌ 未通过'}")
        
        return result
    
    def validate_profit_distribution(self) -> Dict:
        """验证利润分布"""
        print("\n" + "=" * 80)
        print("利润分布验证")
        print("=" * 80)
        
        # 历史利润
        if 'expert_decision' in self.historical.columns:
            hist_approved = self.historical[self.historical['expert_decision'] == 'approve']
        else:
            hist_approved = self.historical
        hist_profit = hist_approved['actual_profit']
        
        # 模拟利润
        if 'expert_decision' in self.simulated.columns:
            sim_approved = self.simulated[self.simulated['expert_decision'] == 'approve']
        elif 'decision' in self.simulated.columns:
            sim_approved = self.simulated[self.simulated['decision'] == 'approve']
        else:
            sim_approved = self.simulated
        sim_profit = sim_approved['actual_profit']
        
        if len(hist_profit) == 0 or len(sim_profit) == 0:
            return {'error': '数据不足'}
        
        # 统计量对比
        hist_mean = hist_profit.mean()
        hist_std = hist_profit.std()
        sim_mean = sim_profit.mean()
        sim_std = sim_profit.std()
        
        # KS检验（分布相似性）
        ks_statistic, ks_p_value = stats.ks_2samp(hist_profit, sim_profit)
        
        # 均值差异检验（t检验）
        t_statistic, t_p_value = stats.ttest_ind(hist_profit, sim_profit)
        
        result = {
            'historical_mean': hist_mean,
            'historical_std': hist_std,
            'simulated_mean': sim_mean,
            'simulated_std': sim_std,
            'mean_difference': abs(hist_mean - sim_mean),
            'relative_mean_difference': abs(hist_mean - sim_mean) / abs(hist_mean) if hist_mean != 0 else 0,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            't_statistic': t_statistic,
            't_p_value': t_p_value,
            'is_acceptable': ks_p_value > 0.05 and abs(hist_mean - sim_mean) / abs(hist_mean) < 0.3 if hist_mean != 0 else False
        }
        
        print(f"历史平均利润: ¥{hist_mean:,.2f} (标准差: ¥{hist_std:,.2f})")
        print(f"模拟平均利润: ¥{sim_mean:,.2f} (标准差: ¥{sim_std:,.2f})")
        print(f"均值差异: ¥{result['mean_difference']:,.2f} ({result['relative_mean_difference']:.2%})")
        print(f"KS统计量: {ks_statistic:.4f}, P值: {ks_p_value:.4f}")
        print(f"T统计量: {t_statistic:.4f}, P值: {t_p_value:.4f}")
        print(f"验证结果: {'✅ 通过' if result['is_acceptable'] else '❌ 未通过'}")
        
        return result
    
    def validate_recovery_rate(self) -> Dict:
        """验证回收率"""
        print("\n" + "=" * 80)
        print("回收率验证")
        print("=" * 80)
        
        # 历史回收率
        if 'expert_decision' in self.historical.columns:
            hist_defaulted = self.historical[
                (self.historical['expert_decision'] == 'approve') &
                (self.historical['actual_defaulted'] == True)
            ]
        else:
            hist_defaulted = self.historical[self.historical['actual_defaulted'] == True]
        
        if len(hist_defaulted) == 0:
            print("⚠️  历史数据中没有违约记录")
            return {'error': '数据不足'}
        
        hist_recovery_rate = hist_defaulted['recovery_rate'].mean() if 'recovery_rate' in hist_defaulted.columns else 0
        hist_recovery_amount = hist_defaulted['recovery_amount'].sum() if 'recovery_amount' in hist_defaulted.columns else 0
        hist_default_amount = hist_defaulted['default_amount'].sum() if 'default_amount' in hist_defaulted.columns else 0
        
        # 模拟回收率
        if 'expert_decision' in self.simulated.columns:
            sim_defaulted = self.simulated[
                (self.simulated['expert_decision'] == 'approve') &
                (self.simulated['actual_defaulted'] == True)
            ]
        elif 'decision' in self.simulated.columns:
            sim_defaulted = self.simulated[
                (self.simulated['decision'] == 'approve') &
                (self.simulated['actual_defaulted'] == True)
            ]
        else:
            sim_defaulted = self.simulated[self.simulated['actual_defaulted'] == True]
        
        if len(sim_defaulted) == 0:
            print("⚠️  模拟数据中没有违约记录")
            return {'error': '数据不足'}
        
        sim_recovery_rate = sim_defaulted['recovery_rate'].mean()
        sim_recovery_amount = sim_defaulted['recovery_amount'].sum()
        sim_default_amount = sim_defaulted['default_amount'].sum()
        
        # 差异
        rate_diff = abs(hist_recovery_rate - sim_recovery_rate)
        relative_diff = rate_diff / hist_recovery_rate if hist_recovery_rate > 0 else 0
        
        result = {
            'historical_recovery_rate': hist_recovery_rate,
            'simulated_recovery_rate': sim_recovery_rate,
            'rate_difference': rate_diff,
            'relative_difference': relative_diff,
            'historical_recovery_amount': hist_recovery_amount,
            'simulated_recovery_amount': sim_recovery_amount,
            'historical_default_amount': hist_default_amount,
            'simulated_default_amount': sim_default_amount,
            'is_acceptable': relative_diff < 0.3
        }
        
        print(f"历史平均回收率: {hist_recovery_rate:.2%}")
        print(f"模拟平均回收率: {sim_recovery_rate:.2%}")
        print(f"回收率差异: {rate_diff:.2%} ({relative_diff:.2%})")
        print(f"历史回收金额: ¥{hist_recovery_amount:,.2f} / 违约金额: ¥{hist_default_amount:,.2f}")
        print(f"模拟回收金额: ¥{sim_recovery_amount:,.2f} / 违约金额: ¥{sim_default_amount:,.2f}")
        print(f"验证结果: {'✅ 通过' if result['is_acceptable'] else '❌ 未通过'}")
        
        return result
    
    def comprehensive_validation(self) -> Dict:
        """综合验证"""
        print("\n" + "=" * 80)
        print("综合验证")
        print("=" * 80)
        
        results = {
            'default_rate': self.validate_default_rate(),
            'profit_distribution': self.validate_profit_distribution(),
            'recovery_rate': self.validate_recovery_rate()
        }
        
        # 综合评估
        all_acceptable = all(
            r.get('is_acceptable', False) if isinstance(r, dict) and 'is_acceptable' in r else False
            for r in results.values()
        )
        
        print("\n" + "=" * 80)
        print("验证总结")
        print("=" * 80)
        print(f"违约率验证: {'✅' if results['default_rate'].get('is_acceptable') else '❌'}")
        print(f"利润分布验证: {'✅' if results['profit_distribution'].get('is_acceptable') else '❌'}")
        print(f"回收率验证: {'✅' if results['recovery_rate'].get('is_acceptable') else '❌'}")
        print(f"综合结果: {'✅ 全部通过' if all_acceptable else '❌ 部分未通过'}")
        
        results['overall_acceptable'] = all_acceptable
        
        return results
    
    def save_validation_report(self, results: Dict, output_path: str = 'data/historical/validation_report.json'):
        """保存验证报告"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 验证报告已保存到: {output_path}")


def main():
    """主函数：测试结果验证"""
    import sys
    
    # 加载数据
    hist_path = 'data/historical/historical_loans_engineered.csv'
    if not os.path.exists(hist_path):
        print(f"❌ 历史数据文件不存在: {hist_path}")
        sys.exit(1)
    
    print("正在加载历史数据...")
    historical_data = pd.read_csv(hist_path)
    print(f"✅ 已加载 {len(historical_data)} 条历史记录")
    
    # 使用历史数据作为模拟数据（实际应该使用模拟结果）
    print("⚠️  使用历史数据作为模拟数据（实际应使用模拟结果）")
    simulated_data = historical_data.copy()
    
    # 执行验证
    validator = ResultValidator(historical_data, simulated_data)
    results = validator.comprehensive_validation()
    
    # 保存报告
    validator.save_validation_report(results)
    
    return validator, results


if __name__ == '__main__':
    main()

