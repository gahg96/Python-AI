"""
数据质量检查模块
用于检查历史贷款数据的完整性、一致性和时间一致性
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化数据质量检查器
        
        Args:
            data: 历史贷款数据DataFrame
        """
        self.data = data.copy()
        self.issues = []
        self.warnings = []
        self.statistics = {}
    
    def check_completeness(self) -> Dict:
        """
        检查数据完整性
        
        Returns:
            完整性检查结果
        """
        print("=" * 80)
        print("1. 数据完整性检查")
        print("=" * 80)
        
        # 检查缺失值
        missing_rates = self.data.isnull().mean()
        missing_counts = self.data.isnull().sum()
        
        # 关键字段列表
        critical_fields = {
            'personal': [
                'customer_id', 'customer_type', 'age', 'monthly_income',
                'credit_score', 'debt_ratio', 'application_date',
                'expert_decision', 'actual_defaulted', 'actual_profit'
            ],
            'corporate': [
                'customer_id', 'customer_type', 'registered_capital',
                'operating_years', 'annual_revenue', 'debt_to_asset_ratio',
                'application_date', 'expert_decision', 'actual_defaulted', 'actual_profit'
            ]
        }
        
        # 检查关键字段（只检查对应客户类型的字段）
        critical_issues = []
        for customer_type, fields in critical_fields.items():
            type_data = self.data[self.data['customer_type'] == customer_type]
            if len(type_data) == 0:
                continue
            for field in fields:
                if field in type_data.columns:
                    missing_rate = type_data[field].isnull().mean()
                    if missing_rate > 0.05:  # 只报告缺失率>5%的
                        critical_issues.append({
                            'field': field,
                            'customer_type': customer_type,
                            'missing_rate': missing_rate,
                            'missing_count': type_data[field].isnull().sum()
                        })
        
        # 计算完整性得分
        total_cells = len(self.data) * len(self.data.columns)
        missing_cells = self.data.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        result = {
            'completeness_score': completeness_score,
            'total_missing_cells': int(missing_cells),
            'total_cells': total_cells,
            'missing_rates': missing_rates[missing_rates > 0].to_dict(),
            'missing_counts': missing_counts[missing_counts > 0].to_dict(),
            'critical_issues': critical_issues,
            'is_acceptable': completeness_score > 0.95 and len(critical_issues) == 0
        }
        
        # 打印结果
        print(f"完整性得分: {completeness_score:.4f}")
        print(f"缺失单元格数: {missing_cells} / {total_cells}")
        
        if len(critical_issues) > 0:
            print(f"\n⚠️  发现 {len(critical_issues)} 个关键字段缺失问题:")
            for issue in critical_issues:
                print(f"  - {issue['field']} ({issue['customer_type']}): "
                      f"缺失率 {issue['missing_rate']:.2%} ({issue['missing_count']} 条)")
        else:
            print("✅ 关键字段完整性检查通过")
        
        if missing_rates.sum() > 0:
            print(f"\n缺失值统计 (缺失率 > 0):")
            for field, rate in missing_rates[missing_rates > 0].items():
                print(f"  - {field}: {rate:.2%} ({missing_counts[field]} 条)")
        
        self.statistics['completeness'] = result
        return result
    
    def check_consistency(self) -> Dict:
        """
        检查数据一致性
        
        Returns:
            一致性检查结果
        """
        print("\n" + "=" * 80)
        print("2. 数据一致性检查")
        print("=" * 80)
        
        issues = []
        
        # 1. 检查逻辑一致性
        # 如果defaulted=True，应该有default_date
        if 'actual_defaulted' in self.data.columns and 'default_date' in self.data.columns:
            inconsistent = self.data[
                (self.data['actual_defaulted'] == True) & 
                (self.data['default_date'].isnull())
            ]
            if len(inconsistent) > 0:
                issues.append({
                    'type': 'logical',
                    'description': '违约标记为True但缺少违约日期',
                    'count': len(inconsistent),
                    'severity': 'high'
                })
        
        # 如果defaulted=False，不应该有default_date
        if 'actual_defaulted' in self.data.columns and 'default_date' in self.data.columns:
            inconsistent = self.data[
                (self.data['actual_defaulted'] == False) & 
                (self.data['default_date'].notna())
            ]
            if len(inconsistent) > 0:
                issues.append({
                    'type': 'logical',
                    'description': '违约标记为False但有违约日期',
                    'count': len(inconsistent),
                    'severity': 'medium'
                })
        
        # 2. 检查数值范围
        # 信用分范围
        if 'credit_score' in self.data.columns:
            invalid_scores = self.data[
                (self.data['credit_score'] < 300) | 
                (self.data['credit_score'] > 850)
            ]
            if len(invalid_scores) > 0:
                issues.append({
                    'type': 'range',
                    'field': 'credit_score',
                    'description': '信用分超出有效范围 (300-850)',
                    'count': len(invalid_scores),
                    'severity': 'high'
                })
        
        # 年龄范围
        if 'age' in self.data.columns:
            invalid_ages = self.data[
                (self.data['age'] < 18) | 
                (self.data['age'] > 100)
            ]
            if len(invalid_ages) > 0:
                issues.append({
                    'type': 'range',
                    'field': 'age',
                    'description': '年龄超出合理范围 (18-100)',
                    'count': len(invalid_ages),
                    'severity': 'high'
                })
        
        # 负债率范围 (0-1)
        if 'debt_ratio' in self.data.columns:
            invalid_ratios = self.data[
                (self.data['debt_ratio'] < 0) | 
                (self.data['debt_ratio'] > 1)
            ]
            if len(invalid_ratios) > 0:
                issues.append({
                    'type': 'range',
                    'field': 'debt_ratio',
                    'description': '负债率超出有效范围 (0-1)',
                    'count': len(invalid_ratios),
                    'severity': 'high'
                })
        
        # 3. 检查审批决策一致性
        if 'expert_decision' in self.data.columns:
            # 如果decision=reject，approved_amount应该为0
            inconsistent = self.data[
                (self.data['expert_decision'] == 'reject') & 
                (self.data['approved_amount'] > 0)
            ]
            if len(inconsistent) > 0:
                issues.append({
                    'type': 'logical',
                    'description': '拒绝决策但批准金额>0',
                    'count': len(inconsistent),
                    'severity': 'high'
                })
            
            # 如果decision=approve，approved_amount应该>0
            inconsistent = self.data[
                (self.data['expert_decision'] == 'approve') & 
                (self.data['approved_amount'] <= 0)
            ]
            if len(inconsistent) > 0:
                issues.append({
                    'type': 'logical',
                    'description': '批准决策但批准金额<=0',
                    'count': len(inconsistent),
                    'severity': 'high'
                })
        
        # 4. 检查利润计算一致性
        if all(col in self.data.columns for col in ['total_interest_paid', 'total_principal_paid', 
                                                     'default_amount', 'recovery_amount', 'actual_profit']):
            approved_loans = self.data[self.data['expert_decision'] == 'approve']
            if len(approved_loans) > 0:
                # 计算预期利润
                calculated_profit = (
                    approved_loans['total_interest_paid'] +
                    approved_loans['recovery_amount'] -
                    approved_loans['default_amount']
                )
                
                # 允许小误差（浮点数精度）
                profit_diff = abs(calculated_profit - approved_loans['actual_profit'])
                inconsistent = approved_loans[profit_diff > 1.0]  # 误差>1元
                
                if len(inconsistent) > 0:
                    issues.append({
                        'type': 'calculation',
                        'description': '利润计算不一致',
                        'count': len(inconsistent),
                        'severity': 'medium',
                        'max_diff': profit_diff.max()
                    })
        
        # 计算一致性得分
        total_issues = sum(issue['count'] for issue in issues)
        consistency_score = 1 - (total_issues / len(self.data)) if len(self.data) > 0 else 0
        consistency_score = max(0, min(1, consistency_score))
        
        result = {
            'consistency_score': consistency_score,
            'total_issues': total_issues,
            'issues': issues,
            'is_acceptable': consistency_score > 0.99 and len([i for i in issues if i['severity'] == 'high']) == 0
        }
        
        # 打印结果
        print(f"一致性得分: {consistency_score:.4f}")
        print(f"发现问题数: {total_issues}")
        
        if len(issues) > 0:
            print(f"\n⚠️  发现 {len(issues)} 类一致性问题:")
            for issue in issues:
                print(f"  [{issue['severity'].upper()}] {issue['description']}: {issue['count']} 条")
        else:
            print("✅ 数据一致性检查通过")
        
        self.statistics['consistency'] = result
        return result
    
    def check_temporal_consistency(self) -> Dict:
        """
        检查时间一致性
        
        Returns:
            时间一致性检查结果
        """
        print("\n" + "=" * 80)
        print("3. 时间一致性检查")
        print("=" * 80)
        
        issues = []
        
        # 1. 检查时间顺序
        if 'application_date' in self.data.columns and 'approval_date' in self.data.columns:
            # 转换为日期类型
            app_dates = pd.to_datetime(self.data['application_date'], errors='coerce')
            approval_dates = pd.to_datetime(self.data['approval_date'], errors='coerce')
            
            # 审批日期应该在申请日期之后
            invalid_order = self.data[approval_dates < app_dates]
            if len(invalid_order) > 0:
                issues.append({
                    'type': 'temporal_order',
                    'description': '审批日期早于申请日期',
                    'count': len(invalid_order),
                    'severity': 'high'
                })
            
            # 审批日期不应该太晚（比如超过30天）
            days_diff = (approval_dates - app_dates).dt.days
            too_late = self.data[days_diff > 30]
            if len(too_late) > 0:
                issues.append({
                    'type': 'temporal_range',
                    'description': '审批日期距离申请日期超过30天',
                    'count': len(too_late),
                    'severity': 'medium'
                })
        
        # 2. 检查违约日期
        if 'approval_date' in self.data.columns and 'default_date' in self.data.columns:
            approval_dates = pd.to_datetime(self.data['approval_date'], errors='coerce')
            default_dates = pd.to_datetime(self.data['default_date'], errors='coerce')
            
            # 违约日期应该在审批日期之后
            defaulted_loans = self.data[self.data['actual_defaulted'] == True]
            if len(defaulted_loans) > 0:
                invalid_default_dates = defaulted_loans[
                    default_dates[defaulted_loans.index] < approval_dates[defaulted_loans.index]
                ]
                if len(invalid_default_dates) > 0:
                    issues.append({
                        'type': 'temporal_order',
                        'description': '违约日期早于审批日期',
                        'count': len(invalid_default_dates),
                        'severity': 'high'
                    })
        
        # 3. 检查时间范围合理性
        if 'application_date' in self.data.columns:
            app_dates = pd.to_datetime(self.data['application_date'], errors='coerce')
            date_range = app_dates.max() - app_dates.min()
            print(f"数据时间跨度: {date_range.days} 天")
            
            # 检查是否有未来日期
            today = pd.Timestamp.now()
            future_dates = self.data[app_dates > today]
            if len(future_dates) > 0:
                issues.append({
                    'type': 'temporal_range',
                    'description': '申请日期在未来',
                    'count': len(future_dates),
                    'severity': 'high'
                })
        
        # 计算时间一致性得分
        total_issues = sum(issue['count'] for issue in issues)
        temporal_score = 1 - (total_issues / len(self.data)) if len(self.data) > 0 else 0
        temporal_score = max(0, min(1, temporal_score))
        
        result = {
            'temporal_consistency_score': temporal_score,
            'total_issues': total_issues,
            'issues': issues,
            'is_acceptable': temporal_score > 0.99 and len([i for i in issues if i['severity'] == 'high']) == 0
        }
        
        # 打印结果
        print(f"时间一致性得分: {temporal_score:.4f}")
        print(f"发现问题数: {total_issues}")
        
        if len(issues) > 0:
            print(f"\n⚠️  发现 {len(issues)} 类时间一致性问题:")
            for issue in issues:
                print(f"  [{issue['severity'].upper()}] {issue['description']}: {issue['count']} 条")
        else:
            print("✅ 时间一致性检查通过")
        
        self.statistics['temporal_consistency'] = result
        return result
    
    def check_business_rules(self) -> Dict:
        """
        检查业务规则一致性
        
        Returns:
            业务规则检查结果
        """
        print("\n" + "=" * 80)
        print("4. 业务规则检查")
        print("=" * 80)
        
        violations = []
        
        # 1. 对私贷款规则
        personal_loans = self.data[self.data['customer_type'] == 'personal']
        if len(personal_loans) > 0:
            # 年龄限制
            invalid_age = personal_loans[
                (personal_loans['age'] < 18) | 
                (personal_loans['age'] > 65)
            ]
            if len(invalid_age) > 0:
                violations.append({
                    'rule': '对私贷款年龄限制 (18-65)',
                    'count': len(invalid_age),
                    'severity': 'high'
                })
            
            # 贷款金额/收入比
            if 'loan_amount' in personal_loans.columns and 'monthly_income' in personal_loans.columns:
                loan_to_income = personal_loans['loan_amount'] / (personal_loans['monthly_income'] * 12 + 1e-6)
                excessive_ratio = personal_loans[loan_to_income > 0.5]
                if len(excessive_ratio) > 0:
                    violations.append({
                        'rule': '对私贷款金额/年收入比 <= 0.5',
                        'count': len(excessive_ratio),
                        'severity': 'medium'
                    })
        
        # 2. 对公贷款规则
        corporate_loans = self.data[self.data['customer_type'] == 'corporate']
        if len(corporate_loans) > 0:
            # 经营年限
            if 'operating_years' in corporate_loans.columns:
                invalid_years = corporate_loans[corporate_loans['operating_years'] < 1]
                if len(invalid_years) > 0:
                    violations.append({
                        'rule': '对公贷款经营年限 >= 1年',
                        'count': len(invalid_years),
                        'severity': 'high'
                    })
            
            # 贷款金额/年营收比
            if 'loan_amount' in corporate_loans.columns and 'annual_revenue' in corporate_loans.columns:
                loan_to_revenue = corporate_loans['loan_amount'] / (corporate_loans['annual_revenue'] + 1e-6)
                excessive_ratio = corporate_loans[loan_to_revenue > 0.3]
                if len(excessive_ratio) > 0:
                    violations.append({
                        'rule': '对公贷款金额/年营收比 <= 0.3',
                        'count': len(excessive_ratio),
                        'severity': 'medium'
                    })
        
        # 3. 利率合理性
        if 'approved_rate' in self.data.columns:
            approved_loans = self.data[self.data['expert_decision'] == 'approve']
            if len(approved_loans) > 0:
                # 对私利率应该在合理范围
                personal_approved = approved_loans[approved_loans['customer_type'] == 'personal']
                if len(personal_approved) > 0:
                    invalid_rate = personal_approved[
                        (personal_approved['approved_rate'] < 0.05) | 
                        (personal_approved['approved_rate'] > 0.2)
                    ]
                    if len(invalid_rate) > 0:
                        violations.append({
                            'rule': '对私贷款利率范围 (5%-20%)',
                            'count': len(invalid_rate),
                            'severity': 'medium'
                        })
                
                # 对公利率应该在合理范围
                corporate_approved = approved_loans[approved_loans['customer_type'] == 'corporate']
                if len(corporate_approved) > 0:
                    invalid_rate = corporate_approved[
                        (corporate_approved['approved_rate'] < 0.03) | 
                        (corporate_approved['approved_rate'] > 0.15)
                    ]
                    if len(invalid_rate) > 0:
                        violations.append({
                            'rule': '对公贷款利率范围 (3%-15%)',
                            'count': len(invalid_rate),
                            'severity': 'medium'
                        })
        
        # 计算业务规则得分
        total_violations = sum(v['count'] for v in violations)
        rule_score = 1 - (total_violations / len(self.data)) if len(self.data) > 0 else 0
        rule_score = max(0, min(1, rule_score))
        
        result = {
            'rule_score': rule_score,
            'total_violations': total_violations,
            'violations': violations,
            'is_acceptable': rule_score > 0.95 and len([v for v in violations if v['severity'] == 'high']) == 0
        }
        
        # 打印结果
        print(f"业务规则得分: {rule_score:.4f}")
        print(f"违规数量: {total_violations}")
        
        if len(violations) > 0:
            print(f"\n⚠️  发现 {len(violations)} 类业务规则违规:")
            for violation in violations:
                print(f"  [{violation['severity'].upper()}] {violation['rule']}: {violation['count']} 条")
        else:
            print("✅ 业务规则检查通过")
        
        self.statistics['business_rules'] = result
        return result
    
    def check_distribution(self) -> Dict:
        """
        检查数据分布合理性
        
        Returns:
            分布检查结果
        """
        print("\n" + "=" * 80)
        print("5. 数据分布检查")
        print("=" * 80)
        
        warnings = []
        
        # 1. 检查对私客户特征分布
        personal_data = self.data[self.data['customer_type'] == 'personal']
        if len(personal_data) > 0:
            # 信用分分布
            if 'credit_score' in personal_data.columns:
                mean_score = personal_data['credit_score'].mean()
                std_score = personal_data['credit_score'].std()
                if mean_score < 500 or mean_score > 750:
                    warnings.append({
                        'field': 'credit_score',
                        'customer_type': 'personal',
                        'description': f'信用分均值异常: {mean_score:.1f} (正常范围: 500-750)',
                        'severity': 'medium'
                    })
            
            # 收入分布
            if 'monthly_income' in personal_data.columns:
                mean_income = personal_data['monthly_income'].mean()
                if mean_income < 5000 or mean_income > 20000:
                    warnings.append({
                        'field': 'monthly_income',
                        'customer_type': 'personal',
                        'description': f'月收入均值异常: {mean_income:.2f} (正常范围: 5000-20000)',
                        'severity': 'low'
                    })
        
        # 2. 检查对公客户特征分布
        corporate_data = self.data[self.data['customer_type'] == 'corporate']
        if len(corporate_data) > 0:
            # 年营收分布
            if 'annual_revenue' in corporate_data.columns:
                mean_revenue = corporate_data['annual_revenue'].mean()
                if mean_revenue < 1000000 or mean_revenue > 100000000:
                    warnings.append({
                        'field': 'annual_revenue',
                        'customer_type': 'corporate',
                        'description': f'年营收均值异常: {mean_revenue:.2f}',
                        'severity': 'low'
                    })
        
        # 3. 检查审批率
        if 'expert_decision' in self.data.columns:
            approval_rate = (self.data['expert_decision'] == 'approve').mean()
            if approval_rate < 0.5 or approval_rate > 0.95:
                warnings.append({
                    'field': 'approval_rate',
                    'description': f'审批率异常: {approval_rate:.2%} (正常范围: 50%-95%)',
                    'severity': 'medium'
                })
        
        # 4. 检查违约率
        approved_loans = self.data[self.data['expert_decision'] == 'approve']
        if len(approved_loans) > 0 and 'actual_defaulted' in approved_loans.columns:
            default_rate = approved_loans['actual_defaulted'].mean()
            if default_rate < 0.01 or default_rate > 0.2:
                warnings.append({
                    'field': 'default_rate',
                    'description': f'违约率异常: {default_rate:.2%} (正常范围: 1%-20%)',
                    'severity': 'medium'
                })
        
        result = {
            'warnings': warnings,
            'warning_count': len(warnings)
        }
        
        # 打印结果
        if len(warnings) > 0:
            print(f"⚠️  发现 {len(warnings)} 个分布警告:")
            for warning in warnings:
                print(f"  [{warning['severity'].upper()}] {warning['description']}")
        else:
            print("✅ 数据分布检查通过")
        
        self.statistics['distribution'] = result
        return result
    
    def comprehensive_check(self) -> Dict:
        """
        执行综合数据质量检查
        
        Returns:
            完整的检查结果
        """
        print("\n" + "=" * 80)
        print("数据质量综合检查")
        print("=" * 80)
        print()
        
        # 执行所有检查
        completeness = self.check_completeness()
        consistency = self.check_consistency()
        temporal = self.check_temporal_consistency()
        business_rules = self.check_business_rules()
        distribution = self.check_distribution()
        
        # 计算综合得分
        scores = [
            completeness['completeness_score'],
            consistency['consistency_score'],
            temporal['temporal_consistency_score'],
            business_rules['rule_score']
        ]
        overall_score = np.mean(scores)
        
        # 综合评估
        is_acceptable = all([
            completeness['is_acceptable'],
            consistency['is_acceptable'],
            temporal['is_acceptable'],
            business_rules['is_acceptable']
        ])
        
        result = {
            'overall_score': overall_score,
            'is_acceptable': is_acceptable,
            'completeness': completeness,
            'consistency': consistency,
            'temporal_consistency': temporal,
            'business_rules': business_rules,
            'distribution': distribution,
            'summary': {
                'total_records': len(self.data),
                'total_issues': (
                    completeness.get('total_missing_cells', 0) +
                    consistency.get('total_issues', 0) +
                    temporal.get('total_issues', 0) +
                    business_rules.get('total_violations', 0)
                ),
                'critical_issues': len([
                    i for i in consistency.get('issues', []) if i.get('severity') == 'high'
                ]) + len([
                    i for i in temporal.get('issues', []) if i.get('severity') == 'high'
                ]) + len([
                    v for v in business_rules.get('violations', []) if v.get('severity') == 'high'
                ])
            }
        }
        
        # 打印总结
        print("\n" + "=" * 80)
        print("数据质量检查总结")
        print("=" * 80)
        print(f"综合得分: {overall_score:.4f}")
        print(f"总记录数: {len(self.data)}")
        print(f"总问题数: {result['summary']['total_issues']}")
        print(f"严重问题数: {result['summary']['critical_issues']}")
        print(f"数据质量: {'✅ 合格' if is_acceptable else '❌ 不合格'}")
        print("=" * 80)
        
        self.statistics['overall'] = result
        return result
    
    def save_report(self, output_path: str = 'data/historical/quality_report.json'):
        """保存质量检查报告"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ 质量检查报告已保存到: {output_path}")


def main():
    """主函数：执行数据质量检查"""
    import sys
    
    # 加载数据
    data_path = 'data/historical/historical_loans.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行 historical_data_generator.py 生成数据")
        sys.exit(1)
    
    print("正在加载数据...")
    data = pd.read_csv(data_path)
    print(f"✅ 已加载 {len(data)} 条记录")
    
    # 执行检查
    checker = DataQualityChecker(data)
    result = checker.comprehensive_check()
    
    # 保存报告
    checker.save_report()
    
    return result


if __name__ == '__main__':
    main()

