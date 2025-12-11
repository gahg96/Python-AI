"""
业务规则提取模块
从历史数据中自动提取业务规则（阈值规则、范围规则、比例规则、复合规则）
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ExtractedRule:
    """提取的规则"""
    rule_id: str
    rule_name: str
    rule_type: str  # 'threshold', 'range', 'ratio', 'composite'
    customer_type: str  # 'personal', 'corporate', 'both'
    field: str  # 字段名
    operator: str  # '>', '<', '>=', '<=', '==', 'in', 'between'
    value: any  # 阈值或值
    conditions: List[Dict] = None  # 复合规则的条件列表
    confidence: float = 0.0  # 置信度
    support: float = 0.0  # 支持度（满足该规则的数据比例）
    description: str = ""
    priority: int = 0  # 优先级
    
    def to_dict(self):
        """转换为字典"""
        return asdict(self)


class RuleExtractor:
    """规则提取器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化规则提取器
        
        Args:
            data: 历史贷款数据（已进行特征工程）
        """
        self.data = data.copy()
        self.rules = []
        self.rule_counter = 0
    
    def extract_threshold_rules(self, field: str, target: str = 'is_defaulted',
                               customer_type: str = 'personal', 
                               min_support: float = 0.05) -> List[ExtractedRule]:
        """
        提取阈值规则
        
        例如：信用分 >= 600 时违约率显著降低
        
        Args:
            field: 字段名
            target: 目标变量（'is_defaulted', 'is_approved', 'actual_profit'）
            customer_type: 客户类型
            min_support: 最小支持度
        """
        rules = []
        
        # 筛选数据
        if customer_type != 'both':
            data = self.data[self.data['customer_type'] == customer_type]
        else:
            data = self.data
        
        if len(data) == 0 or field not in data.columns:
            return rules
        
        # 只考虑批准的贷款（对于违约率）
        if target == 'is_defaulted':
            data = data[data['expert_decision'] == 'approve']
            if len(data) == 0:
                return rules
        
        # 计算不同阈值下的目标变量均值
        field_values = data[field].dropna()
        if len(field_values) == 0:
            return rules
        
        # 使用分位数作为候选阈值
        percentiles = list(range(10, 100, 10))
        thresholds = [np.percentile(field_values, p) for p in percentiles]
        
        for threshold in thresholds:
            # 大于等于阈值
            above_mask = data[field] >= threshold
            below_mask = data[field] < threshold
            
            above_count = above_mask.sum()
            below_count = below_mask.sum()
            
            if above_count < len(data) * min_support or below_count < len(data) * min_support:
                continue
            
            if target in ['is_defaulted', 'is_approved']:
                # 二分类目标
                above_rate = data[above_mask][target].mean() if above_count > 0 else 0
                below_rate = data[below_mask][target].mean() if below_count > 0 else 0
                
                # 如果差异显著（>5%）
                if abs(above_rate - below_rate) > 0.05:
                    if target == 'is_defaulted':
                        # 违约率：越低越好
                        if above_rate < below_rate:
                            # 大于等于阈值时违约率更低
                            rule = ExtractedRule(
                                rule_id=f"R{self.rule_counter:04d}",
                                rule_name=f"{field}_gte_{threshold:.2f}",
                                rule_type='threshold',
                                customer_type=customer_type,
                                field=field,
                                operator='>=',
                                value=round(threshold, 2),
                                confidence=1 - above_rate,  # 置信度 = 1 - 违约率
                                support=above_count / len(data),
                                description=f"{field} >= {threshold:.2f} 时违约率 {above_rate:.2%} (低于 {below_rate:.2%})",
                                priority=1 if above_rate < 0.1 else 2
                            )
                            rules.append(rule)
                            self.rule_counter += 1
                    else:
                        # 审批率：越高越好
                        if above_rate > below_rate:
                            rule = ExtractedRule(
                                rule_id=f"R{self.rule_counter:04d}",
                                rule_name=f"{field}_gte_{threshold:.2f}",
                                rule_type='threshold',
                                customer_type=customer_type,
                                field=field,
                                operator='>=',
                                value=round(threshold, 2),
                                confidence=above_rate,
                                support=above_count / len(data),
                                description=f"{field} >= {threshold:.2f} 时审批率 {above_rate:.2%} (高于 {below_rate:.2%})",
                                priority=1
                            )
                            rules.append(rule)
                            self.rule_counter += 1
            
            elif target == 'actual_profit':
                # 连续目标（利润）
                above_mean = data[above_mask][target].mean() if above_count > 0 else 0
                below_mean = data[below_mask][target].mean() if below_count > 0 else 0
                
                # 如果差异显著（>10%）
                if abs(above_mean - below_mean) > abs(below_mean) * 0.1:
                    if above_mean > below_mean:
                        rule = ExtractedRule(
                            rule_id=f"R{self.rule_counter:04d}",
                            rule_name=f"{field}_gte_{threshold:.2f}",
                            rule_type='threshold',
                            customer_type=customer_type,
                            field=field,
                            operator='>=',
                            value=round(threshold, 2),
                            confidence=min(above_mean / (abs(below_mean) + 1e-6), 1.0),
                            support=above_count / len(data),
                            description=f"{field} >= {threshold:.2f} 时平均利润 {above_mean:.2f} (高于 {below_mean:.2f})",
                            priority=1
                        )
                        rules.append(rule)
                        self.rule_counter += 1
        
        return rules
    
    def extract_range_rules(self, field: str, target: str = 'is_defaulted',
                           customer_type: str = 'personal',
                           min_support: float = 0.05) -> List[ExtractedRule]:
        """
        提取范围规则
        
        例如：年龄在 25-55 之间时违约率最低
        """
        rules = []
        
        # 筛选数据
        if customer_type != 'both':
            data = self.data[self.data['customer_type'] == customer_type]
        else:
            data = self.data
        
        if len(data) == 0 or field not in data.columns:
            return rules
        
        # 只考虑批准的贷款
        if target == 'is_defaulted':
            data = data[data['expert_decision'] == 'approve']
            if len(data) == 0:
                return rules
        
        field_values = data[field].dropna()
        if len(field_values) == 0:
            return rules
        
        # 使用分位数划分范围
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        values = [np.percentile(field_values, p) for p in percentiles]
        
        best_range = None
        best_score = None
        
        # 寻找最优范围
        for i in range(len(values) - 1):
            for j in range(i + 1, len(values)):
                range_mask = (data[field] >= values[i]) & (data[field] <= values[j])
                range_count = range_mask.sum()
                
                if range_count < len(data) * min_support:
                    continue
                
                if target in ['is_defaulted', 'is_approved']:
                    range_rate = data[range_mask][target].mean()
                    if target == 'is_defaulted':
                        # 违约率越低越好
                        if best_score is None or range_rate < best_score:
                            best_score = range_rate
                            best_range = (values[i], values[j])
                    else:
                        # 审批率越高越好
                        if best_score is None or range_rate > best_score:
                            best_score = range_rate
                            best_range = (values[i], values[j])
                elif target == 'actual_profit':
                    range_mean = data[range_mask][target].mean()
                    if best_score is None or range_mean > best_score:
                        best_score = range_mean
                        best_range = (values[i], values[j])
        
        if best_range:
            range_mask = (data[field] >= best_range[0]) & (data[field] <= best_range[1])
            support = range_mask.sum() / len(data)
            
            if target == 'is_defaulted':
                confidence = 1 - best_score
                description = f"{field} 在 {best_range[0]:.2f}-{best_range[1]:.2f} 时违约率 {best_score:.2%}"
            elif target == 'is_approved':
                confidence = best_score
                description = f"{field} 在 {best_range[0]:.2f}-{best_range[1]:.2f} 时审批率 {best_score:.2%}"
            else:
                confidence = min(best_score / (abs(data[target].mean()) + 1e-6), 1.0)
                description = f"{field} 在 {best_range[0]:.2f}-{best_range[1]:.2f} 时平均利润 {best_score:.2f}"
            
            rule = ExtractedRule(
                rule_id=f"R{self.rule_counter:04d}",
                rule_name=f"{field}_range_{best_range[0]:.2f}_{best_range[1]:.2f}",
                rule_type='range',
                customer_type=customer_type,
                field=field,
                operator='between',
                value=best_range,
                confidence=confidence,
                support=support,
                description=description,
                priority=1 if confidence > 0.8 else 2
            )
            rules.append(rule)
            self.rule_counter += 1
        
        return rules
    
    def extract_ratio_rules(self, field1: str, field2: str, target: str = 'is_defaulted',
                           customer_type: str = 'personal',
                           min_support: float = 0.05) -> List[ExtractedRule]:
        """
        提取比例规则
        
        例如：贷款金额/月收入 <= 0.5 时违约率更低
        """
        rules = []
        
        # 筛选数据
        if customer_type != 'both':
            data = self.data[self.data['customer_type'] == customer_type]
        else:
            data = self.data
        
        if len(data) == 0 or field1 not in data.columns or field2 not in data.columns:
            return rules
        
        # 只考虑批准的贷款
        if target == 'is_defaulted':
            data = data[data['expert_decision'] == 'approve']
            if len(data) == 0:
                return rules
        
        # 计算比例
        ratio = data[field1] / (data[field2] + 1e-6)
        ratio = ratio.dropna()
        
        if len(ratio) == 0:
            return rules
        
        # 使用分位数作为候选阈值
        percentiles = list(range(10, 100, 10))
        thresholds = [np.percentile(ratio, p) for p in percentiles]
        
        for threshold in thresholds:
            below_mask = ratio <= threshold
            above_mask = ratio > threshold
            
            below_count = below_mask.sum()
            above_count = above_mask.sum()
            
            if below_count < len(ratio) * min_support or above_count < len(ratio) * min_support:
                continue
            
            if target in ['is_defaulted', 'is_approved']:
                below_rate = data[below_mask][target].mean() if below_count > 0 else 0
                above_rate = data[above_mask][target].mean() if above_count > 0 else 0
                
                if abs(below_rate - above_rate) > 0.05:
                    if target == 'is_defaulted':
                        if below_rate < above_rate:
                            rule = ExtractedRule(
                                rule_id=f"R{self.rule_counter:04d}",
                                rule_name=f"{field1}_div_{field2}_lte_{threshold:.2f}",
                                rule_type='ratio',
                                customer_type=customer_type,
                                field=f"{field1}/{field2}",
                                operator='<=',
                                value=round(threshold, 2),
                                confidence=1 - below_rate,
                                support=below_count / len(ratio),
                                description=f"{field1}/{field2} <= {threshold:.2f} 时违约率 {below_rate:.2%} (低于 {above_rate:.2%})",
                                priority=1
                            )
                            rules.append(rule)
                            self.rule_counter += 1
                    else:
                        if below_rate > above_rate:
                            rule = ExtractedRule(
                                rule_id=f"R{self.rule_counter:04d}",
                                rule_name=f"{field1}_div_{field2}_lte_{threshold:.2f}",
                                rule_type='ratio',
                                customer_type=customer_type,
                                field=f"{field1}/{field2}",
                                operator='<=',
                                value=round(threshold, 2),
                                confidence=below_rate,
                                support=below_count / len(ratio),
                                description=f"{field1}/{field2} <= {threshold:.2f} 时审批率 {below_rate:.2%} (高于 {above_rate:.2%})",
                                priority=1
                            )
                            rules.append(rule)
                            self.rule_counter += 1
        
        return rules
    
    def extract_composite_rules(self, fields: List[str], target: str = 'is_defaulted',
                               customer_type: str = 'personal',
                               max_depth: int = 3,
                               min_samples_leaf: int = 100) -> List[ExtractedRule]:
        """
        提取复合规则（多条件组合）
        
        例如：信用分 >= 700 AND 月收入 >= 10000 AND 负债率 < 0.5
        """
        rules = []
        
        # 筛选数据
        if customer_type != 'both':
            data = self.data[self.data['customer_type'] == customer_type]
        else:
            data = self.data
        
        if len(data) == 0:
            return rules
        
        # 只考虑批准的贷款
        if target == 'is_defaulted':
            data = data[data['expert_decision'] == 'approve']
            if len(data) == 0:
                return rules
        
        # 检查字段是否存在
        available_fields = [f for f in fields if f in data.columns]
        if len(available_fields) == 0:
            return rules
        
        X = data[available_fields].fillna(0)
        y = data[target]
        
        if len(y.unique()) < 2:
            return rules
        
        # 使用决策树提取规则
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        tree.fit(X, y)
        
        # 从决策树中提取规则
        rules = self._extract_rules_from_tree(
            tree, available_fields, target, customer_type, data, X, y
        )
        
        return rules
    
    def _extract_rules_from_tree(self, tree, feature_names: List[str], target: str,
                                 customer_type: str, data: pd.DataFrame,
                                 X: pd.DataFrame, y: pd.Series) -> List[ExtractedRule]:
        """从决策树中提取规则"""
        rules = []
        
        def traverse_tree(node, depth, path):
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                # 叶子节点
                samples = tree.tree_.n_node_samples[node]
                value = tree.tree_.value[node][0]
                predicted_class = np.argmax(value)
                confidence = value[predicted_class] / samples
                
                # 只保留高置信度的规则
                if confidence > 0.7 and samples > 100:
                    # 构建条件描述
                    conditions_desc = ' AND '.join([
                        f"{f} {op} {v:.2f}" for f, op, v in path
                    ])
                    
                    if target == 'is_defaulted':
                        description = f"{conditions_desc} -> 违约率 {1-confidence:.2%}"
                    elif target == 'is_approved':
                        description = f"{conditions_desc} -> 审批率 {confidence:.2%}"
                    else:
                        description = f"{conditions_desc} -> 目标值 {predicted_class}"
                    
                    rule = ExtractedRule(
                        rule_id=f"R{self.rule_counter:04d}",
                        rule_name=f"composite_{self.rule_counter}",
                        rule_type='composite',
                        customer_type=customer_type,
                        field='composite',
                        operator='AND',
                        value=None,
                        conditions=path.copy(),
                        confidence=confidence,
                        support=samples / len(data),
                        description=description,
                        priority=1 if confidence > 0.8 else 2
                    )
                    rules.append(rule)
                    self.rule_counter += 1
            else:
                # 内部节点
                feature = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                feature_name = feature_names[feature]
                
                # 左子树（<= threshold）
                traverse_tree(
                    tree.tree_.children_left[node],
                    depth + 1,
                    path + [(feature_name, '<=', threshold)]
                )
                # 右子树（> threshold）
                traverse_tree(
                    tree.tree_.children_right[node],
                    depth + 1,
                    path + [(feature_name, '>', threshold)]
                )
        
        traverse_tree(0, 0, [])
        return rules
    
    def extract_all_rules(self, customer_type: str = 'both') -> List[ExtractedRule]:
        """
        提取所有规则
        
        Args:
            customer_type: 'personal', 'corporate', 'both'
        """
        print("=" * 80)
        print(f"提取业务规则 (客户类型: {customer_type})")
        print("=" * 80)
        
        all_rules = []
        
        if customer_type in ['personal', 'both']:
            print("\n提取对私贷款规则...")
            
            # 对私字段
            personal_fields = ['age', 'monthly_income', 'credit_score', 'debt_ratio', 
                             'years_in_job', 'loan_to_annual_income_ratio']
            available_personal_fields = [f for f in personal_fields if f in self.data.columns]
            
            # 阈值规则
            for field in available_personal_fields[:4]:  # 限制数量
                rules = self.extract_threshold_rules(field, 'is_defaulted', 'personal')
                all_rules.extend(rules)
                print(f"  ✅ {field}: 提取 {len(rules)} 条阈值规则")
            
            # 范围规则
            for field in available_personal_fields[:2]:
                rules = self.extract_range_rules(field, 'is_defaulted', 'personal')
                all_rules.extend(rules)
                print(f"  ✅ {field}: 提取 {len(rules)} 条范围规则")
            
            # 比例规则
            if 'loan_amount' in self.data.columns and 'monthly_income' in self.data.columns:
                rules = self.extract_ratio_rules('loan_amount', 'monthly_income', 'is_defaulted', 'personal')
                all_rules.extend(rules)
                print(f"  ✅ loan_amount/monthly_income: 提取 {len(rules)} 条比例规则")
            
            # 复合规则
            if len(available_personal_fields) >= 3:
                rules = self.extract_composite_rules(available_personal_fields[:5], 'is_defaulted', 'personal')
                all_rules.extend(rules)
                print(f"  ✅ 复合规则: 提取 {len(rules)} 条")
        
        if customer_type in ['corporate', 'both']:
            print("\n提取对公贷款规则...")
            
            # 对公字段
            corporate_fields = ['registered_capital', 'operating_years', 'annual_revenue',
                             'debt_to_asset_ratio', 'current_ratio', 'loan_to_revenue_ratio']
            available_corporate_fields = [f for f in corporate_fields if f in self.data.columns]
            
            # 阈值规则
            for field in available_corporate_fields[:4]:
                rules = self.extract_threshold_rules(field, 'is_defaulted', 'corporate')
                all_rules.extend(rules)
                print(f"  ✅ {field}: 提取 {len(rules)} 条阈值规则")
            
            # 范围规则
            for field in available_corporate_fields[:2]:
                rules = self.extract_range_rules(field, 'is_defaulted', 'corporate')
                all_rules.extend(rules)
                print(f"  ✅ {field}: 提取 {len(rules)} 条范围规则")
            
            # 比例规则
            if 'loan_amount' in self.data.columns and 'annual_revenue' in self.data.columns:
                rules = self.extract_ratio_rules('loan_amount', 'annual_revenue', 'is_defaulted', 'corporate')
                all_rules.extend(rules)
                print(f"  ✅ loan_amount/annual_revenue: 提取 {len(rules)} 条比例规则")
            
            # 复合规则
            if len(available_corporate_fields) >= 3:
                rules = self.extract_composite_rules(available_corporate_fields[:5], 'is_defaulted', 'corporate')
                all_rules.extend(rules)
                print(f"  ✅ 复合规则: 提取 {len(rules)} 条")
        
        # 按置信度和支持度排序
        all_rules.sort(key=lambda r: r.confidence * r.support, reverse=True)
        
        print(f"\n✅ 总共提取 {len(all_rules)} 条规则")
        print(f"   按置信度×支持度排序，Top 10:")
        for i, rule in enumerate(all_rules[:10], 1):
            print(f"   {i}. {rule.description} (置信度: {rule.confidence:.2%}, 支持度: {rule.support:.2%})")
        
        self.rules = all_rules
        return all_rules
    
    def save_rules(self, output_path: str = 'data/historical/extracted_rules.json'):
        """保存提取的规则"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        rules_dict = [rule.to_dict() for rule in self.rules]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_dict, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ 已保存 {len(self.rules)} 条规则到: {output_path}")


def main():
    """主函数：执行规则提取"""
    import sys
    import os
    
    # 加载特征工程后的数据
    data_path = 'data/historical/historical_loans_engineered.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行 feature_engineer.py 进行特征工程")
        sys.exit(1)
    
    print("正在加载数据...")
    data = pd.read_csv(data_path)
    print(f"✅ 已加载 {len(data)} 条记录，{len(data.columns)} 个特征")
    
    # 执行规则提取
    extractor = RuleExtractor(data)
    rules = extractor.extract_all_rules(customer_type='both')
    
    # 保存规则
    extractor.save_rules()
    
    return rules


if __name__ == '__main__':
    main()

