# 端到端贷款审批Demo完整设计方案

## 一、项目概述

### 1.1 项目目标

构建一个完整的端到端贷款审批演示系统，包括：
1. **业务规则提取**：从历史数据中自动提取和量化业务规则
2. **模拟审批**：基于规则和模型进行贷款审批决策
3. **结果验证**：验证模拟结果与真实情况的一致性

### 1.2 核心价值

- ✅ **真实性**：模拟环境尽可能接近真实业务场景
- ✅ **完整性**：覆盖从申请到回收的全流程
- ✅ **可验证性**：可以对比模拟结果与历史真实结果
- ✅ **可解释性**：每个决策都有明确的规则和依据

---

## 二、系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    端到端贷款审批Demo系统                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  规则提取模块  │   │  模拟审批模块  │   │  结果验证模块  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  历史数据     │   │  模拟环境     │   │  验证报告     │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 2.2 模块划分

#### 模块一：历史数据管理模块
- 数据加载和预处理
- 数据质量检查
- 特征工程

#### 模块二：业务规则提取模块
- 规则挖掘算法
- 规则量化
- 规则库管理

#### 模块三：模拟环境模块
- 客户生成器
- 市场环境模拟
- 世界模型（预测客户行为）

#### 模块四：审批决策模块
- 规则引擎
- 模型决策
- 决策融合

#### 模块五：贷款回收模拟模块
- 还款行为模拟
- 违约事件模拟
- 回收率计算

#### 模块六：结果验证模块
- 对比分析
- 拟合度评估
- 报告生成

---

## 三、详细设计

### 3.1 模块一：历史数据管理模块

#### 3.1.1 数据结构设计

```python
class HistoricalLoanData:
    """历史贷款数据结构"""
    
    # 申请时点数据（申请时已知）
    application_data = {
        'customer_id': str,
        'application_date': datetime,
        'customer_type': str,  # 'personal' or 'corporate'
        
        # 对私客户特征
        'age': int,
        'monthly_income': float,
        'credit_score': int,
        'debt_ratio': float,
        'employment_status': str,
        'years_in_job': int,
        'education_level': str,
        'marital_status': str,
        'has_collateral': bool,
        'collateral_value': float,
        
        # 对公客户特征
        'registered_capital': float,
        'operating_years': int,
        'annual_revenue': float,
        'debt_to_asset_ratio': float,
        'current_ratio': float,
        'industry': str,
        'company_size': str,
        
        # 贷款申请信息
        'loan_amount': float,
        'loan_purpose': str,
        'requested_term_months': int,
    }
    
    # 审批决策数据
    approval_data = {
        'expert_decision': str,  # 'approve' or 'reject'
        'approved_amount': float,
        'approved_rate': float,
        'approved_term': int,
        'conditions': dict,  # 附加条件
        'approval_date': datetime,
        'approver_id': str,
    }
    
    # 实际结果数据（申请后发生）
    outcome_data = {
        'actual_defaulted': bool,
        'default_date': datetime,
        'default_amount': float,
        'recovery_amount': float,
        'recovery_rate': float,
        'total_interest_paid': float,
        'total_principal_paid': float,
        'actual_profit': float,
        'actual_roi': float,
        'payment_history': list,  # 还款历史
    }
```

#### 3.1.2 数据质量检查

```python
class DataQualityChecker:
    """数据质量检查器"""
    
    def check_completeness(self, data: pd.DataFrame) -> Dict:
        """检查数据完整性"""
        missing_rates = data.isnull().mean()
        return {
            'missing_rates': missing_rates.to_dict(),
            'completeness_score': 1 - missing_rates.mean(),
            'critical_missing': missing_rates[missing_rates > 0.1].to_dict()
        }
    
    def check_consistency(self, data: pd.DataFrame) -> Dict:
        """检查数据一致性"""
        issues = []
        
        # 检查逻辑一致性
        # 例如：如果defaulted=True，应该有default_date
        if 'defaulted' in data.columns and 'default_date' in data.columns:
            inconsistent = data[
                (data['defaulted'] == True) & 
                (data['default_date'].isnull())
            ]
            if len(inconsistent) > 0:
                issues.append(f"{len(inconsistent)} cases: defaulted=True but no default_date")
        
        # 检查数值范围
        if 'credit_score' in data.columns:
            invalid_scores = data[
                (data['credit_score'] < 300) | 
                (data['credit_score'] > 850)
            ]
            if len(invalid_scores) > 0:
                issues.append(f"{len(invalid_scores)} cases: credit_score out of range")
        
        return {
            'issues': issues,
            'consistency_score': 1 - len(issues) / 10  # 归一化
        }
    
    def check_temporal_consistency(self, data: pd.DataFrame) -> Dict:
        """检查时间一致性"""
        issues = []
        
        # 检查时间顺序
        if 'application_date' in data.columns and 'approval_date' in data.columns:
            invalid_order = data[
                data['approval_date'] < data['application_date']
            ]
            if len(invalid_order) > 0:
                issues.append(f"{len(invalid_order)} cases: approval_date before application_date")
        
        return {
            'issues': issues,
            'temporal_consistency_score': 1 - len(issues) / len(data) * 0.1
        }
```

#### 3.1.3 特征工程

```python
class FeatureEngineer:
    """特征工程器"""
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建特征"""
        df = data.copy()
        
        # 1. 基础特征（已有）
        # 保持原样
        
        # 2. 衍生特征
        if 'monthly_income' in df.columns and 'loan_amount' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amount'] / (df['monthly_income'] * 12)
        
        if 'debt_ratio' in df.columns:
            df['debt_ratio_category'] = pd.cut(
                df['debt_ratio'],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        # 3. 时间特征
        if 'application_date' in df.columns:
            df['application_year'] = df['application_date'].dt.year
            df['application_month'] = df['application_date'].dt.month
            df['application_quarter'] = df['application_date'].dt.quarter
        
        # 4. 交互特征
        if 'credit_score' in df.columns and 'debt_ratio' in df.columns:
            df['credit_debt_interaction'] = df['credit_score'] * (1 - df['debt_ratio'])
        
        return df
```

---

### 3.2 模块二：业务规则提取模块

#### 3.2.1 规则类型定义

```python
class BusinessRule:
    """业务规则基类"""
    
    def __init__(self):
        self.rule_id = None
        self.rule_name = str
        self.rule_type = str  # 'threshold', 'range', 'ratio', 'conditional', 'composite'
        self.customer_type = str  # 'personal', 'corporate', 'both'
        self.priority = int  # 优先级
        self.confidence = float  # 置信度
        self.support = float  # 支持度
        self.description = str
        self.conditions = list  # 条件列表
        self.action = dict  # 动作
        self.penalty = dict  # 违反规则的惩罚
```

#### 3.2.2 规则提取算法

```python
class RuleExtractor:
    """规则提取器"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.data = historical_data
        self.rules = []
    
    def extract_threshold_rules(self, field: str, target: str = 'defaulted', 
                               customer_type: str = 'personal') -> List[BusinessRule]:
        """
        提取阈值规则
        
        例如：信用分 >= 600 时违约率显著降低
        """
        rules = []
        data = self.data[self.data['customer_type'] == customer_type] if customer_type != 'both' else self.data
        
        # 使用决策树或分位数方法找最优阈值
        from sklearn.tree import DecisionTreeRegressor
        
        X = data[[field]]
        y = data[target].astype(int)
        
        # 简单方法：遍历分位数
        for percentile in range(10, 100, 10):
            threshold = np.percentile(data[field], percentile)
            
            above = data[data[field] >= threshold]
            below = data[data[field] < threshold]
            
            if len(above) > 0 and len(below) > 0:
                above_rate = above[target].mean()
                below_rate = below[target].mean()
                
                # 如果差异显著
                if abs(above_rate - below_rate) > 0.05:
                    rule = BusinessRule()
                    rule.rule_type = 'threshold'
                    rule.field = field
                    rule.operator = '>=' if above_rate < below_rate else '<'
                    rule.threshold = threshold
                    rule.confidence = 1 - min(above_rate, below_rate)
                    rule.support = len(above) / len(data) if above_rate < below_rate else len(below) / len(data)
                    rule.description = f"{field} {rule.operator} {threshold:.2f} 时违约率 {min(above_rate, below_rate):.2%}"
                    rules.append(rule)
        
        return rules
    
    def extract_composite_rules(self, fields: List[str], target: str = 'defaulted',
                               customer_type: str = 'personal') -> List[BusinessRule]:
        """
        提取复合规则（多条件组合）
        
        例如：信用分 >= 700 AND 月收入 >= 10000 AND 负债率 < 0.5
        """
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        data = self.data[self.data['customer_type'] == customer_type] if customer_type != 'both' else self.data
        
        X = data[fields]
        y = data[target].astype(int)
        
        # 使用决策树提取规则
        tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100)
        tree.fit(X, y)
        
        rules = self.extract_rules_from_tree(tree, fields, customer_type)
        
        return rules
    
    def extract_rules_from_tree(self, tree, feature_names: List[str], 
                               customer_type: str) -> List[BusinessRule]:
        """从决策树中提取规则"""
        rules = []
        
        def traverse_tree(node, depth, path):
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                # 叶子节点
                samples = tree.tree_.n_node_samples[node]
                value = tree.tree_.value[node][0]
                predicted_class = np.argmax(value)
                confidence = value[predicted_class] / samples
                
                if confidence > 0.7 and samples > 100:
                    rule = BusinessRule()
                    rule.rule_type = 'composite'
                    rule.conditions = path
                    rule.confidence = confidence
                    rule.support = samples / len(self.data)
                    rule.customer_type = customer_type
                    rule.description = ' AND '.join([f"{f} {op} {v:.2f}" for f, op, v in path])
                    rules.append(rule)
            else:
                # 内部节点
                feature = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                feature_name = feature_names[feature]
                
                # 左子树
                traverse_tree(
                    tree.tree_.children_left[node],
                    depth + 1,
                    path + [(feature_name, '<=', threshold)]
                )
                # 右子树
                traverse_tree(
                    tree.tree_.children_right[node],
                    depth + 1,
                    path + [(feature_name, '>', threshold)]
                )
        
        traverse_tree(0, 0, [])
        return rules
    
    def extract_all_rules(self, customer_type: str = 'personal') -> List[BusinessRule]:
        """提取所有规则"""
        all_rules = []
        
        if customer_type == 'personal':
            # 对私规则
            fields = ['age', 'monthly_income', 'credit_score', 'debt_ratio', 
                     'years_in_job', 'loan_to_income_ratio']
            for field in fields:
                if field in self.data.columns:
                    all_rules.extend(self.extract_threshold_rules(field, customer_type='personal'))
            
            # 复合规则
            all_rules.extend(self.extract_composite_rules(fields, customer_type='personal'))
        
        elif customer_type == 'corporate':
            # 对公规则
            fields = ['registered_capital', 'operating_years', 'annual_revenue',
                     'debt_to_asset_ratio', 'current_ratio']
            for field in fields:
                if field in self.data.columns:
                    all_rules.extend(self.extract_threshold_rules(field, customer_type='corporate'))
            
            # 复合规则
            all_rules.extend(self.extract_composite_rules(fields, customer_type='corporate'))
        
        # 按置信度和支持度排序
        all_rules.sort(key=lambda r: r.confidence * r.support, reverse=True)
        
        return all_rules
```

#### 3.2.3 规则量化

```python
class RuleQuantifier:
    """规则量化器"""
    
    def quantify_rule(self, rule: BusinessRule) -> Dict:
        """量化规则为可执行的约束"""
        if rule.rule_type == 'threshold':
            def check(state):
                value = state.get(rule.field, 0)
                if rule.operator == '>=':
                    return value >= rule.threshold
                elif rule.operator == '<=':
                    return value <= rule.threshold
                elif rule.operator == '>':
                    return value > rule.threshold
                elif rule.operator == '<':
                    return value < rule.threshold
            
            def penalty(state):
                if not check(state):
                    value = state.get(rule.field, 0)
                    if rule.operator == '>=':
                        diff = rule.threshold - value
                    elif rule.operator == '<=':
                        diff = value - rule.threshold
                    else:
                        diff = abs(value - rule.threshold)
                    return -rule.confidence * diff * 10
                return 0
        
        elif rule.rule_type == 'composite':
            def check(state):
                for field, op, val in rule.conditions:
                    field_value = state.get(field, 0)
                    if op == '<=' and field_value > val:
                        return False
                    elif op == '>' and field_value <= val:
                        return False
                return True
            
            def penalty(state):
                if not check(state):
                    return -rule.confidence * 100
                return 0
        
        return {
            'rule_id': rule.rule_id,
            'rule_name': rule.rule_name,
            'check': check,
            'penalty': penalty,
            'weight': rule.confidence * rule.support,
            'description': rule.description
        }
```

---

### 3.3 模块三：模拟环境模块

#### 3.3.1 客户生成器（增强版）

```python
class EnhancedCustomerGenerator:
    """增强版客户生成器"""
    
    def __init__(self, historical_data: pd.DataFrame, seed: int = 42):
        self.data = historical_data
        self.rng = np.random.default_rng(seed)
        
        # 学习历史数据的分布
        self.personal_distributions = self.learn_distributions('personal')
        self.corporate_distributions = self.learn_distributions('corporate')
    
    def learn_distributions(self, customer_type: str) -> Dict:
        """学习历史数据的分布"""
        data = self.data[self.data['customer_type'] == customer_type]
        
        distributions = {}
        
        if customer_type == 'personal':
            # 年龄分布（可能不是正态分布）
            distributions['age'] = {
                'type': 'beta',  # 使用Beta分布
                'params': self.fit_beta_distribution(data['age'] / 100)
            }
            
            # 收入分布（对数正态）
            distributions['monthly_income'] = {
                'type': 'lognormal',
                'params': self.fit_lognormal_distribution(data['monthly_income'])
            }
            
            # 信用分分布
            distributions['credit_score'] = {
                'type': 'normal',
                'params': (data['credit_score'].mean(), data['credit_score'].std())
            }
            
            # 负债率分布（Beta分布，0-1之间）
            distributions['debt_ratio'] = {
                'type': 'beta',
                'params': self.fit_beta_distribution(data['debt_ratio'])
            }
            
            # 分类特征的概率分布
            distributions['employment_status'] = data['employment_status'].value_counts(normalize=True).to_dict()
            distributions['education_level'] = data['education_level'].value_counts(normalize=True).to_dict()
            distributions['marital_status'] = data['marital_status'].value_counts(normalize=True).to_dict()
        
        elif customer_type == 'corporate':
            # 注册资本（对数正态）
            distributions['registered_capital'] = {
                'type': 'lognormal',
                'params': self.fit_lognormal_distribution(data['registered_capital'])
            }
            
            # 经营年限
            distributions['operating_years'] = {
                'type': 'poisson',
                'params': data['operating_years'].mean()
            }
            
            # 年营收（对数正态）
            distributions['annual_revenue'] = {
                'type': 'lognormal',
                'params': self.fit_lognormal_distribution(data['annual_revenue'])
            }
            
            # 资产负债率（Beta分布）
            distributions['debt_to_asset_ratio'] = {
                'type': 'beta',
                'params': self.fit_beta_distribution(data['debt_to_asset_ratio'])
            }
            
            # 行业分布
            distributions['industry'] = data['industry'].value_counts(normalize=True).to_dict()
        
        return distributions
    
    def generate_personal_customer(self) -> Dict:
        """生成对私客户"""
        dist = self.personal_distributions
        
        customer = {
            'customer_type': 'personal',
            'customer_id': f"P{self.rng.integers(1000000, 9999999)}",
            'age': int(self.sample_from_distribution(dist['age'])),
            'monthly_income': self.sample_from_distribution(dist['monthly_income']),
            'credit_score': int(self.sample_from_distribution(dist['credit_score'])),
            'debt_ratio': self.sample_from_distribution(dist['debt_ratio']),
            'employment_status': self.rng.choice(
                list(dist['employment_status'].keys()),
                p=list(dist['employment_status'].values())
            ),
            'education_level': self.rng.choice(
                list(dist['education_level'].keys()),
                p=list(dist['education_level'].values())
            ),
            'marital_status': self.rng.choice(
                list(dist['marital_status'].keys()),
                p=list(dist['marital_status'].values())
            ),
            'years_in_job': self.rng.integers(0, 30),
            'has_collateral': self.rng.random() < 0.3,
            'collateral_value': self.rng.exponential(50000) if self.rng.random() < 0.3 else 0,
        }
        
        return customer
    
    def generate_corporate_customer(self) -> Dict:
        """生成对公客户"""
        dist = self.corporate_distributions
        
        customer = {
            'customer_type': 'corporate',
            'customer_id': f"C{self.rng.integers(1000000, 9999999)}",
            'registered_capital': self.sample_from_distribution(dist['registered_capital']),
            'operating_years': int(self.sample_from_distribution(dist['operating_years'])),
            'annual_revenue': self.sample_from_distribution(dist['annual_revenue']),
            'debt_to_asset_ratio': self.sample_from_distribution(dist['debt_to_asset_ratio']),
            'current_ratio': self.rng.normal(1.5, 0.5),
            'industry': self.rng.choice(
                list(dist['industry'].keys()),
                p=list(dist['industry'].values())
            ),
            'company_size': self.rng.choice(['small', 'medium', 'large'], p=[0.6, 0.3, 0.1]),
        }
        
        return customer
```

#### 3.3.2 市场环境模拟

```python
class MarketEnvironment:
    """市场环境模拟器"""
    
    def __init__(self, historical_macro_data: pd.DataFrame):
        self.macro_data = historical_macro_data
        self.current_state = None
    
    def initialize(self, date: datetime):
        """初始化市场环境"""
        # 从历史数据中找到最接近的日期
        closest_date = self.macro_data.iloc[
            (self.macro_data['date'] - date).abs().argsort()[:1]
        ]
        
        self.current_state = {
            'date': date,
            'gdp_growth': closest_date['gdp_growth'].values[0],
            'base_interest_rate': closest_date['base_interest_rate'].values[0],
            'unemployment_rate': closest_date['unemployment_rate'].values[0],
            'inflation_rate': closest_date['inflation_rate'].values[0],
            'credit_spread': closest_date['credit_spread'].values[0],
        }
    
    def evolve(self, months: int = 1):
        """市场环境演化"""
        # 模拟市场环境的变化
        # 可以使用随机游走或ARIMA模型
        pass
```

#### 3.3.3 世界模型（预测客户行为）

```python
class WorldModel:
    """世界模型：预测客户未来行为"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.data = historical_data
        self.default_model = self.train_default_model()
        self.repayment_model = self.train_repayment_model()
    
    def train_default_model(self):
        """训练违约预测模型"""
        from sklearn.ensemble import GradientBoostingClassifier
        
        # 准备训练数据
        approved_loans = self.data[self.data['expert_decision'] == 'approve']
        
        features = ['age', 'monthly_income', 'credit_score', 'debt_ratio',
                   'loan_amount', 'interest_rate', 'term_months']
        X = approved_loans[features]
        y = approved_loans['actual_defaulted'].astype(int)
        
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
        model.fit(X, y)
        
        return model
    
    def train_repayment_model(self):
        """训练还款行为模型"""
        # 使用历史还款数据训练
        # 预测每期还款金额、提前还款概率等
        pass
    
    def predict_customer_future(self, customer: Dict, loan: Dict, 
                                market: Dict) -> Dict:
        """预测客户未来行为"""
        # 1. 预测违约概率
        features = np.array([[
            customer.get('age', 0),
            customer.get('monthly_income', 0),
            customer.get('credit_score', 0),
            customer.get('debt_ratio', 0),
            loan['amount'],
            loan['interest_rate'],
            loan['term_months']
        ]])
        
        default_prob = self.default_model.predict_proba(features)[0, 1]
        
        # 2. 预测还款行为
        repayment_behavior = self.predict_repayment_behavior(customer, loan, market)
        
        return {
            'default_probability': default_prob,
            'repayment_behavior': repayment_behavior,
            'expected_profit': self.calculate_expected_profit(loan, default_prob, repayment_behavior)
        }
```

---

### 3.4 模块四：审批决策模块

#### 3.4.1 规则引擎（已有，增强）

```python
class EnhancedRuleEngine:
    """增强版规则引擎"""
    
    def __init__(self, rules: List[Dict]):
        self.rules = rules
        self.rule_cache = {}  # 规则缓存
    
    def apply_rules(self, customer: Dict, loan_request: Dict) -> Dict:
        """应用所有规则"""
        adjustments = {
            'approval_threshold': 0.18,  # 默认阈值
            'rate_spread': 0.01,
            'loan_amount': loan_request['amount'],
            'term_months': loan_request['term_months'],
            'force_approve': False,
            'force_reject': False,
            'require_collateral': False,
            'require_guarantor': False,
        }
        
        triggered_rules = []
        total_penalty = 0
        
        for rule in self.rules:
            # 检查规则是否适用
            if self.is_rule_applicable(rule, customer):
                # 应用规则
                rule_result = rule['check']({**customer, **loan_request})
                
                if not rule_result:
                    # 规则被违反
                    penalty = rule['penalty']({**customer, **loan_request})
                    total_penalty += penalty * rule['weight']
                    triggered_rules.append(rule['rule_name'])
                else:
                    # 规则通过，可能调整参数
                    if 'adjustment' in rule:
                        adjustments = self.apply_adjustment(adjustments, rule['adjustment'])
        
        return {
            'adjustments': adjustments,
            'triggered_rules': triggered_rules,
            'total_penalty': total_penalty
        }
```

#### 3.4.2 模型决策

```python
class ModelDecisionMaker:
    """模型决策器"""
    
    def __init__(self, model):
        self.model = model
    
    def make_decision(self, customer: Dict, loan_request: Dict, 
                     market: Dict, world_model: WorldModel) -> Dict:
        """做出决策"""
        # 1. 预测客户未来行为
        future = world_model.predict_customer_future(customer, loan_request, market)
        
        # 2. 计算风险指标
        risk_score = self.calculate_risk_score(customer, loan_request, future)
        
        # 3. 计算预期利润
        expected_profit = future['expected_profit']
        
        # 4. 决策
        if risk_score <= 0.18 and expected_profit > 0:
            decision = 'approve'
        else:
            decision = 'reject'
        
        return {
            'decision': decision,
            'risk_score': risk_score,
            'expected_profit': expected_profit,
            'default_probability': future['default_probability'],
            'confidence': 1 - abs(risk_score - 0.18)  # 置信度
        }
```

#### 3.4.3 决策融合

```python
class DecisionFusion:
    """决策融合器"""
    
    def fuse_decisions(self, rule_decision: Dict, model_decision: Dict) -> Dict:
        """融合规则决策和模型决策"""
        # 如果规则强制拒绝，直接拒绝
        if rule_decision['adjustments']['force_reject']:
            return {
                'decision': 'reject',
                'reason': 'rule_force_reject',
                'confidence': 1.0
            }
        
        # 如果规则强制通过，但需要检查风险
        if rule_decision['adjustments']['force_approve']:
            if model_decision['risk_score'] > 0.3:  # 风险太高
                return {
                    'decision': 'reject',
                    'reason': 'high_risk_override',
                    'confidence': 0.8
                }
            else:
                return {
                    'decision': 'approve',
                    'reason': 'rule_force_approve',
                    'confidence': 0.9
                }
        
        # 正常决策融合
        # 规则权重 0.3，模型权重 0.7
        rule_weight = 0.3
        model_weight = 0.7
        
        # 综合风险评分
        combined_risk = (
            rule_decision.get('risk_score', 0.18) * rule_weight +
            model_decision['risk_score'] * model_weight
        )
        
        # 综合利润
        combined_profit = model_decision['expected_profit']
        
        # 最终决策
        if combined_risk <= 0.18 and combined_profit > 0:
            decision = 'approve'
        else:
            decision = 'reject'
        
        return {
            'decision': decision,
            'combined_risk': combined_risk,
            'combined_profit': combined_profit,
            'confidence': (rule_decision.get('confidence', 0.5) * rule_weight +
                          model_decision['confidence'] * model_weight),
            'rule_decision': rule_decision,
            'model_decision': model_decision
        }
```

---

### 3.5 模块五：贷款回收模拟模块

#### 3.5.1 还款行为模拟

```python
class RepaymentSimulator:
    """还款模拟器"""
    
    def __init__(self, historical_repayment_data: pd.DataFrame):
        self.data = historical_repayment_data
        self.repayment_model = self.train_repayment_model()
    
    def simulate_repayment(self, customer: Dict, loan: Dict, 
                          market: Dict, default_prob: float) -> List[Dict]:
        """模拟还款过程"""
        term_months = loan['term_months']
        monthly_payment = self.calculate_monthly_payment(loan)
        
        repayment_history = []
        remaining_principal = loan['amount']
        
        for month in range(term_months):
            # 检查是否违约
            if self.check_default(month, default_prob, market):
                # 违约
                repayment_history.append({
                    'month': month + 1,
                    'status': 'defaulted',
                    'principal_paid': remaining_principal * 0.1,  # 假设回收10%
                    'interest_paid': 0,
                    'total_paid': remaining_principal * 0.1
                })
                break
            
            # 正常还款
            # 可能提前还款
            prepay_prob = self.calculate_prepay_probability(customer, loan, market, month)
            if np.random.random() < prepay_prob:
                # 提前还款
                repayment_history.append({
                    'month': month + 1,
                    'status': 'prepaid',
                    'principal_paid': remaining_principal,
                    'interest_paid': monthly_payment - remaining_principal,
                    'total_paid': remaining_principal + monthly_payment - remaining_principal
                })
                break
            
            # 正常还款
            interest_payment = remaining_principal * loan['interest_rate'] / 12
            principal_payment = monthly_payment - interest_payment
            
            repayment_history.append({
                'month': month + 1,
                'status': 'paid',
                'principal_paid': principal_payment,
                'interest_paid': interest_payment,
                'total_paid': monthly_payment
            })
            
            remaining_principal -= principal_payment
        
        return repayment_history
    
    def calculate_monthly_payment(self, loan: Dict) -> float:
        """计算月还款额"""
        r = loan['interest_rate'] / 12
        n = loan['term_months']
        p = loan['amount']
        
        if r == 0:
            return p / n
        else:
            return p * r * (1 + r) ** n / ((1 + r) ** n - 1)
    
    def check_default(self, month: int, default_prob: float, market: Dict) -> bool:
        """检查是否违约"""
        # 违约概率可能随时间变化
        # 考虑市场环境的影响
        adjusted_prob = default_prob * (1 + market.get('unemployment_rate', 0) * 0.5)
        
        # 前几个月违约概率较低
        if month < 6:
            adjusted_prob *= 0.5
        
        return np.random.random() < adjusted_prob
```

#### 3.5.2 回收率计算

```python
class RecoveryCalculator:
    """回收率计算器"""
    
    def calculate_recovery(self, defaulted_loan: Dict, 
                          repayment_history: List[Dict]) -> Dict:
        """计算回收情况"""
        # 已回收金额
        recovered_amount = sum([r['total_paid'] for r in repayment_history])
        
        # 原始贷款金额
        original_amount = defaulted_loan['amount']
        
        # 回收率
        recovery_rate = recovered_amount / original_amount
        
        # 回收时间
        recovery_time = len(repayment_history)
        
        return {
            'original_amount': original_amount,
            'recovered_amount': recovered_amount,
            'loss_amount': original_amount - recovered_amount,
            'recovery_rate': recovery_rate,
            'recovery_time_months': recovery_time
        }
```

---

### 3.6 模块六：结果验证模块

#### 3.6.1 对比分析

```python
class ResultValidator:
    """结果验证器"""
    
    def __init__(self, simulated_results: pd.DataFrame, 
                 historical_results: pd.DataFrame):
        self.simulated = simulated_results
        self.historical = historical_results
    
    def compare_default_rates(self) -> Dict:
        """对比违约率"""
        sim_default_rate = self.simulated['defaulted'].mean()
        hist_default_rate = self.historical['defaulted'].mean()
        
        return {
            'simulated_default_rate': sim_default_rate,
            'historical_default_rate': hist_default_rate,
            'difference': abs(sim_default_rate - hist_default_rate),
            'is_acceptable': abs(sim_default_rate - hist_default_rate) < 0.02
        }
    
    def compare_profit_distribution(self) -> Dict:
        """对比利润分布"""
        from scipy import stats
        
        sim_profit = self.simulated['profit']
        hist_profit = self.historical['actual_profit']
        
        # KS检验
        ks_stat, ks_pvalue = stats.ks_2samp(sim_profit, hist_profit)
        
        # 分位数对比
        percentiles = [10, 25, 50, 75, 90]
        sim_percentiles = np.percentile(sim_profit, percentiles)
        hist_percentiles = np.percentile(hist_profit, percentiles)
        
        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'is_distribution_similar': ks_pvalue > 0.05,
            'percentile_comparison': {
                'simulated': dict(zip(percentiles, sim_percentiles)),
                'historical': dict(zip(percentiles, hist_percentiles))
            }
        }
    
    def compare_recovery_rates(self) -> Dict:
        """对比回收率"""
        sim_defaulted = self.simulated[self.simulated['defaulted'] == True]
        hist_defaulted = self.historical[self.historical['actual_defaulted'] == True]
        
        if len(sim_defaulted) == 0 or len(hist_defaulted) == 0:
            return {'error': 'No defaulted cases'}
        
        sim_recovery_rate = sim_defaulted['recovery_rate'].mean()
        hist_recovery_rate = hist_defaulted['recovery_rate'].mean()
        
        return {
            'simulated_recovery_rate': sim_recovery_rate,
            'historical_recovery_rate': hist_recovery_rate,
            'difference': abs(sim_recovery_rate - hist_recovery_rate),
            'is_acceptable': abs(sim_recovery_rate - hist_recovery_rate) < 0.1
        }
```

---

## 四、实施计划

### 4.1 阶段划分

#### 阶段一：数据准备和规则提取（2周）
- 任务1.1：历史数据加载和预处理（3天）
- 任务1.2：数据质量检查（2天）
- 任务1.3：特征工程（2天）
- 任务1.4：规则提取算法实现（4天）
- 任务1.5：规则量化和规则库构建（3天）

#### 阶段二：模拟环境构建（2周）
- 任务2.1：客户生成器增强（3天）
- 任务2.2：市场环境模拟（2天）
- 任务2.3：世界模型训练（4天）
- 任务2.4：还款行为模拟（3天）

#### 阶段三：审批决策模块（1.5周）
- 任务3.1：规则引擎增强（3天）
- 任务3.2：模型决策实现（3天）
- 任务3.3：决策融合（2天）

#### 阶段四：回收模拟和验证（1.5周）
- 任务4.1：回收模拟实现（3天）
- 任务4.2：结果验证模块（3天）
- 任务4.3：报告生成（2天）

#### 阶段五：集成测试和优化（1周）
- 任务5.1：端到端集成（3天）
- 任务5.2：性能优化（2天）
- 任务5.3：文档和演示准备（2天）

**总时间：8周**

---

## 五、技术栈

### 5.1 核心技术

- **Python 3.9+**
- **Pandas/NumPy**：数据处理
- **Scikit-learn**：机器学习
- **Matplotlib/Seaborn**：可视化
- **Flask**：API服务（如果需要）

### 5.2 数据存储

- **CSV/Parquet**：历史数据存储
- **JSON**：规则库存储
- **SQLite**：结果存储（可选）

---

## 六、交付物

### 6.1 代码交付

1. **数据管理模块**：`src/data_management/`
2. **规则提取模块**：`src/rule_extraction/`
3. **模拟环境模块**：`src/simulation/`
4. **审批决策模块**：`src/decision/`
5. **回收模拟模块**：`src/recovery/`
6. **结果验证模块**：`src/validation/`
7. **主程序**：`demo_main.py`

### 6.2 文档交付

1. **设计文档**（本文档）
2. **API文档**
3. **使用手册**
4. **测试报告**

### 6.3 演示交付

1. **Jupyter Notebook演示**
2. **可视化报告**
3. **对比分析图表**

---

## 七、风险评估

### 7.1 技术风险

- **数据质量**：历史数据可能不完整或不准确
  - 缓解：加强数据质量检查
- **模型准确性**：预测模型可能不够准确
  - 缓解：使用多个模型集成，持续优化

### 7.2 业务风险

- **规则复杂性**：业务规则可能过于复杂
  - 缓解：分阶段实现，先实现核心规则
- **真实性**：模拟环境可能不够真实
  - 缓解：持续对比历史数据，调整参数

---

## 八、下一步行动

### 8.1 立即行动

1. **评审设计文档**：确认设计方案
2. **准备开发环境**：搭建开发环境
3. **准备测试数据**：准备历史数据样本

### 8.2 开发顺序

按照阶段划分，逐步实施：
1. 先完成数据准备和规则提取
2. 再构建模拟环境
3. 最后集成和验证

---

## 九、总结

本设计方案提供了一个完整的端到端贷款审批Demo系统，包括：

1. ✅ **完整的模块划分**：6大核心模块
2. ✅ **详细的技术设计**：每个模块的具体实现
3. ✅ **清晰的实施计划**：8周开发计划
4. ✅ **可验证的结果**：完整的验证方案

**关键成功因素：**
- 数据质量
- 规则准确性
- 模拟真实性
- 验证完整性

**预期成果：**
- 一个可运行的端到端Demo
- 完整的验证报告
- 可复用的代码库

