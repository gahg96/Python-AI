# AlphaZero 业务规则提取与多模型博弈方案

## 一、核心思路

### 1.1 问题理解

在使用 AlphaZero 进行贷款审批决策时，我们需要：

1. **业务规则作为约束**：对公、对私等不同类型的贷款有不同的业务规则
2. **规则量化**：将业务规则转化为可计算的约束条件
3. **规则学习**：从历史数据中自动提取和优化规则
4. **多模型博弈**：让不同模型在相同规则约束下博弈，找出最优策略

### 1.2 整体架构

```
历史数据
    ↓
规则提取引擎（Rule Extraction Engine）
    ↓
规则量化（Rule Quantification）
    ↓
规则库（Rule Base）
    ↓
AlphaZero环境（集成规则约束）
    ↓
多模型博弈（Multi-Model Arena）
    ↓
最优策略选择
```

---

## 二、业务规则提取

### 2.1 规则类型分类

#### 2.1.1 对私贷款规则

```python
class PersonalLoanRules:
    """对私贷款业务规则"""
    
    # 基础准入规则
    MIN_AGE = 18
    MAX_AGE = 65
    MIN_INCOME = 3000  # 最低月收入
    MIN_CREDIT_SCORE = 600  # 最低信用分
    
    # 额度规则
    MAX_LOAN_AMOUNT = 500000  # 最高额度
    MAX_LOAN_TO_INCOME_RATIO = 0.5  # 贷款金额/月收入 <= 0.5
    
    # 利率规则
    BASE_RATE = 0.08  # 基准利率
    RATE_SPREAD_BY_SCORE = {
        (750, 850): 0.01,  # 信用分750-850，利差1%
        (650, 750): 0.02,  # 信用分650-750，利差2%
        (600, 650): 0.03,  # 信用分600-650，利差3%
    }
    
    # 期限规则
    MIN_TERM_MONTHS = 6
    MAX_TERM_MONTHS = 60
    
    # 风险规则
    MAX_DEBT_RATIO = 0.7  # 最大负债率
    MAX_DEFAULT_PROB = 0.15  # 最大违约概率
```

#### 2.1.2 对公贷款规则

```python
class CorporateLoanRules:
    """对公贷款业务规则"""
    
    # 基础准入规则
    MIN_REGISTERED_CAPITAL = 1000000  # 最低注册资本
    MIN_OPERATING_YEARS = 1  # 最低经营年限
    MIN_ANNUAL_REVENUE = 5000000  # 最低年营收
    
    # 额度规则
    MAX_LOAN_AMOUNT = 10000000  # 最高额度
    MAX_LOAN_TO_REVENUE_RATIO = 0.3  # 贷款金额/年营收 <= 0.3
    
    # 利率规则
    BASE_RATE = 0.06  # 基准利率（对公通常更低）
    RATE_SPREAD_BY_RATING = {
        'AAA': 0.005,
        'AA': 0.01,
        'A': 0.015,
        'BBB': 0.02,
        'BB': 0.03,
    }
    
    # 期限规则
    MIN_TERM_MONTHS = 12
    MAX_TERM_MONTHS = 120
    
    # 风险规则
    MAX_DEBT_TO_ASSET_RATIO = 0.7  # 最大资产负债率
    MIN_CURRENT_RATIO = 1.2  # 最小流动比率
    REQUIRE_COLLATERAL = True  # 要求抵押
    MIN_COLLATERAL_RATIO = 0.5  # 最低抵押率
```

### 2.2 从历史数据中提取规则

#### 2.2.1 规则提取引擎

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ExtractedRule:
    """提取的规则"""
    rule_type: str  # 'threshold', 'range', 'ratio', 'conditional'
    field: str  # 字段名
    operator: str  # '>', '<', '>=', '<=', '==', 'in'
    value: any  # 阈值或值
    confidence: float  # 置信度
    support: float  # 支持度（满足该规则的数据比例）
    description: str  # 规则描述

class RuleExtractionEngine:
    """规则提取引擎"""
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        初始化规则提取引擎
        
        Args:
            historical_data: 历史贷款数据，包含：
                - 客户特征（收入、信用分、年龄等）
                - 审批决策（通过/拒绝）
                - 贷款条件（金额、利率、期限）
                - 实际结果（是否违约、利润等）
        """
        self.data = historical_data
        self.rules = []
    
    def extract_threshold_rules(self, field: str, target: str = 'approved'):
        """
        提取阈值规则
        
        例如：信用分 >= 600 的客户通过率更高
        """
        rules = []
        
        # 计算不同阈值下的通过率
        for threshold in np.percentile(self.data[field], [10, 20, 30, 40, 50, 60, 70, 80, 90]):
            # 大于阈值的数据
            above_threshold = self.data[self.data[field] >= threshold]
            # 小于阈值的数据
            below_threshold = self.data[self.data[field] < threshold]
            
            if len(above_threshold) > 0 and len(below_threshold) > 0:
                above_rate = above_threshold[target].mean()
                below_rate = below_threshold[target].mean()
                
                # 如果差异显著，提取规则
                if abs(above_rate - below_rate) > 0.1:  # 差异超过10%
                    if above_rate > below_rate:
                        rule = ExtractedRule(
                            rule_type='threshold',
                            field=field,
                            operator='>=',
                            value=threshold,
                            confidence=above_rate,
                            support=len(above_threshold) / len(self.data),
                            description=f'{field} >= {threshold:.2f} 时通过率 {above_rate:.2%}'
                        )
                    else:
                        rule = ExtractedRule(
                            rule_type='threshold',
                            field=field,
                            operator='<',
                            value=threshold,
                            confidence=below_rate,
                            support=len(below_threshold) / len(self.data),
                            description=f'{field} < {threshold:.2f} 时通过率 {below_rate:.2%}'
                        )
                    rules.append(rule)
        
        return rules
    
    def extract_range_rules(self, field: str, target: str = 'approved'):
        """
        提取范围规则
        
        例如：年龄在 25-55 之间的客户通过率最高
        """
        rules = []
        
        # 计算不同分位数的通过率
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        values = [np.percentile(self.data[field], p) for p in percentiles]
        
        best_range = None
        best_rate = 0
        
        # 寻找通过率最高的范围
        for i in range(len(values) - 1):
            for j in range(i + 1, len(values)):
                range_data = self.data[
                    (self.data[field] >= values[i]) & 
                    (self.data[field] <= values[j])
                ]
                
                if len(range_data) > len(self.data) * 0.1:  # 至少10%的数据
                    rate = range_data[target].mean()
                    if rate > best_rate:
                        best_rate = rate
                        best_range = (values[i], values[j])
        
        if best_range:
            rule = ExtractedRule(
                rule_type='range',
                field=field,
                operator='between',
                value=best_range,
                confidence=best_rate,
                support=len(self.data[
                    (self.data[field] >= best_range[0]) & 
                    (self.data[field] <= best_range[1])
                ]) / len(self.data),
                description=f'{field} 在 {best_range[0]:.2f}-{best_range[1]:.2f} 时通过率 {best_rate:.2%}'
            )
            rules.append(rule)
        
        return rules
    
    def extract_ratio_rules(self, field1: str, field2: str, target: str = 'approved'):
        """
        提取比例规则
        
        例如：贷款金额/月收入 <= 0.5 时通过率更高
        """
        rules = []
        
        # 计算比例
        ratio = self.data[field1] / (self.data[field2] + 1e-6)
        
        # 计算不同阈值下的通过率
        for threshold in np.percentile(ratio, [10, 20, 30, 40, 50, 60, 70, 80, 90]):
            below_threshold = self.data[ratio <= threshold]
            above_threshold = self.data[ratio > threshold]
            
            if len(below_threshold) > 0 and len(above_threshold) > 0:
                below_rate = below_threshold[target].mean()
                above_rate = above_threshold[target].mean()
                
                if abs(below_rate - above_rate) > 0.1:
                    if below_rate > above_rate:
                        rule = ExtractedRule(
                            rule_type='ratio',
                            field=f'{field1}/{field2}',
                            operator='<=',
                            value=threshold,
                            confidence=below_rate,
                            support=len(below_threshold) / len(self.data),
                            description=f'{field1}/{field2} <= {threshold:.2f} 时通过率 {below_rate:.2%}'
                        )
                    else:
                        rule = ExtractedRule(
                            rule_type='ratio',
                            field=f'{field1}/{field2}',
                            operator='>',
                            value=threshold,
                            confidence=above_rate,
                            support=len(above_threshold) / len(self.data),
                            description=f'{field1}/{field2} > {threshold:.2f} 时通过率 {above_rate:.2%}'
                        )
                    rules.append(rule)
        
        return rules
    
    def extract_conditional_rules(self, condition_fields: List[str], target: str = 'approved'):
        """
        提取条件规则（组合规则）
        
        例如：信用分 >= 700 AND 月收入 >= 10000 时通过率更高
        """
        rules = []
        
        # 使用决策树或关联规则挖掘
        from sklearn.tree import DecisionTreeClassifier
        
        X = self.data[condition_fields]
        y = self.data[target]
        
        # 训练决策树
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100)
        tree.fit(X, y)
        
        # 提取规则
        feature_names = condition_fields
        
        def extract_tree_rules(tree, node=0, depth=0, path=[]):
            """递归提取决策树规则"""
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                # 叶子节点
                samples = tree.tree_.n_node_samples[node]
                value = tree.tree_.value[node][0]
                predicted_class = np.argmax(value)
                confidence = value[predicted_class] / samples
                
                if confidence > 0.6 and samples > len(self.data) * 0.05:
                    rule = ExtractedRule(
                        rule_type='conditional',
                        field=' AND '.join([f'{f} {op} {val}' for f, op, val in path]),
                        operator='AND',
                        value=predicted_class,
                        confidence=confidence,
                        support=samples / len(self.data),
                        description=f'{" AND ".join([f"{f} {op} {val:.2f}" for f, op, val in path])} -> 通过率 {confidence:.2%}'
                    )
                    rules.append(rule)
            else:
                # 内部节点
                feature = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                feature_name = feature_names[feature]
                
                # 左子树（<= threshold）
                extract_tree_rules(
                    tree, 
                    tree.tree_.children_left[node], 
                    depth + 1, 
                    path + [(feature_name, '<=', threshold)]
                )
                
                # 右子树（> threshold）
                extract_tree_rules(
                    tree, 
                    tree.tree_.children_right[node], 
                    depth + 1, 
                    path + [(feature_name, '>', threshold)]
                )
        
        extract_tree_rules(tree)
        
        return rules
    
    def extract_all_rules(self, customer_type: str = 'personal'):
        """
        提取所有规则
        
        Args:
            customer_type: 'personal' 或 'corporate'
        """
        all_rules = []
        
        if customer_type == 'personal':
            # 对私贷款规则提取
            numeric_fields = ['age', 'monthly_income', 'credit_score', 'debt_ratio']
            for field in numeric_fields:
                if field in self.data.columns:
                    all_rules.extend(self.extract_threshold_rules(field))
                    all_rules.extend(self.extract_range_rules(field))
            
            # 比例规则
            if 'loan_amount' in self.data.columns and 'monthly_income' in self.data.columns:
                all_rules.extend(self.extract_ratio_rules('loan_amount', 'monthly_income'))
            
            # 条件规则
            all_rules.extend(self.extract_conditional_rules(numeric_fields))
        
        elif customer_type == 'corporate':
            # 对公贷款规则提取
            numeric_fields = ['registered_capital', 'operating_years', 'annual_revenue', 
                            'debt_to_asset_ratio', 'current_ratio']
            for field in numeric_fields:
                if field in self.data.columns:
                    all_rules.extend(self.extract_threshold_rules(field))
                    all_rules.extend(self.extract_range_rules(field))
            
            # 比例规则
            if 'loan_amount' in self.data.columns and 'annual_revenue' in self.data.columns:
                all_rules.extend(self.extract_ratio_rules('loan_amount', 'annual_revenue'))
            
            # 条件规则
            all_rules.extend(self.extract_conditional_rules(numeric_fields))
        
        # 按置信度和支持度排序
        all_rules.sort(key=lambda r: r.confidence * r.support, reverse=True)
        
        return all_rules
```

### 2.3 规则量化

```python
class RuleQuantifier:
    """规则量化器：将提取的规则转化为可计算的约束"""
    
    def __init__(self, rules: List[ExtractedRule]):
        self.rules = rules
    
    def quantify_rule(self, rule: ExtractedRule) -> Dict:
        """
        量化规则，返回可执行的约束函数
        
        Returns:
            {
                'check': callable,  # 检查函数
                'penalty': callable,  # 违反规则的惩罚函数
                'weight': float,  # 规则权重
            }
        """
        if rule.rule_type == 'threshold':
            if rule.operator == '>=':
                def check(state):
                    return state.get(rule.field, 0) >= rule.value
                def penalty(state):
                    if not check(state):
                        return -rule.confidence * 100  # 惩罚
                    return 0
            elif rule.operator == '<':
                def check(state):
                    return state.get(rule.field, 0) < rule.value
                def penalty(state):
                    if not check(state):
                        return -rule.confidence * 100
                    return 0
        
        elif rule.rule_type == 'range':
            min_val, max_val = rule.value
            def check(state):
                val = state.get(rule.field, 0)
                return min_val <= val <= max_val
            def penalty(state):
                if not check(state):
                    val = state.get(rule.field, 0)
                    if val < min_val:
                        return -rule.confidence * abs(val - min_val) * 10
                    else:
                        return -rule.confidence * abs(val - max_val) * 10
                return 0
        
        elif rule.rule_type == 'ratio':
            field1, field2 = rule.field.split('/')
            if rule.operator == '<=':
                def check(state):
                    val1 = state.get(field1, 0)
                    val2 = state.get(field2, 1)
                    return (val1 / val2) <= rule.value
                def penalty(state):
                    if not check(state):
                        val1 = state.get(field1, 0)
                        val2 = state.get(field2, 1)
                        ratio = val1 / val2
                        return -rule.confidence * (ratio - rule.value) * 1000
                    return 0
        
        elif rule.rule_type == 'conditional':
            # 解析条件表达式
            conditions = rule.field.split(' AND ')
            def check(state):
                for cond in conditions:
                    # 解析条件（简化版）
                    if '<=' in cond:
                        field, val = cond.split('<=')
                        if state.get(field.strip(), 0) > float(val.strip()):
                            return False
                    elif '>' in cond:
                        field, val = cond.split('>')
                        if state.get(field.strip(), 0) <= float(val.strip()):
                            return False
                return True
            def penalty(state):
                if not check(state):
                    return -rule.confidence * 200
                return 0
        
        return {
            'check': check,
            'penalty': penalty,
            'weight': rule.confidence * rule.support,
            'description': rule.description
        }
    
    def quantify_all_rules(self) -> List[Dict]:
        """量化所有规则"""
        return [self.quantify_rule(rule) for rule in self.rules]
```

---

## 三、规则集成到 AlphaZero

### 3.1 带规则约束的环境

```python
class RuleConstrainedLoanEnvironment:
    """带规则约束的贷款审批环境"""
    
    def __init__(self, rules: List[Dict], customer_type: str = 'personal'):
        """
        Args:
            rules: 量化后的规则列表
            customer_type: 'personal' 或 'corporate'
        """
        self.rules = rules
        self.customer_type = customer_type
        self.customer_generator = CustomerGenerator()
        self.world_model = WorldModel()
        self.portfolio = Portfolio()
    
    def check_rules(self, state: Dict, action: Dict) -> Tuple[bool, float]:
        """
        检查规则约束
        
        Returns:
            (is_valid, total_penalty)
        """
        total_penalty = 0.0
        violated_rules = []
        
        # 合并状态和动作
        full_state = {**state, **action}
        
        for rule in self.rules:
            if not rule['check'](full_state):
                penalty = rule['penalty'](full_state)
                total_penalty += penalty * rule['weight']
                violated_rules.append(rule['description'])
        
        is_valid = total_penalty == 0.0
        
        return is_valid, total_penalty, violated_rules
    
    def step(self, state: Dict, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作（带规则检查）
        
        Returns:
            (next_state, reward, done, info)
        """
        # 检查规则约束
        is_valid, penalty, violated_rules = self.check_rules(state, action)
        
        # 如果违反规则，给予严重惩罚
        if not is_valid:
            reward = penalty
            info = {
                'violated_rules': violated_rules,
                'action_valid': False
            }
            # 仍然生成下一个客户（但当前动作被拒绝）
            next_customer = self.customer_generator.generate()
            next_state = self.extract_state(next_customer)
            return next_state, reward, False, info
        
        # 规则通过，执行动作
        if action['decision'] == 'approve':
            loan = self.create_loan(action)
            outcome = self.world_model.simulate(loan)
            
            # 计算基础奖励
            base_reward = self.calculate_reward(action, outcome)
            
            # 规则奖励（遵守规则有额外奖励）
            rule_reward = sum([rule['weight'] * 10 for rule in self.rules 
                             if rule['check']({**state, **action})])
            
            reward = base_reward + rule_reward
            
            self.portfolio.add_loan(loan)
        else:
            reward = 0
        
        # 生成下一个客户
        next_customer = self.customer_generator.generate()
        next_state = self.extract_state(next_customer)
        
        info = {
            'violated_rules': [],
            'action_valid': True
        }
        
        return next_state, reward, False, info
```

### 3.2 规则感知的奖励函数

```python
class RuleAwareRewardFunction:
    """规则感知的奖励函数"""
    
    def __init__(self, rules: List[Dict]):
        self.rules = rules
    
    def calculate_reward(self, state: Dict, action: Dict, outcome: Dict) -> float:
        """
        计算奖励（考虑规则约束）
        """
        # 基础奖励（利润）
        if action['decision'] == 'approve':
            interest_income = action['loan_amount'] * action['interest_rate']
            default_loss = action['loan_amount'] * outcome['default_probability']
            base_reward = interest_income * (1 - outcome['default_probability']) - default_loss
        else:
            base_reward = 0
        
        # 规则奖励/惩罚
        rule_reward = 0.0
        for rule in self.rules:
            full_state = {**state, **action}
            if rule['check'](full_state):
                # 遵守规则有奖励
                rule_reward += rule['weight'] * 10
            else:
                # 违反规则有惩罚
                penalty = rule['penalty'](full_state)
                rule_reward += penalty * rule['weight']
        
        # 总奖励
        total_reward = base_reward + rule_reward
        
        return total_reward
```

---

## 四、多模型博弈框架

### 4.1 多模型 AlphaZero 环境

```python
class MultiModelAlphaZeroArena:
    """多模型 AlphaZero 博弈场"""
    
    def __init__(self, models: List[str], rules: List[Dict], customer_type: str = 'personal'):
        """
        Args:
            models: 模型ID列表，例如 ['claude-3-opus', 'gpt-4', 'qwen-plus']
            rules: 量化后的业务规则
            customer_type: 'personal' 或 'corporate'
        """
        self.models = models
        self.rules = rules
        self.customer_type = customer_type
        
        # 为每个模型创建环境
        self.environments = {
            model_id: RuleConstrainedLoanEnvironment(rules, customer_type)
            for model_id in models
        }
        
        # 为每个模型创建网络
        self.networks = {
            model_id: AlphaZeroNetwork(input_dim, action_dim)
            for model_id in models
        }
        
        # 模型网关（用于调用真实LLM）
        self.model_gateway = ModelGateway()
    
    def self_play_round(self, model_id: str, num_games: int = 100):
        """
        单个模型的自我对弈
        
        Args:
            model_id: 模型ID
            num_games: 对弈局数
        """
        env = self.environments[model_id]
        network = self.networks[model_id]
        mcts = MCTS(network, env)
        
        games = []
        
        for game in range(num_games):
            state = env.reset()
            game_history = []
            
            while not env.is_terminal():
                # 使用MCTS搜索
                improved_policy = mcts.search(state)
                
                # 选择动作
                action = sample_action(improved_policy)
                
                # 执行动作（带规则检查）
                next_state, reward, done, info = env.step(state, action)
                
                # 记录
                game_history.append({
                    'state': state,
                    'policy': improved_policy,
                    'action': action,
                    'reward': reward,
                    'violated_rules': info.get('violated_rules', [])
                })
                
                state = next_state
            
            # 计算回报
            returns = calculate_returns(game_history)
            
            # 添加到训练数据
            for i, step in enumerate(game_history):
                games.append({
                    'state': step['state'],
                    'policy': step['policy'],
                    'value': returns[i],
                    'model_id': model_id
                })
        
        return games
    
    def cross_model_evaluation(self, num_games: int = 50):
        """
        跨模型评估：让不同模型在相同客户集上博弈
        
        Returns:
            Dict[model_id, performance]
        """
        results = {}
        
        # 生成统一的客户集
        customer_generator = CustomerGenerator()
        test_customers = [customer_generator.generate() for _ in range(num_games)]
        
        for model_id in self.models:
            env = self.environments[model_id]
            network = self.networks[model_id]
            mcts = MCTS(network, env)
            
            total_reward = 0.0
            total_violations = 0
            approved_count = 0
            
            for customer in test_customers:
                state = env.extract_state(customer)
                
                # 使用MCTS搜索最优动作
                improved_policy = mcts.search(state)
                action = sample_action(improved_policy, temperature=0.1)  # 低温度，更确定
                
                # 执行动作
                next_state, reward, done, info = env.step(state, action)
                
                total_reward += reward
                total_violations += len(info.get('violated_rules', []))
                if action['decision'] == 'approve':
                    approved_count += 1
            
            results[model_id] = {
                'avg_reward': total_reward / num_games,
                'approval_rate': approved_count / num_games,
                'avg_violations': total_violations / num_games,
                'performance_score': total_reward / num_games - total_violations * 10
            }
        
        return results
    
    def train_all_models(self, iterations: int = 100):
        """
        训练所有模型
        
        Args:
            iterations: 训练迭代次数
        """
        for iteration in range(iterations):
            print(f"Iteration {iteration}")
            
            # 1. 每个模型自我对弈
            all_games = []
            for model_id in self.models:
                print(f"  Self-playing {model_id}...")
                games = self.self_play_round(model_id, num_games=50)
                all_games.extend(games)
            
            # 2. 训练每个模型
            for model_id in self.models:
                print(f"  Training {model_id}...")
                model_games = [g for g in all_games if g['model_id'] == model_id]
                self.train_model(model_id, model_games)
            
            # 3. 跨模型评估
            if iteration % 10 == 0:
                print(f"  Cross-model evaluation...")
                results = self.cross_model_evaluation()
                print(f"  Results: {results}")
    
    def train_model(self, model_id: str, games: List[Dict]):
        """训练单个模型"""
        network = self.networks[model_id]
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        
        for epoch in range(10):
            for batch in create_batches(games, batch_size=32):
                states = batch['states']
                policy_targets = batch['policies']
                value_targets = batch['values']
                
                # 前向传播
                policy_pred, value_pred = network(states)
                
                # 计算损失
                loss, policy_loss, value_loss = alpha_zero_loss(
                    policy_pred, value_pred,
                    policy_targets, value_targets
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

---

## 五、完整实施流程

### 5.1 步骤一：历史数据准备

```python
# 1. 加载历史数据
historical_data = pd.read_csv('historical_loans.csv')

# 2. 数据预处理
historical_data = preprocess_data(historical_data)

# 3. 按客户类型分组
personal_data = historical_data[historical_data['customer_type'] == 'personal']
corporate_data = historical_data[historical_data['customer_type'] == 'corporate']
```

### 5.2 步骤二：规则提取

```python
# 1. 对私贷款规则提取
personal_engine = RuleExtractionEngine(personal_data)
personal_rules = personal_engine.extract_all_rules(customer_type='personal')

# 2. 对公贷款规则提取
corporate_engine = RuleExtractionEngine(corporate_data)
corporate_rules = corporate_engine.extract_all_rules(customer_type='corporate')

# 3. 规则量化
personal_quantifier = RuleQuantifier(personal_rules)
quantified_personal_rules = personal_quantifier.quantify_all_rules()

corporate_quantifier = RuleQuantifier(corporate_rules)
quantified_corporate_rules = corporate_quantifier.quantify_all_rules()
```

### 5.3 步骤三：创建多模型博弈场

```python
# 1. 定义要对比的模型
models = ['claude-3-opus', 'gpt-4', 'qwen-plus', 'gemini-pro']

# 2. 创建对私贷款博弈场
personal_arena = MultiModelAlphaZeroArena(
    models=models,
    rules=quantified_personal_rules,
    customer_type='personal'
)

# 3. 创建对公贷款博弈场
corporate_arena = MultiModelAlphaZeroArena(
    models=models,
    rules=quantified_corporate_rules,
    customer_type='corporate'
)
```

### 5.4 步骤四：训练和评估

```python
# 1. 训练对私贷款模型
print("Training personal loan models...")
personal_arena.train_all_models(iterations=100)

# 2. 训练对公贷款模型
print("Training corporate loan models...")
corporate_arena.train_all_models(iterations=100)

# 3. 最终评估
print("Final evaluation...")
personal_results = personal_arena.cross_model_evaluation(num_games=1000)
corporate_results = corporate_arena.cross_model_evaluation(num_games=1000)

# 4. 选择最优模型
best_personal_model = max(personal_results.items(), key=lambda x: x[1]['performance_score'])
best_corporate_model = max(corporate_results.items(), key=lambda x: x[1]['performance_score'])

print(f"Best personal loan model: {best_personal_model[0]}")
print(f"Best corporate loan model: {best_corporate_model[0]}")
```

---

## 六、规则优化和学习

### 6.1 规则动态调整

```python
class AdaptiveRuleEngine:
    """自适应规则引擎：根据模型表现动态调整规则"""
    
    def __init__(self, initial_rules: List[Dict]):
        self.rules = initial_rules
        self.rule_performance = {i: [] for i in range(len(initial_rules))}
    
    def update_rule_weights(self, model_performance: Dict):
        """
        根据模型表现更新规则权重
        
        Args:
            model_performance: {
                'model_id': {
                    'avg_reward': float,
                    'avg_violations': float,
                    'rule_violations': Dict[rule_id, count]
                }
            }
        """
        # 分析哪些规则被违反最多
        violation_counts = {}
        for model_id, perf in model_performance.items():
            for rule_id, count in perf.get('rule_violations', {}).items():
                violation_counts[rule_id] = violation_counts.get(rule_id, 0) + count
        
        # 调整规则权重
        for rule_id, rule in enumerate(self.rules):
            violation_count = violation_counts.get(rule_id, 0)
            
            if violation_count > 0:
                # 如果规则经常被违反，可能需要调整阈值
                # 或者增加惩罚权重
                rule['weight'] *= 1.1  # 增加10%权重
            else:
                # 如果规则很少被违反，可能太严格
                # 可以适当降低权重
                rule['weight'] *= 0.99
    
    def learn_new_rules(self, successful_decisions: List[Dict]):
        """
        从成功的决策中学习新规则
        
        Args:
            successful_decisions: 成功的决策列表，包含状态和动作
        """
        # 分析成功决策的共同特征
        # 提取新的规则模式
        new_rules = extract_patterns(successful_decisions)
        
        # 添加到规则库
        for new_rule in new_rules:
            self.rules.append(new_rule)
```

---

## 七、总结

### 7.1 核心要点

1. **规则提取**：从历史数据中自动提取业务规则
2. **规则量化**：将规则转化为可计算的约束
3. **规则集成**：在AlphaZero环境中集成规则约束
4. **多模型博弈**：让不同模型在相同规则下博弈
5. **规则优化**：根据模型表现动态调整规则

### 7.2 优势

- ✅ **可解释性**：规则明确，决策可解释
- ✅ **合规性**：规则确保符合业务要求
- ✅ **公平性**：所有模型在相同规则下博弈
- ✅ **可优化**：规则可以根据表现动态调整

### 7.3 实施建议

1. **先提取规则**：从历史数据中提取基础规则
2. **量化规则**：将规则转化为约束函数
3. **集成环境**：在AlphaZero环境中集成规则
4. **多模型训练**：让不同模型在规则约束下训练
5. **评估对比**：在相同客户集上评估各模型表现
6. **规则优化**：根据结果优化规则权重和阈值

