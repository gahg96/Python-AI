"""
模型决策模块
实现基于风险评分和预期利润的决策逻辑
"""
import pickle
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DecisionResult:
    """决策结果"""
    decision: str  # 'approve', 'reject', 'manual_review'
    confidence: float  # 决策置信度
    default_probability: float  # 预测违约概率
    expected_profit: float  # 预期利润
    risk_score: float  # 风险评分
    profit_score: float  # 利润评分
    reasoning: str  # 决策理由


class ModelDecisionMaker:
    """模型决策器"""
    
    def __init__(self, models_dir: str = 'data/historical/models'):
        """
        初始化决策器
        
        Args:
            models_dir: 模型文件目录
        """
        self.models_dir = models_dir
        self.default_model = None
        self.profit_model = None
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型"""
        default_model_path = os.path.join(self.models_dir, 'default_prediction.pkl')
        profit_model_path = os.path.join(self.models_dir, 'profit_prediction.pkl')
        
        if os.path.exists(default_model_path):
            with open(default_model_path, 'rb') as f:
                self.default_model = pickle.load(f)
            print(f"✅ 已加载违约预测模型")
        else:
            print("⚠️  违约预测模型文件不存在")
        
        if os.path.exists(profit_model_path):
            with open(profit_model_path, 'rb') as f:
                self.profit_model = pickle.load(f)
            print(f"✅ 已加载利润预测模型")
        else:
            print("⚠️  利润预测模型文件不存在")
    
    def predict_default_probability(self, customer: Dict, loan: Dict, 
                                   market: Dict) -> float:
        """
        预测违约概率
        
        Args:
            customer: 客户数据
            loan: 贷款数据
            market: 市场数据
        
        Returns:
            违约概率
        """
        if self.default_model is None:
            # 如果没有模型，使用简单规则估算
            return self._estimate_default_probability_simple(customer, loan, market)
        
        # 构建特征向量
        features = self._build_features(customer, loan, market)
        
        # 预测
        try:
            proba = self.default_model.predict_proba([features])[0, 1]
            return float(proba)
        except Exception as e:
            print(f"⚠️  模型预测失败: {e}，使用简单估算")
            return self._estimate_default_probability_simple(customer, loan, market)
    
    def predict_expected_profit(self, customer: Dict, loan: Dict, 
                               market: Dict, default_prob: float) -> float:
        """
        预测预期利润
        
        Args:
            customer: 客户数据
            loan: 贷款数据
            market: 市场数据
            default_prob: 违约概率
        
        Returns:
            预期利润
        """
        if self.profit_model is None:
            # 如果没有模型，使用简单计算
            return self._calculate_expected_profit_simple(customer, loan, market, default_prob)
        
        # 构建特征向量
        features = self._build_profit_features(customer, loan, market)
        
        # 预测
        try:
            profit = self.profit_model.predict([features])[0]
            # 根据违约概率调整
            adjusted_profit = profit * (1 - default_prob)
            return float(adjusted_profit)
        except Exception as e:
            print(f"⚠️  利润预测失败: {e}，使用简单计算")
            return self._calculate_expected_profit_simple(customer, loan, market, default_prob)
    
    def _estimate_default_probability_simple(self, customer: Dict, loan: Dict, 
                                            market: Dict) -> float:
        """简单估算违约概率"""
        base_prob = 0.1
        
        if customer.get('customer_type') == 'personal':
            # 信用分影响
            credit_score = customer.get('credit_score', 650)
            credit_factor = 1 - (credit_score - 300) / 550 * 0.5
            
            # 负债率影响
            debt_ratio = customer.get('debt_ratio', 0.5)
            debt_factor = 1 + debt_ratio * 0.5
            
            # 收入稳定性
            years_in_job = customer.get('years_in_job', 5)
            stability_factor = 1 - min(years_in_job / 10, 0.3)
            
            default_prob = base_prob * credit_factor * debt_factor * stability_factor
        else:
            # 对公客户
            operating_years = customer.get('operating_years', 5)
            years_factor = 1 - min(operating_years / 10, 0.3)
            
            debt_to_asset = customer.get('debt_to_asset_ratio', 0.6)
            debt_factor = 1 + (debt_to_asset - 0.5) * 0.5
            
            default_prob = base_prob * years_factor * debt_factor
        
        # 市场环境影响
        unemployment = market.get('unemployment_rate', 0.05)
        market_factor = 1 + unemployment * 2
        
        default_prob *= market_factor
        return max(0.01, min(default_prob, 0.5))
    
    def _calculate_expected_profit_simple(self, customer: Dict, loan: Dict, 
                                         market: Dict, default_prob: float) -> float:
        """简单计算预期利润"""
        loan_amount = loan.get('loan_amount', 0)
        rate = loan.get('approved_rate', 0.08)
        term_months = loan.get('approved_term_months', 12)
        
        # 预期利息收入
        expected_interest = loan_amount * rate * (term_months / 12)
        
        # 预期损失（违约概率 × 违约损失）
        expected_loss = loan_amount * default_prob * 0.5  # 假设违约损失50%
        
        # 预期利润
        expected_profit = expected_interest - expected_loss
        
        return expected_profit
    
    def _build_features(self, customer: Dict, loan: Dict, market: Dict) -> np.ndarray:
        """构建特征向量（用于违约预测）"""
        # 这里需要与训练时的特征顺序一致
        # 简化处理，使用常见特征
        features = []
        
        # 客户特征
        features.append(customer.get('age', 35))
        features.append(customer.get('monthly_income', 8000))
        features.append(customer.get('credit_score', 650))
        features.append(customer.get('debt_ratio', 0.5))
        features.append(customer.get('years_in_job', 5))
        
        # 贷款特征
        features.append(loan.get('loan_amount', 50000))
        features.append(loan.get('requested_term_months', 12))
        
        # 市场特征
        features.append(market.get('gdp_growth', 0.03))
        features.append(market.get('base_interest_rate', 0.05))
        features.append(market.get('unemployment_rate', 0.05))
        features.append(market.get('inflation_rate', 0.02))
        features.append(market.get('credit_spread', 0.02))
        
        return np.array(features)
    
    def _build_profit_features(self, customer: Dict, loan: Dict, market: Dict) -> np.ndarray:
        """构建特征向量（用于利润预测）"""
        features = []
        
        # 客户特征
        features.append(customer.get('age', 35))
        features.append(customer.get('monthly_income', 8000))
        features.append(customer.get('credit_score', 650))
        features.append(customer.get('debt_ratio', 0.5))
        features.append(customer.get('years_in_job', 5))
        
        # 贷款特征
        features.append(loan.get('loan_amount', 50000))
        features.append(loan.get('approved_rate', 0.08))
        features.append(loan.get('approved_term_months', 12))
        
        # 市场特征
        features.append(market.get('gdp_growth', 0.03))
        features.append(market.get('base_interest_rate', 0.05))
        features.append(market.get('unemployment_rate', 0.05))
        features.append(market.get('credit_spread', 0.02))
        
        return np.array(features)
    
    def make_decision(self, customer: Dict, loan: Dict, market: Dict,
                     approval_threshold: float = 0.18,
                     min_profit: float = 0.0,
                     risk_tolerance: float = 0.15) -> DecisionResult:
        """
        做出审批决策
        
        Args:
            customer: 客户数据
            loan: 贷款数据
            market: 市场数据
            approval_threshold: 审批阈值（违约概率）
            min_profit: 最小利润要求
            risk_tolerance: 风险容忍度
        
        Returns:
            决策结果
        """
        # 1. 预测违约概率
        default_prob = self.predict_default_probability(customer, loan, market)
        
        # 2. 预测预期利润
        expected_profit = self.predict_expected_profit(customer, loan, market, default_prob)
        
        # 3. 计算风险评分（0-1，越高风险越大）
        risk_score = default_prob
        
        # 4. 计算利润评分（归一化到0-1）
        loan_amount = loan.get('loan_amount', 50000)
        profit_score = min(expected_profit / (loan_amount * 0.1), 1.0) if loan_amount > 0 else 0
        profit_score = max(0, profit_score)
        
        # 5. 决策逻辑
        decision = 'reject'
        confidence = 0.0
        reasoning = ""
        
        # 强制拒绝条件
        if default_prob > approval_threshold:
            decision = 'reject'
            confidence = 1.0 - default_prob
            reasoning = f"违约概率 {default_prob:.2%} 超过阈值 {approval_threshold:.2%}"
        
        # 强制通过条件
        elif default_prob < approval_threshold * 0.6 and expected_profit > min_profit:
            decision = 'approve'
            confidence = 1.0 - default_prob
            reasoning = f"违约概率 {default_prob:.2%} 较低，预期利润 ¥{expected_profit:,.2f} 满足要求"
        
        # 需要人工审核
        elif default_prob < approval_threshold and expected_profit > min_profit * 0.5:
            decision = 'manual_review'
            confidence = 0.5
            reasoning = f"违约概率 {default_prob:.2%} 接近阈值，预期利润 ¥{expected_profit:,.2f}，建议人工审核"
        
        # 拒绝（利润不足）
        elif expected_profit < min_profit:
            decision = 'reject'
            confidence = 0.7
            reasoning = f"预期利润 ¥{expected_profit:,.2f} 低于最小要求 ¥{min_profit:,.2f}"
        
        # 拒绝（风险过高）
        else:
            decision = 'reject'
            confidence = 0.8
            reasoning = f"风险评分 {risk_score:.2%} 超过容忍度 {risk_tolerance:.2%}"
        
        return DecisionResult(
            decision=decision,
            confidence=confidence,
            default_probability=default_prob,
            expected_profit=expected_profit,
            risk_score=risk_score,
            profit_score=profit_score,
            reasoning=reasoning
        )


def main():
    """主函数：测试模型决策"""
    print("=" * 80)
    print("模型决策模块测试")
    print("=" * 80)
    
    # 创建决策器
    decision_maker = ModelDecisionMaker()
    
    # 测试用例1：优质客户
    print("\n测试用例1：优质客户")
    customer1 = {
        'customer_type': 'personal',
        'age': 35,
        'monthly_income': 15000,
        'credit_score': 750,
        'debt_ratio': 0.3,
        'years_in_job': 8
    }
    loan1 = {
        'loan_amount': 100000,
        'approved_rate': 0.08,
        'approved_term_months': 24
    }
    market1 = {
        'gdp_growth': 0.03,
        'base_interest_rate': 0.05,
        'unemployment_rate': 0.05,
        'inflation_rate': 0.02,
        'credit_spread': 0.02
    }
    
    result1 = decision_maker.make_decision(customer1, loan1, market1)
    print(f"决策: {result1.decision}")
    print(f"置信度: {result1.confidence:.2%}")
    print(f"违约概率: {result1.default_probability:.2%}")
    print(f"预期利润: ¥{result1.expected_profit:,.2f}")
    print(f"理由: {result1.reasoning}")
    
    # 测试用例2：高风险客户
    print("\n测试用例2：高风险客户")
    customer2 = {
        'customer_type': 'personal',
        'age': 25,
        'monthly_income': 5000,
        'credit_score': 550,
        'debt_ratio': 0.7,
        'years_in_job': 1
    }
    loan2 = {
        'loan_amount': 50000,
        'approved_rate': 0.12,
        'approved_term_months': 36
    }
    
    result2 = decision_maker.make_decision(customer2, loan2, market1)
    print(f"决策: {result2.decision}")
    print(f"置信度: {result2.confidence:.2%}")
    print(f"违约概率: {result2.default_probability:.2%}")
    print(f"预期利润: ¥{result2.expected_profit:,.2f}")
    print(f"理由: {result2.reasoning}")
    
    return decision_maker


if __name__ == '__main__':
    main()

