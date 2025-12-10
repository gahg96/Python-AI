"""
评分系统模块
用于计算和分解演武场参赛者的综合得分
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ScoreBreakdown:
    """评分分解"""
    profit_score: float = 0.0  # 利润得分
    risk_score: float = 0.0  # 风险得分
    stability_score: float = 0.0  # 稳定性得分
    compliance_score: float = 0.0  # 合规得分
    efficiency_score: float = 0.0  # 效率得分
    explainability_score: float = 0.0  # 可解释性得分
    overall_score: float = 0.0  # 综合得分
    
    # 详细指标
    profit_amount: float = 0.0  # 利润金额
    raroc: float = 0.0  # 风险调整后收益
    default_rate: float = 0.0  # 违约率
    max_drawdown: float = 0.0  # 最大回撤
    recovery_time: float = 0.0  # 恢复时间
    profit_volatility: float = 0.0  # 利润波动率
    compliance_violations: int = 0  # 合规违规次数
    avg_latency: float = 0.0  # 平均延迟
    rule_coverage: float = 0.0  # 规则覆盖率


class ScoringSystem:
    """评分系统"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        初始化评分系统
        
        Args:
            weights: 各维度权重，默认值：
                profit: 0.30, risk: 0.30, stability: 0.10,
                compliance: 0.20, efficiency: 0.05, explainability: 0.05
        """
        self.weights = weights or {
            'profit': 0.30,
            'risk': 0.30,
            'stability': 0.10,
            'compliance': 0.20,
            'efficiency': 0.05,
            'explainability': 0.05
        }
        # 归一化权重
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def normalize_score(self, value: float, min_val: float, max_val: float, reverse: bool = False) -> float:
        """
        归一化得分到 [0, 1] 区间
        
        Args:
            value: 原始值
            min_val: 最小值
            max_val: 最大值
            reverse: 是否反向（值越大得分越低）
        
        Returns:
            归一化后的得分 [0, 1]
        """
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        
        if reverse:
            normalized = 1.0 - normalized
        
        return normalized
    
    def calculate_profit_score(self, profit: float, max_profit: float, min_profit: float = 0.0) -> float:
        """计算利润得分"""
        if max_profit <= min_profit:
            return 0.5
        return self.normalize_score(profit, min_profit, max_profit, reverse=False)
    
    def calculate_risk_score(self, default_rate: float, max_default_rate: float = 0.5) -> float:
        """计算风险得分（违约率越低得分越高）"""
        return self.normalize_score(default_rate, 0.0, max_default_rate, reverse=True)
    
    def calculate_stability_score(self, volatility: float, max_volatility: float = 1.0,
                                  drawdown: float = 0.0, max_drawdown: float = 1.0) -> float:
        """计算稳定性得分（波动率和回撤越低得分越高）"""
        vol_score = self.normalize_score(volatility, 0.0, max_volatility, reverse=True)
        dd_score = self.normalize_score(drawdown, 0.0, max_drawdown, reverse=True)
        return (vol_score + dd_score) / 2.0
    
    def calculate_compliance_score(self, violations: int, total_decisions: int) -> float:
        """计算合规得分（违规次数越少得分越高）"""
        if total_decisions == 0:
            return 1.0
        violation_rate = violations / total_decisions
        return self.normalize_score(violation_rate, 0.0, 0.1, reverse=True)  # 违规率超过10%得0分
    
    def calculate_efficiency_score(self, avg_latency: float, max_latency: float = 5.0) -> float:
        """计算效率得分（延迟越低得分越高）"""
        return self.normalize_score(avg_latency, 0.0, max_latency, reverse=True)
    
    def calculate_explainability_score(self, rule_coverage: float, 
                                      triggered_rules_count: int, 
                                      total_rules_count: int) -> float:
        """计算可解释性得分（规则覆盖率越高得分越高）"""
        if total_rules_count == 0:
            return 1.0 if rule_coverage > 0 else 0.5
        rule_usage = triggered_rules_count / total_rules_count
        return (rule_coverage + rule_usage) / 2.0
    
    def calculate_overall_score(self, breakdown: ScoreBreakdown, 
                                all_results: List[Dict[str, Any]] = None) -> float:
        """
        计算综合得分
        
        Args:
            breakdown: 评分分解
            all_results: 所有参赛者的结果（用于归一化）
        
        Returns:
            综合得分 [0, 1]
        """
        # 如果没有其他结果，使用默认范围
        if not all_results:
            profit_score = self.calculate_profit_score(breakdown.profit_amount, 1e6, 0.0)
            risk_score = self.calculate_risk_score(breakdown.default_rate, 0.5)
            stability_score = self.calculate_stability_score(
                breakdown.profit_volatility, 1.0,
                breakdown.max_drawdown, 1.0
            )
            compliance_score = self.calculate_compliance_score(
                breakdown.compliance_violations, 100
            )
            efficiency_score = self.calculate_efficiency_score(breakdown.avg_latency, 5.0)
            explainability_score = self.calculate_explainability_score(
                breakdown.rule_coverage, 0, 10
            )
        else:
            # 从所有结果中获取最大值和最小值用于归一化
            profits = [r.get('est_profit', 0) for r in all_results]
            default_rates = [r.get('avg_default_prob', 0) for r in all_results]
            volatilities = [r.get('profit_volatility', 0) for r in all_results]
            drawdowns = [r.get('max_drawdown', 0) for r in all_results]
            violations = [r.get('compliance_violations', 0) for r in all_results]
            latencies = [r.get('avg_latency', 0) for r in all_results]
            
            max_profit = max(profits) if profits else 1e6
            min_profit = min(profits) if profits else 0.0
            max_default = max(default_rates) if default_rates else 0.5
            max_vol = max(volatilities) if volatilities else 1.0
            max_dd = max(drawdowns) if drawdowns else 1.0
            total_decisions = sum([r.get('sample_size', 0) for r in all_results])
            max_latency = max(latencies) if latencies else 5.0
            
            profit_score = self.calculate_profit_score(breakdown.profit_amount, max_profit, min_profit)
            risk_score = self.calculate_risk_score(breakdown.default_rate, max_default)
            stability_score = self.calculate_stability_score(
                breakdown.profit_volatility, max_vol,
                breakdown.max_drawdown, max_dd
            )
            compliance_score = self.calculate_compliance_score(
                breakdown.compliance_violations, total_decisions
            )
            efficiency_score = self.calculate_efficiency_score(breakdown.avg_latency, max_latency)
            explainability_score = self.calculate_explainability_score(
                breakdown.rule_coverage, breakdown.compliance_violations, 10
            )
        
        # 加权求和
        overall = (
            profit_score * self.weights['profit'] +
            risk_score * self.weights['risk'] +
            stability_score * self.weights['stability'] +
            compliance_score * self.weights['compliance'] +
            efficiency_score * self.weights['efficiency'] +
            explainability_score * self.weights['explainability']
        )
        
        breakdown.profit_score = profit_score
        breakdown.risk_score = risk_score
        breakdown.stability_score = stability_score
        breakdown.compliance_score = compliance_score
        breakdown.efficiency_score = efficiency_score
        breakdown.explainability_score = explainability_score
        breakdown.overall_score = overall
        
        return overall
    
    def create_score_breakdown(self, result: Dict[str, Any], 
                              triggered_rules: List[str] = None,
                              all_results: List[Dict[str, Any]] = None) -> ScoreBreakdown:
        """
        创建评分分解
        
        Args:
            result: 单个参赛者的结果
            triggered_rules: 触发的规则列表
            all_results: 所有参赛者的结果（用于归一化）
        
        Returns:
            评分分解对象
        """
        breakdown = ScoreBreakdown()
        
        # 基础指标
        breakdown.profit_amount = result.get('est_profit', 0.0)
        breakdown.raroc = result.get('raroc', 0.0)
        breakdown.default_rate = result.get('avg_default_prob', 0.0)
        breakdown.max_drawdown = result.get('max_drawdown', 0.0)
        breakdown.recovery_time = result.get('recovery_time', 0.0)
        breakdown.profit_volatility = result.get('profit_volatility', 0.0)
        breakdown.compliance_violations = result.get('compliance_violations', 0)
        breakdown.avg_latency = result.get('avg_latency', 0.0)
        
        # 规则覆盖率
        total_rules = result.get('total_rules_count', 0)
        triggered_count = len(triggered_rules) if triggered_rules else 0
        breakdown.rule_coverage = triggered_count / total_rules if total_rules > 0 else 0.0
        
        # 计算综合得分
        self.calculate_overall_score(breakdown, all_results)
        
        return breakdown

