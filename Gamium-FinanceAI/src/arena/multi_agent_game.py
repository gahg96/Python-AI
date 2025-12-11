"""
多智能体博弈模块
支持同池竞争客户、竞价利差/额度、锦标赛或积分赛
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from data_distillation.world_model import WorldModel, MarketConditions, LoanOffer
from data_distillation.customer_generator import CustomerGenerator
from arena.rule_engine import RuleEngine
from arena.scoring_system import ScoringSystem


@dataclass
class AgentDecision:
    """智能体决策"""
    agent_id: str
    customer_id: str
    decision: str  # approve / reject
    interest_rate: float
    loan_amount: float
    term_months: int
    estimated_profit: float
    default_probability: float
    risk_factors: Dict[str, Any]
    compliance_status: bool
    reasoning: str  # 决策理由


@dataclass
class RoundResult:
    """单轮博弈结果"""
    round_number: int
    customer_id: str
    customer_profile: Dict[str, Any]
    agent_decisions: List[AgentDecision]
    winner: Optional[str]  # 获得客户的智能体ID
    market_conditions: Dict[str, float]


class MultiAgentGame:
    """多智能体博弈系统"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.world_model = WorldModel(seed=seed)
        self.generator = CustomerGenerator(seed=seed)
        self.round_history: List[RoundResult] = []
    
    def simulate_competitive_round(self, round_num: int, customer: Any,
                                   agents: List[Dict], market: MarketConditions,
                                   base_rate: float, base_loan_amount: float,
                                   rule_engine: Optional[RuleEngine] = None) -> RoundResult:
        """
        模拟竞争性回合：多个智能体竞争同一个客户
        
        Args:
            round_num: 回合编号
            customer: 客户对象
            agents: 智能体列表，每个包含 {id, name, strategy, approval_threshold, rate_spread, ...}
            market: 市场条件
            base_rate: 基准利率
            base_loan_amount: 基准贷款金额
            rule_engine: 规则引擎
        
        Returns:
            回合结果
        """
        agent_decisions = []
        
        # 每个智能体做出决策
        for agent in agents:
            agent_id = agent.get('id', agent.get('name', 'unknown'))
            name = agent.get('name', agent_id)
            threshold = float(agent.get('approval_threshold', 0.18))
            spread = float(agent.get('rate_spread', 0.01))
            strategy = agent.get('strategy', 'rule_based')  # rule_based, aggressive, conservative
            
            # 应用规则引擎
            if rule_engine:
                adjustments, triggered, score_adjustments = rule_engine.process_customer(
                    customer, threshold, spread, base_loan_amount, 24
                )
                final_threshold = adjustments['approval_threshold']
                final_spread = adjustments['rate_spread']
                final_loan_amount = adjustments['loan_amount']
            else:
                final_threshold = threshold
                final_spread = spread
                final_loan_amount = base_loan_amount
            
            # 根据策略调整参数
            if strategy == 'aggressive':
                final_threshold += 0.05  # 更激进，提高阈值
                final_spread -= 0.002  # 降低利差竞争
                final_loan_amount *= 1.1
            elif strategy == 'conservative':
                final_threshold -= 0.03  # 更保守，降低阈值
                final_spread += 0.003  # 提高利差
                final_loan_amount *= 0.9
            
            # 预测客户未来
            rate = base_rate + final_spread
            loan = LoanOffer(amount=final_loan_amount, interest_rate=rate, term_months=24)
            future = self.world_model.predict_customer_future(customer, loan, market, add_noise=False)
            dp = float(future.default_probability)
            
            # 决策
            decision = 'approve' if dp <= final_threshold else 'reject'
            
            # 计算预估利润
            if decision == 'approve':
                interest_income = final_loan_amount * rate
                expected_loss = final_loan_amount * dp
                estimated_profit = interest_income * (1 - dp) - expected_loss
            else:
                estimated_profit = 0.0
            
            # 生成决策理由
            reasoning = self._generate_reasoning(customer, dp, final_threshold, decision, strategy)
            
            agent_decisions.append(AgentDecision(
                agent_id=agent_id,
                customer_id=customer.customer_id,
                decision=decision,
                interest_rate=rate,
                loan_amount=final_loan_amount,
                term_months=24,
                estimated_profit=estimated_profit,
                default_probability=dp,
                risk_factors=future.risk_factors or {},
                compliance_status=True,  # 简化：假设都合规
                reasoning=reasoning
            ))
        
        # 选择获胜者（如果多个智能体都批准，选择利润最高的）
        approved_decisions = [d for d in agent_decisions if d.decision == 'approve']
        if approved_decisions:
            winner_decision = max(approved_decisions, key=lambda x: x.estimated_profit)
            winner = winner_decision.agent_id
        else:
            winner = None  # 所有智能体都拒绝
        
        return RoundResult(
            round_number=round_num,
            customer_id=customer.customer_id,
            customer_profile={
                'customer_id': customer.customer_id,
                'monthly_income': getattr(customer, 'monthly_income', 0),
                'age': getattr(customer, 'age', 0),
                'credit_score': getattr(customer, 'credit_score', 0),
                'debt_ratio': getattr(customer, 'debt_ratio', 0)
            },
            agent_decisions=agent_decisions,
            winner=winner,
            market_conditions={
                'gdp_growth': market.gdp_growth,
                'base_interest_rate': market.base_interest_rate,
                'unemployment_rate': market.unemployment_rate
            }
        )
    
    def _generate_reasoning(self, customer: Any, dp: float, threshold: float,
                           decision: str, strategy: str) -> str:
        """生成决策理由"""
        if decision == 'approve':
            return f"违约概率 {dp:.2%} 低于阈值 {threshold:.2%}，符合{strategy}策略，批准放款"
        else:
            return f"违约概率 {dp:.2%} 高于阈值 {threshold:.2%}，风险过高，拒绝放款"
    
    def run_tournament(self, agents: List[Dict], customers: List[Any],
                      market: MarketConditions, base_rate: float,
                      base_loan_amount: float, rounds: int = 10,
                      rule_engine: Optional[RuleEngine] = None,
                      scoring_system: Optional[ScoringSystem] = None) -> Dict[str, Any]:
        """
        运行锦标赛：多轮竞争，累积得分
        
        Args:
            agents: 智能体列表
            customers: 客户列表
            market: 市场条件
            base_rate: 基准利率
            base_loan_amount: 基准贷款金额
            rounds: 回合数
            rule_engine: 规则引擎
            scoring_system: 评分系统
        
        Returns:
            锦标赛结果
        """
        self.round_history = []
        agent_scores = {agent.get('id', agent.get('name')): 0.0 for agent in agents}
        agent_stats = {agent.get('id', agent.get('name')): {
            'wins': 0,
            'approvals': 0,
            'rejects': 0,
            'total_profit': 0.0,
            'total_risk': 0.0
        } for agent in agents}
        
        # 运行多轮
        for round_num in range(1, rounds + 1):
            if round_num > len(customers):
                break
            
            customer = customers[round_num - 1]
            round_result = self.simulate_competitive_round(
                round_num, customer, agents, market,
                base_rate, base_loan_amount, rule_engine
            )
            
            self.round_history.append(round_result)
            
            # 更新统计
            if round_result.winner:
                agent_scores[round_result.winner] += 1.0
                agent_stats[round_result.winner]['wins'] += 1
                
                # 找到获胜决策
                winner_decision = next(
                    d for d in round_result.agent_decisions
                    if d.agent_id == round_result.winner
                )
                agent_stats[round_result.winner]['total_profit'] += winner_decision.estimated_profit
                agent_stats[round_result.winner]['total_risk'] += winner_decision.default_probability
            
            # 更新所有智能体的审批统计
            for decision in round_result.agent_decisions:
                agent_id = decision.agent_id
                if decision.decision == 'approve':
                    agent_stats[agent_id]['approvals'] += 1
                else:
                    agent_stats[agent_id]['rejects'] += 1
        
        # 计算最终排名
        final_rankings = sorted(
            agent_scores.items(),
            key=lambda x: (x[1], agent_stats[x[0]]['total_profit']),
            reverse=True
        )
        
        champion = final_rankings[0][0] if final_rankings else None
        
        return {
            'rounds': rounds,
            'round_history': [
                {
                    'round_number': r.round_number,
                    'customer_id': r.customer_id,
                    'customer_profile': r.customer_profile,
                    'agent_decisions': [
                        {
                            'agent_id': d.agent_id,
                            'decision': d.decision,
                            'interest_rate': d.interest_rate,
                            'loan_amount': d.loan_amount,
                            'estimated_profit': d.estimated_profit,
                            'default_probability': d.default_probability,
                            'reasoning': d.reasoning
                        }
                        for d in r.agent_decisions
                    ],
                    'winner': r.winner
                }
                for r in self.round_history
            ],
            'agent_scores': agent_scores,
            'agent_stats': agent_stats,
            'final_rankings': final_rankings,
            'champion': champion
        }



