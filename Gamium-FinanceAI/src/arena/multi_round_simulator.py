"""
多轮多场景演武场模拟器
支持回合数、宏观扰动、压力情景、累积得分与回放
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
# 这些导入在app.py中已经可用，这里仅作类型提示
# 实际使用时，这些类会通过app.py的导入传递
try:
    from data_distillation.world_model import WorldModel, MarketConditions, LoanOffer
    from data_distillation.customer_generator import CustomerGenerator
except ImportError:
    # 如果直接运行此模块，使用相对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data_distillation.world_model import WorldModel, MarketConditions, LoanOffer
    from src.data_distillation.customer_generator import CustomerGenerator
from arena.rule_engine import RuleEngine
from arena.scoring_system import ScoringSystem


@dataclass
class RoundResult:
    """单轮结果"""
    round_number: int
    scenario: str  # normal / stress / black_swan
    market_conditions: Dict[str, float]
    participant_results: List[Dict[str, Any]]
    cumulative_scores: Dict[str, float]
    timestamp: str


class MultiRoundSimulator:
    """多轮模拟器"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.world_model = WorldModel(seed=seed)
        self.generator = CustomerGenerator(seed=seed)
        self.round_history: List[RoundResult] = []
    
    def simulate_round(self, round_num: int, participants: List[Dict],
                      customers: List, rules: List[Dict],
                      base_rate: float, loan_amount: float,
                      scenario: str = 'normal', black_swan: bool = False,
                      rule_engine: Optional[RuleEngine] = None,
                      scoring_system: Optional[ScoringSystem] = None) -> RoundResult:
        """
        模拟单轮
        
        Args:
            round_num: 回合编号
            participants: 参赛者列表
            customers: 客户列表
            rules: 规则列表
            base_rate: 基准利率
            loan_amount: 贷款金额
            scenario: 场景类型
            black_swan: 是否黑天鹅事件
            rule_engine: 规则引擎
            scoring_system: 评分系统
        
        Returns:
            单轮结果
        """
        # 根据场景调整宏观参数
        market = self._create_market_conditions(base_rate, scenario, black_swan)
        
        participant_results = []
        
        for p in participants:
            threshold = float(p.get('approval_threshold', 0.18))
            spread = float(p.get('rate_spread', 0.01))
            name = p.get('name', '未命名')
            
            approved = 0
            rejected = 0
            profit = 0.0
            default_probs = []
            triggered_rules_list = []
            
            for cust in customers:
                # 应用规则引擎
                if rule_engine:
                    adjustments, triggered, score_adjustments = rule_engine.process_customer(
                        cust, threshold, spread, loan_amount, 24
                    )
                    final_threshold = adjustments['approval_threshold']
                    final_spread = adjustments['rate_spread']
                    triggered_rules_list.extend(triggered)
                else:
                    final_threshold = threshold
                    final_spread = spread
                
                rate = base_rate + final_spread
                loan = LoanOffer(amount=loan_amount, interest_rate=rate, term_months=24)
                future = self.world_model.predict_customer_future(cust, loan, market, add_noise=False)
                dp = float(future.default_probability)
                
                if dp <= final_threshold:
                    approved += 1
                    interest_income = loan_amount * rate
                    expected_loss = loan_amount * dp
                    net_profit = interest_income * (1 - dp) - expected_loss
                    profit += net_profit
                    default_probs.append(dp)
                else:
                    rejected += 1
            
            total = approved + rejected
            avg_dp = np.mean(default_probs) if default_probs else 0.0
            approval_rate = approved / total if total > 0 else 0
            raroc = profit / max(1.0, (avg_dp * loan_amount * approved) + 1e3) if approved > 0 else 0.0
            
            result = {
                'name': name,
                'approval_rate': approval_rate,
                'avg_default_prob': avg_dp,
                'est_profit': profit,
                'raroc': raroc,
                'sample_size': total,
                'triggered_rules_list': list(set(triggered_rules_list))
            }
            
            participant_results.append(result)
        
        # 计算累积得分
        cumulative_scores = {}
        if scoring_system:
            for result in participant_results:
                breakdown = scoring_system.create_score_breakdown(
                    result,
                    triggered_rules=result.get('triggered_rules_list', []),
                    all_results=participant_results
                )
                name = result['name']
                if name not in cumulative_scores:
                    cumulative_scores[name] = 0.0
                cumulative_scores[name] += breakdown.overall_score
        
        round_result = RoundResult(
            round_number=round_num,
            scenario=scenario,
            market_conditions={
                'gdp_growth': market.gdp_growth,
                'base_interest_rate': market.base_interest_rate,
                'unemployment_rate': market.unemployment_rate,
                'inflation_rate': market.inflation_rate,
                'credit_spread': market.credit_spread
            },
            participant_results=participant_results,
            cumulative_scores=cumulative_scores,
            timestamp=datetime.now().isoformat()
        )
        
        return round_result
    
    def _create_market_conditions(self, base_rate: float, scenario: str, black_swan: bool) -> MarketConditions:
        """创建市场条件"""
        gdp = 0.03
        unemp = 0.05
        base_ir = base_rate
        infl = 0.02
        credit_spread = 0.02
        
        if scenario == 'stress':
            gdp = -0.02
            unemp = 0.09
            base_ir = base_rate + 0.01
            credit_spread = 0.05
        
        if black_swan:
            gdp -= 0.02
            unemp += 0.02
            credit_spread += 0.03
        
        return MarketConditions(
            gdp_growth=gdp,
            base_interest_rate=base_ir,
            unemployment_rate=unemp,
            inflation_rate=infl,
            credit_spread=credit_spread,
        )
    
    def simulate_multi_rounds(self, rounds: int, participants: List[Dict],
                              customer_count: int, rules: List[Dict],
                              base_rate: float, loan_amount: float,
                              seed: int, scenario_sequence: List[str] = None,
                              black_swan_rounds: List[int] = None,
                              rule_engine: Optional[RuleEngine] = None,
                              scoring_system: Optional[ScoringSystem] = None) -> Dict[str, Any]:
        """
        模拟多轮
        
        Args:
            rounds: 回合数
            participants: 参赛者列表
            customer_count: 每轮客户数
            rules: 规则列表
            base_rate: 基准利率
            loan_amount: 贷款金额
            seed: 随机种子
            scenario_sequence: 场景序列（如 ['normal', 'stress', 'normal']）
            black_swan_rounds: 黑天鹅事件发生的回合列表（如 [3, 7]）
            rule_engine: 规则引擎
            scoring_system: 评分系统
        
        Returns:
            多轮模拟结果
        """
        self.round_history = []
        rng = np.random.default_rng(seed)
        
        if scenario_sequence is None:
            scenario_sequence = ['normal'] * rounds
        
        if black_swan_rounds is None:
            black_swan_rounds = []
        
        # 生成统一的客户池（每轮使用不同的子集）
        all_customers = [self.generator.generate_one() for _ in range(customer_count * rounds)]
        
        cumulative_scores = {p['name']: 0.0 for p in participants}
        
        for round_num in range(1, rounds + 1):
            # 选择场景
            scenario = scenario_sequence[(round_num - 1) % len(scenario_sequence)]
            black_swan = round_num in black_swan_rounds
            
            # 选择客户子集
            start_idx = (round_num - 1) * customer_count
            end_idx = start_idx + customer_count
            round_customers = all_customers[start_idx:end_idx]
            
            # 模拟本轮
            round_result = self.simulate_round(
                round_num, participants, round_customers, rules,
                base_rate, loan_amount, scenario, black_swan,
                rule_engine, scoring_system
            )
            
            # 更新累积得分
            for name, score in round_result.cumulative_scores.items():
                cumulative_scores[name] += score
            
            self.round_history.append(round_result)
        
        # 汇总结果
        final_scores = cumulative_scores
        winner = max(final_scores.items(), key=lambda x: x[1])[0] if final_scores else None
        
        return {
            'rounds': rounds,
            'round_history': [
                {
                    'round_number': r.round_number,
                    'scenario': r.scenario,
                    'market_conditions': r.market_conditions,
                    'participant_results': r.participant_results,
                    'cumulative_scores': r.cumulative_scores,
                    'timestamp': r.timestamp
                }
                for r in self.round_history
            ],
            'final_scores': final_scores,
            'winner': winner
        }

