"""
经济周期模拟器 - 模拟宏观经济的周期性波动
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class CyclePhase(Enum):
    """经济周期阶段"""
    BOOM = "繁荣"        # 高增长、低失业、可能过热
    RECESSION = "衰退"   # 增长放缓、失业上升
    DEPRESSION = "萧条"  # 负增长、高失业、信用紧缩
    RECOVERY = "复苏"    # 触底反弹、逐步恢复


@dataclass
class EconomicState:
    """经济状态 - 扩展版"""
    # 基础指标
    gdp_growth: float           # GDP 增长率 (-0.1 ~ 0.15)
    interest_rate: float        # 基准利率 (0.01 ~ 0.15)
    unemployment_rate: float    # 失业率 (0.03 ~ 0.15)
    inflation_rate: float       # 通胀率 (-0.02 ~ 0.10)
    credit_spread: float        # 信用利差 (0.01 ~ 0.08)
    phase: CyclePhase           # 当前周期阶段
    
    # 扩展指标
    consumer_confidence: float = 0.5
    manufacturing_pmi: float = 50.0
    housing_price_index: float = 100.0
    stock_index: float = 3000.0
    m2_growth: float = 0.10
    exchange_rate: float = 7.0
    trade_balance: float = 0.0
    fiscal_policy_stance: float = 0.5
    monetary_policy_stance: float = 0.5
    risk_appetite: float = 0.5
    liquidity_index: float = 0.5
    market_volatility: float = 0.15
    
    def to_array(self) -> np.ndarray:
        """转换为数组，用于神经网络输入"""
        phase_onehot = [0, 0, 0, 0]
        phase_onehot[list(CyclePhase).index(self.phase)] = 1
        return np.array([
            self.gdp_growth,
            self.interest_rate,
            self.unemployment_rate,
            self.inflation_rate,
            self.credit_spread,
            self.consumer_confidence,
            self.manufacturing_pmi / 100.0,
            (self.housing_price_index - 100) / 100.0,
            (self.stock_index - 3000) / 3000.0,
            self.m2_growth,
            self.exchange_rate / 10.0,
            self.trade_balance / 1000.0,
            self.fiscal_policy_stance,
            self.monetary_policy_stance,
            self.risk_appetite,
            self.liquidity_index,
            self.market_volatility,
            *phase_onehot
        ], dtype=np.float32)


class EconomicCycle:
    """
    经济周期模拟器
    
    使用马尔可夫链模拟经济周期转换，每个阶段有不同的经济参数分布
    """
    
    # 周期转换概率矩阵 (每月转换概率)
    # 从 [BOOM, RECESSION, DEPRESSION, RECOVERY] 到各阶段的概率
    TRANSITION_MATRIX = np.array([
        [0.92, 0.08, 0.00, 0.00],  # BOOM -> 
        [0.00, 0.85, 0.15, 0.00],  # RECESSION ->
        [0.00, 0.00, 0.88, 0.12],  # DEPRESSION ->
        [0.10, 0.00, 0.00, 0.90],  # RECOVERY ->
    ])
    
    # 各阶段的经济参数 (均值, 标准差)
    PHASE_PARAMS = {
        CyclePhase.BOOM: {
            'gdp_growth': (0.08, 0.02),
            'interest_rate': (0.06, 0.01),
            'unemployment_rate': (0.04, 0.01),
            'inflation_rate': (0.04, 0.01),
            'credit_spread': (0.02, 0.005),
            'consumer_confidence': (0.75, 0.05),
            'manufacturing_pmi': (55.0, 3.0),
            'housing_price_index': (120.0, 5.0),
            'stock_index': (3500.0, 200.0),
            'm2_growth': (0.12, 0.02),
            'exchange_rate': (6.8, 0.2),
            'trade_balance': (300.0, 50.0),
            'fiscal_policy_stance': (0.4, 0.1),  # 相对中性
            'monetary_policy_stance': (0.4, 0.1),
            'risk_appetite': (0.7, 0.1),
            'liquidity_index': (0.7, 0.1),
            'market_volatility': (0.12, 0.03),
        },
        CyclePhase.RECESSION: {
            'gdp_growth': (0.01, 0.02),
            'interest_rate': (0.04, 0.01),
            'unemployment_rate': (0.07, 0.02),
            'inflation_rate': (0.02, 0.01),
            'credit_spread': (0.04, 0.01),
            'consumer_confidence': (0.55, 0.05),
            'manufacturing_pmi': (48.0, 2.0),
            'housing_price_index': (105.0, 3.0),
            'stock_index': (2800.0, 150.0),
            'm2_growth': (0.08, 0.02),
            'exchange_rate': (7.2, 0.2),
            'trade_balance': (100.0, 50.0),
            'fiscal_policy_stance': (0.6, 0.1),  # 开始扩张
            'monetary_policy_stance': (0.6, 0.1),
            'risk_appetite': (0.4, 0.1),
            'liquidity_index': (0.5, 0.1),
            'market_volatility': (0.18, 0.03),
        },
        CyclePhase.DEPRESSION: {
            'gdp_growth': (-0.03, 0.02),
            'interest_rate': (0.02, 0.01),
            'unemployment_rate': (0.12, 0.02),
            'inflation_rate': (0.00, 0.01),
            'credit_spread': (0.06, 0.015),
            'consumer_confidence': (0.35, 0.05),
            'manufacturing_pmi': (42.0, 3.0),
            'housing_price_index': (95.0, 5.0),
            'stock_index': (2400.0, 200.0),
            'm2_growth': (0.15, 0.02),  # 货币宽松
            'exchange_rate': (7.5, 0.3),
            'trade_balance': (-50.0, 50.0),
            'fiscal_policy_stance': (0.8, 0.1),  # 强扩张
            'monetary_policy_stance': (0.8, 0.1),
            'risk_appetite': (0.2, 0.1),
            'liquidity_index': (0.3, 0.1),
            'market_volatility': (0.25, 0.05),
        },
        CyclePhase.RECOVERY: {
            'gdp_growth': (0.04, 0.02),
            'interest_rate': (0.03, 0.01),
            'unemployment_rate': (0.08, 0.02),
            'inflation_rate': (0.02, 0.01),
            'credit_spread': (0.03, 0.01),
            'consumer_confidence': (0.65, 0.05),
            'manufacturing_pmi': (52.0, 2.0),
            'housing_price_index': (110.0, 4.0),
            'stock_index': (3200.0, 150.0),
            'm2_growth': (0.10, 0.02),
            'exchange_rate': (7.0, 0.2),
            'trade_balance': (200.0, 50.0),
            'fiscal_policy_stance': (0.5, 0.1),
            'monetary_policy_stance': (0.5, 0.1),
            'risk_appetite': (0.6, 0.1),
            'liquidity_index': (0.6, 0.1),
            'market_volatility': (0.15, 0.03),
        },
    }
    
    def __init__(self, initial_phase: CyclePhase = CyclePhase.BOOM, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.phase = initial_phase
        self.month = 0
        self.state = self._generate_state()
    
    def _generate_state(self) -> EconomicState:
        """根据当前阶段生成经济状态"""
        params = self.PHASE_PARAMS[self.phase]
        
        def sample_param(key, default_mean=0.5, default_std=0.1):
            if key in params:
                mean, std = params[key]
                return np.clip(self.rng.normal(mean, std), 0, 1) if key in ['consumer_confidence', 'fiscal_policy_stance', 'monetary_policy_stance', 'risk_appetite', 'liquidity_index', 'market_volatility'] else self.rng.normal(mean, std)
            return self.rng.normal(default_mean, default_std)
        
        return EconomicState(
            gdp_growth=np.clip(
                self.rng.normal(*params['gdp_growth']), -0.10, 0.15
            ),
            interest_rate=np.clip(
                self.rng.normal(*params['interest_rate']), 0.01, 0.15
            ),
            unemployment_rate=np.clip(
                self.rng.normal(*params['unemployment_rate']), 0.03, 0.20
            ),
            inflation_rate=np.clip(
                self.rng.normal(*params['inflation_rate']), -0.02, 0.10
            ),
            credit_spread=np.clip(
                self.rng.normal(*params['credit_spread']), 0.01, 0.10
            ),
            phase=self.phase,
            consumer_confidence=np.clip(
                self.rng.normal(*params['consumer_confidence']), 0.0, 1.0
            ),
            manufacturing_pmi=np.clip(
                self.rng.normal(*params['manufacturing_pmi']), 30.0, 70.0
            ),
            housing_price_index=np.clip(
                self.rng.normal(*params['housing_price_index']), 80.0, 150.0
            ),
            stock_index=np.clip(
                self.rng.normal(*params['stock_index']), 2000.0, 5000.0
            ),
            m2_growth=np.clip(
                self.rng.normal(*params['m2_growth']), 0.05, 0.20
            ),
            exchange_rate=np.clip(
                self.rng.normal(*params['exchange_rate']), 6.0, 8.0
            ),
            trade_balance=self.rng.normal(*params['trade_balance']),
            fiscal_policy_stance=np.clip(
                self.rng.normal(*params['fiscal_policy_stance']), 0.0, 1.0
            ),
            monetary_policy_stance=np.clip(
                self.rng.normal(*params['monetary_policy_stance']), 0.0, 1.0
            ),
            risk_appetite=np.clip(
                self.rng.normal(*params['risk_appetite']), 0.0, 1.0
            ),
            liquidity_index=np.clip(
                self.rng.normal(*params['liquidity_index']), 0.0, 1.0
            ),
            market_volatility=np.clip(
                self.rng.normal(*params['market_volatility']), 0.05, 0.40
            ),
        )
    
    def step(self) -> EconomicState:
        """推进一个月，可能发生周期转换"""
        self.month += 1
        
        # 周期转换
        phase_idx = list(CyclePhase).index(self.phase)
        transition_probs = self.TRANSITION_MATRIX[phase_idx]
        new_phase_idx = self.rng.choice(4, p=transition_probs)
        self.phase = list(CyclePhase)[new_phase_idx]
        
        # 生成新状态
        self.state = self._generate_state()
        return self.state
    
    def inject_shock(self, shock_type: str = 'financial_crisis') -> EconomicState:
        """
        注入黑天鹅事件
        
        Args:
            shock_type: 'financial_crisis', 'pandemic', 'geopolitical'
        """
        if shock_type == 'financial_crisis':
            self.phase = CyclePhase.DEPRESSION
            self.state = EconomicState(
                gdp_growth=-0.08,
                interest_rate=0.01,
                unemployment_rate=0.15,
                inflation_rate=-0.01,
                credit_spread=0.08,
                phase=CyclePhase.DEPRESSION
            )
        elif shock_type == 'pandemic':
            self.phase = CyclePhase.DEPRESSION
            self.state = EconomicState(
                gdp_growth=-0.06,
                interest_rate=0.01,
                unemployment_rate=0.14,
                inflation_rate=0.01,
                credit_spread=0.05,
                phase=CyclePhase.DEPRESSION
            )
        elif shock_type == 'geopolitical':
            self.phase = CyclePhase.RECESSION
            self.state = EconomicState(
                gdp_growth=0.00,
                interest_rate=0.05,
                unemployment_rate=0.08,
                inflation_rate=0.06,
                credit_spread=0.04,
                phase=CyclePhase.RECESSION
            )
        
        return self.state
    
    def reset(self, initial_phase: CyclePhase = CyclePhase.BOOM) -> EconomicState:
        """重置经济周期"""
        self.phase = initial_phase
        self.month = 0
        self.state = self._generate_state()
        return self.state


if __name__ == "__main__":
    # 测试经济周期模拟
    cycle = EconomicCycle(seed=42)
    
    print("经济周期模拟测试 (120个月 = 10年)")
    print("=" * 60)
    
    phase_counts = {phase: 0 for phase in CyclePhase}
    
    for month in range(120):
        state = cycle.step()
        phase_counts[state.phase] += 1
        
        if month % 12 == 0:
            year = month // 12 + 1
            print(f"第{year:2d}年: {state.phase.value:4s} | "
                  f"GDP:{state.gdp_growth:+.1%} | "
                  f"利率:{state.interest_rate:.1%} | "
                  f"失业:{state.unemployment_rate:.1%}")
    
    print("\n周期阶段分布:")
    for phase, count in phase_counts.items():
        print(f"  {phase.value}: {count} 个月 ({count/120:.1%})")


