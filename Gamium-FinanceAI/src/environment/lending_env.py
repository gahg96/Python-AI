"""
信贷策略环境 - Gamium POC 核心模拟器

这是一个简化版的信贷经营模拟环境，模拟银行在10年期间的信贷决策。
智能体需要在不同经济周期下做出：定价、风控、客群选择等决策。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

from .economic_cycle import EconomicCycle, EconomicState, CyclePhase


class CustomerSegment(Enum):
    """客户分层"""
    PRIME = "优质客户"      # 低风险，低收益
    NEAR_PRIME = "次优客户"  # 中等风险，中等收益
    SUBPRIME = "次级客户"   # 高风险，高收益


@dataclass
class LoanPortfolio:
    """贷款组合"""
    total_loans: float = 0.0           # 总贷款余额
    performing_loans: float = 0.0      # 正常贷款
    delinquent_loans: float = 0.0      # 逾期贷款 (30-90天)
    non_performing_loans: float = 0.0  # 不良贷款 (>90天)
    
    # 按客群分布
    loans_by_segment: Dict[CustomerSegment, float] = field(
        default_factory=lambda: {seg: 0.0 for seg in CustomerSegment}
    )
    
    # 收益相关
    interest_income: float = 0.0       # 利息收入
    provision_expense: float = 0.0     # 拨备费用
    write_offs: float = 0.0            # 核销损失
    
    @property
    def npl_ratio(self) -> float:
        """不良贷款率"""
        if self.total_loans <= 0:
            return 0.0
        return self.non_performing_loans / self.total_loans
    
    @property
    def net_profit(self) -> float:
        """净利润"""
        return self.interest_income - self.provision_expense - self.write_offs


@dataclass
class BankState:
    """银行经营状态"""
    capital: float = 100.0             # 资本金 (初始100亿)
    total_assets: float = 1000.0       # 总资产 (初始1000亿)
    loan_portfolio: LoanPortfolio = field(default_factory=LoanPortfolio)
    
    # 监管指标
    capital_adequacy_ratio: float = 0.12  # 资本充足率 (监管要求 >= 10.5%)
    liquidity_ratio: float = 0.30         # 流动性比率
    
    # 累计指标
    cumulative_profit: float = 0.0
    cumulative_write_offs: float = 0.0
    
    @property
    def roa(self) -> float:
        """资产回报率 (年化)"""
        if self.total_assets <= 0:
            return 0.0
        return (self.loan_portfolio.net_profit * 12) / self.total_assets
    
    def is_bankrupt(self) -> bool:
        """是否破产"""
        return (self.capital <= 0 or 
                self.capital_adequacy_ratio < 0.08 or 
                self.loan_portfolio.npl_ratio > 0.25)


class LendingEnv(gym.Env):
    """
    信贷策略环境
    
    状态空间 (22维):
        - 经济状态 (9维): GDP增长率, 利率, 失业率, 通胀率, 信用利差, 周期阶段(4维one-hot)
        - 银行状态 (10维): 资本, 总资产, 贷款余额, NPL率, 资本充足率, 各客群贷款占比(3), 累计利润, 累计核销
        - 时间 (3维): 当前月份, 当前年份, 剩余年份
    
    动作空间 (连续, 5维):
        - 利率调整: [-0.02, +0.02] 相对于基准利率
        - 审批通过率: [0.3, 0.9] 
        - 客群分配权重 (3维): 各客群的贷款投放权重
    
    奖励:
        - 月度净利润 (归一化)
        - 风险惩罚 (NPL超标)
        - 终局奖励 (10年累计表现)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # 模拟参数
    SIMULATION_YEARS = 10
    MONTHS_PER_YEAR = 12
    TOTAL_MONTHS = SIMULATION_YEARS * MONTHS_PER_YEAR
    
    # 客群特征 (基础违约率, 利率敏感度, 市场规模权重)
    SEGMENT_PARAMS = {
        CustomerSegment.PRIME: {
            'base_default_rate': 0.01,    # 1% 基础违约率
            'rate_sensitivity': 0.5,       # 利率敏感度
            'recovery_rate': 0.60,         # 违约回收率
            'market_weight': 0.3,          # 市场规模
        },
        CustomerSegment.NEAR_PRIME: {
            'base_default_rate': 0.03,
            'rate_sensitivity': 0.7,
            'recovery_rate': 0.40,
            'market_weight': 0.5,
        },
        CustomerSegment.SUBPRIME: {
            'base_default_rate': 0.08,
            'rate_sensitivity': 0.9,
            'recovery_rate': 0.25,
            'market_weight': 0.2,
        },
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        initial_capital: float = 100.0,
        initial_assets: float = 1000.0,
        seed: int = None
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.initial_capital = initial_capital
        self.initial_assets = initial_assets
        
        # 随机数生成器
        self.rng = np.random.default_rng(seed)
        
        # 定义状态空间 (22维)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )
        
        # 定义动作空间 (5维连续)
        self.action_space = spaces.Box(
            low=np.array([-0.02, 0.3, 0.0, 0.0, 0.0]),
            high=np.array([0.02, 0.9, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 初始化
        self.economy = None
        self.bank = None
        self.month = 0
        self.history = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # 重置经济周期
        initial_phase = self.rng.choice(list(CyclePhase))
        self.economy = EconomicCycle(initial_phase=initial_phase, seed=self.rng.integers(0, 10000))
        
        # 重置银行状态
        self.bank = BankState(
            capital=self.initial_capital,
            total_assets=self.initial_assets,
            loan_portfolio=LoanPortfolio(
                total_loans=self.initial_assets * 0.6,
                performing_loans=self.initial_assets * 0.58,
                delinquent_loans=self.initial_assets * 0.015,
                non_performing_loans=self.initial_assets * 0.005,
            )
        )
        
        # 初始化各客群贷款
        total = self.bank.loan_portfolio.total_loans
        self.bank.loan_portfolio.loans_by_segment = {
            CustomerSegment.PRIME: total * 0.4,
            CustomerSegment.NEAR_PRIME: total * 0.45,
            CustomerSegment.SUBPRIME: total * 0.15,
        }
        
        self.month = 0
        self.history = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步 (一个月)
        
        Args:
            action: [利率调整, 审批通过率, prime权重, near_prime权重, subprime权重]
        """
        # 解析动作
        rate_adjustment = np.clip(action[0], -0.02, 0.02)
        approval_rate = np.clip(action[1], 0.3, 0.9)
        
        # 归一化客群权重
        segment_weights_raw = action[2:5]
        segment_weights = segment_weights_raw / (segment_weights_raw.sum() + 1e-8)
        segment_weights = {
            CustomerSegment.PRIME: segment_weights[0],
            CustomerSegment.NEAR_PRIME: segment_weights[1],
            CustomerSegment.SUBPRIME: segment_weights[2],
        }
        
        # 推进经济周期
        eco_state = self.economy.step()
        
        # 随机黑天鹅事件 (低概率)
        if self.rng.random() < 0.002:  # 0.2% 月概率 ≈ 每40年一次
            eco_state = self.economy.inject_shock('financial_crisis')
        
        # 计算本月信贷业务
        lending_rate = eco_state.interest_rate + eco_state.credit_spread + rate_adjustment
        
        # 1. 新增贷款
        new_loans = self._process_new_loans(
            eco_state, lending_rate, approval_rate, segment_weights
        )
        
        # 2. 贷款表现 (还款、逾期、违约)
        portfolio_changes = self._process_loan_performance(eco_state, lending_rate)
        
        # 3. 更新银行状态
        self._update_bank_state(new_loans, portfolio_changes, lending_rate)
        
        # 4. 计算奖励
        reward = self._calculate_reward(eco_state)
        
        # 5. 检查终止条件
        self.month += 1
        terminated = self.month >= self.TOTAL_MONTHS
        truncated = self.bank.is_bankrupt()
        
        if truncated:
            reward -= 100  # 破产大额惩罚
        
        # 记录历史
        self.history.append({
            'month': self.month,
            'eco_phase': eco_state.phase.value,
            'gdp_growth': eco_state.gdp_growth,
            'npl_ratio': self.bank.loan_portfolio.npl_ratio,
            'roa': self.bank.roa,
            'profit': self.bank.loan_portfolio.net_profit,
            'action': action.copy(),
            'reward': reward,
        })
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _process_new_loans(
        self,
        eco_state: EconomicState,
        lending_rate: float,
        approval_rate: float,
        segment_weights: Dict[CustomerSegment, float]
    ) -> Dict[CustomerSegment, float]:
        """处理新增贷款"""
        new_loans = {}
        
        # 基础月新增贷款规模 (总资产的1-3%)
        base_volume = self.bank.total_assets * self.rng.uniform(0.01, 0.03)
        
        # 经济周期调整
        if eco_state.phase == CyclePhase.BOOM:
            base_volume *= 1.3
        elif eco_state.phase == CyclePhase.DEPRESSION:
            base_volume *= 0.5
        
        for segment in CustomerSegment:
            params = self.SEGMENT_PARAMS[segment]
            
            # 该客群的申请量
            demand = base_volume * params['market_weight'] * segment_weights[segment]
            
            # 利率对需求的影响 (利率越高，需求越低)
            rate_factor = 1.0 - params['rate_sensitivity'] * (lending_rate - 0.08)
            rate_factor = np.clip(rate_factor, 0.3, 1.5)
            
            # 实际发放 = 需求 × 通过率 × 利率因子
            new_loans[segment] = demand * approval_rate * rate_factor
        
        return new_loans
    
    def _process_loan_performance(
        self,
        eco_state: EconomicState,
        lending_rate: float
    ) -> Dict[str, float]:
        """处理贷款表现 (还款、逾期、违约)"""
        portfolio = self.bank.loan_portfolio
        
        # 月还款率 (正常还本)
        repayment_rate = 0.02  # 假设平均贷款期限50个月
        
        # 计算各客群的违约率
        total_new_delinquent = 0.0
        total_new_npl = 0.0
        total_recovery = 0.0
        total_interest = 0.0
        
        for segment in CustomerSegment:
            params = self.SEGMENT_PARAMS[segment]
            segment_loans = portfolio.loans_by_segment[segment]
            
            if segment_loans <= 0:
                continue
            
            # 基础违约率 + 经济周期影响
            cycle_factor = {
                CyclePhase.BOOM: 0.7,
                CyclePhase.RECOVERY: 0.9,
                CyclePhase.RECESSION: 1.3,
                CyclePhase.DEPRESSION: 2.0,
            }[eco_state.phase]
            
            # 失业率影响
            unemployment_factor = 1.0 + (eco_state.unemployment_rate - 0.05) * 3
            
            monthly_default_rate = (
                params['base_default_rate'] / 12 * 
                cycle_factor * 
                unemployment_factor
            )
            monthly_default_rate = np.clip(monthly_default_rate, 0, 0.05)
            
            # 新增逾期
            new_delinquent = segment_loans * monthly_default_rate * self.rng.uniform(0.8, 1.2)
            total_new_delinquent += new_delinquent
            
            # 利息收入
            monthly_rate = lending_rate / 12
            interest = segment_loans * monthly_rate
            total_interest += interest
        
        # 逾期 -> 不良 转化 (每月约10%的逾期变成不良)
        delinquent_to_npl = portfolio.delinquent_loans * 0.10
        total_new_npl += delinquent_to_npl
        
        # 不良 -> 核销 (超过一定期限的不良要核销)
        write_off = portfolio.non_performing_loans * 0.05  # 每月5%核销
        
        # 回收 (部分不良可以回收)
        recovery = write_off * 0.30  # 平均30%回收率
        total_recovery += recovery
        
        return {
            'repayment': portfolio.performing_loans * repayment_rate,
            'new_delinquent': total_new_delinquent,
            'delinquent_to_npl': delinquent_to_npl,
            'write_off': write_off,
            'recovery': total_recovery,
            'interest_income': total_interest,
        }
    
    def _update_bank_state(
        self,
        new_loans: Dict[CustomerSegment, float],
        changes: Dict[str, float],
        lending_rate: float
    ):
        """更新银行状态"""
        portfolio = self.bank.loan_portfolio
        
        # 更新贷款组合
        total_new = sum(new_loans.values())
        
        # 正常贷款变化
        portfolio.performing_loans += total_new
        portfolio.performing_loans -= changes['repayment']
        portfolio.performing_loans -= changes['new_delinquent']
        portfolio.performing_loans = max(0, portfolio.performing_loans)
        
        # 逾期贷款变化
        portfolio.delinquent_loans += changes['new_delinquent']
        portfolio.delinquent_loans -= changes['delinquent_to_npl']
        portfolio.delinquent_loans = max(0, portfolio.delinquent_loans)
        
        # 不良贷款变化
        portfolio.non_performing_loans += changes['delinquent_to_npl']
        portfolio.non_performing_loans -= changes['write_off']
        portfolio.non_performing_loans = max(0, portfolio.non_performing_loans)
        
        # 更新客群分布
        for segment, amount in new_loans.items():
            portfolio.loans_by_segment[segment] += amount
        
        # 总贷款
        portfolio.total_loans = (
            portfolio.performing_loans + 
            portfolio.delinquent_loans + 
            portfolio.non_performing_loans
        )
        
        # 收益
        portfolio.interest_income = changes['interest_income']
        portfolio.provision_expense = changes['new_delinquent'] * 0.5  # 50%拨备
        portfolio.write_offs = changes['write_off'] - changes['recovery']
        
        # 更新资本
        net_profit = portfolio.net_profit
        self.bank.capital += net_profit
        self.bank.cumulative_profit += net_profit
        self.bank.cumulative_write_offs += changes['write_off']
        
        # 更新总资产
        self.bank.total_assets = self.bank.capital / 0.10  # 简化：保持10%资本充足率目标
        
        # 更新监管指标
        self.bank.capital_adequacy_ratio = self.bank.capital / (portfolio.total_loans + 1e-8)
    
    def _calculate_reward(self, eco_state: EconomicState) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 1. 利润奖励 (归一化到[-1, 1])
        profit = self.bank.loan_portfolio.net_profit
        profit_reward = np.clip(profit / 5.0, -2.0, 2.0)  # 5亿为基准
        reward += profit_reward
        
        # 2. NPL惩罚
        npl_ratio = self.bank.loan_portfolio.npl_ratio
        if npl_ratio > 0.05:  # NPL超过5%开始惩罚
            npl_penalty = (npl_ratio - 0.05) * 20
            reward -= npl_penalty
        
        # 3. 资本充足率惩罚
        car = self.bank.capital_adequacy_ratio
        if car < 0.105:  # 低于监管红线
            car_penalty = (0.105 - car) * 50
            reward -= car_penalty
        
        # 4. 经济周期适应性奖励
        if eco_state.phase == CyclePhase.DEPRESSION and npl_ratio < 0.08:
            reward += 0.5  # 萧条期控制住风险
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """获取状态观测"""
        eco = self.economy.state
        bank = self.bank
        portfolio = bank.loan_portfolio
        
        # 经济状态 (9维)
        eco_obs = eco.to_array()
        
        # 银行状态 (10维)
        segment_loans = [
            portfolio.loans_by_segment.get(seg, 0.0) / (portfolio.total_loans + 1e-8)
            for seg in CustomerSegment
        ]
        
        bank_obs = np.array([
            bank.capital / 100,  # 归一化
            bank.total_assets / 1000,
            portfolio.total_loans / 1000,
            portfolio.npl_ratio,
            bank.capital_adequacy_ratio,
            *segment_loans,
            bank.cumulative_profit / 100,
            bank.cumulative_write_offs / 100,
        ], dtype=np.float32)
        
        # 时间 (3维)
        time_obs = np.array([
            self.month / self.TOTAL_MONTHS,
            (self.month // 12) / self.SIMULATION_YEARS,
            1 - self.month / self.TOTAL_MONTHS,
        ], dtype=np.float32)
        
        return np.concatenate([eco_obs, bank_obs, time_obs])
    
    def _get_info(self) -> dict:
        """获取附加信息"""
        return {
            'month': self.month,
            'year': self.month // 12,
            'eco_phase': self.economy.state.phase.value,
            'npl_ratio': self.bank.loan_portfolio.npl_ratio,
            'roa': self.bank.roa,
            'capital': self.bank.capital,
            'cumulative_profit': self.bank.cumulative_profit,
            'is_bankrupt': self.bank.is_bankrupt(),
        }
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"\n{'='*60}")
            print(f"第 {info['year']+1} 年 第 {self.month % 12 + 1} 月 | {info['eco_phase']}")
            print(f"{'='*60}")
            print(f"资本金: {info['capital']:.1f}亿 | 累计利润: {info['cumulative_profit']:.1f}亿")
            print(f"不良率: {info['npl_ratio']:.2%} | ROA: {info['roa']:.2%}")
            if info['is_bankrupt']:
                print("⚠️  银行已破产!")


if __name__ == "__main__":
    # 测试环境
    env = LendingEnv(render_mode="human", seed=42)
    obs, info = env.reset()
    
    print("=" * 60)
    print("信贷策略环境测试")
    print(f"状态空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space.shape}")
    print("=" * 60)
    
    total_reward = 0
    for step in range(120):  # 10年
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 12 == 0:
            env.render()
        
        if terminated or truncated:
            print(f"\n模拟结束! 总奖励: {total_reward:.2f}")
            break
    
    print(f"\n最终状态:")
    print(f"  累计利润: {info['cumulative_profit']:.1f}亿")
    print(f"  最终不良率: {info['npl_ratio']:.2%}")
    print(f"  是否破产: {info['is_bankrupt']}")


