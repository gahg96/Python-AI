"""
基线智能体 - 用于与 AlphaZero 对比的传统策略

包括：
1. RandomAgent: 随机策略
2. RuleBasedAgent: 基于规则的策略 (模拟现有银行系统)
3. ConservativeAgent: 保守策略
4. AggressiveAgent: 激进策略
"""

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态观测
            info: 附加信息
            
        Returns:
            action: [利率调整, 审批率, prime权重, near_prime权重, subprime权重]
        """
        pass


class RandomAgent(BaseAgent):
    """随机智能体"""
    
    def __init__(self, seed: int = None):
        super().__init__("随机策略")
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        return np.array([
            self.rng.uniform(-0.02, 0.02),   # 利率调整
            self.rng.uniform(0.3, 0.9),       # 审批率
            self.rng.uniform(0.1, 0.6),       # prime
            self.rng.uniform(0.2, 0.6),       # near_prime
            self.rng.uniform(0.1, 0.4),       # subprime
        ], dtype=np.float32)


class RuleBasedAgent(BaseAgent):
    """
    基于规则的智能体 - 模拟传统银行决策系统
    
    规则：
    - 经济繁荣期：提高利率，放宽审批，增加次级客户
    - 经济衰退期：降低利率，收紧审批，聚焦优质客户
    - NPL 过高时：大幅收紧
    """
    
    def __init__(self):
        super().__init__("规则策略")
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        # 解析状态
        # 经济状态: [gdp_growth, interest_rate, unemployment, inflation, spread, phase(4)]
        gdp_growth = state[0]
        unemployment = state[2]
        
        # 周期阶段 (one-hot: boom, recession, depression, recovery)
        phase_idx = np.argmax(state[5:9])
        
        # 银行状态
        # [capital, assets, loans, npl_ratio, car, seg1, seg2, seg3, profit, writeoffs]
        npl_ratio = state[12]  # 索引需要根据实际调整
        
        # 默认动作
        rate_adj = 0.0
        approval = 0.6
        segments = [0.4, 0.4, 0.2]  # prime, near_prime, subprime
        
        # 根据经济周期调整
        if phase_idx == 0:  # BOOM
            rate_adj = 0.01
            approval = 0.75
            segments = [0.35, 0.40, 0.25]
        elif phase_idx == 1:  # RECESSION
            rate_adj = -0.005
            approval = 0.55
            segments = [0.50, 0.35, 0.15]
        elif phase_idx == 2:  # DEPRESSION
            rate_adj = -0.015
            approval = 0.40
            segments = [0.60, 0.30, 0.10]
        elif phase_idx == 3:  # RECOVERY
            rate_adj = 0.0
            approval = 0.65
            segments = [0.40, 0.40, 0.20]
        
        # NPL 过高时收紧
        if npl_ratio > 0.05:
            approval *= 0.85
            segments = [0.55, 0.35, 0.10]
        if npl_ratio > 0.08:
            approval *= 0.75
            segments = [0.65, 0.30, 0.05]
        
        return np.array([rate_adj, approval] + segments, dtype=np.float32)


class ConservativeAgent(BaseAgent):
    """
    保守策略智能体
    
    特点：
    - 始终低风险
    - 专注优质客户
    - 低审批率
    - 反周期利率 (衰退期保持利率)
    """
    
    def __init__(self):
        super().__init__("保守策略")
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        # 基础保守设置
        rate_adj = 0.005  # 略高于市场
        approval = 0.45   # 低通过率
        segments = [0.65, 0.30, 0.05]  # 重点优质，极少次级
        
        # 经济萧条时更保守
        phase_idx = np.argmax(state[5:9])
        if phase_idx == 2:  # DEPRESSION
            approval = 0.35
            segments = [0.75, 0.22, 0.03]
        
        return np.array([rate_adj, approval] + segments, dtype=np.float32)


class AggressiveAgent(BaseAgent):
    """
    激进策略智能体
    
    特点：
    - 追求高收益
    - 增加次级客户
    - 高审批率
    - 顺周期 (繁荣期更激进)
    """
    
    def __init__(self):
        super().__init__("激进策略")
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        # 基础激进设置
        rate_adj = 0.015  # 高利率
        approval = 0.80   # 高通过率
        segments = [0.25, 0.40, 0.35]  # 大量次级客户
        
        # 经济繁荣时更激进
        phase_idx = np.argmax(state[5:9])
        if phase_idx == 0:  # BOOM
            approval = 0.90
            segments = [0.20, 0.35, 0.45]
        elif phase_idx == 2:  # DEPRESSION
            # 即使萧条也维持较高风险偏好
            approval = 0.65
            segments = [0.35, 0.40, 0.25]
        
        return np.array([rate_adj, approval] + segments, dtype=np.float32)


class AdaptiveAgent(BaseAgent):
    """
    自适应策略智能体
    
    特点：
    - 根据历史表现动态调整
    - 记住过去的成功策略
    """
    
    def __init__(self):
        super().__init__("自适应策略")
        self.history = []
        self.best_params = None
        self.exploration_rate = 0.3
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        # 有一定概率探索
        if np.random.random() < self.exploration_rate or self.best_params is None:
            # 随机生成
            action = np.array([
                np.random.uniform(-0.01, 0.01),
                np.random.uniform(0.4, 0.8),
                np.random.uniform(0.3, 0.5),
                np.random.uniform(0.3, 0.5),
                np.random.uniform(0.1, 0.3),
            ], dtype=np.float32)
        else:
            # 使用最佳参数 + 小扰动
            action = self.best_params + np.random.normal(0, 0.02, 5)
        
        return np.clip(action, [-0.02, 0.3, 0, 0, 0], [0.02, 0.9, 1, 1, 1])
    
    def update(self, reward: float, action: np.ndarray):
        """更新策略"""
        self.history.append((reward, action))
        
        if len(self.history) > 10:
            # 找到最佳动作
            best_idx = np.argmax([h[0] for h in self.history[-50:]])
            self.best_params = self.history[-50:][best_idx][1]
            
            # 降低探索率
            self.exploration_rate = max(0.1, self.exploration_rate * 0.99)


def evaluate_agent(agent: BaseAgent, env, num_episodes: int = 10) -> dict:
    """
    评估智能体性能
    
    Args:
        agent: 智能体实例
        env: 环境实例
        num_episodes: 评估回合数
        
    Returns:
        评估统计
    """
    results = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, info)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'cumulative_profit': info['cumulative_profit'],
            'final_npl': info['npl_ratio'],
            'is_bankrupt': info['is_bankrupt'],
        })
    
    # 汇总
    return {
        'agent_name': agent.name,
        'avg_reward': np.mean([r['total_reward'] for r in results]),
        'avg_profit': np.mean([r['cumulative_profit'] for r in results]),
        'avg_npl': np.mean([r['final_npl'] for r in results]),
        'bankruptcy_rate': np.mean([r['is_bankrupt'] for r in results]),
        'std_reward': np.std([r['total_reward'] for r in results]),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from environment.lending_env import LendingEnv
    
    print("=" * 60)
    print("基线智能体评估")
    print("=" * 60)
    
    env = LendingEnv(seed=42)
    
    agents = [
        RandomAgent(seed=42),
        RuleBasedAgent(),
        ConservativeAgent(),
        AggressiveAgent(),
    ]
    
    for agent in agents:
        result = evaluate_agent(agent, env, num_episodes=5)
        print(f"\n{result['agent_name']}:")
        print(f"  平均奖励: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  平均利润: {result['avg_profit']:.1f}亿")
        print(f"  平均不良率: {result['avg_npl']:.2%}")
        print(f"  破产率: {result['bankruptcy_rate']:.1%}")


