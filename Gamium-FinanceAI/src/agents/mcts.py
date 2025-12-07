"""
蒙特卡洛树搜索 (MCTS) - AlphaZero 核心决策算法

通过模拟和回溯，在动作空间中找到最优决策
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy


@dataclass
class MCTSConfig:
    """MCTS 配置"""
    num_simulations: int = 50       # 每次决策的模拟次数 (POC降低以加速)
    c_puct: float = 1.5             # 探索常数
    dirichlet_alpha: float = 0.3    # Dirichlet 噪声参数
    dirichlet_epsilon: float = 0.25 # 噪声权重
    temperature: float = 1.0        # 动作选择温度
    temperature_threshold: int = 30 # 温度降为0的步数阈值


class MCTSNode:
    """MCTS 树节点"""
    
    def __init__(
        self,
        prior: float,
        parent: Optional['MCTSNode'] = None,
        action: Optional[int] = None
    ):
        self.prior = prior           # 先验概率 (来自策略网络)
        self.parent = parent
        self.action = action         # 到达此节点的动作
        
        self.visit_count = 0
        self.total_value = 0.0       # 累计价值
        self.children: Dict[int, 'MCTSNode'] = {}
        
        self.is_expanded = False
        self.is_terminal = False
    
    @property
    def q_value(self) -> float:
        """平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """使用 PUCT 公式选择子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for action, child in self.children.items():
            # PUCT 公式: Q + c * P * sqrt(N_parent) / (1 + N_child)
            exploration = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = child.q_value + exploration
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs: np.ndarray):
        """扩展节点"""
        for action, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action] = MCTSNode(
                    prior=prob,
                    parent=self,
                    action=action
                )
        self.is_expanded = True
    
    def backup(self, value: float):
        """回溯更新"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
            value = -value  # 对抗性质 (这里是单玩家，保持正值)
            value = value  # 单玩家环境，不取反


class MCTS:
    """
    蒙特卡洛树搜索
    
    用于在给定状态下，通过神经网络指导的模拟搜索，
    找到最优动作
    """
    
    def __init__(self, network, config: MCTSConfig = None):
        """
        Args:
            network: GamiumNetwork 实例
            config: MCTS 配置
        """
        self.network = network
        self.config = config or MCTSConfig()
    
    def search(self, env, root_state: np.ndarray) -> np.ndarray:
        """
        执行 MCTS 搜索
        
        Args:
            env: 环境实例 (需要支持 clone)
            root_state: 当前状态
            
        Returns:
            action_probs: 动作概率分布
        """
        # 创建根节点
        root = MCTSNode(prior=1.0)
        
        # 用策略网络评估根节点
        policy, value = self.network.predict(root_state)
        
        # 添加 Dirichlet 噪声增加探索
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(policy)
        )
        policy = (
            (1 - self.config.dirichlet_epsilon) * policy +
            self.config.dirichlet_epsilon * noise
        )
        
        root.expand(policy)
        
        # 执行模拟
        for _ in range(self.config.num_simulations):
            node = root
            sim_env = self._clone_env(env)
            search_path = [node]
            
            # 1. 选择 (Selection) - 沿树向下直到叶节点
            while node.is_expanded and not node.is_terminal:
                action, node = node.select_child(self.config.c_puct)
                
                # 在模拟环境中执行动作
                continuous_action = self.network.action_to_continuous(action)
                _, _, terminated, truncated, _ = sim_env.step(continuous_action)
                
                if terminated or truncated:
                    node.is_terminal = True
                
                search_path.append(node)
            
            # 2. 评估 (Evaluation)
            if node.is_terminal:
                # 终止状态，使用实际奖励
                value = 0.0  # 可以用累计奖励
            else:
                # 非终止状态，使用网络估值
                leaf_state = sim_env._get_observation()
                policy, value = self.network.predict(leaf_state)
                
                # 3. 扩展 (Expansion)
                node.expand(policy)
            
            # 4. 回溯 (Backpropagation)
            for node in reversed(search_path):
                node.visit_count += 1
                node.total_value += value
        
        # 计算动作概率 (基于访问次数)
        action_probs = np.zeros(self.network.NUM_ACTIONS)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        # 温度调节
        if self.config.temperature == 0:
            # 贪婪选择
            best_action = np.argmax(action_probs)
            action_probs = np.zeros_like(action_probs)
            action_probs[best_action] = 1.0
        else:
            # 根据温度调整
            action_probs = action_probs ** (1 / self.config.temperature)
            action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        return action_probs
    
    def get_action(self, env, state: np.ndarray, step: int = 0) -> Tuple[int, np.ndarray]:
        """
        获取动作
        
        Args:
            env: 环境实例
            state: 当前状态
            step: 当前步数 (用于温度退火)
            
        Returns:
            action: 选择的动作索引
            action_probs: 动作概率分布 (用于训练)
        """
        # 温度退火
        if step < self.config.temperature_threshold:
            temp = self.config.temperature
        else:
            temp = 0.1  # 后期降低温度，更贪婪
        
        old_temp = self.config.temperature
        self.config.temperature = temp
        
        action_probs = self.search(env, state)
        
        self.config.temperature = old_temp
        
        # 采样动作
        if temp > 0:
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            action = np.argmax(action_probs)
        
        return action, action_probs
    
    def _clone_env(self, env):
        """克隆环境用于模拟"""
        return copy.deepcopy(env)


class SimpleMCTS:
    """
    简化版 MCTS (不需要完整树，只做单步前瞻)
    
    用于 POC 快速验证
    """
    
    def __init__(self, network, num_simulations: int = 20):
        self.network = network
        self.num_simulations = num_simulations
    
    def get_action(self, env, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        简化版动作选择：
        1. 用网络获取先验概率
        2. 对 top-k 动作进行少量模拟
        3. 选择平均奖励最高的动作
        """
        policy, value = self.network.predict(state)
        
        # 选择概率最高的 10 个动作进行模拟
        top_k = 10
        top_actions = np.argsort(policy)[-top_k:]
        
        action_values = {}
        
        for action in top_actions:
            values = []
            
            for _ in range(self.num_simulations // top_k):
                sim_env = copy.deepcopy(env)
                continuous_action = self.network.action_to_continuous(action)
                _, reward, _, _, _ = sim_env.step(continuous_action)
                values.append(reward)
            
            action_values[action] = np.mean(values) if values else 0.0
        
        # 选择最优动作
        best_action = max(action_values.keys(), key=lambda a: action_values[a])
        
        # 构建概率分布 (用于训练)
        action_probs = np.zeros(self.network.NUM_ACTIONS)
        for action, value in action_values.items():
            action_probs[action] = max(0, value + 5)  # 确保非负
        action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        return best_action, action_probs


if __name__ == "__main__":
    print("MCTS 模块测试")
    print("=" * 60)
    
    # 需要在有环境和网络的情况下测试
    print("MCTS 类已定义，可通过主程序测试")


