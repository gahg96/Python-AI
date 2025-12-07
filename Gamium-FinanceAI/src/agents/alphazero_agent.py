"""
AlphaZero 智能体 - 整合网络和 MCTS 的完整智能体
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random

from .network import GamiumNetwork
from .mcts import MCTS, MCTSConfig, SimpleMCTS


@dataclass
class TrainingExample:
    """训练样本"""
    state: np.ndarray
    action_probs: np.ndarray
    value: float


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples: List[TrainingExample]):
        """添加一局游戏的所有样本"""
        self.buffer.extend(examples)
    
    def sample(self, batch_size: int) -> List[TrainingExample]:
        """随机采样"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class AlphaZeroAgent:
    """
    AlphaZero 智能体
    
    整合神经网络和 MCTS，支持：
    - 自我对弈生成训练数据
    - 网络训练
    - 动作选择
    """
    
    def __init__(
        self,
        state_dim: int = 22,
        hidden_dim: int = 256,
        lr: float = 0.001,
        mcts_config: MCTSConfig = None,
        use_simple_mcts: bool = True  # POC 使用简化版 MCTS
    ):
        self.network = GamiumNetwork(state_dim=state_dim, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        self.mcts_config = mcts_config or MCTSConfig()
        self.use_simple_mcts = use_simple_mcts
        
        if use_simple_mcts:
            self.mcts = SimpleMCTS(self.network, num_simulations=20)
        else:
            self.mcts = MCTS(self.network, self.mcts_config)
        
        self.replay_buffer = ReplayBuffer(max_size=50000)
        
        # 训练统计
        self.train_step = 0
        self.losses = []
    
    def self_play(self, env) -> Tuple[List[TrainingExample], dict]:
        """
        执行一局自我对弈
        
        Returns:
            examples: 训练样本列表
            stats: 游戏统计信息
        """
        examples = []
        
        state, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            # MCTS 搜索获取动作
            if self.use_simple_mcts:
                action, action_probs = self.mcts.get_action(env, state)
            else:
                action, action_probs = self.mcts.get_action(env, state, step=step)
            
            # 记录样本 (value 稍后填充)
            examples.append(TrainingExample(
                state=state.copy(),
                action_probs=action_probs.copy(),
                value=0.0
            ))
            
            # 执行动作
            continuous_action = self.network.action_to_continuous(action)
            next_state, reward, terminated, truncated, info = env.step(continuous_action)
            
            total_reward += reward
            state = next_state
            done = terminated or truncated
            step += 1
        
        # 回填最终价值 (使用折扣累计奖励)
        final_value = total_reward / (step + 1)  # 平均每步奖励
        gamma = 0.99
        
        for i, example in enumerate(reversed(examples)):
            example.value = final_value * (gamma ** i)
        
        stats = {
            'steps': step,
            'total_reward': total_reward,
            'final_npl': info.get('npl_ratio', 0),
            'cumulative_profit': info.get('cumulative_profit', 0),
            'is_bankrupt': info.get('is_bankrupt', False),
        }
        
        return examples, stats
    
    def train_step_batch(self, batch_size: int = 64) -> Optional[float]:
        """执行一步批量训练"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # 采样
        examples = self.replay_buffer.sample(batch_size)
        
        # 准备数据
        states = torch.FloatTensor(np.array([e.state for e in examples]))
        target_policies = torch.FloatTensor(np.array([e.action_probs for e in examples]))
        target_values = torch.FloatTensor(np.array([e.value for e in examples])).unsqueeze(1)
        
        # 前向传播
        self.network.train()
        pred_policies, pred_values = self.network(states)
        
        # 计算损失
        # 策略损失: 交叉熵
        policy_loss = -torch.sum(target_policies * torch.log(pred_policies + 1e-8), dim=1).mean()
        
        # 价值损失: MSE
        value_loss = nn.functional.mse_loss(pred_values, target_values)
        
        # 总损失
        loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.train_step += 1
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        
        return loss_val
    
    def train(
        self,
        env,
        num_iterations: int = 100,
        games_per_iteration: int = 5,
        train_steps_per_iteration: int = 50,
        batch_size: int = 64,
        verbose: bool = True
    ) -> dict:
        """
        完整训练循环
        
        Args:
            env: 环境实例
            num_iterations: 训练迭代次数
            games_per_iteration: 每次迭代的自我对弈局数
            train_steps_per_iteration: 每次迭代的训练步数
            batch_size: 批量大小
            verbose: 是否打印进度
            
        Returns:
            训练统计信息
        """
        all_stats = []
        
        for iteration in range(num_iterations):
            # 1. 自我对弈收集数据
            iteration_examples = []
            iteration_stats = []
            
            for game in range(games_per_iteration):
                examples, stats = self.self_play(env)
                iteration_examples.extend(examples)
                iteration_stats.append(stats)
                
                if verbose:
                    print(f"  游戏 {game+1}/{games_per_iteration}: "
                          f"步数={stats['steps']}, "
                          f"奖励={stats['total_reward']:.1f}, "
                          f"利润={stats['cumulative_profit']:.1f}亿")
            
            # 添加到回放缓冲区
            self.replay_buffer.add(iteration_examples)
            
            # 2. 训练网络
            avg_loss = 0
            for _ in range(train_steps_per_iteration):
                loss = self.train_step_batch(batch_size)
                if loss is not None:
                    avg_loss += loss
            
            avg_loss /= max(1, train_steps_per_iteration)
            self.scheduler.step()
            
            # 3. 统计
            avg_reward = np.mean([s['total_reward'] for s in iteration_stats])
            avg_profit = np.mean([s['cumulative_profit'] for s in iteration_stats])
            bankruptcy_rate = np.mean([s['is_bankrupt'] for s in iteration_stats])
            
            all_stats.append({
                'iteration': iteration,
                'avg_reward': avg_reward,
                'avg_profit': avg_profit,
                'bankruptcy_rate': bankruptcy_rate,
                'avg_loss': avg_loss,
                'buffer_size': len(self.replay_buffer),
            })
            
            if verbose:
                print(f"\n迭代 {iteration+1}/{num_iterations}:")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  平均利润: {avg_profit:.1f}亿")
                print(f"  破产率: {bankruptcy_rate:.1%}")
                print(f"  平均损失: {avg_loss:.4f}")
                print(f"  缓冲区大小: {len(self.replay_buffer)}")
                print("-" * 50)
        
        return {
            'iterations': all_stats,
            'final_avg_reward': all_stats[-1]['avg_reward'] if all_stats else 0,
            'final_avg_profit': all_stats[-1]['avg_profit'] if all_stats else 0,
        }
    
    def select_action(self, state: np.ndarray, env=None, deterministic: bool = False) -> np.ndarray:
        """
        选择动作 (用于评估)
        
        Args:
            state: 当前状态
            env: 环境 (MCTS 需要)
            deterministic: 是否确定性选择
            
        Returns:
            continuous_action: 连续动作
        """
        if deterministic or env is None:
            # 直接使用网络输出
            self.network.eval()
            policy, _ = self.network.predict(state)
            action = np.argmax(policy)
        else:
            # 使用 MCTS
            action, _ = self.mcts.get_action(env, state)
        
        return self.network.action_to_continuous(action)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'losses': self.losses[-1000:],  # 只保存最近的损失
        }, path)
        print(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = checkpoint.get('train_step', 0)
        self.losses = checkpoint.get('losses', [])
        print(f"模型已加载: {path}")


if __name__ == "__main__":
    print("AlphaZero 智能体模块")
    print("=" * 60)
    print("请运行 train.py 进行完整训练测试")


