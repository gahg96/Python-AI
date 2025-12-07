"""
Gamium 神经网络 - AlphaZero 风格的双头网络

策略头 (Policy Head): 输出动作概率分布
价值头 (Value Head): 评估当前状态的价值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = F.relu(self.fc1(x))
        x = self.ln2(x)
        x = self.fc2(x)
        return F.relu(x + residual)


class GamiumNetwork(nn.Module):
    """
    Gamium 决策网络
    
    输入: 状态向量 (22维)
    输出: 
        - 策略 (policy): 动作空间的概率分布
        - 价值 (value): 状态价值估计 [-1, 1]
    
    对于连续动作空间，我们将其离散化为固定的动作选项
    """
    
    # 动作离散化配置
    # [利率调整, 审批率, prime权重, near_prime权重, subprime权重]
    RATE_OPTIONS = [-0.02, -0.01, 0.0, 0.01, 0.02]  # 5个选项
    APPROVAL_OPTIONS = [0.3, 0.5, 0.7, 0.9]          # 4个选项
    SEGMENT_OPTIONS = [                               # 6个组合
        [0.6, 0.3, 0.1],  # 保守：重点优质客户
        [0.4, 0.4, 0.2],  # 平衡
        [0.3, 0.5, 0.2],  # 次优为主
        [0.2, 0.4, 0.4],  # 进取：增加次级
        [0.3, 0.3, 0.4],  # 高风险高收益
        [0.5, 0.35, 0.15], # 稳健
    ]
    
    # 总动作数 = 5 * 4 * 6 = 120
    NUM_ACTIONS = len(RATE_OPTIONS) * len(APPROVAL_OPTIONS) * len(SEGMENT_OPTIONS)
    
    def __init__(
        self,
        state_dim: int = 22,
        hidden_dim: int = 256,
        num_residual_blocks: int = 4
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.NUM_ACTIONS),
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        
        # 构建动作映射表
        self._build_action_map()
    
    def _build_action_map(self):
        """构建动作索引到连续动作的映射"""
        self.action_map = []
        for rate in self.RATE_OPTIONS:
            for approval in self.APPROVAL_OPTIONS:
                for segment in self.SEGMENT_OPTIONS:
                    action = np.array([rate, approval] + segment, dtype=np.float32)
                    self.action_map.append(action)
        self.action_map = np.array(self.action_map)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            policy: 动作概率 [batch_size, num_actions]
            value: 状态价值 [batch_size, 1]
        """
        x = self.input_layer(state)
        
        for block in self.residual_blocks:
            x = block(x)
        
        # 策略输出
        policy_logits = self.policy_head(x)
        policy = F.softmax(policy_logits, dim=-1)
        
        # 价值输出
        value = self.value_head(x)
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测 (用于 MCTS)
        
        Args:
            state: numpy 状态数组
            
        Returns:
            policy: 动作概率分布
            value: 状态价值
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, value = self.forward(state_tensor)
            return policy.squeeze(0).numpy(), value.item()
    
    def action_to_continuous(self, action_idx: int) -> np.ndarray:
        """将离散动作索引转换为连续动作"""
        return self.action_map[action_idx].copy()
    
    def get_action_description(self, action_idx: int) -> str:
        """获取动作的可读描述"""
        action = self.action_map[action_idx]
        rate_adj = action[0]
        approval = action[1]
        prime, near_prime, subprime = action[2], action[3], action[4]
        
        return (f"利率调整:{rate_adj:+.2f} | "
                f"通过率:{approval:.0%} | "
                f"客群[优质:{prime:.0%}, 次优:{near_prime:.0%}, 次级:{subprime:.0%}]")


class GamiumNetworkContinuous(nn.Module):
    """
    连续动作空间版本 (使用 Actor-Critic 风格)
    
    直接输出连续动作的均值和标准差
    """
    
    ACTION_DIM = 5  # [利率调整, 审批率, prime, near_prime, subprime]
    
    def __init__(self, state_dim: int = 22, hidden_dim: int = 256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (策略)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.ACTION_DIM),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(self.ACTION_DIM))
        
        # Critic (价值)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        
        # 动作均值
        action_mean = self.actor_mean(features)
        # 动作标准差
        action_std = torch.exp(self.actor_log_std)
        
        # 状态价值
        value = self.critic(features)
        
        return action_mean, action_std, value


if __name__ == "__main__":
    # 测试网络
    print("=" * 60)
    print("Gamium 神经网络测试")
    print("=" * 60)
    
    network = GamiumNetwork()
    
    print(f"网络参数量: {sum(p.numel() for p in network.parameters()):,}")
    print(f"状态维度: {network.state_dim}")
    print(f"动作数量: {network.NUM_ACTIONS}")
    
    # 测试前向传播
    batch_size = 4
    dummy_state = torch.randn(batch_size, 22)
    
    policy, value = network(dummy_state)
    print(f"\n输入形状: {dummy_state.shape}")
    print(f"策略输出形状: {policy.shape}")
    print(f"价值输出形状: {value.shape}")
    
    # 检查策略是否为有效概率分布
    print(f"\n策略概率和: {policy.sum(dim=-1)}")
    print(f"价值范围: [{value.min():.3f}, {value.max():.3f}]")
    
    # 测试动作转换
    action_idx = 42
    continuous_action = network.action_to_continuous(action_idx)
    print(f"\n动作 {action_idx}: {continuous_action}")
    print(f"描述: {network.get_action_description(action_idx)}")


