#!/usr/bin/env python3
"""
Gamium AlphaZero 训练脚本

用法:
    python train.py [--iterations N] [--games G] [--save-dir DIR]

示例:
    python train.py --iterations 50 --games 10  # 快速测试
    python train.py --iterations 200 --games 20  # 完整训练
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.lending_env import LendingEnv
from agents.alphazero_agent import AlphaZeroAgent
from agents.mcts import MCTSConfig
from agents.baseline_agents import (
    RandomAgent, RuleBasedAgent, ConservativeAgent, AggressiveAgent,
    evaluate_agent
)
from utils.visualization import plot_training_progress, print_comparison_table
from utils.logger import GamiumLogger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Gamium AlphaZero 训练")
    
    parser.add_argument("--iterations", type=int, default=30,
                        help="训练迭代次数 (默认: 30)")
    parser.add_argument("--games", type=int, default=5,
                        help="每次迭代的自我对弈局数 (默认: 5)")
    parser.add_argument("--train-steps", type=int, default=50,
                        help="每次迭代的训练步数 (默认: 50)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批量大小 (默认: 64)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率 (默认: 0.001)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="网络隐藏层维度 (默认: 256)")
    parser.add_argument("--mcts-simulations", type=int, default=20,
                        help="MCTS 模拟次数 (默认: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--save-dir", type=str, default="experiments",
                        help="保存目录 (默认: experiments)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="评估回合数 (默认: 5)")
    parser.add_argument("--no-plot", action="store_true",
                        help="不显示图表")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"run_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志
    logger = GamiumLogger("Gamium-Train", log_dir=str(save_dir / "logs"))
    
    logger.info("=" * 60)
    logger.info("🎮 Gamium AlphaZero 训练开始")
    logger.info("=" * 60)
    logger.info(f"训练参数:")
    logger.info(f"  - 迭代次数: {args.iterations}")
    logger.info(f"  - 每次迭代对弈局数: {args.games}")
    logger.info(f"  - 批量大小: {args.batch_size}")
    logger.info(f"  - 学习率: {args.lr}")
    logger.info(f"  - 保存目录: {save_dir}")
    
    # 创建环境
    env = LendingEnv(seed=args.seed)
    logger.info(f"环境已创建: 状态维度={env.observation_space.shape}, 动作维度={env.action_space.shape}")
    
    # 创建 MCTS 配置
    mcts_config = MCTSConfig(
        num_simulations=args.mcts_simulations,
        c_puct=1.5,
        temperature=1.0,
    )
    
    # 创建 AlphaZero 智能体
    agent = AlphaZeroAgent(
        state_dim=22,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        mcts_config=mcts_config,
        use_simple_mcts=True  # POC 使用简化版
    )
    
    logger.info(f"AlphaZero 智能体已创建")
    logger.info(f"  - 网络参数量: {sum(p.numel() for p in agent.network.parameters()):,}")
    logger.info(f"  - 动作空间大小: {agent.network.NUM_ACTIONS}")
    
    # 训练
    logger.info("\n" + "=" * 60)
    logger.info("🚀 开始训练...")
    logger.info("=" * 60 + "\n")
    
    train_stats = agent.train(
        env=env,
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        train_steps_per_iteration=args.train_steps,
        batch_size=args.batch_size,
        verbose=True
    )
    
    # 保存模型
    model_path = save_dir / "alphazero_model.pt"
    agent.save(str(model_path))
    
    # 记录训练指标
    for stat in train_stats['iterations']:
        logger.log_metric("reward", stat['avg_reward'], step=stat['iteration'])
        logger.log_metric("profit", stat['avg_profit'], step=stat['iteration'])
        logger.log_metric("bankruptcy_rate", stat['bankruptcy_rate'], step=stat['iteration'])
        logger.log_metric("loss", stat['avg_loss'], step=stat['iteration'])
    
    logger.save_metrics()
    
    # 评估并与基线对比
    logger.info("\n" + "=" * 60)
    logger.info("📊 评估 AlphaZero vs 基线策略")
    logger.info("=" * 60 + "\n")
    
    # 创建基线智能体
    baseline_agents = [
        RandomAgent(seed=args.seed),
        RuleBasedAgent(),
        ConservativeAgent(),
        AggressiveAgent(),
    ]
    
    results = []
    
    # 评估基线
    for baseline in baseline_agents:
        result = evaluate_agent(baseline, env, num_episodes=args.eval_episodes)
        results.append(result)
        logger.info(f"{result['agent_name']}: 奖励={result['avg_reward']:.2f}, "
                    f"利润={result['avg_profit']:.1f}亿, NPL={result['avg_npl']:.2%}")
    
    # 评估 AlphaZero
    class AlphaZeroWrapper:
        def __init__(self, agent, env):
            self.name = "AlphaZero"
            self.agent = agent
            self.env = env
        
        def select_action(self, state, info=None):
            return self.agent.select_action(state, env=self.env, deterministic=True)
    
    az_wrapper = AlphaZeroWrapper(agent, env)
    az_result = evaluate_agent(az_wrapper, env, num_episodes=args.eval_episodes)
    az_result['agent_name'] = "AlphaZero"
    results.append(az_result)
    
    logger.info(f"AlphaZero: 奖励={az_result['avg_reward']:.2f}, "
                f"利润={az_result['avg_profit']:.1f}亿, NPL={az_result['avg_npl']:.2%}")
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 绘制训练曲线
    if not args.no_plot:
        plot_path = save_dir / "training_progress.png"
        plot_training_progress(
            train_stats['iterations'],
            save_path=str(plot_path),
            show=True
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 训练完成!")
    logger.info(f"   模型保存于: {model_path}")
    logger.info(f"   日志保存于: {save_dir / 'logs'}")
    logger.info("=" * 60)
    
    return train_stats


if __name__ == "__main__":
    main()

