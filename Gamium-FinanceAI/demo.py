#!/usr/bin/env python3
"""
Gamium 演示脚本

交互式演示 AlphaZero 决策过程，可视化经济周期与银行经营

用法:
    python demo.py                    # 快速演示
    python demo.py --mode compare     # 策略对比模式
    python demo.py --mode interactive # 交互模式
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.lending_env import LendingEnv
from environment.economic_cycle import CyclePhase
from agents.alphazero_agent import AlphaZeroAgent
from agents.baseline_agents import (
    RandomAgent, RuleBasedAgent, ConservativeAgent, AggressiveAgent
)
from utils.visualization import (
    plot_episode_comparison, 
    plot_economic_cycle,
    print_comparison_table
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("提示: 安装 rich 库可获得更好的显示效果: pip install rich")


def format_percent(value, width=8):
    """格式化百分比"""
    return f"{value*100:>{width}.2f}%"


def format_money(value, width=10):
    """格式化金额"""
    return f"{value:>{width}.1f}亿"


def print_state(month, info, action=None, reward=None):
    """打印当前状态"""
    year = month // 12 + 1
    month_in_year = month % 12 + 1
    
    if RICH_AVAILABLE:
        console = Console()
        
        # 周期阶段颜色
        phase_colors = {
            '繁荣': 'green',
            '衰退': 'yellow',
            '萧条': 'red',
            '复苏': 'blue',
        }
        phase = info.get('eco_phase', '未知')
        phase_color = phase_colors.get(phase, 'white')
        
        # 构建状态表格
        table = Table(title=f"📅 第 {year} 年 第 {month_in_year} 月", show_header=False)
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="white")
        
        table.add_row("经济周期", f"[{phase_color}]{phase}[/{phase_color}]")
        table.add_row("资本金", format_money(info.get('capital', 0)))
        table.add_row("累计利润", format_money(info.get('cumulative_profit', 0)))
        table.add_row("不良率 (NPL)", format_percent(info.get('npl_ratio', 0)))
        table.add_row("资产回报率 (ROA)", format_percent(info.get('roa', 0)))
        
        if reward is not None:
            table.add_row("本月奖励", f"{reward:.2f}")
        
        console.print(table)
        
        if action is not None:
            console.print(f"  决策: 利率调整={action[0]:+.2f}, 审批率={action[1]:.0%}, "
                          f"客群=[优质:{action[2]:.0%}, 次优:{action[3]:.0%}, 次级:{action[4]:.0%}]")
    else:
        print(f"\n{'='*50}")
        print(f"📅 第 {year} 年 第 {month_in_year} 月 | 周期: {info.get('eco_phase', '?')}")
        print(f"{'='*50}")
        print(f"  资本金: {info.get('capital', 0):.1f}亿")
        print(f"  累计利润: {info.get('cumulative_profit', 0):.1f}亿")
        print(f"  不良率: {info.get('npl_ratio', 0):.2%}")
        print(f"  ROA: {info.get('roa', 0):.2%}")
        if reward is not None:
            print(f"  奖励: {reward:.2f}")
        if action is not None:
            print(f"  决策: 利率={action[0]:+.3f}, 审批={action[1]:.0%}")


def run_demo(agent, env, verbose=True, sleep_time=0.1):
    """运行单局演示"""
    state, info = env.reset()
    history = []
    total_reward = 0
    
    if verbose:
        print("\n" + "🎮 " * 20)
        print("开始 10 年经营模拟...")
        print("🎮 " * 20 + "\n")
    
    step = 0
    while True:
        # 获取动作
        if hasattr(agent, 'select_action'):
            action = agent.select_action(state, info)
        else:
            action = agent(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 记录历史
        history.append({
            'month': step,
            'eco_phase': info.get('eco_phase', ''),
            'gdp_growth': env.economy.state.gdp_growth,
            'npl_ratio': info.get('npl_ratio', 0),
            'roa': info.get('roa', 0),
            'profit': env.bank.loan_portfolio.net_profit,
            'reward': reward,
            'action': action.copy(),
        })
        
        # 打印状态
        if verbose and step % 6 == 0:  # 每半年打印一次
            print_state(step, info, action, reward)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        state = next_state
        step += 1
        
        if terminated or truncated:
            break
    
    # 最终结果
    if verbose:
        print("\n" + "=" * 60)
        print("📊 模拟结束 - 最终报告")
        print("=" * 60)
        print(f"  总步数: {step} 个月 ({step // 12} 年)")
        print(f"  累计利润: {info.get('cumulative_profit', 0):.1f} 亿")
        print(f"  最终不良率: {info.get('npl_ratio', 0):.2%}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  是否破产: {'是 ❌' if info.get('is_bankrupt') else '否 ✅'}")
        print("=" * 60)
    
    return history, total_reward, info


def compare_strategies(env, num_episodes=3):
    """对比不同策略"""
    print("\n" + "=" * 60)
    print("🏆 策略对比模式")
    print("=" * 60 + "\n")
    
    # 创建智能体
    agents = {
        '随机策略': RandomAgent(seed=42),
        '规则策略': RuleBasedAgent(),
        '保守策略': ConservativeAgent(),
        '激进策略': AggressiveAgent(),
    }
    
    # 尝试加载训练好的 AlphaZero
    try:
        az_agent = AlphaZeroAgent(use_simple_mcts=True)
        model_paths = list(Path("experiments").glob("**/alphazero_model.pt"))
        if model_paths:
            az_agent.load(str(model_paths[-1]))
            
            class AZWrapper:
                def __init__(self, agent, env):
                    self.name = "AlphaZero"
                    self.agent = agent
                    self.env = env
                
                def select_action(self, state, info=None):
                    return self.agent.select_action(state, env=self.env, deterministic=True)
            
            agents['AlphaZero'] = AZWrapper(az_agent, env)
            print("✅ 已加载训练好的 AlphaZero 模型")
        else:
            print("⚠️  未找到训练好的模型，跳过 AlphaZero 对比")
    except Exception as e:
        print(f"⚠️  加载 AlphaZero 失败: {e}")
    
    # 运行对比
    all_results = []
    all_histories = {}
    
    for name, agent in agents.items():
        print(f"\n▶ 评估 {name}...")
        
        episode_rewards = []
        episode_profits = []
        episode_npls = []
        bankruptcies = 0
        
        for ep in range(num_episodes):
            history, reward, info = run_demo(agent, env, verbose=False)
            episode_rewards.append(reward)
            episode_profits.append(info.get('cumulative_profit', 0))
            episode_npls.append(info.get('npl_ratio', 0))
            if info.get('is_bankrupt'):
                bankruptcies += 1
            
            if ep == 0:
                all_histories[name] = history
        
        result = {
            'agent_name': name,
            'avg_reward': np.mean(episode_rewards),
            'avg_profit': np.mean(episode_profits),
            'avg_npl': np.mean(episode_npls),
            'bankruptcy_rate': bankruptcies / num_episodes,
        }
        all_results.append(result)
        
        print(f"   平均奖励: {result['avg_reward']:.2f}, "
              f"平均利润: {result['avg_profit']:.1f}亿, "
              f"破产率: {result['bankruptcy_rate']:.0%}")
    
    # 打印对比表格
    print_comparison_table(all_results)
    
    # 绘制对比图
    print("\n绘制对比图...")
    plot_episode_comparison(all_histories, save_path="strategy_comparison.png", show=True)
    
    return all_results


def interactive_mode(env):
    """交互模式 - 手动决策"""
    print("\n" + "=" * 60)
    print("🎮 交互模式 - 你来当行长!")
    print("=" * 60)
    print("\n说明: 每个月你需要做出决策")
    print("  - 利率调整: -0.02 到 +0.02")
    print("  - 审批通过率: 0.3 到 0.9")
    print("  - 客群分配: 优质/次优/次级的权重 (自动归一化)")
    print("\n输入 'q' 退出, 'auto' 切换到自动决策")
    
    state, info = env.reset()
    total_reward = 0
    auto_mode = False
    auto_agent = RuleBasedAgent()
    
    step = 0
    while True:
        print_state(step, info)
        
        if auto_mode:
            action = auto_agent.select_action(state, info)
            print(f"[自动] 决策: 利率={action[0]:+.3f}, 审批={action[1]:.0%}")
        else:
            try:
                user_input = input("\n请输入决策 (利率,审批,优质,次优,次级) 或命令: ").strip()
                
                if user_input.lower() == 'q':
                    print("退出游戏")
                    break
                elif user_input.lower() == 'auto':
                    auto_mode = True
                    action = auto_agent.select_action(state, info)
                    print("切换到自动模式")
                elif user_input == '':
                    # 默认动作
                    action = np.array([0.0, 0.6, 0.4, 0.4, 0.2], dtype=np.float32)
                    print("使用默认决策")
                else:
                    parts = [float(x) for x in user_input.split(',')]
                    if len(parts) == 2:
                        action = np.array([parts[0], parts[1], 0.4, 0.4, 0.2], dtype=np.float32)
                    elif len(parts) == 5:
                        action = np.array(parts, dtype=np.float32)
                    else:
                        print("输入格式错误，使用默认值")
                        action = np.array([0.0, 0.6, 0.4, 0.4, 0.2], dtype=np.float32)
            except Exception as e:
                print(f"输入错误: {e}，使用默认值")
                action = np.array([0.0, 0.6, 0.4, 0.4, 0.2], dtype=np.float32)
        
        # 执行动作
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"  -> 获得奖励: {reward:.2f}, 累计: {total_reward:.2f}")
        
        if terminated or truncated:
            print("\n" + "=" * 60)
            print("游戏结束!")
            print(f"  总奖励: {total_reward:.2f}")
            print(f"  累计利润: {info.get('cumulative_profit', 0):.1f}亿")
            print(f"  是否破产: {'是' if info.get('is_bankrupt') else '否'}")
            print("=" * 60)
            break
        
        if not auto_mode:
            input("\n按 Enter 继续下个月...")


def main():
    parser = argparse.ArgumentParser(description="Gamium 演示")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "compare", "interactive"],
                        help="运行模式: quick(快速演示), compare(策略对比), interactive(交互)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=3, help="对比模式的评估回合数")
    
    args = parser.parse_args()
    
    # 创建环境
    env = LendingEnv(seed=args.seed)
    
    if args.mode == "quick":
        print("\n🚀 快速演示模式")
        print("使用规则策略模拟 10 年银行经营\n")
        
        agent = RuleBasedAgent()
        history, reward, info = run_demo(agent, env, verbose=True, sleep_time=0.05)
        
        # 绘制经济周期图
        plot_economic_cycle(history, save_path="economic_cycle.png", show=True)
        
    elif args.mode == "compare":
        compare_strategies(env, num_episodes=args.episodes)
        
    elif args.mode == "interactive":
        interactive_mode(env)


if __name__ == "__main__":
    main()

