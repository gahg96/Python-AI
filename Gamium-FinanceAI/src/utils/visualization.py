"""
Gamium 可视化工具

用于训练过程和结果的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_progress(
    stats: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制训练进度曲线
    
    Args:
        stats: 训练统计列表，每项包含 iteration, avg_reward, avg_profit 等
        save_path: 保存路径
        show: 是否显示图表
    """
    iterations = [s['iteration'] for s in stats]
    rewards = [s['avg_reward'] for s in stats]
    profits = [s['avg_profit'] for s in stats]
    losses = [s.get('avg_loss', 0) for s in stats]
    bankruptcy = [s.get('bankruptcy_rate', 0) for s in stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gamium AlphaZero 训练进度', fontsize=14, fontweight='bold')
    
    # 奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(iterations, rewards, 'b-', linewidth=2, label='平均奖励')
    ax1.fill_between(iterations, rewards, alpha=0.3)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('训练奖励趋势')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 利润曲线
    ax2 = axes[0, 1]
    ax2.plot(iterations, profits, 'g-', linewidth=2, label='平均利润(亿)')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(iterations, profits, alpha=0.3, color='green')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('累计利润 (亿元)')
    ax2.set_title('经营利润趋势')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 损失曲线
    ax3 = axes[1, 0]
    if any(losses):
        ax3.plot(iterations, losses, 'r-', linewidth=2, label='训练损失')
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('损失值')
        ax3.set_title('网络训练损失')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '暂无损失数据', ha='center', va='center', fontsize=12)
        ax3.set_title('网络训练损失')
    
    # 破产率
    ax4 = axes[1, 1]
    ax4.plot(iterations, [b * 100 for b in bankruptcy], 'orange', linewidth=2, label='破产率(%)')
    ax4.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='警戒线')
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('破产率 (%)')
    ax4.set_title('风险控制 - 破产率')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_episode_comparison(
    episodes: Dict[str, List[Dict]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    对比不同策略的单局表现
    
    Args:
        episodes: {策略名称: [月度数据列表]}
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('策略对比分析', fontsize=14, fontweight='bold')
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    
    for idx, (name, history) in enumerate(episodes.items()):
        color = colors[idx % len(colors)]
        months = [h['month'] for h in history]
        
        # NPL 对比
        axes[0, 0].plot(months, [h['npl_ratio'] * 100 for h in history], 
                        color=color, linewidth=2, label=name)
        
        # 利润对比
        cumulative_profit = np.cumsum([h['profit'] for h in history])
        axes[0, 1].plot(months, cumulative_profit, 
                        color=color, linewidth=2, label=name)
        
        # ROA 对比
        axes[1, 0].plot(months, [h['roa'] * 100 for h in history], 
                        color=color, linewidth=2, label=name)
        
        # 奖励对比
        cumulative_reward = np.cumsum([h['reward'] for h in history])
        axes[1, 1].plot(months, cumulative_reward, 
                        color=color, linewidth=2, label=name)
    
    # 设置标签
    axes[0, 0].set_title('不良贷款率 (NPL)')
    axes[0, 0].set_xlabel('月份')
    axes[0, 0].set_ylabel('NPL (%)')
    axes[0, 0].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='警戒线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('累计利润')
    axes[0, 1].set_xlabel('月份')
    axes[0, 1].set_ylabel('利润 (亿元)')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('资产回报率 (ROA)')
    axes[1, 0].set_xlabel('月份')
    axes[1, 0].set_ylabel('ROA (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('累计奖励')
    axes[1, 1].set_xlabel('月份')
    axes[1, 1].set_ylabel('奖励')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_economic_cycle(
    history: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制经济周期与银行表现
    
    Args:
        history: 月度历史数据
        save_path: 保存路径
        show: 是否显示
    """
    months = [h['month'] for h in history]
    gdp = [h['gdp_growth'] * 100 for h in history]
    npl = [h['npl_ratio'] * 100 for h in history]
    profit = [h['profit'] for h in history]
    phases = [h['eco_phase'] for h in history]
    
    # 映射周期阶段到颜色
    phase_colors = {
        '繁荣': '#2ecc71',
        '衰退': '#f39c12', 
        '萧条': '#e74c3c',
        '复苏': '#3498db',
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('经济周期与银行经营表现', fontsize=14, fontweight='bold')
    
    # 为每个阶段添加背景色
    for i in range(len(months) - 1):
        color = phase_colors.get(phases[i], 'gray')
        for ax in axes:
            ax.axvspan(months[i], months[i+1], alpha=0.2, color=color)
    
    # GDP 增长率
    axes[0].plot(months, gdp, 'b-', linewidth=2)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].fill_between(months, gdp, alpha=0.3, 
                         color=['green' if g > 0 else 'red' for g in gdp])
    axes[0].set_ylabel('GDP 增长率 (%)')
    axes[0].set_title('宏观经济：GDP 增长率')
    axes[0].grid(True, alpha=0.3)
    
    # NPL
    axes[1].plot(months, npl, 'r-', linewidth=2)
    axes[1].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='警戒线 5%')
    axes[1].axhline(y=8, color='red', linestyle='--', alpha=0.7, label='危险线 8%')
    axes[1].fill_between(months, npl, alpha=0.3, color='red')
    axes[1].set_ylabel('不良率 (%)')
    axes[1].set_title('风险指标：不良贷款率')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 利润
    axes[2].bar(months, profit, color=['green' if p > 0 else 'red' for p in profit], alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_xlabel('月份')
    axes[2].set_ylabel('月利润 (亿元)')
    axes[2].set_title('经营成果：月度利润')
    axes[2].grid(True, alpha=0.3)
    
    # 添加图例说明周期阶段
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.5, label=p) 
                       for p, c in phase_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', 
               title='经济周期', bbox_to_anchor=(0.99, 0.99))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_comparison_table(results: List[Dict]):
    """
    打印策略对比表格
    
    Args:
        results: 评估结果列表
    """
    print("\n" + "=" * 80)
    print("📊 策略对比分析")
    print("=" * 80)
    print(f"{'策略名称':<15} {'平均奖励':>12} {'平均利润(亿)':>15} {'平均NPL':>12} {'破产率':>10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['avg_reward'], reverse=True):
        print(f"{r['agent_name']:<15} {r['avg_reward']:>12.2f} {r['avg_profit']:>15.1f} "
              f"{r['avg_npl']*100:>11.2f}% {r['bankruptcy_rate']*100:>9.1f}%")
    
    print("=" * 80)


if __name__ == "__main__":
    print("可视化模块测试")
    
    # 生成测试数据
    test_stats = [
        {'iteration': i, 'avg_reward': np.random.randn() + i * 0.1, 
         'avg_profit': np.random.randn() * 10 + i * 0.5, 
         'avg_loss': 0.5 / (i + 1), 'bankruptcy_rate': max(0, 0.3 - i * 0.02)}
        for i in range(50)
    ]
    
    plot_training_progress(test_stats, show=True)

