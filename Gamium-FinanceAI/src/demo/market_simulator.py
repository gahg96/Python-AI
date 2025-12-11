"""
市场环境模拟器
实现GDP、利率、失业率、通胀率等市场因子模拟
支持时间序列变化和宏观经济扰动
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os


@dataclass
class MarketCondition:
    """市场环境"""
    date: str
    gdp_growth: float  # GDP增长率
    base_interest_rate: float  # 基准利率
    unemployment_rate: float  # 失业率
    inflation_rate: float  # 通胀率
    credit_spread: float  # 信用利差
    market_sentiment: float  # 市场情绪 (-1到1)
    economic_cycle: str  # 经济周期: 'expansion', 'peak', 'recession', 'trough'


class MarketSimulator:
    """市场环境模拟器"""
    
    def __init__(self, seed: int = 42, start_date: str = '2020-01-01'):
        """
        初始化市场模拟器
        
        Args:
            seed: 随机种子
            start_date: 开始日期
        """
        self.rng = np.random.default_rng(seed)
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # 基准参数
        self.base_gdp_growth = 0.03  # 3%基准GDP增长
        self.base_interest_rate = 0.05  # 5%基准利率
        self.base_unemployment = 0.05  # 5%基准失业率
        self.base_inflation = 0.02  # 2%基准通胀率
        self.base_credit_spread = 0.02  # 2%基准信用利差
        
        # 经济周期参数
        self.cycle_period = 84  # 7年周期（月）
        self.cycle_phase = 0  # 当前周期相位
    
    def generate_market_condition(self, date: datetime, 
                                  shock: Optional[Dict] = None) -> MarketCondition:
        """
        生成指定日期的市场环境
        
        Args:
            date: 日期
            shock: 外部冲击（可选）
        
        Returns:
            市场环境对象
        """
        # 计算时间偏移（天数）
        days_from_start = (date - self.start_date).days
        months_from_start = days_from_start / 30.44
        
        # 1. 经济周期（7年周期）
        cycle_phase = (months_from_start % self.cycle_period) / self.cycle_period * 2 * np.pi
        
        # 确定经济周期阶段
        if 0 <= cycle_phase < np.pi / 2:
            cycle_stage = 'expansion'  # 扩张期
            cycle_factor = np.sin(cycle_phase)
        elif np.pi / 2 <= cycle_phase < np.pi:
            cycle_stage = 'peak'  # 峰值期
            cycle_factor = 1.0
        elif np.pi <= cycle_phase < 3 * np.pi / 2:
            cycle_stage = 'recession'  # 衰退期
            cycle_factor = -np.sin(cycle_phase)
        else:
            cycle_stage = 'trough'  # 低谷期
            cycle_factor = -1.0
        
        # 2. GDP增长率（受周期影响）
        gdp_growth = self.base_gdp_growth + cycle_factor * 0.02 + \
                    self.rng.normal(0, 0.005)
        gdp_growth = max(0.01, min(gdp_growth, 0.08))
        
        # 3. 基准利率（与GDP相关，滞后效应）
        # 利率通常滞后GDP变化3-6个月
        lagged_cycle_factor = np.sin(cycle_phase - np.pi / 6)
        base_interest_rate = self.base_interest_rate + lagged_cycle_factor * 0.02 + \
                           self.rng.normal(0, 0.002)
        base_interest_rate = max(0.02, min(base_interest_rate, 0.1))
        
        # 4. 失业率（与GDP负相关）
        unemployment_rate = self.base_unemployment - (gdp_growth - self.base_gdp_growth) * 0.5 + \
                          self.rng.normal(0, 0.002)
        unemployment_rate = max(0.02, min(unemployment_rate, 0.1))
        
        # 5. 通胀率（与GDP和利率相关）
        inflation_rate = self.base_inflation + (gdp_growth - self.base_gdp_growth) * 0.3 + \
                        (base_interest_rate - self.base_interest_rate) * 0.2 + \
                        self.rng.normal(0, 0.003)
        inflation_rate = max(0.0, min(inflation_rate, 0.1))
        
        # 6. 信用利差（与失业率和经济周期相关）
        credit_spread = self.base_credit_spread + (unemployment_rate - self.base_unemployment) * 0.5 + \
                       abs(cycle_factor) * 0.01 + \
                       self.rng.normal(0, 0.002)
        credit_spread = max(0.01, min(credit_spread, 0.05))
        
        # 7. 市场情绪（综合指标）
        market_sentiment = cycle_factor * 0.5 + \
                          (gdp_growth - self.base_gdp_growth) * 10 + \
                          (self.base_unemployment - unemployment_rate) * 5 + \
                          self.rng.normal(0, 0.1)
        market_sentiment = max(-1, min(market_sentiment, 1))
        
        # 8. 应用外部冲击
        if shock:
            gdp_growth += shock.get('gdp_shock', 0)
            base_interest_rate += shock.get('rate_shock', 0)
            unemployment_rate += shock.get('unemployment_shock', 0)
            inflation_rate += shock.get('inflation_shock', 0)
            credit_spread += shock.get('credit_spread_shock', 0)
            market_sentiment += shock.get('sentiment_shock', 0)
            
            # 限制范围
            gdp_growth = max(0.01, min(gdp_growth, 0.08))
            base_interest_rate = max(0.02, min(base_interest_rate, 0.1))
            unemployment_rate = max(0.02, min(unemployment_rate, 0.1))
            inflation_rate = max(0.0, min(inflation_rate, 0.1))
            credit_spread = max(0.01, min(credit_spread, 0.05))
            market_sentiment = max(-1, min(market_sentiment, 1))
        
        return MarketCondition(
            date=date.strftime('%Y-%m-%d'),
            gdp_growth=round(gdp_growth, 4),
            base_interest_rate=round(base_interest_rate, 4),
            unemployment_rate=round(unemployment_rate, 4),
            inflation_rate=round(inflation_rate, 4),
            credit_spread=round(credit_spread, 4),
            market_sentiment=round(market_sentiment, 4),
            economic_cycle=cycle_stage
        )
    
    def generate_market_series(self, start_date: str, end_date: str,
                              frequency: str = 'daily',
                              shocks: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        生成市场环境时间序列
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率 ('daily', 'weekly', 'monthly')
            shocks: 外部冲击列表，格式: [{'date': '2020-03-15', 'type': 'crisis', ...}]
        
        Returns:
            市场环境DataFrame
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 确定日期间隔
        if frequency == 'daily':
            delta = timedelta(days=1)
        elif frequency == 'weekly':
            delta = timedelta(weeks=1)
        elif frequency == 'monthly':
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)
        
        # 创建冲击字典（按日期索引）
        shock_dict = {}
        if shocks:
            for shock in shocks:
                shock_date = datetime.strptime(shock['date'], '%Y-%m-%d')
                shock_dict[shock_date] = shock
        
        # 生成时间序列
        conditions = []
        current_date = start
        
        while current_date <= end:
            # 检查是否有冲击
            shock = shock_dict.get(current_date, None)
            
            condition = self.generate_market_condition(current_date, shock)
            conditions.append({
                'date': condition.date,
                'gdp_growth': condition.gdp_growth,
                'base_interest_rate': condition.base_interest_rate,
                'unemployment_rate': condition.unemployment_rate,
                'inflation_rate': condition.inflation_rate,
                'credit_spread': condition.credit_spread,
                'market_sentiment': condition.market_sentiment,
                'economic_cycle': condition.economic_cycle
            })
            
            current_date += delta
        
        df = pd.DataFrame(conditions)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def create_stress_scenario(self, scenario_type: str) -> Dict:
        """
        创建压力测试场景
        
        Args:
            scenario_type: 场景类型
                - 'mild_recession': 轻度衰退
                - 'severe_recession': 严重衰退
                - 'financial_crisis': 金融危机
                - 'inflation_spike': 通胀飙升
                - 'credit_crunch': 信贷紧缩
        
        Returns:
            冲击字典
        """
        scenarios = {
            'mild_recession': {
                'gdp_shock': -0.01,  # GDP下降1%
                'rate_shock': -0.005,  # 利率下降0.5%
                'unemployment_shock': 0.01,  # 失业率上升1%
                'inflation_shock': -0.005,  # 通胀下降0.5%
                'credit_spread_shock': 0.005,  # 信用利差上升0.5%
                'sentiment_shock': -0.3
            },
            'severe_recession': {
                'gdp_shock': -0.03,  # GDP下降3%
                'rate_shock': -0.01,  # 利率下降1%
                'unemployment_shock': 0.03,  # 失业率上升3%
                'inflation_shock': -0.01,  # 通胀下降1%
                'credit_spread_shock': 0.01,  # 信用利差上升1%
                'sentiment_shock': -0.6
            },
            'financial_crisis': {
                'gdp_shock': -0.05,  # GDP下降5%
                'rate_shock': -0.02,  # 利率下降2%
                'unemployment_shock': 0.05,  # 失业率上升5%
                'inflation_shock': -0.02,  # 通胀下降2%
                'credit_spread_shock': 0.02,  # 信用利差上升2%
                'sentiment_shock': -0.8
            },
            'inflation_spike': {
                'gdp_shock': 0.01,  # GDP上升1%
                'rate_shock': 0.02,  # 利率上升2%
                'unemployment_shock': 0.005,  # 失业率略微上升
                'inflation_shock': 0.03,  # 通胀上升3%
                'credit_spread_shock': 0.005,  # 信用利差略微上升
                'sentiment_shock': -0.2
            },
            'credit_crunch': {
                'gdp_shock': -0.02,  # GDP下降2%
                'rate_shock': 0.01,  # 利率上升1%
                'unemployment_shock': 0.02,  # 失业率上升2%
                'inflation_shock': 0.0,
                'credit_spread_shock': 0.02,  # 信用利差大幅上升2%
                'sentiment_shock': -0.5
            }
        }
        
        return scenarios.get(scenario_type, {})
    
    def apply_macro_disturbance(self, base_condition: MarketCondition,
                               disturbance: Dict) -> MarketCondition:
        """
        应用宏观经济扰动
        
        Args:
            base_condition: 基础市场环境
            disturbance: 扰动参数
        
        Returns:
            扰动后的市场环境
        """
        date = datetime.strptime(base_condition.date, '%Y-%m-%d')
        return self.generate_market_condition(date, shock=disturbance)


def main():
    """主函数：测试市场模拟器"""
    print("=" * 80)
    print("市场环境模拟器测试")
    print("=" * 80)
    
    # 创建模拟器
    simulator = MarketSimulator(seed=42, start_date='2020-01-01')
    
    # 生成单日市场环境
    test_date = datetime(2023, 6, 15)
    condition = simulator.generate_market_condition(test_date)
    
    print("\n单日市场环境示例:")
    print(f"日期: {condition.date}")
    print(f"GDP增长率: {condition.gdp_growth:.2%}")
    print(f"基准利率: {condition.base_interest_rate:.2%}")
    print(f"失业率: {condition.unemployment_rate:.2%}")
    print(f"通胀率: {condition.inflation_rate:.2%}")
    print(f"信用利差: {condition.credit_spread:.2%}")
    print(f"市场情绪: {condition.market_sentiment:.2f}")
    print(f"经济周期: {condition.economic_cycle}")
    
    # 生成时间序列
    print("\n生成市场环境时间序列...")
    market_series = simulator.generate_market_series(
        start_date='2020-01-01',
        end_date='2024-12-31',
        frequency='monthly'
    )
    
    print(f"✅ 生成了 {len(market_series)} 个月度数据点")
    print("\n时间序列统计:")
    print(market_series.describe())
    
    # 测试压力场景
    print("\n" + "=" * 80)
    print("压力测试场景")
    print("=" * 80)
    
    scenarios = ['mild_recession', 'severe_recession', 'financial_crisis']
    for scenario_type in scenarios:
        shock = simulator.create_stress_scenario(scenario_type)
        disturbed_condition = simulator.apply_macro_disturbance(condition, shock)
        
        print(f"\n{scenario_type}:")
        print(f"  GDP增长率: {disturbed_condition.gdp_growth:.2%} "
              f"(变化: {disturbed_condition.gdp_growth - condition.gdp_growth:+.2%})")
        print(f"  基准利率: {disturbed_condition.base_interest_rate:.2%} "
              f"(变化: {disturbed_condition.base_interest_rate - condition.base_interest_rate:+.2%})")
        print(f"  失业率: {disturbed_condition.unemployment_rate:.2%} "
              f"(变化: {disturbed_condition.unemployment_rate - condition.unemployment_rate:+.2%})")
        print(f"  信用利差: {disturbed_condition.credit_spread:.2%} "
              f"(变化: {disturbed_condition.credit_spread - condition.credit_spread:+.2%})")
    
    # 保存示例数据
    output_path = 'data/historical/market_conditions.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    market_series.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存市场环境数据到: {output_path}")
    
    return simulator, market_series


if __name__ == '__main__':
    main()

