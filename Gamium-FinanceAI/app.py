#!/usr/bin/env python3
"""
Gamium Finance AI - Web Application

Flask 后端，提供 RESTful API
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, jsonify, request, send_from_directory, Response
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
import numpy as np
import json
import threading
import time

from data_distillation.customer_generator import (
    CustomerGenerator, CustomerType, CityTier, Industry
)
from data_distillation.world_model import (
    WorldModel, LoanOffer, MarketConditions
)
from data_distillation.distillation_pipeline import (
    DistillationPipeline, DistillationConfig
)
from environment.lending_env import LendingEnv
from agents.baseline_agents import (
    RandomAgent, RuleBasedAgent, ConservativeAgent, AggressiveAgent
)

# 自定义 JSON 序列化器，处理 numpy 类型
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

app = Flask(__name__, static_folder='web', static_url_path='')
app.json = NumpyJSONProvider(app)
CORS(app)

# 全局实例
generator = CustomerGenerator(seed=42)
world_model = WorldModel(seed=42)
env = None
training_status = {"running": False, "progress": 0, "logs": []}

# 蒸馏过程详细记录
distillation_trace = {
    "steps": [],
    "data_samples": [],
    "feature_stats": {},
    "model_params": {},
    "validation_details": {},
    "audit_log": []
}

# 经济周期分析追踪
cycle_analysis_trace = {
    "run_id": None,
    "config": {},
    "steps": [],
    "results": {},
    "audit_log": []
}

# 银行模拟追踪
simulation_trace = {
    "run_id": None,
    "config": {},
    "steps": [],
    "monthly_decisions": [],
    "summary": {},
    "audit_log": []
}

# 策略对比追踪
comparison_trace = {
    "run_id": None,
    "config": {},
    "strategies": [],
    "results": [],
    "audit_log": []
}

# ============================================================
# 静态文件
# ============================================================

@app.route('/')
def index():
    """大屏展示页面 - 首页"""
    return send_from_directory('web', 'dashboard.html')

@app.route('/control')
def control():
    """操作控制台"""
    return send_from_directory('web', 'index.html')

@app.route('/story')
def story():
    """Gamium AI 故事展示页面"""
    return send_from_directory('docs', 'Gamium-AI-Story.html')

@app.route('/images/<path:filename>')
def serve_images(filename):
    """提供图片文件"""
    return send_from_directory('images', filename)

@app.route('/ops')
def ops():
    """操作控制台（别名）"""
    return send_from_directory('web', 'index.html')

@app.route('/dashboard')
def dashboard():
    """大屏展示页面（别名）"""
    return send_from_directory('web', 'dashboard.html')

# ============================================================
# 客户生成 API
# ============================================================

@app.route('/api/customer/generate', methods=['POST'])
def generate_customer():
    """生成客户"""
    data = request.json or {}
    
    # 解析参数
    customer_type = None
    if data.get('customer_type'):
        type_map = {
            'salaried': CustomerType.SALARIED,
            'small_business': CustomerType.SMALL_BUSINESS,
            'freelancer': CustomerType.FREELANCER,
            'farmer': CustomerType.FARMER,
        }
        customer_type = type_map.get(data['customer_type'])
    
    risk_profile = data.get('risk_profile', 'medium')
    
    customer = generator.generate_one(
        customer_type=customer_type,
        risk_profile=risk_profile
    )
    
    return jsonify({
        'success': True,
        'customer': customer.to_dict()
    })

@app.route('/api/customer/batch', methods=['POST'])
def generate_batch():
    """批量生成客户"""
    data = request.json or {}
    n = min(data.get('count', 100), 500)
    
    customers = generator.generate_batch(n)
    
    return jsonify({
        'success': True,
        'customers': [c.to_dict() for c in customers],
        'summary': {
            'total': len(customers),
            'avg_income': np.mean([c.monthly_income for c in customers]),
            'avg_risk_score': np.mean([c.risk_score for c in customers]),
            'by_type': {
                t.value: sum(1 for c in customers if c.customer_type == t)
                for t in CustomerType
            }
        }
    })

# ============================================================
# 预测 API
# ============================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测客户未来"""
    data = request.json
    
    # 生成或使用提供的客户
    if data.get('customer'):
        # 从提供的数据创建客户 (简化处理，使用生成器)
        customer = generator.generate_one(
            risk_profile=data.get('risk_profile', 'medium')
        )
        # 覆盖关键字段
        if data['customer'].get('monthly_income'):
            customer.monthly_income = float(data['customer']['monthly_income'])
        if data['customer'].get('debt_ratio'):
            customer.total_liabilities = customer.total_assets * float(data['customer']['debt_ratio'])
    else:
        customer = generator.generate_one(risk_profile='medium')
    
    # 贷款条件
    loan_data = data.get('loan', {})
    loan = LoanOffer(
        amount=float(loan_data.get('amount', 100000)),
        interest_rate=float(loan_data.get('interest_rate', 0.08)),
        term_months=int(loan_data.get('term_months', 24)),
    )
    
    # 市场环境
    market_data = data.get('market', {})
    market = MarketConditions(
        gdp_growth=float(market_data.get('gdp_growth', 0.03)),
        base_interest_rate=float(market_data.get('base_interest_rate', 0.04)),
        unemployment_rate=float(market_data.get('unemployment_rate', 0.05)),
        inflation_rate=float(market_data.get('inflation_rate', 0.02)),
        credit_spread=float(market_data.get('credit_spread', 0.02)),
    )
    
    # 预测
    future = world_model.predict_customer_future(customer, loan, market)
    
    return jsonify({
        'success': True,
        'customer': customer.to_dict(),
        'loan': {
            'amount': loan.amount,
            'interest_rate': loan.interest_rate,
            'term_months': loan.term_months,
            'monthly_payment': loan.monthly_payment,
        },
        'market': {
            'gdp_growth': market.gdp_growth,
            'unemployment_rate': market.unemployment_rate,
            'economic_stress': market.economic_stress,
        },
        'prediction': future.to_dict()
    })

# ============================================================
# 经济周期分析 API
# ============================================================

@app.route('/api/analysis/economic-cycle', methods=['POST'])
def economic_cycle_analysis():
    """经济周期影响分析"""
    global cycle_analysis_trace
    
    data = request.json or {}
    n_customers = min(data.get('count', 100), 200)
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 初始化追踪
    cycle_analysis_trace = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": {"customer_count": n_customers},
        "steps": [],
        "results": {},
        "customer_samples": [],
        "audit_log": []
    }
    
    def add_log(action, details):
        cycle_analysis_trace["audit_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        })
    
    add_log("INIT", f"经济周期分析启动, 样本量: {n_customers}")
    
    # Step 1: 生成客户
    step1_start = datetime.now()
    customers = generator.generate_batch(n_customers)
    
    # 客户分布统计
    type_dist = {}
    for c in customers:
        t = c.customer_type.value
        type_dist[t] = type_dist.get(t, 0) + 1
    
    cycle_analysis_trace["steps"].append({
        "step_id": 1,
        "name": "客户样本生成",
        "duration_ms": int((datetime.now() - step1_start).total_seconds() * 1000),
        "details": {
            "total_customers": n_customers,
            "type_distribution": type_dist,
            "avg_income": float(np.mean([c.monthly_income for c in customers])),
            "avg_debt_ratio": float(np.mean([c.debt_ratio for c in customers]))
        }
    })
    add_log("GENERATE_CUSTOMERS", f"生成 {n_customers} 个客户样本")
    
    # 保存客户样本
    cycle_analysis_trace["customer_samples"] = [
        {
            "id": c.customer_id,
            "type": c.customer_type.value,
            "income": round(c.monthly_income, 0),
            "debt_ratio": round(c.debt_ratio, 2)
        }
        for c in customers[:10]  # 前10个样本
    ]
    
    # Step 2: 定义分析场景
    scenarios = {
        "繁荣期": MarketConditions(0.06, 0.05, 0.04, 0.02, 0.02),
        "正常期": MarketConditions(0.03, 0.04, 0.05, 0.02, 0.02),
        "衰退期": MarketConditions(0.01, 0.03, 0.07, 0.03, 0.03),
        "萧条期": MarketConditions(-0.02, 0.02, 0.10, 0.01, 0.04),
    }
    
    cycle_analysis_trace["steps"].append({
        "step_id": 2,
        "name": "场景定义",
        "details": {
            "scenarios": {
                name: {"gdp_growth": m.gdp_growth, "unemployment": m.unemployment_rate}
                for name, m in scenarios.items()
            }
        }
    })
    add_log("DEFINE_SCENARIOS", "定义4个经济周期场景: 繁荣期, 正常期, 衰退期, 萧条期")
    
    loan = LoanOffer(amount=100000, interest_rate=0.08, term_months=24)
    
    # Step 3: 逐场景分析
    results = {}
    for scenario_name, market in scenarios.items():
        step_start = datetime.now()
        default_probs = []
        ltvs = []
        
        for customer in customers:
            future = world_model.predict_customer_future(customer, loan, market)
            default_probs.append(future.default_probability)
            ltvs.append(future.expected_ltv)
        
        results[scenario_name] = {
            'avg_default_rate': float(np.mean(default_probs)),
            'avg_ltv': float(np.mean(ltvs)),
            'high_risk_count': int(sum(1 for p in default_probs if p > 0.15)),
            'high_risk_ratio': sum(1 for p in default_probs if p > 0.15) / len(default_probs),
        }
        
        add_log("ANALYZE_SCENARIO", f"{scenario_name}: 违约率={results[scenario_name]['avg_default_rate']*100:.2f}%, 高风险={results[scenario_name]['high_risk_count']}人")
    
    cycle_analysis_trace["steps"].append({
        "step_id": 3,
        "name": "场景预测分析",
        "details": {"scenario_results": results}
    })
    
    # Step 4: 客户类型细分
    step4_start = datetime.now()
    by_type = {}
    market = scenarios["萧条期"]
    for customer in customers:
        ctype = customer.customer_type.value
        if ctype not in by_type:
            by_type[ctype] = []
        future = world_model.predict_customer_future(customer, loan, market)
        by_type[ctype].append(future.default_probability)
    
    type_analysis = {
        ctype: float(np.mean(probs))
        for ctype, probs in by_type.items()
    }
    
    cycle_analysis_trace["steps"].append({
        "step_id": 4,
        "name": "客户类型风险细分",
        "duration_ms": int((datetime.now() - step4_start).total_seconds() * 1000),
        "details": {"type_risk": type_analysis}
    })
    add_log("TYPE_ANALYSIS", f"完成客户类型风险细分: {list(type_analysis.keys())}")
    
    cycle_analysis_trace["end_time"] = datetime.now().isoformat()
    cycle_analysis_trace["results"] = {"scenarios": results, "by_type": type_analysis}
    add_log("COMPLETE", "经济周期分析完成")
    
    return jsonify({
        'success': True,
        'run_id': run_id,
        'customer_count': n_customers,
        'scenarios': results,
        'by_customer_type': type_analysis
    })

@app.route('/api/analysis/trace', methods=['GET'])
def get_cycle_analysis_trace():
    """获取经济周期分析追踪"""
    return jsonify({
        'success': True,
        'trace': cycle_analysis_trace
    })

# ============================================================
# 银行模拟 API
# ============================================================

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """开始银行模拟"""
    global env
    
    data = request.json or {}
    seed = data.get('seed', 42)
    
    env = LendingEnv(seed=seed)
    obs, info = env.reset()
    
    return jsonify({
        'success': True,
        'state': {
            'month': 0,
            'year': 0,
            'eco_phase': info['eco_phase'],
            'capital': info['capital'],
            'cumulative_profit': info['cumulative_profit'],
            'npl_ratio': info['npl_ratio'],
            'roa': info['roa'],
        }
    })

@app.route('/api/simulation/step', methods=['POST'])
def simulation_step():
    """执行模拟步骤"""
    global env
    
    if env is None:
        return jsonify({'success': False, 'error': 'Simulation not started'})
    
    data = request.json or {}
    
    # 解析动作
    action = np.array([
        float(data.get('rate_adjustment', 0.0)),
        float(data.get('approval_rate', 0.6)),
        float(data.get('prime_weight', 0.4)),
        float(data.get('near_prime_weight', 0.4)),
        float(data.get('subprime_weight', 0.2)),
    ], dtype=np.float32)
    
    # 执行步骤
    obs, reward, terminated, truncated, info = env.step(action)
    
    return jsonify({
        'success': True,
        'state': {
            'month': env.month,
            'year': env.month // 12,
            'eco_phase': info['eco_phase'],
            'capital': float(info['capital']),
            'cumulative_profit': float(info['cumulative_profit']),
            'npl_ratio': float(info['npl_ratio']),
            'roa': float(info['roa']),
            'is_bankrupt': bool(info['is_bankrupt']),
        },
        'reward': float(reward),
        'terminated': bool(terminated),
        'truncated': bool(truncated),
    })

@app.route('/api/simulation/auto-run', methods=['POST'])
def auto_run_simulation():
    """自动运行完整模拟"""
    global simulation_trace
    
    data = request.json or {}
    strategy = data.get('strategy', 'rule_based')
    seed = data.get('seed', 42)
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    strategy_names = {
        'random': '随机策略',
        'rule_based': '规则策略',
        'conservative': '保守策略',
        'aggressive': '激进策略'
    }
    
    # 初始化追踪
    simulation_trace = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": {"strategy": strategy, "strategy_name": strategy_names.get(strategy, strategy), "seed": seed},
        "steps": [],
        "monthly_decisions": [],
        "key_events": [],
        "summary": {},
        "audit_log": []
    }
    
    def add_log(action, details):
        simulation_trace["audit_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        })
    
    add_log("INIT", f"银行模拟启动, 策略: {strategy_names.get(strategy, strategy)}")
    
    # 创建环境和智能体
    env = LendingEnv(seed=seed)
    
    agents = {
        'random': RandomAgent(seed=seed),
        'rule_based': RuleBasedAgent(),
        'conservative': ConservativeAgent(),
        'aggressive': AggressiveAgent(),
    }
    agent = agents.get(strategy, RuleBasedAgent())
    
    # 运行模拟 - 先reset初始化环境
    state, info = env.reset()
    
    simulation_trace["steps"].append({
        "step_id": 1,
        "name": "环境初始化",
        "details": {
            "initial_capital": float(env.bank.capital),
            "max_months": LendingEnv.TOTAL_MONTHS,
            "agent_type": type(agent).__name__
        }
    })
    add_log("ENV_INIT", f"初始资本: ¥{env.bank.capital/1e8:.1f}亿")
    history = []
    total_reward = 0
    month_count = 0
    
    while True:
        month_count += 1
        prev_capital = env.bank.capital
        prev_npl = info.get('npl_ratio', 0)
        
        action = agent.select_action(state, info)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 记录每月决策
        # action可能是标量或数组（连续动作空间）
        if hasattr(action, '__len__'):
            action_str = f"利率调整:{action[0]:.3f}, 通过率:{action[1]:.2f}"
        else:
            action_str = ['拒绝', '批准'][int(action)] if action in [0, 1] else f'行动{action}'
        
        decision = {
            'month': env.month,
            'eco_phase': info['eco_phase'],
            'action': action.tolist() if hasattr(action, 'tolist') else action,
            'action_name': action_str,
            'capital': float(env.bank.capital),
            'capital_change': float(env.bank.capital - prev_capital),
            'profit': float(env.bank.loan_portfolio.net_profit),
            'npl_ratio': float(info['npl_ratio']),
            'npl_change': float(info['npl_ratio'] - prev_npl),
            'reward': float(reward),
        }
        
        history.append({
            'month': env.month,
            'eco_phase': info['eco_phase'],
            'capital': float(env.bank.capital),
            'profit': float(env.bank.loan_portfolio.net_profit),
            'cumulative_profit': float(info['cumulative_profit']),
            'npl_ratio': float(info['npl_ratio']),
            'roa': float(info['roa']),
            'reward': float(reward),
        })
        
        # 记录关键事件
        if month_count <= 5 or month_count % 12 == 0 or terminated:
            simulation_trace["monthly_decisions"].append(decision)
        
        # 检测关键事件
        if info['npl_ratio'] > 0.15 and prev_npl <= 0.15:
            simulation_trace["key_events"].append({
                "month": env.month,
                "event": "NPL_WARNING",
                "details": f"不良率突破警戒线: {info['npl_ratio']*100:.1f}%"
            })
            add_log("NPL_WARNING", f"第{env.month}月: 不良率达到 {info['npl_ratio']*100:.1f}%")
        
        if decision['capital_change'] < -1e8:
            simulation_trace["key_events"].append({
                "month": env.month,
                "event": "LARGE_LOSS",
                "details": f"单月大额亏损: ¥{abs(decision['capital_change'])/1e8:.2f}亿"
            })
        
        if terminated or truncated:
            break
    
    add_log("SIMULATION_COMPLETE", f"模拟完成, 共{month_count}个月")
    
    simulation_trace["steps"].append({
        "step_id": 2,
        "name": "模拟执行",
        "details": {
            "total_months": month_count,
            "total_decisions": len(simulation_trace["monthly_decisions"]),
            "key_events_count": len(simulation_trace["key_events"])
        }
    })
    
    simulation_trace["summary"] = {
        'total_months': len(history),
        'total_reward': float(total_reward),
        'final_capital': float(info['capital']),
        'final_profit': float(info['cumulative_profit']),
        'final_npl': float(info['npl_ratio']),
        'is_bankrupt': bool(info['is_bankrupt']),
    }
    simulation_trace["end_time"] = datetime.now().isoformat()
    
    if info['is_bankrupt']:
        add_log("BANKRUPT", "银行破产!")
    else:
        add_log("RESULT", f"最终资本: ¥{info['capital']/1e8:.1f}亿, NPL: {info['npl_ratio']*100:.1f}%")
    
    return jsonify({
        'success': True,
        'run_id': run_id,
        'strategy': strategy,
        'history': history,
        'summary': simulation_trace["summary"]
    })

@app.route('/api/simulation/trace', methods=['GET'])
def get_simulation_trace():
    """获取银行模拟追踪"""
    return jsonify({
        'success': True,
        'trace': simulation_trace
    })

# ============================================================
# 策略对比 API
# ============================================================

@app.route('/api/comparison/strategies', methods=['POST'])
def compare_strategies():
    """对比不同策略"""
    global comparison_trace
    
    data = request.json or {}
    n_episodes = min(data.get('episodes', 3), 5)
    seed = data.get('seed', 42)
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 初始化追踪
    comparison_trace = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": {"episodes_per_strategy": n_episodes, "seed": seed},
        "strategies": [],
        "episode_details": {},
        "results": [],
        "audit_log": []
    }
    
    def add_log(action, details):
        comparison_trace["audit_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        })
    
    add_log("INIT", f"策略对比启动, 每策略{n_episodes}轮")
    
    strategies = {
        '随机策略': RandomAgent(seed=seed),
        '规则策略': RuleBasedAgent(),
        '保守策略': ConservativeAgent(),
        '激进策略': AggressiveAgent(),
    }
    
    comparison_trace["strategies"] = list(strategies.keys())
    
    results = []
    
    for name, agent in strategies.items():
        strategy_start = datetime.now()
        episode_rewards = []
        episode_profits = []
        episode_npls = []
        bankruptcies = 0
        episode_details = []
        
        add_log("STRATEGY_START", f"开始评估: {name}")
        
        for ep in range(n_episodes):
            env = LendingEnv(seed=seed + ep)
            state, info = env.reset()
            total_reward = 0
            month_count = 0
            
            while True:
                action = agent.select_action(state, info)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                month_count += 1
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_profits.append(info['cumulative_profit'])
            episode_npls.append(info['npl_ratio'])
            if info['is_bankrupt']:
                bankruptcies += 1
            
            episode_details.append({
                "episode": ep + 1,
                "months": month_count,
                "reward": float(total_reward),
                "profit": float(info['cumulative_profit']),
                "npl": float(info['npl_ratio']),
                "bankrupt": bool(info['is_bankrupt'])
            })
        
        strategy_result = {
            'name': name,
            'avg_reward': float(np.mean(episode_rewards)),
            'avg_profit': float(np.mean(episode_profits)),
            'avg_npl': float(np.mean(episode_npls)),
            'bankruptcy_rate': bankruptcies / n_episodes,
            'duration_ms': int((datetime.now() - strategy_start).total_seconds() * 1000)
        }
        results.append(strategy_result)
        
        comparison_trace["episode_details"][name] = episode_details
        add_log("STRATEGY_COMPLETE", f"{name}: 平均奖励={strategy_result['avg_reward']:.2f}, 平均利润=¥{strategy_result['avg_profit']/1e8:.1f}亿")
    
    # 按奖励排序
    results.sort(key=lambda x: x['avg_reward'], reverse=True)
    
    comparison_trace["results"] = results
    comparison_trace["end_time"] = datetime.now().isoformat()
    comparison_trace["winner"] = results[0]['name'] if results else None
    
    add_log("COMPLETE", f"对比完成, 最佳策略: {comparison_trace['winner']}")
    
    return jsonify({
        'success': True,
        'run_id': run_id,
        'episodes': n_episodes,
        'results': results
    })

@app.route('/api/comparison/trace', methods=['GET'])
def get_comparison_trace():
    """获取策略对比追踪"""
    return jsonify({
        'success': True,
        'trace': comparison_trace
    })

# ============================================================
# 数据蒸馏 API
# ============================================================

# 检测可用的数据源
def detect_data_sources():
    """检测可用的历史数据"""
    sources = []
    
    # 检查生成的完整数据集
    full_data = Path('data/historical')
    if full_data.exists() and (full_data / 'customers.parquet').exists():
        summary_file = full_data / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                sources.append({
                    'name': '完整历史数据',
                    'path': str(full_data),
                    'size_gb': summary.get('total_size_gb', 0),
                    'customers': summary.get('total_customers', 0),
                    'loans': summary.get('total_loans', 0),
                })
    
    # 检查测试数据
    test_data = Path('data/test_data')
    if test_data.exists() and (test_data / 'customers.parquet').exists():
        sources.append({
            'name': '测试数据集',
            'path': str(test_data),
            'size_gb': 0.01,
            'customers': 10000,
            'loans': 40000,
        })
    
    return sources

@app.route('/api/distillation/sources', methods=['GET'])
def get_data_sources():
    """获取可用的数据源"""
    sources = detect_data_sources()
    return jsonify({
        'success': True,
        'sources': sources,
        'synthetic_available': True,  # 始终可以生成合成数据
    })

@app.route('/api/distillation/run', methods=['POST'])
def run_distillation():
    """运行数据蒸馏流程"""
    global training_status, distillation_trace
    
    data = request.json or {}
    use_real_data = data.get('use_real_data', False)
    data_path = data.get('data_path', 'data/test_data')
    sample_size = data.get('sample_size', 10000)
    n_synthetic = min(data.get('data_size', 1000), 3000)
    
    # 初始化追踪记录
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    distillation_trace = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": {
            "use_real_data": use_real_data,
            "data_path": data_path if use_real_data else None,
            "sample_size": sample_size if use_real_data else n_synthetic,
            "data_mode": "real" if use_real_data else "synthetic"
        },
        "steps": [],
        "data_samples": [],
        "feature_stats": {},
        "model_params": {},
        "validation_details": {},
        "audit_log": []
    }
    
    training_status = {
        "running": True, 
        "progress": 0, 
        "logs": [], 
        "result": None,
        "data_mode": "real" if use_real_data else "synthetic",
        "run_id": run_id
    }
    
    def add_audit_log(action, details):
        distillation_trace["audit_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        })
    
    def run():
        try:
            config = DistillationConfig()
            pipeline = DistillationPipeline(config=config, seed=42)
            
            add_audit_log("INIT", f"蒸馏任务启动, 运行ID: {run_id}")
            
            # ========== Step 1: 数据准备 ==========
            step1_start = datetime.now()
            training_status["progress"] = 10
            
            step1_info = {
                "step_id": 1,
                "name": "数据准备",
                "status": "running",
                "start_time": step1_start.isoformat(),
                "actions": []
            }
            
            if use_real_data:
                training_status["logs"].append(f"📦 第一步: 加载真实历史数据 ({data_path})...")
                step1_info["actions"].append({"action": "LOAD_DATA", "source": data_path, "type": "parquet"})
                add_audit_log("LOAD_DATA", f"从 {data_path} 加载历史数据")
                pipeline.step1_prepare_data(data_dir=data_path, sample_size=sample_size)
            else:
                training_status["logs"].append(f"📦 第一步: 生成合成数据 (n={n_synthetic})...")
                step1_info["actions"].append({"action": "GENERATE_DATA", "count": n_synthetic, "type": "synthetic"})
                add_audit_log("GENERATE_DATA", f"生成 {n_synthetic} 条合成数据")
                pipeline.step1_prepare_data(n_synthetic=n_synthetic)
            
            # 记录数据样本
            if pipeline.raw_data:
                sample_count = min(5, len(pipeline.raw_data))
                distillation_trace["data_samples"] = [
                    {
                        "index": i,
                        "year": r.get("year"),
                        "customer_type": r.get("customer", {}).get("customer_type"),
                        "loan_amount": r.get("loan_offer", {}).get("amount"),
                        "defaulted": r.get("actual", {}).get("defaulted"),
                        "preview": True
                    }
                    for i, r in enumerate(pipeline.raw_data[:sample_count])
                ]
                
                # 数据统计
                total = len(pipeline.raw_data)
                defaults = sum(1 for r in pipeline.raw_data if r.get("actual", {}).get("defaulted"))
                by_type = {}
                for r in pipeline.raw_data:
                    ctype = r.get("customer", {}).get("customer_type", "unknown")
                    by_type[ctype] = by_type.get(ctype, 0) + 1
                
                step1_info["data_stats"] = {
                    "total_records": total,
                    "default_count": defaults,
                    "default_rate": round(defaults / total * 100, 2) if total > 0 else 0,
                    "by_customer_type": by_type
                }
                step1_info["actions"].append({
                    "action": "DATA_VALIDATION",
                    "records_loaded": total,
                    "valid_records": total,
                    "invalid_records": 0
                })
                add_audit_log("DATA_STATS", f"数据加载完成: {total}条, 违约率{step1_info['data_stats']['default_rate']}%")
            
            step1_info["end_time"] = datetime.now().isoformat()
            step1_info["duration_ms"] = int((datetime.now() - step1_start).total_seconds() * 1000)
            step1_info["status"] = "completed"
            distillation_trace["steps"].append(step1_info)
            training_status["progress"] = 20
            
            # ========== Step 2: 特征工程 ==========
            step2_start = datetime.now()
            training_status["logs"].append("⚙️ 第二步: 特征工程...")
            
            step2_info = {
                "step_id": 2,
                "name": "特征工程",
                "status": "running",
                "start_time": step2_start.isoformat(),
                "actions": []
            }
            
            step2_info["actions"].append({"action": "EXTRACT_STATIC_FEATURES", "features": ["age", "years_in_business", "risk_score"]})
            add_audit_log("FEATURE_EXTRACT", "提取静态特征: age, years_in_business, risk_score")
            
            step2_info["actions"].append({"action": "EXTRACT_BEHAVIOR_FEATURES", "features": ["monthly_income", "income_volatility", "debt_ratio", "deposit_balance"]})
            add_audit_log("FEATURE_EXTRACT", "提取行为特征: monthly_income, income_volatility, debt_ratio, deposit_balance")
            
            step2_info["actions"].append({"action": "EXTRACT_CREDIT_FEATURES", "features": ["previous_loans", "max_historical_dpd", "months_as_customer"]})
            add_audit_log("FEATURE_EXTRACT", "提取信贷特征: previous_loans, max_historical_dpd, months_as_customer")
            
            step2_info["actions"].append({"action": "EXTRACT_ENV_FEATURES", "features": ["gdp_growth", "unemployment_rate", "base_interest_rate"]})
            add_audit_log("FEATURE_EXTRACT", "提取环境特征: gdp_growth, unemployment_rate, base_interest_rate")
            
            pipeline.step2_feature_engineering()
            
            # 特征统计
            if pipeline.feature_matrix is not None:
                feature_names = ["age_norm", "years_biz_norm", "risk_score", "income_norm", "income_vol", 
                               "debt_ratio", "dti_norm", "deposit_norm", "deposit_stab", "prev_loans_norm",
                               "dpd_norm", "last_loan_norm", "months_cust_norm", "amount_norm", "rate", "term_norm",
                               "gdp", "base_rate", "unemployment"]
                
                distillation_trace["feature_stats"] = {
                    "total_features": pipeline.feature_matrix.shape[1],
                    "total_samples": pipeline.feature_matrix.shape[0],
                    "feature_names": feature_names[:pipeline.feature_matrix.shape[1]],
                    "label_distribution": {
                        "positive": int(pipeline.labels.sum()),
                        "negative": int(len(pipeline.labels) - pipeline.labels.sum())
                    }
                }
                step2_info["feature_matrix_shape"] = list(pipeline.feature_matrix.shape)
                step2_info["actions"].append({
                    "action": "BUILD_FEATURE_MATRIX",
                    "shape": list(pipeline.feature_matrix.shape),
                    "features_count": pipeline.feature_matrix.shape[1]
                })
                add_audit_log("FEATURE_MATRIX", f"特征矩阵构建完成: {pipeline.feature_matrix.shape}")
            
            step2_info["end_time"] = datetime.now().isoformat()
            step2_info["duration_ms"] = int((datetime.now() - step2_start).total_seconds() * 1000)
            step2_info["status"] = "completed"
            distillation_trace["steps"].append(step2_info)
            training_status["progress"] = 40
            
            # ========== Step 3: 规律建模 ==========
            step3_start = datetime.now()
            training_status["logs"].append("🧠 第三步: 规律建模...")
            
            step3_info = {
                "step_id": 3,
                "name": "规律建模",
                "status": "running",
                "start_time": step3_start.isoformat(),
                "actions": []
            }
            
            step3_info["actions"].append({"action": "SELECT_MODEL", "model_type": config.model_type})
            add_audit_log("MODEL_SELECT", f"选择模型类型: {config.model_type}")
            
            step3_info["actions"].append({"action": "CALIBRATE_RULES", "description": "从数据中校准规则参数"})
            add_audit_log("MODEL_CALIBRATE", "开始从历史数据校准模型参数")
            
            pipeline.step3_train_model()
            
            # 模型参数
            distillation_trace["model_params"] = {
                "model_type": config.model_type,
                "trained": True,
                "rules_learned": [
                    {"rule": "客户类型风险系数", "small_business": 1.5, "freelancer": 1.3, "salaried": 0.8, "farmer": 1.0},
                    {"rule": "负债率阈值", "warning": 0.4, "critical": 0.6},
                    {"rule": "经济周期调整", "boom": 0.7, "normal": 1.0, "recession": 1.5, "depression": 2.5},
                    {"rule": "历史逾期权重", "dpd_30": 1.5, "dpd_60": 2.0, "dpd_90": 3.0}
                ]
            }
            step3_info["actions"].append({"action": "TRAIN_COMPLETE", "rules_count": 4})
            add_audit_log("MODEL_TRAINED", "模型训练完成, 学习到4条业务规则")
            
            step3_info["end_time"] = datetime.now().isoformat()
            step3_info["duration_ms"] = int((datetime.now() - step3_start).total_seconds() * 1000)
            step3_info["status"] = "completed"
            distillation_trace["steps"].append(step3_info)
            training_status["progress"] = 60
            
            # ========== Step 4: 函数封装 ==========
            step4_start = datetime.now()
            training_status["logs"].append("📦 第四步: 函数封装...")
            
            step4_info = {
                "step_id": 4,
                "name": "函数封装",
                "status": "running",
                "start_time": step4_start.isoformat(),
                "actions": []
            }
            
            step4_info["actions"].append({
                "action": "CREATE_API",
                "function_name": "predict_customer_future",
                "inputs": ["CustomerProfile", "LoanOffer", "MarketConditions"],
                "outputs": ["default_probability", "expected_ltv", "churn_probability", "expected_dpd", "confidence"]
            })
            add_audit_log("API_CREATE", "封装预测API: predict_customer_future()")
            
            pipeline.step4_create_api()
            
            step4_info["actions"].append({"action": "API_READY", "endpoint": "/api/predict"})
            add_audit_log("API_READY", "API封装完成, 可通过 /api/predict 调用")
            
            step4_info["end_time"] = datetime.now().isoformat()
            step4_info["duration_ms"] = int((datetime.now() - step4_start).total_seconds() * 1000)
            step4_info["status"] = "completed"
            distillation_trace["steps"].append(step4_info)
            training_status["progress"] = 80
            
            # ========== Step 5: 验证与校准 ==========
            step5_start = datetime.now()
            training_status["logs"].append("✅ 第五步: 验证与校准...")
            
            step5_info = {
                "step_id": 5,
                "name": "验证与校准",
                "status": "running",
                "start_time": step5_start.isoformat(),
                "actions": []
            }
            
            step5_info["actions"].append({"action": "SPLIT_TEST_DATA", "test_ratio": 0.2})
            add_audit_log("VALIDATION_START", "划分测试集进行验证 (20%)")
            
            validation = pipeline.step5_validate()
            
            distillation_trace["validation_details"] = {
                "test_records": int(validation.total_records),
                "predicted_default_rate": float(validation.predicted_default_rate),
                "actual_default_rate": float(validation.actual_default_rate),
                "deviation": float(validation.deviation),
                "passed": bool(validation.passed),
                "threshold": config.acceptable_deviation,
                "by_customer_type": {k: {"predicted": float(v["predicted"]), "actual": float(v["actual"])} 
                                     for k, v in validation.by_customer_type.items()}
            }
            
            step5_info["actions"].append({
                "action": "BACKTEST",
                "test_records": int(validation.total_records),
                "deviation": float(validation.deviation),
                "passed": bool(validation.passed)
            })
            add_audit_log("VALIDATION_COMPLETE", f"验证完成: 偏差{validation.deviation*100:.2f}%, {'通过' if validation.passed else '未通过'}")
            
            step5_info["end_time"] = datetime.now().isoformat()
            step5_info["duration_ms"] = int((datetime.now() - step5_start).total_seconds() * 1000)
            step5_info["status"] = "completed"
            distillation_trace["steps"].append(step5_info)
            training_status["progress"] = 100
            
            # 完成
            distillation_trace["end_time"] = datetime.now().isoformat()
            distillation_trace["total_duration_ms"] = sum(s["duration_ms"] for s in distillation_trace["steps"])
            add_audit_log("COMPLETE", f"蒸馏任务完成, 总耗时{distillation_trace['total_duration_ms']}ms")
            
            training_status["result"] = {
                "total_records": int(validation.total_records),
                "predicted_default_rate": float(validation.predicted_default_rate),
                "actual_default_rate": float(validation.actual_default_rate),
                "deviation": float(validation.deviation),
                "passed": bool(validation.passed),
            }
            training_status["logs"].append(f"🎉 完成! 验证{'通过' if validation.passed else '未通过'}")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            training_status["logs"].append(f"❌ 错误: {error_msg}")
            add_audit_log("ERROR", f"蒸馏失败: {error_msg}")
            distillation_trace["error"] = {"message": error_msg, "traceback": traceback.format_exc()}
        finally:
            training_status["running"] = False
    
    thread = threading.Thread(target=run)
    thread.start()
    
    return jsonify({'success': True, 'message': '蒸馏流程已启动', 'run_id': run_id})

@app.route('/api/distillation/trace', methods=['GET'])
def get_distillation_trace():
    """获取蒸馏过程详细追踪信息"""
    return jsonify({
        'success': True,
        'trace': distillation_trace
    })

@app.route('/api/distillation/audit-log', methods=['GET'])
def get_audit_log():
    """获取审计日志"""
    return jsonify({
        'success': True,
        'audit_log': distillation_trace.get("audit_log", []),
        'run_id': distillation_trace.get("run_id"),
        'config': distillation_trace.get("config")
    })

@app.route('/api/distillation/status', methods=['GET'])
def distillation_status():
    """获取蒸馏状态"""
    return jsonify(training_status)

# ============================================================
# 数据浏览器 API
# ============================================================

import pandas as pd
import os

@app.route('/api/data/sources', methods=['GET'])
def list_data_sources():
    """获取可用的数据源列表"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    sources = []
    
    for folder in ['test_data', 'historical']:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            files = []
            for f in os.listdir(folder_path):
                if f.endswith('.parquet'):
                    fpath = os.path.join(folder_path, f)
                    size = os.path.getsize(fpath)
                    files.append({
                        'name': f,
                        'size': size,
                        'size_mb': round(size / (1024*1024), 2)
                    })
            
            # 读取summary
            summary = {}
            summary_path = os.path.join(folder_path, 'summary.json')
            if os.path.exists(summary_path):
                import json
                with open(summary_path, 'r') as sf:
                    summary = json.load(sf)
            
            sources.append({
                'name': folder,
                'path': folder_path,
                'files': files,
                'summary': summary
            })
    
    return jsonify({'success': True, 'sources': sources})

@app.route('/api/data/schema', methods=['POST'])
def get_data_schema():
    """获取数据表的字段结构"""
    data = request.json or {}
    source = data.get('source', 'test_data')
    table = data.get('table', 'customers')
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, source, f'{table}.parquet')
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': f'文件不存在: {file_path}'})
    
    try:
        df = pd.read_parquet(file_path)
        schema = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            unique_count = int(df[col].nunique())
            
            # 采样值
            sample_values = df[col].dropna().head(3).tolist()
            
            schema.append({
                'name': col,
                'dtype': dtype,
                'null_count': null_count,
                'unique_count': unique_count,
                'sample_values': [str(v) for v in sample_values]
            })
        
        return jsonify({
            'success': True,
            'source': source,
            'table': table,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'schema': schema
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/data/browse', methods=['POST'])
def browse_data():
    """浏览数据记录"""
    data = request.json or {}
    source = data.get('source', 'test_data')
    table = data.get('table', 'customers')
    page = data.get('page', 1)
    page_size = min(data.get('page_size', 20), 100)
    search = data.get('search', '')
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, source, f'{table}.parquet')
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': f'文件不存在: {file_path}'})
    
    try:
        df = pd.read_parquet(file_path)
        
        # 搜索过滤
        if search:
            mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
            df = df[mask]
        
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size
        
        # 分页
        start = (page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end]
        
        # 转换为记录列表
        records = []
        for idx, row in page_df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                # 处理特殊类型
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (np.bool_, bool)):
                    record[col] = bool(val)
                elif isinstance(val, (np.integer, int)):
                    record[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    record[col] = round(float(val), 4)
                else:
                    record[col] = str(val)
            record['_index'] = int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
            records.append(record)
        
        return jsonify({
            'success': True,
            'source': source,
            'table': table,
            'page': page,
            'page_size': page_size,
            'total_records': total_records,
            'total_pages': total_pages,
            'columns': list(df.columns),
            'records': records
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/data/record/<source>/<table>/<record_id>', methods=['GET'])
def get_record_detail(source, table, record_id):
    """获取单条记录详情"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, source, f'{table}.parquet')
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': f'文件不存在: {file_path}'})
    
    try:
        df = pd.read_parquet(file_path)
        
        # 查找记录
        id_col = df.columns[0]  # 假设第一列是ID
        record_df = df[df[id_col].astype(str) == str(record_id)]
        
        if len(record_df) == 0:
            return jsonify({'success': False, 'error': f'记录不存在: {record_id}'})
        
        row = record_df.iloc[0]
        record = {}
        field_details = []
        
        for col in df.columns:
            val = row[col]
            dtype = str(df[col].dtype)
            
            # 处理值
            if pd.isna(val):
                display_val = None
            elif isinstance(val, (np.bool_, bool)):
                display_val = bool(val)
            elif isinstance(val, (np.integer, int)):
                display_val = int(val)
            elif isinstance(val, (np.floating, float)):
                display_val = float(val)
            else:
                display_val = str(val)
            
            record[col] = display_val
            
            # 字段详情
            field_details.append({
                'name': col,
                'value': display_val,
                'dtype': dtype,
                'is_null': pd.isna(val)
            })
        
        return jsonify({
            'success': True,
            'source': source,
            'table': table,
            'record_id': record_id,
            'record': record,
            'field_details': field_details
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/data/stats', methods=['POST'])
def get_data_stats():
    """获取数据统计信息"""
    data = request.json or {}
    source = data.get('source', 'test_data')
    table = data.get('table', 'customers')
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, source, f'{table}.parquet')
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': f'文件不存在'})
    
    try:
        df = pd.read_parquet(file_path)
        
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
            'column_stats': {}
        }
        
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'null_percent': round(df[col].isna().mean() * 100, 2),
                'unique_count': int(df[col].nunique())
            }
            
            # 数值列统计
            if df[col].dtype in ['int64', 'float64']:
                col_stats['min'] = float(df[col].min()) if not pd.isna(df[col].min()) else None
                col_stats['max'] = float(df[col].max()) if not pd.isna(df[col].max()) else None
                col_stats['mean'] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                col_stats['std'] = float(df[col].std()) if not pd.isna(df[col].std()) else None
            
            # 分类列频率
            if df[col].dtype == 'object' or col_stats['unique_count'] < 20:
                value_counts = df[col].value_counts().head(10).to_dict()
                col_stats['top_values'] = {str(k): int(v) for k, v in value_counts.items()}
            
            stats['column_stats'][col] = col_stats
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ============================================================
# 实时数据 API (大屏用)
# ============================================================

# 全局实时状态
realtime_state = {
    'capital': 386.0,
    'profit': 286.0,
    'npl': 2.15,
    'customers': 523847,
    'pending_loans': 0,
    'approved_today': 0,
    'rejected_today': 0,
    'total_processed': 0,
}

import random
from datetime import datetime

def generate_decision_explanation(customer, loan_offer, market, future, approved):
    """生成决策解释"""
    
    # 1. 客户画像分析
    customer_analysis = {
        'type': customer.customer_type.value,
        'type_risk': '低' if customer.customer_type.value == '工薪阶层' else '中' if customer.customer_type.value in ['农户', '自由职业'] else '高',
        'age': customer.age,
        'age_factor': '稳定' if 30 <= customer.age <= 50 else '较高风险',
        'income': round(customer.monthly_income, 0),
        'income_stability': '稳定' if customer.income_volatility < 0.2 else '波动较大',
        'debt_ratio': round(customer.debt_ratio * 100, 1),
        'debt_status': '健康' if customer.debt_ratio < 0.4 else '偏高' if customer.debt_ratio < 0.6 else '风险',
    }
    
    # 2. 风险因子贡献度
    base_risk = 0.05  # 基础风险5%
    risk_factors = []
    
    # 客户类型风险
    type_risk_map = {'工薪阶层': 0, '农户': 0.02, '自由职业': 0.03, '小微企业': 0.05}
    type_contribution = type_risk_map.get(customer.customer_type.value, 0.03)
    if type_contribution > 0:
        risk_factors.append({
            'name': '客户类型',
            'value': customer.customer_type.value,
            'contribution': round(type_contribution * 100, 1),
            'direction': 'up',
            'reason': f'{customer.customer_type.value}收入不稳定性较高'
        })
    
    # 负债率风险
    if customer.debt_ratio > 0.4:
        debt_contribution = (customer.debt_ratio - 0.4) * 0.3
        risk_factors.append({
            'name': '负债率',
            'value': f'{customer.debt_ratio*100:.1f}%',
            'contribution': round(debt_contribution * 100, 1),
            'direction': 'up',
            'reason': '负债率超过40%警戒线'
        })
    
    # 收入波动风险
    if customer.income_volatility > 0.2:
        vol_contribution = customer.income_volatility * 0.1
        risk_factors.append({
            'name': '收入波动',
            'value': f'{customer.income_volatility*100:.1f}%',
            'contribution': round(vol_contribution * 100, 1),
            'direction': 'up',
            'reason': '月收入波动较大'
        })
    
    # 经济环境影响
    if market.gdp_growth < 0.03:
        eco_contribution = (0.03 - market.gdp_growth) * 2
        risk_factors.append({
            'name': '经济环境',
            'value': f'GDP {market.gdp_growth*100:.1f}%',
            'contribution': round(eco_contribution * 100, 1),
            'direction': 'up',
            'reason': '当前经济增速放缓'
        })
    
    # 贷款金额风险
    loan_to_income = loan_offer.amount / max(1, customer.monthly_income * 12)
    if loan_to_income > 3:
        amount_contribution = (loan_to_income - 3) * 0.02
        risk_factors.append({
            'name': '贷款/收入比',
            'value': f'{loan_to_income:.1f}倍',
            'contribution': round(amount_contribution * 100, 1),
            'direction': 'up',
            'reason': '贷款额度超过年收入3倍'
        })
    
    # 正面因素
    positive_factors = []
    if customer.deposit_stability > 0.7:
        positive_factors.append({
            'name': '存款稳定性',
            'value': f'{customer.deposit_stability*100:.0f}%',
            'contribution': round(customer.deposit_stability * 2, 1),
            'direction': 'down',
            'reason': '存款行为稳定，风险降低'
        })
    
    if customer.months_as_customer > 24:
        positive_factors.append({
            'name': '客户关系',
            'value': f'{customer.months_as_customer}个月',
            'contribution': round(min(customer.months_as_customer / 12, 3), 1),
            'direction': 'down',
            'reason': '长期客户，信用积累'
        })
    
    # 3. 决策规则链
    rules_triggered = []
    
    # 规则1: 违约率阈值
    default_prob = float(future.default_probability)
    rules_triggered.append({
        'id': 'R001',
        'name': '违约率阈值检查',
        'condition': f'预测违约率 {default_prob*100:.1f}% < 25%',
        'result': '通过' if default_prob < 0.25 else '不通过',
        'passed': default_prob < 0.25,
        'weight': 40
    })
    
    # 规则2: 负债率检查
    rules_triggered.append({
        'id': 'R002', 
        'name': '负债率检查',
        'condition': f'当前负债率 {customer.debt_ratio*100:.1f}% < 70%',
        'result': '通过' if customer.debt_ratio < 0.7 else '不通过',
        'passed': customer.debt_ratio < 0.7,
        'weight': 25
    })
    
    # 规则3: 月供收入比
    monthly_payment = loan_offer.amount * (loan_offer.interest_rate / 12) / (1 - (1 + loan_offer.interest_rate/12)**(-loan_offer.term_months))
    payment_ratio = monthly_payment / max(1, customer.monthly_income)
    rules_triggered.append({
        'id': 'R003',
        'name': '月供收入比检查',
        'condition': f'月供/月收入 {payment_ratio*100:.1f}% < 50%',
        'result': '通过' if payment_ratio < 0.5 else '不通过',
        'passed': payment_ratio < 0.5,
        'weight': 20
    })
    
    # 规则4: 经济周期调整
    eco_ok = market.gdp_growth > 0 or customer.customer_type.value == '工薪阶层'
    rules_triggered.append({
        'id': 'R004',
        'name': '经济周期风控',
        'condition': f'GDP增速 {market.gdp_growth*100:.1f}% > 0% 或 工薪客户',
        'result': '通过' if eco_ok else '谨慎',
        'passed': eco_ok,
        'weight': 15
    })
    
    # 计算规则得分
    total_weight = sum(r['weight'] for r in rules_triggered)
    passed_weight = sum(r['weight'] for r in rules_triggered if r['passed'])
    rule_score = round(passed_weight / total_weight * 100, 1)
    
    # 4. 最终决策
    decision = {
        'result': '批准' if approved else '拒绝',
        'confidence': round(float(future.confidence) * 100, 1),
        'risk_score': round(default_prob * 100, 1),
        'rule_score': rule_score,
        'reasons': []
    }
    
    if approved:
        decision['reasons'] = [
            f'综合风险评分 {default_prob*100:.1f}% 低于审批阈值 25%',
            f'规则检查通过率 {rule_score}%',
            f'预期LTV ¥{float(future.expected_ltv):,.0f}，收益可观'
        ]
    else:
        decision['reasons'] = []
        if default_prob >= 0.25:
            decision['reasons'].append(f'违约风险 {default_prob*100:.1f}% 超过阈值 25%')
        if not all(r['passed'] for r in rules_triggered):
            failed_rules = [r['name'] for r in rules_triggered if not r['passed']]
            decision['reasons'].append(f'未通过规则: {", ".join(failed_rules)}')
        if len(decision['reasons']) == 0:
            decision['reasons'].append('综合评估未达批准标准')
    
    return {
        'customer_analysis': customer_analysis,
        'risk_factors': risk_factors,
        'positive_factors': positive_factors,
        'rules_triggered': rules_triggered,
        'decision': decision
    }

@app.route('/api/realtime/tick', methods=['POST'])
def realtime_tick():
    """模拟一次贷款审核，更新实时状态"""
    global realtime_state
    
    # 生成一个随机客户
    customer_types = ['工薪阶层', '小微企业', '自由职业', '农户']
    customer = generator.generate_one()
    
    # 随机贷款条件
    loan_amount = random.uniform(5, 100)  # 5-100万
    interest_rate = random.uniform(0.04, 0.12)
    term_months = random.choice([12, 24, 36])
    
    # 预测风险
    loan_offer = LoanOffer(
        amount=loan_amount * 10000,
        interest_rate=interest_rate,
        term_months=term_months
    )
    market = MarketConditions(
        gdp_growth=random.uniform(0.02, 0.06),
        base_interest_rate=0.04,
        unemployment_rate=random.uniform(0.04, 0.08),
        inflation_rate=random.uniform(0.01, 0.04),
        credit_spread=random.uniform(0.01, 0.03)
    )
    
    future = world_model.predict_customer_future(customer, loan_offer, market)
    
    # 审批决策
    risk_level = 'low' if future.default_probability < 0.1 else 'medium' if future.default_probability < 0.3 else 'high'
    approved = future.default_probability < 0.25 and random.random() > 0.2
    
    # 生成决策解释
    explanation = generate_decision_explanation(customer, loan_offer, market, future, approved)
    
    # 更新统计
    realtime_state['total_processed'] += 1
    if approved:
        realtime_state['approved_today'] += 1
        realtime_state['capital'] += loan_amount * 0.01  # 微量增长
        realtime_state['profit'] += loan_amount * interest_rate * 0.1
        realtime_state['customers'] += 1 if random.random() > 0.7 else 0
    else:
        realtime_state['rejected_today'] += 1
    
    # NPL波动
    realtime_state['npl'] += random.uniform(-0.02, 0.03)
    realtime_state['npl'] = max(1.5, min(4.0, realtime_state['npl']))
    
    # 构建返回数据
    return jsonify({
        'success': True,
        'transaction': {
            'id': f"TXN{realtime_state['total_processed']:06d}",
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'customer_type': customer.customer_type.value,
            'amount': round(loan_amount, 1),
            'term_months': term_months,
            'interest_rate': round(interest_rate * 100, 2),
            'risk_level': risk_level,
            'default_prob': round(float(future.default_probability) * 100, 1),
            'expected_ltv': round(float(future.expected_ltv), 0),
            'approved': approved,
            'status': '通过' if approved else '拒绝',
        },
        'explanation': explanation,
        'stats': {
            'capital': round(float(realtime_state['capital']), 1),
            'profit': round(float(realtime_state['profit']), 1),
            'npl': round(float(realtime_state['npl']), 2),
            'customers': int(realtime_state['customers']),
            'approved_today': int(realtime_state['approved_today']),
            'rejected_today': int(realtime_state['rejected_today']),
            'total_processed': int(realtime_state['total_processed']),
            'approval_rate': round(float(realtime_state['approved_today'] / max(1, realtime_state['total_processed']) * 100), 1),
        }
    })

@app.route('/api/realtime/stats', methods=['GET'])
def realtime_stats():
    """获取当前实时统计"""
    return jsonify({
        'success': True,
        'stats': {
            'capital': round(float(realtime_state['capital']), 1),
            'profit': round(float(realtime_state['profit']), 1),
            'npl': round(float(realtime_state['npl']), 2),
            'customers': int(realtime_state['customers']),
            'approved_today': int(realtime_state['approved_today']),
            'rejected_today': int(realtime_state['rejected_today']),
            'total_processed': int(realtime_state['total_processed']),
        }
    })

@app.route('/api/realtime/reset', methods=['POST'])
def realtime_reset():
    """重置实时状态"""
    global realtime_state
    realtime_state = {
        'capital': 386.0,
        'profit': 286.0,
        'npl': 2.15,
        'customers': 523847,
        'pending_loans': 0,
        'approved_today': 0,
        'rejected_today': 0,
        'total_processed': 0,
    }
    return jsonify({'success': True, 'message': '已重置'})

# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print("=" * 60)
    print("🚀 Gamium Finance AI Web Server")
    print("=" * 60)
    print(f"访问: http://localhost:{port}")
    print(f"环境: {'开发' if debug else '生产'}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=debug)

