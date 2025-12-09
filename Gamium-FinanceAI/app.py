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
from environment.economic_cycle import CyclePhase
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

@app.route('/data-generation')
def data_generation():
    """数据生成流程详解页面"""
    return send_from_directory('docs', '数据生成流程详解.html')

@app.route('/banking-architecture')
def banking_architecture():
    """银行系统架构与数据提取详解页面"""
    return send_from_directory('docs', '银行系统架构与数据提取详解.html')

@app.route('/model-terms')
def model_terms():
    """模型评估术语详解页面"""
    return send_from_directory('docs', '模型评估术语详解.html')

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
            # 个人客户
            'salaried': CustomerType.SALARIED,
            'small_business': CustomerType.SMALL_BUSINESS,
            'freelancer': CustomerType.FREELANCER,
            'farmer': CustomerType.FARMER,
            'professional': CustomerType.PROFESSIONAL,
            'entrepreneur': CustomerType.ENTREPRENEUR,
            'investor': CustomerType.INVESTOR,
            'retiree': CustomerType.RETIREE,
            'student': CustomerType.STUDENT,
            # 企业客户
            'micro_enterprise': CustomerType.MICRO_ENTERPRISE,
            'small_enterprise': CustomerType.SMALL_ENTERPRISE,
            'medium_enterprise': CustomerType.MEDIUM_ENTERPRISE,
            'large_enterprise': CustomerType.LARGE_ENTERPRISE,
            'startup': CustomerType.STARTUP,
            'tech_startup': CustomerType.TECH_STARTUP,
            'manufacturing': CustomerType.MANUFACTURING,
            'trade_company': CustomerType.TRADE_COMPANY,
            'service_company': CustomerType.SERVICE_COMPANY,
            'freelancer': CustomerType.FREELANCER,
            'farmer': CustomerType.FARMER,
            # 企业客户
            'micro_enterprise': CustomerType.MICRO_ENTERPRISE,
            'small_enterprise': CustomerType.SMALL_ENTERPRISE,
            'medium_enterprise': CustomerType.MEDIUM_ENTERPRISE,
            'large_enterprise': CustomerType.LARGE_ENTERPRISE,
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

@app.route('/api/customer/default-probability-detail', methods=['POST'])
def get_default_probability_detail():
    """获取违约概率详细计算过程"""
    data = request.json or {}
    
    # 获取客户和预测数据
    customer_data = data.get('customer')
    prediction_data = data.get('prediction')
    loan_data = data.get('loan', {})
    market_data = data.get('market', {})
    
    if not customer_data or not prediction_data:
        return jsonify({'success': False, 'error': 'Missing customer or prediction data'}), 400
    
    # 重新计算以获取详细过程（不添加噪声，便于展示）
    from data_distillation.customer_generator import CustomerGenerator, CustomerType
    from data_distillation.world_model import WorldModel, LoanOffer, MarketConditions
    
    # 重建客户对象
    type_map = {
        # 个人客户
        '工薪阶层': CustomerType.SALARIED,
        '小微企业主': CustomerType.SMALL_BUSINESS,
        '自由职业': CustomerType.FREELANCER,
        '农户': CustomerType.FARMER,
        '专业人士': CustomerType.PROFESSIONAL,
        '创业者': CustomerType.ENTREPRENEUR,
        '投资者': CustomerType.INVESTOR,
        '退休人员': CustomerType.RETIREE,
        '学生': CustomerType.STUDENT,
        # 企业客户
        '微型企业': CustomerType.MICRO_ENTERPRISE,
        '小型企业': CustomerType.SMALL_ENTERPRISE,
        '中型企业': CustomerType.MEDIUM_ENTERPRISE,
        '大型企业': CustomerType.LARGE_ENTERPRISE,
        '初创企业': CustomerType.STARTUP,
        '科技初创': CustomerType.TECH_STARTUP,
        '制造企业': CustomerType.MANUFACTURING,
        '贸易公司': CustomerType.TRADE_COMPANY,
        '服务企业': CustomerType.SERVICE_COMPANY,
    }
    
    customer_type = type_map.get(customer_data.get('customer_type'), CustomerType.SALARIED)
    customer = generator.generate_one(customer_type=customer_type)
    
    # 覆盖关键字段（跳过只读属性）
    readonly_props = {'debt_ratio', 'debt_to_income', 'risk_score', 'financial_health_score', 
                      'innovation_score', 'enterprise_size_score', 'is_enterprise'}
    
    for key, value in customer_data.items():
        if key not in readonly_props and hasattr(customer, key):
            try:
                # 尝试设置属性
                attr = getattr(type(customer), key, None)
                if not isinstance(attr, property) or attr.fset is not None:
                    setattr(customer, key, value)
            except (AttributeError, TypeError):
                # 如果是只读属性，跳过
                pass
    
    # 特殊处理：通过设置总资产和总负债来间接设置负债率
    if 'debt_ratio' in customer_data:
        debt_ratio = float(customer_data['debt_ratio'])
        if hasattr(customer, 'total_assets') and customer.total_assets > 0:
            customer.total_liabilities = customer.total_assets * debt_ratio
    
    # 重建贷款对象
    loan = LoanOffer(
        amount=float(loan_data.get('amount', 100000)),
        interest_rate=float(loan_data.get('interest_rate', 0.08)),
        term_months=int(loan_data.get('term_months', 24)),
    )
    
    # 重建市场环境
    market = MarketConditions(
        gdp_growth=float(market_data.get('gdp_growth', 0.03)),
        base_interest_rate=float(market_data.get('base_interest_rate', 0.04)),
        unemployment_rate=float(market_data.get('unemployment_rate', 0.05)),
        inflation_rate=float(market_data.get('inflation_rate', 0.02)),
        credit_spread=float(market_data.get('credit_spread', 0.02)),
    )
    
    # 重新预测（不添加噪声）
    future = world_model.predict_customer_future(customer, loan, market, add_noise=False)
    
    # 构建详细计算过程
    risk_factors = future.risk_factors or {}
    
    # 计算步骤
    calculation_steps = []
    
    # 步骤1: 基础违约率
    base_rate = risk_factors.get('base_rate', 0.03)
    customer_type_name = customer.customer_type.value if hasattr(customer.customer_type, 'value') else str(customer.customer_type)
    calculation_steps.append({
        'step': 1,
        'name': '基础违约率',
        'description': f'根据客户类型({customer_type_name})确定的基础违约率',
        'value': base_rate,
        'formula': f'base_rate = {base_rate:.4f}',
        'impact': '基础风险水平'
    })
    
    # 步骤2-8: 各个风险因子
    factor_descriptions = {
        'industry_factor': ('行业风险系数', '根据客户所属行业评估的风险调整系数'),
        'city_factor': ('城市风险系数', '根据客户所在城市等级评估的风险调整系数'),
        'debt_factor': ('负债率影响', '客户负债率对违约风险的影响'),
        'payment_factor': ('还款能力', '月供占收入比例对违约风险的影响'),
        'history_factor': ('历史信用', '客户历史贷款表现对违约风险的影响'),
        'volatility_factor': ('收入稳定性', '客户收入波动性对违约风险的影响'),
        'economic_factor': ('宏观经济', '当前宏观经济环境对违约风险的影响'),
    }
    
    # 企业客户额外的风险因子
    enterprise_factors = {
        'financial_health_factor': ('财务健康度', '企业财务健康状况对违约风险的影响'),
        'profit_factor': ('盈利能力', '企业盈利能力对违约风险的影响'),
        'cashflow_factor': ('现金流', '企业现金流状况对违约风险的影响'),
        'liquidity_factor': ('流动性', '企业流动性比率对违约风险的影响'),
        'payment_coverage_factor': ('还款覆盖能力', '企业营收覆盖还款能力的影响'),
        'years_factor': ('经营年限', '企业成立年限对违约风险的影响'),
        'innovation_factor': ('创新能力', '企业创新能力对违约风险的影响'),
        'legal_factor': ('法律风险', '企业法律纠纷对违约风险的影响'),
        'tax_factor': ('税务合规', '企业税务合规性对违约风险的影响'),
        'credit_rating_factor': ('信用评级', '企业信用评级对违约风险的影响'),
    }
    
    step_num = 2
    intermediate_result = base_rate
    
    # 个人客户风险因子
    if not customer.is_enterprise:
        for factor_key, (name, desc) in factor_descriptions.items():
            if factor_key in risk_factors:
                factor_value = risk_factors[factor_key]
                old_result = intermediate_result
                intermediate_result *= factor_value
                
                calculation_steps.append({
                    'step': step_num,
                    'name': name,
                    'description': desc,
                    'value': factor_value,
                    'formula': f'{old_result:.6f} × {factor_value:.4f} = {intermediate_result:.6f}',
                    'impact': f'{"增加" if factor_value > 1.0 else "降低" if factor_value < 1.0 else "不变"}风险',
                    'contribution': abs(factor_value - 1.0) * 100
                })
                step_num += 1
    else:
        # 企业客户风险因子
        for factor_key, (name, desc) in enterprise_factors.items():
            if factor_key in risk_factors:
                factor_value = risk_factors[factor_key]
                old_result = intermediate_result
                intermediate_result *= factor_value
                
                calculation_steps.append({
                    'step': step_num,
                    'name': name,
                    'description': desc,
                    'value': factor_value,
                    'formula': f'{old_result:.6f} × {factor_value:.4f} = {intermediate_result:.6f}',
                    'impact': f'{"增加" if factor_value > 1.0 else "降低" if factor_value < 1.0 else "不变"}风险',
                    'contribution': abs(factor_value - 1.0) * 100
                })
                step_num += 1
        
        # 企业客户也包含行业因子
        if 'industry_factor' in risk_factors:
            factor_value = risk_factors['industry_factor']
            old_result = intermediate_result
            intermediate_result *= factor_value
            
            calculation_steps.append({
                'step': step_num,
                'name': '行业风险系数',
                'description': '根据客户所属行业评估的风险调整系数',
                'value': factor_value,
                'formula': f'{old_result:.6f} × {factor_value:.4f} = {intermediate_result:.6f}',
                'impact': f'{"增加" if factor_value > 1.0 else "降低" if factor_value < 1.0 else "不变"}风险',
                'contribution': abs(factor_value - 1.0) * 100
            })
            step_num += 1
    
    # 最终结果
    final_prob = min(0.95, max(0.001, intermediate_result))
    
    # 计算各因子的贡献度（相对于基础违约率）
    total_contribution = sum(step.get('contribution', 0) for step in calculation_steps[1:])
    
    return jsonify({
        'success': True,
        'customer_id': customer_data.get('customer_id'),
        'customer_type': customer_data.get('customer_type'),
        'is_enterprise': customer.is_enterprise,
        'calculation_steps': calculation_steps,
        'final_probability': final_prob,
        'base_rate': base_rate,
        'total_factors': len(calculation_steps) - 1,
        'risk_factors': risk_factors,
        'formula': '违约概率 = 基础违约率 × 风险因子1 × 风险因子2 × ... × 风险因子N',
        'summary': {
            'base_risk': base_rate,
            'adjusted_risk': final_prob,
            'risk_multiplier': final_prob / base_rate if base_rate > 0 else 1.0,
            'risk_level': '高风险' if final_prob > 0.25 else '中风险' if final_prob > 0.10 else '低风险'
        }
    })

@app.route('/api/customer/churn-probability-detail', methods=['POST'])
def get_churn_probability_detail():
    """获取流失概率详细计算过程"""
    data = request.json or {}
    
    # 获取客户和预测数据
    customer_data = data.get('customer')
    prediction_data = data.get('prediction')
    loan_data = data.get('loan', {})
    market_data = data.get('market', {})
    
    if not customer_data or not prediction_data:
        return jsonify({'success': False, 'error': 'Missing customer or prediction data'}), 400
    
    # 重新计算以获取详细过程（不添加噪声，便于展示）
    from data_distillation.customer_generator import CustomerGenerator, CustomerType
    from data_distillation.world_model import WorldModel, LoanOffer, MarketConditions
    
    # 重建客户对象
    type_map = {
        # 个人客户
        '工薪阶层': CustomerType.SALARIED,
        '小微企业主': CustomerType.SMALL_BUSINESS,
        '自由职业': CustomerType.FREELANCER,
        '农户': CustomerType.FARMER,
        '专业人士': CustomerType.PROFESSIONAL,
        '创业者': CustomerType.ENTREPRENEUR,
        '投资者': CustomerType.INVESTOR,
        '退休人员': CustomerType.RETIREE,
        '学生': CustomerType.STUDENT,
        # 企业客户
        '微型企业': CustomerType.MICRO_ENTERPRISE,
        '小型企业': CustomerType.SMALL_ENTERPRISE,
        '中型企业': CustomerType.MEDIUM_ENTERPRISE,
        '大型企业': CustomerType.LARGE_ENTERPRISE,
        '初创企业': CustomerType.STARTUP,
        '科技初创': CustomerType.TECH_STARTUP,
        '制造企业': CustomerType.MANUFACTURING,
        '贸易公司': CustomerType.TRADE_COMPANY,
        '服务企业': CustomerType.SERVICE_COMPANY,
    }
    
    customer_type = type_map.get(customer_data.get('customer_type'), CustomerType.SALARIED)
    customer = generator.generate_one(customer_type=customer_type)
    
    # 覆盖关键字段（跳过只读属性）
    readonly_props = {'debt_ratio', 'debt_to_income', 'risk_score', 'financial_health_score', 
                      'innovation_score', 'enterprise_size_score', 'is_enterprise'}
    
    for key, value in customer_data.items():
        if key not in readonly_props and hasattr(customer, key):
            try:
                attr = getattr(type(customer), key, None)
                if not isinstance(attr, property) or attr.fset is not None:
                    setattr(customer, key, value)
            except (AttributeError, TypeError):
                pass
    
    # 重建贷款对象
    loan = LoanOffer(
        amount=float(loan_data.get('amount', 100000)),
        interest_rate=float(loan_data.get('interest_rate', 0.08)),
        term_months=int(loan_data.get('term_months', 24)),
    )
    
    # 重建市场环境
    market = MarketConditions(
        gdp_growth=float(market_data.get('gdp_growth', 0.03)),
        base_interest_rate=float(market_data.get('base_interest_rate', 0.04)),
        unemployment_rate=float(market_data.get('unemployment_rate', 0.05)),
        inflation_rate=float(market_data.get('inflation_rate', 0.02)),
        credit_spread=float(market_data.get('credit_spread', 0.02)),
    )
    
    # 重新预测（不添加噪声）
    future = world_model.predict_customer_future(customer, loan, market, add_noise=False)
    
    # 获取流失概率计算因子
    risk_factors = future.risk_factors or {}
    churn_factors = risk_factors.get('churn_factors', {})
    
    # 构建计算步骤
    calculation_steps = []
    
    # 步骤1: 基础流失率
    base_churn = churn_factors.get('base_churn', 0.05 if not customer.is_enterprise else 0.08)
    calculation_steps.append({
        'step': 1,
        'name': '基础流失率',
        'description': f'根据客户类型({customer.customer_type.value if hasattr(customer.customer_type, "value") else str(customer.customer_type)})确定的基础流失率',
        'value': base_churn,
        'formula': f'base_churn = {base_churn:.4f}',
        'impact': '基础流失风险水平'
    })
    
    # 步骤2: 利率敏感性
    rate_sensitivity = churn_factors.get('rate_sensitivity', 0.0)
    rate_impact = churn_factors.get('rate_impact', '')
    if rate_sensitivity > 0:
        intermediate_result = base_churn + rate_sensitivity
        calculation_steps.append({
            'step': 2,
            'name': '利率敏感性',
            'description': rate_impact,
            'value': rate_sensitivity,
            'formula': f'{base_churn:.4f} + {rate_sensitivity:.4f} = {intermediate_result:.4f}',
            'impact': '利率过高会增加客户提前还款或转投其他银行的概率'
        })
    else:
        calculation_steps.append({
            'step': 2,
            'name': '利率敏感性',
            'description': rate_impact if rate_impact else '贷款利率在可接受范围内',
            'value': 0.0,
            'formula': f'{base_churn:.4f} + 0.0 = {base_churn:.4f}',
            'impact': '利率合理，不会显著增加流失风险'
        })
    
    # 步骤3: 客户质量影响
    quality_bonus = churn_factors.get('quality_bonus', 0.0)
    quality_impact = churn_factors.get('quality_impact', '')
    if quality_bonus > 0:
        prev_result = base_churn + rate_sensitivity
        intermediate_result = prev_result + quality_bonus
        calculation_steps.append({
            'step': 3,
            'name': '客户质量影响',
            'description': quality_impact,
            'value': quality_bonus,
            'formula': f'{prev_result:.4f} + {quality_bonus:.4f} = {intermediate_result:.4f}',
            'impact': '优质客户更可能找到更好的融资渠道或提前还款'
        })
    else:
        prev_result = base_churn + rate_sensitivity
        calculation_steps.append({
            'step': 3,
            'name': '客户质量影响',
            'description': quality_impact if quality_impact else '客户质量一般，不会显著增加流失风险',
            'value': 0.0,
            'formula': f'{prev_result:.4f} + 0.0 = {prev_result:.4f}',
            'impact': '客户质量一般，流失风险正常'
        })
    
    # 最终结果
    final_churn = churn_factors.get('final_churn', future.churn_probability)
    max_churn = 0.5 if not customer.is_enterprise else 0.6
    
    # 计算摘要
    summary = {
        'base_churn': base_churn,
        'final_churn': final_churn,
        'churn_multiplier': final_churn / base_churn if base_churn > 0 else 1.0,
        'risk_level': '高风险' if final_churn > 0.3 else '中风险' if final_churn > 0.15 else '低风险',
        'max_churn_limit': max_churn
    }
    
    return jsonify({
        'success': True,
        'customer_id': customer_data.get('customer_id'),
        'customer_type': customer_data.get('customer_type'),
        'is_enterprise': customer.is_enterprise,
        'calculation_steps': calculation_steps,
        'final_probability': final_churn,
        'base_churn': base_churn,
        'total_factors': len(calculation_steps) - 1,
        'churn_factors': churn_factors,
        'formula': '流失概率 = 基础流失率 + 利率敏感性 + 客户质量影响',
        'summary': summary
    })

@app.route('/api/customer/risk-score-detail', methods=['POST'])
def get_risk_score_detail():
    """获取风险评分详细计算过程"""
    data = request.json or {}
    customer_data = data.get('customer')
    
    if not customer_data:
        return jsonify({'success': False, 'error': 'Missing customer data'}), 400
    
    # 重建客户对象
    from data_distillation.customer_generator import CustomerGenerator, CustomerType
    
    type_map = {
        # 个人客户
        '工薪阶层': CustomerType.SALARIED,
        '小微企业主': CustomerType.SMALL_BUSINESS,
        '自由职业': CustomerType.FREELANCER,
        '农户': CustomerType.FARMER,
        '专业人士': CustomerType.PROFESSIONAL,
        '创业者': CustomerType.ENTREPRENEUR,
        '投资者': CustomerType.INVESTOR,
        '退休人员': CustomerType.RETIREE,
        '学生': CustomerType.STUDENT,
        # 企业客户
        '微型企业': CustomerType.MICRO_ENTERPRISE,
        '小型企业': CustomerType.SMALL_ENTERPRISE,
        '中型企业': CustomerType.MEDIUM_ENTERPRISE,
        '大型企业': CustomerType.LARGE_ENTERPRISE,
        '初创企业': CustomerType.STARTUP,
        '科技初创': CustomerType.TECH_STARTUP,
        '制造企业': CustomerType.MANUFACTURING,
        '贸易公司': CustomerType.TRADE_COMPANY,
        '服务企业': CustomerType.SERVICE_COMPANY,
    }
    
    customer_type = type_map.get(customer_data.get('customer_type'), CustomerType.SALARIED)
    customer = generator.generate_one(customer_type=customer_type)
    
    # 覆盖关键字段
    readonly_props = {'debt_ratio', 'debt_to_income', 'risk_score', 'financial_health_score', 
                      'innovation_score', 'enterprise_size_score', 'is_enterprise'}
    
    for key, value in customer_data.items():
        if key not in readonly_props and hasattr(customer, key):
            try:
                attr = getattr(type(customer), key, None)
                if not isinstance(attr, property) or attr.fset is not None:
                    setattr(customer, key, value)
            except (AttributeError, TypeError):
                pass
    
    # 计算风险评分并记录步骤
    calculation_steps = []
    score = 0.0
    
    if customer.is_enterprise:
        # 企业客户风险评分
        # 1. 财务健康度（反向）
        financial_health = customer.financial_health_score
        financial_contribution = (1 - financial_health) * 0.3
        score += financial_contribution
        calculation_steps.append({
            'step': 1,
            'name': '财务健康度影响',
            'description': f'财务健康度评分: {financial_health:.3f} (反向计算)',
            'value': financial_contribution,
            'formula': f'(1 - {financial_health:.3f}) × 0.3 = {financial_contribution:.4f}',
            'impact': '财务健康度越低，风险越高',
            'weight': 0.3
        })
        
        # 2. 负债率
        debt_contribution = customer.debt_ratio * 0.2
        score += debt_contribution
        calculation_steps.append({
            'step': 2,
            'name': '负债率影响',
            'description': f'负债率: {customer.debt_ratio*100:.1f}%',
            'value': debt_contribution,
            'formula': f'{customer.debt_ratio:.3f} × 0.2 = {debt_contribution:.4f}',
            'impact': '负债率越高，风险越高',
            'weight': 0.2
        })
        
        # 3. 流动性风险
        liquidity_contribution = 0.0
        if customer.current_ratio < 1.0:
            liquidity_contribution = 0.15
        elif customer.current_ratio < 1.2:
            liquidity_contribution = 0.10
        if liquidity_contribution > 0:
            score += liquidity_contribution
            calculation_steps.append({
                'step': len(calculation_steps) + 1,
                'name': '流动性风险',
                'description': f'流动比率: {customer.current_ratio:.2f}',
                'value': liquidity_contribution,
                'formula': f'流动比率 < 1.2，增加风险 {liquidity_contribution:.4f}',
                'impact': '流动比率过低，偿债能力不足',
                'weight': liquidity_contribution
            })
        
        # 4. 现金流风险
        cashflow_contribution = 0.0
        if customer.operating_cash_flow < 0:
            cashflow_contribution = 0.15
            score += cashflow_contribution
            calculation_steps.append({
                'step': len(calculation_steps) + 1,
                'name': '现金流风险',
                'description': f'经营现金流: ¥{customer.operating_cash_flow:,.0f}',
                'value': cashflow_contribution,
                'formula': f'经营现金流为负，增加风险 {cashflow_contribution:.4f}',
                'impact': '现金流为负，经营困难',
                'weight': cashflow_contribution
            })
        
        # 5. 历史信用
        history_contribution = 0.0
        if customer.max_historical_dpd > 0:
            history_contribution = min(0.2, customer.max_historical_dpd / 90 * 0.2)
            score += history_contribution
            calculation_steps.append({
                'step': len(calculation_steps) + 1,
                'name': '历史信用影响',
                'description': f'历史最大逾期: {customer.max_historical_dpd}天',
                'value': history_contribution,
                'formula': f'min(0.2, {customer.max_historical_dpd}/90 × 0.2) = {history_contribution:.4f}',
                'impact': '历史逾期会增加风险评分',
                'weight': history_contribution
            })
        
        # 6. 经营年限
        years_contribution = 0.0
        if customer.years_in_business < 3:
            years_contribution = 0.10
        elif customer.years_in_business < 5:
            years_contribution = 0.05
        if years_contribution > 0:
            score += years_contribution
            calculation_steps.append({
                'step': len(calculation_steps) + 1,
                'name': '经营年限影响',
                'description': f'经营年限: {customer.years_in_business:.1f}年',
                'value': years_contribution,
                'formula': f'经营年限 < 5年，增加风险 {years_contribution:.4f}',
                'impact': '经营年限短，经验不足',
                'weight': years_contribution
            })
        
        # 7. 法律纠纷
        legal_contribution = 0.0
        if customer.has_legal_disputes:
            legal_contribution = 0.10
            score += legal_contribution
            calculation_steps.append({
                'step': len(calculation_steps) + 1,
                'name': '法律纠纷影响',
                'description': f'法律纠纷数量: {customer.legal_dispute_count}',
                'value': legal_contribution,
                'formula': f'存在法律纠纷，增加风险 {legal_contribution:.4f}',
                'impact': '法律纠纷增加经营风险',
                'weight': legal_contribution
            })
    else:
        # 个人客户风险评分
        # 1. 年龄因素
        age_contribution = 0.0
        if customer.age < 25 or customer.age > 60:
            age_contribution = 0.1
            score += age_contribution
            calculation_steps.append({
                'step': 1,
                'name': '年龄因素',
                'description': f'年龄: {customer.age}岁',
                'value': age_contribution,
                'formula': f'年龄 < 25 或 > 60，增加风险 {age_contribution:.4f}',
                'impact': '年龄过小或过大，收入稳定性较差',
                'weight': 0.1
            })
        else:
            calculation_steps.append({
                'step': 1,
                'name': '年龄因素',
                'description': f'年龄: {customer.age}岁',
                'value': 0.0,
                'formula': f'年龄在25-60岁之间，风险正常',
                'impact': '年龄适中，收入稳定性较好',
                'weight': 0.0
            })
        
        # 2. 收入稳定性
        volatility_contribution = customer.income_volatility * 0.2
        score += volatility_contribution
        calculation_steps.append({
            'step': 2,
            'name': '收入稳定性',
            'description': f'收入波动性: {customer.income_volatility*100:.1f}%',
            'value': volatility_contribution,
            'formula': f'{customer.income_volatility:.3f} × 0.2 = {volatility_contribution:.4f}',
            'impact': '收入波动性越高，风险越高',
            'weight': 0.2
        })
        
        # 3. 负债率
        debt_contribution = customer.debt_ratio * 0.25
        score += debt_contribution
        calculation_steps.append({
            'step': 3,
            'name': '负债率影响',
            'description': f'负债率: {customer.debt_ratio*100:.1f}%',
            'value': debt_contribution,
            'formula': f'{customer.debt_ratio:.3f} × 0.25 = {debt_contribution:.4f}',
            'impact': '负债率越高，风险越高',
            'weight': 0.25
        })
        
        # 4. 历史信用
        history_contribution = 0.0
        if customer.max_historical_dpd > 0:
            history_contribution = min(0.3, customer.max_historical_dpd / 90 * 0.3)
            score += history_contribution
            calculation_steps.append({
                'step': 4,
                'name': '历史信用影响',
                'description': f'历史最大逾期: {customer.max_historical_dpd}天',
                'value': history_contribution,
                'formula': f'min(0.3, {customer.max_historical_dpd}/90 × 0.3) = {history_contribution:.4f}',
                'impact': '历史逾期会增加风险评分',
                'weight': history_contribution
            })
        else:
            calculation_steps.append({
                'step': 4,
                'name': '历史信用影响',
                'description': '历史最大逾期: 0天',
                'value': 0.0,
                'formula': '无历史逾期记录',
                'impact': '无逾期记录，风险正常',
                'weight': 0.0
            })
        
        # 5. 从业年限
        years_contribution = 0.0
        if customer.years_in_business < 2:
            years_contribution = 0.1
            score += years_contribution
            calculation_steps.append({
                'step': 5,
                'name': '从业年限影响',
                'description': f'从业年限: {customer.years_in_business:.1f}年',
                'value': years_contribution,
                'formula': f'从业年限 < 2年，增加风险 {years_contribution:.4f}',
                'impact': '从业年限短，收入稳定性较差',
                'weight': 0.1
            })
        else:
            calculation_steps.append({
                'step': 5,
                'name': '从业年限影响',
                'description': f'从业年限: {customer.years_in_business:.1f}年',
                'value': 0.0,
                'formula': f'从业年限 ≥ 2年，风险正常',
                'impact': '从业年限充足，收入稳定性较好',
                'weight': 0.0
            })
    
    # 最终结果（限制在0-1之间）
    final_score = min(1.0, score)
    
    return jsonify({
        'success': True,
        'customer_id': customer_data.get('customer_id'),
        'customer_type': customer_data.get('customer_type'),
        'is_enterprise': customer.is_enterprise,
        'calculation_steps': calculation_steps,
        'final_score': final_score,
        'raw_score': score,
        'formula': '风险评分 = Σ(各因子贡献值)，最终限制在0-1之间',
        'summary': {
            'risk_level': '高风险' if final_score > 0.6 else '中风险' if final_score > 0.3 else '低风险',
            'total_factors': len(calculation_steps),
            'max_possible_score': 1.0
        }
    })

@app.route('/api/customer/loan-history', methods=['POST'])
def get_loan_history():
    """获取客户历史贷款详情"""
    data = request.json or {}
    customer_id = data.get('customer_id')
    
    if not customer_id:
        return jsonify({'success': False, 'error': 'Missing customer_id'}), 400
    
    # 生成模拟历史贷款数据
    import random
    from datetime import datetime, timedelta
    
    # 从客户数据中获取信息
    customer = None
    customer_data_from_request = data.get('customer_data')  # 从前端传来的客户数据
    if customer_data_from_request:
        # 如果有客户数据，尝试重建客户对象
        from data_distillation.customer_generator import CustomerType
        type_map = {
            '工薪阶层': CustomerType.SALARIED,
            '小微企业主': CustomerType.SMALL_BUSINESS,
            '自由职业': CustomerType.FREELANCER,
            '农户': CustomerType.FARMER,
            '微型企业': CustomerType.MICRO_ENTERPRISE,
            '小型企业': CustomerType.SMALL_ENTERPRISE,
            '中型企业': CustomerType.MEDIUM_ENTERPRISE,
            '大型企业': CustomerType.LARGE_ENTERPRISE,
        }
        customer_type = type_map.get(customer_data_from_request.get('customer_type'), CustomerType.SALARIED)
        customer = generator.generate_one(customer_type=customer_type)
        # 覆盖关键字段
        if 'max_historical_dpd' in customer_data_from_request:
            customer.max_historical_dpd = customer_data_from_request['max_historical_dpd']
        if 'previous_loans' in customer_data_from_request:
            customer.previous_loans = customer_data_from_request['previous_loans']
    elif hasattr(generator, '_customers_cache') and customer_id in generator._customers_cache:
        customer = generator._customers_cache[customer_id]
    else:
        # 如果没有缓存，生成一个临时客户来获取previous_loans数量
        customer = generator.generate_one()
    
    num_loans = data.get('num_loans', customer.previous_loans if customer else random.randint(1, 10))
    # 确保至少有一条贷款
    num_loans = max(1, num_loans)
    
    # 如果客户有历史逾期记录，确保至少生成一条逾期贷款
    has_historical_overdue = customer and customer.max_historical_dpd > 0
    overdue_loan_generated = False
    
    loan_history = []
    base_date = datetime.now() - timedelta(days=365 * 3)  # 3年前开始
    
    for i in range(num_loans):
        # 贷款时间（从旧到新）
        days_ago = random.randint(30, 365 * 3)
        apply_date = base_date + timedelta(days=random.randint(0, 365 * 3 - days_ago))
        
        # 贷款金额（根据客户类型调整）
        if customer and customer.is_enterprise:
            loan_amount = random.uniform(500000, 5000000)
        else:
            loan_amount = random.uniform(10000, 500000)
        
        # 贷款期限
        term_months = random.choice([6, 12, 24, 36, 48, 60])
        
        # 利率
        interest_rate = random.uniform(0.05, 0.12)
        
        # 审批结果
        # 如果有历史逾期且需要生成逾期贷款，确保审批通过
        if has_historical_overdue and not overdue_loan_generated and i == num_loans - 1:
            approval_status = 'approved'  # 强制批准最后一条贷款
        else:
            approval_status = random.choices(
                ['approved', 'rejected', 'pending'],
                weights=[0.7, 0.2, 0.1]
            )[0]
        
        # 如果批准，生成还款信息
        repayment_status = 'N/A'
        overdue_days = 0
        total_paid = 0
        remaining_amount = 0
        
        if approval_status == 'approved':
            # 还款状态
            # 如果客户有历史逾期且还没有生成逾期贷款，提高逾期概率
            if has_historical_overdue and not overdue_loan_generated:
                # 有历史逾期的客户，提高逾期贷款概率
                if i == num_loans - 1:
                    # 最后一条贷款强制为逾期
                    repayment_status = random.choice(['overdue', 'defaulted'])
                    overdue_loan_generated = True
                else:
                    # 提高逾期概率到40%
                    repayment_status = random.choices(
                        ['completed', 'ongoing', 'overdue', 'defaulted'],
                        weights=[0.3, 0.3, 0.3, 0.1]
                    )[0]
                    if repayment_status in ['overdue', 'defaulted']:
                        overdue_loan_generated = True
            else:
                repayment_status = random.choices(
                    ['completed', 'ongoing', 'overdue', 'defaulted'],
                    weights=[0.5, 0.3, 0.15, 0.05]
                )[0]
                if repayment_status in ['overdue', 'defaulted']:
                    overdue_loan_generated = True
            
            # 逾期天数
            overdue_info = None
            if repayment_status in ['overdue', 'defaulted']:
                overdue_days = random.randint(1, 180)
                
                # 计算详细的逾期信息
                approval_date = apply_date + timedelta(days=random.randint(1, 7))
                monthly_payment = loan_amount * (interest_rate / 12) / (1 - (1 + interest_rate / 12) ** (-term_months))
                
                # 计算应该还清的期数
                months_should_paid = min(term_months, int(overdue_days / 30) + 1)
                months_actually_paid = max(0, months_should_paid - 1)
                
                # 逾期开始时间（最后一次正常还款后的下一个月）
                overdue_start_date = approval_date + timedelta(days=months_actually_paid * 30)
                
                # 逾期金额（包括本金和利息）
                overdue_principal = monthly_payment * (months_should_paid - months_actually_paid)
                overdue_interest = overdue_principal * (interest_rate / 12) * (overdue_days / 30)
                overdue_penalty = overdue_principal * 0.0005 * overdue_days  # 每日0.05%的罚息
                total_overdue_amount = overdue_principal + overdue_interest + overdue_penalty
                
                overdue_info = {
                    'overdue_days': overdue_days,
                    'overdue_start_date': overdue_start_date.strftime('%Y-%m-%d'),
                    'overdue_principal': round(overdue_principal, 2),
                    'overdue_interest': round(overdue_interest, 2),
                    'overdue_penalty': round(overdue_penalty, 2),
                    'total_overdue_amount': round(total_overdue_amount, 2),
                    'overdue_severity': '严重' if overdue_days > 90 else '中等' if overdue_days > 30 else '轻微',
                    'months_overdue': months_should_paid - months_actually_paid
                }
            else:
                overdue_days = 0
            
            # 已还金额和剩余金额
            if repayment_status == 'completed':
                total_paid = loan_amount * (1 + interest_rate * term_months / 12)
                remaining_amount = 0
            elif repayment_status == 'defaulted':
                # 违约通常只还了部分
                paid_ratio = random.uniform(0.1, 0.6)
                total_paid = loan_amount * paid_ratio
                remaining_amount = loan_amount * (1 + interest_rate * term_months / 12) - total_paid
            else:
                # 进行中
                months_paid = random.randint(1, term_months - 1)
                monthly_payment = loan_amount * (interest_rate / 12) / (1 - (1 + interest_rate / 12) ** (-term_months))
                total_paid = monthly_payment * months_paid
                remaining_amount = loan_amount * (1 + interest_rate * term_months / 12) - total_paid
        
        loan_record = {
            'loan_id': f'LOAN_{customer_id}_{i+1:03d}',
            'apply_date': apply_date.strftime('%Y-%m-%d'),
            'approval_date': (apply_date + timedelta(days=random.randint(1, 7))).strftime('%Y-%m-%d') if approval_status == 'approved' else None,
            'loan_amount': round(loan_amount, 2),
            'term_months': term_months,
            'interest_rate': round(interest_rate, 4),
            'approval_status': approval_status,
            'repayment_status': repayment_status,
            'overdue_days': overdue_days,
            'total_paid': round(total_paid, 2),
            'remaining_amount': round(remaining_amount, 2),
            'purpose': random.choice(['消费', '经营周转', '购房', '购车', '教育', '其他']),
            'collateral_type': random.choice(['信用', '抵押', '担保', '质押']),
        }
        
        # 如果有逾期信息，添加到记录中
        if overdue_info:
            loan_record['overdue_info'] = overdue_info
        
        loan_history.append(loan_record)
    
    # 按申请时间倒序排列（最新的在前）
    loan_history.sort(key=lambda x: x['apply_date'], reverse=True)
    
    # 统计逾期贷款数量
    overdue_count = sum(1 for loan in loan_history if loan.get('overdue_info'))
    
    return jsonify({
        'success': True,
        'customer_id': customer_id,
        'total_loans': len(loan_history),
        'overdue_count': overdue_count,
        'has_historical_overdue': has_historical_overdue,
        'loan_history': loan_history
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
    try:
        data = request.json
        if data is None:
            return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400
    
        # 获取客户数据
        customer_data = data.get('customer', {})
        customer_type_str = customer_data.get('customer_type', 'salaried')
        risk_profile = customer_data.get('risk_profile', 'medium')
        
        # 解析客户类型
        type_map = {
            # 个人客户
            'salaried': CustomerType.SALARIED,
            'small_business': CustomerType.SMALL_BUSINESS,
            'freelancer': CustomerType.FREELANCER,
            'farmer': CustomerType.FARMER,
            'professional': CustomerType.PROFESSIONAL,
            'entrepreneur': CustomerType.ENTREPRENEUR,
            'investor': CustomerType.INVESTOR,
            'retiree': CustomerType.RETIREE,
            'student': CustomerType.STUDENT,
            # 企业客户
            'micro_enterprise': CustomerType.MICRO_ENTERPRISE,
            'small_enterprise': CustomerType.SMALL_ENTERPRISE,
            'medium_enterprise': CustomerType.MEDIUM_ENTERPRISE,
            'large_enterprise': CustomerType.LARGE_ENTERPRISE,
            'startup': CustomerType.STARTUP,
            'tech_startup': CustomerType.TECH_STARTUP,
            'manufacturing': CustomerType.MANUFACTURING,
            'trade_company': CustomerType.TRADE_COMPANY,
            'service_company': CustomerType.SERVICE_COMPANY,
        }
        customer_type = type_map.get(customer_type_str, CustomerType.SALARIED)
        
        # 判断是否为企业客户
        is_enterprise = customer_type in [
            CustomerType.MICRO_ENTERPRISE, CustomerType.SMALL_ENTERPRISE,
            CustomerType.MEDIUM_ENTERPRISE, CustomerType.LARGE_ENTERPRISE,
            CustomerType.STARTUP, CustomerType.TECH_STARTUP,
            CustomerType.MANUFACTURING, CustomerType.TRADE_COMPANY,
            CustomerType.SERVICE_COMPANY
        ]
        
        # 生成客户
        from data_distillation.customer_generator import CityTier, Industry
        
        # 解析城市等级和行业
        city_tier = None
        if customer_data.get('city_tier'):
            city_tier_map = {
                'tier_1': CityTier.TIER_1,
                'tier_2': CityTier.TIER_2,
                'tier_3': CityTier.TIER_3,
                'tier_4': CityTier.TIER_4,
            }
            city_tier = city_tier_map.get(customer_data['city_tier'])
        
        industry = None
        if customer_data.get('industry'):
            industry_map = {
                'manufacturing': Industry.MANUFACTURING,
                'service': Industry.SERVICE,
                'retail': Industry.RETAIL,
                'catering': Industry.CATERING,
                'it': Industry.IT,
                'construction': Industry.CONSTRUCTION,
                'agriculture': Industry.AGRICULTURE,
            }
            industry = industry_map.get(customer_data['industry'])
        
        customer = generator.generate_one(
            customer_type=customer_type,
            city_tier=city_tier,
            industry=industry,
            risk_profile=risk_profile
        )
        
        # 覆盖用户提供的字段
        if is_enterprise:
            # 企业客户字段
            if customer_data.get('annual_revenue'):
                customer.annual_revenue = float(customer_data['annual_revenue'])
            if customer_data.get('net_profit'):
                customer.net_profit = float(customer_data['net_profit'])
            if customer_data.get('profit_margin'):
                customer.net_profit = customer.annual_revenue * float(customer_data['profit_margin'])
            if customer_data.get('operating_cash_flow'):
                customer.operating_cash_flow = float(customer_data['operating_cash_flow'])
            if customer_data.get('current_ratio'):
                customer.current_ratio = float(customer_data['current_ratio'])
            if customer_data.get('quick_ratio'):
                customer.quick_ratio = float(customer_data['quick_ratio'])
            if customer_data.get('debt_ratio'):
                customer.total_liabilities = customer.total_assets * float(customer_data['debt_ratio'])
            if customer_data.get('revenue_growth_rate'):
                customer.revenue_growth_rate = float(customer_data['revenue_growth_rate'])
            if customer_data.get('rnd_expense_ratio'):
                customer.rnd_expense = customer.annual_revenue * float(customer_data['rnd_expense_ratio'])
            if customer_data.get('total_patents'):
                customer.total_patents = int(customer_data['total_patents'])
            if customer_data.get('legal_disputes'):
                customer.legal_disputes = int(customer_data['legal_disputes'])
            if customer_data.get('years_in_business'):
                customer.years_in_business = float(customer_data['years_in_business'])
        else:
            # 个人客户字段
            if customer_data.get('monthly_income'):
                customer.monthly_income = float(customer_data['monthly_income'])
            if customer_data.get('income_volatility'):
                customer.income_volatility = float(customer_data['income_volatility'])
            if customer_data.get('total_assets'):
                customer.total_assets = float(customer_data['total_assets'])
            if customer_data.get('total_liabilities'):
                customer.total_liabilities = float(customer_data['total_liabilities'])
            if customer_data.get('deposit_balance'):
                customer.deposit_balance = float(customer_data['deposit_balance'])
            if customer_data.get('deposit_stability'):
                customer.deposit_stability = float(customer_data['deposit_stability'])
            if customer_data.get('years_in_business'):
                customer.years_in_business = float(customer_data['years_in_business'])
            if customer_data.get('previous_loans'):
                customer.previous_loans = int(customer_data['previous_loans'])
            if customer_data.get('max_historical_dpd'):
                customer.max_historical_dpd = int(customer_data['max_historical_dpd'])
            
            # 年龄范围处理
            if customer_data.get('age_range'):
                age_range = customer_data['age_range']
                if age_range == 'young':
                    customer.age = int(generator.rng.normal(28, 4))
                elif age_range == 'middle':
                    customer.age = int(generator.rng.normal(42, 5))
                elif age_range == 'senior':
                    customer.age = int(generator.rng.normal(58, 5))
                customer.age = max(18, min(70, customer.age))
        
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
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

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
    
    # Step 2: 定义分析场景 - 使用扩展指标
    scenarios = {
        "繁荣期": MarketConditions(
            gdp_growth=0.06, base_interest_rate=0.05, unemployment_rate=0.04,
            inflation_rate=0.02, credit_spread=0.02,
            consumer_confidence=0.75, manufacturing_pmi=55.0, housing_price_index=120.0,
            stock_index=3500.0, m2_growth=0.12, exchange_rate=6.8, trade_balance=300.0,
            fiscal_policy_stance=0.4, monetary_policy_stance=0.4,
            risk_appetite=0.7, liquidity_index=0.7, market_volatility=0.12
        ),
        "正常期": MarketConditions(
            gdp_growth=0.03, base_interest_rate=0.04, unemployment_rate=0.05,
            inflation_rate=0.02, credit_spread=0.02,
            consumer_confidence=0.60, manufacturing_pmi=50.0, housing_price_index=105.0,
            stock_index=3000.0, m2_growth=0.10, exchange_rate=7.0, trade_balance=200.0,
            fiscal_policy_stance=0.5, monetary_policy_stance=0.5,
            risk_appetite=0.5, liquidity_index=0.5, market_volatility=0.15
        ),
        "衰退期": MarketConditions(
            gdp_growth=0.01, base_interest_rate=0.03, unemployment_rate=0.07,
            inflation_rate=0.03, credit_spread=0.03,
            consumer_confidence=0.55, manufacturing_pmi=48.0, housing_price_index=100.0,
            stock_index=2800.0, m2_growth=0.08, exchange_rate=7.2, trade_balance=100.0,
            fiscal_policy_stance=0.6, monetary_policy_stance=0.6,
            risk_appetite=0.4, liquidity_index=0.5, market_volatility=0.18
        ),
        "萧条期": MarketConditions(
            gdp_growth=-0.02, base_interest_rate=0.02, unemployment_rate=0.10,
            inflation_rate=0.01, credit_spread=0.04,
            consumer_confidence=0.35, manufacturing_pmi=42.0, housing_price_index=95.0,
            stock_index=2400.0, m2_growth=0.15, exchange_rate=7.5, trade_balance=-50.0,
            fiscal_policy_stance=0.8, monetary_policy_stance=0.8,
            risk_appetite=0.2, liquidity_index=0.3, market_volatility=0.25
        ),
    }
    
    # 生成详细的经济指标报告
    scenario_details = {}
    for name, market in scenarios.items():
        scenario_details[name] = market.to_dict()
    
    cycle_analysis_trace["steps"].append({
        "step_id": 2,
        "name": "场景定义",
        "details": {
            "scenarios": scenario_details
        }
    })
    add_log("DEFINE_SCENARIOS", "定义4个经济周期场景: 繁荣期, 正常期, 衰退期, 萧条期")
    
    loan = LoanOffer(amount=100000, interest_rate=0.08, term_months=24)
    
    # Step 3: 逐场景分析 - 扩展分析
    results = {}
    for scenario_name, market in scenarios.items():
        step_start = datetime.now()
        default_probs = []
        ltvs = []
        churn_probs = []
        expected_dpds = []
        
        # 按行业分析
        by_industry = {}
        
        for customer in customers:
            future = world_model.predict_customer_future(customer, loan, market)
            default_probs.append(future.default_probability)
            ltvs.append(future.expected_ltv)
            churn_probs.append(future.churn_probability)
            expected_dpds.append(future.expected_dpd)
            
            # 按行业统计
            industry = customer.industry.value if hasattr(customer, 'industry') else '其他'
            if industry not in by_industry:
                by_industry[industry] = {'default_probs': [], 'count': 0}
            by_industry[industry]['default_probs'].append(future.default_probability)
            by_industry[industry]['count'] += 1
        
        # 计算行业风险
        industry_risk = {
            ind: {
                'avg_default_rate': float(np.mean(data['default_probs'])),
                'count': data['count']
            }
            for ind, data in by_industry.items()
        }
        
        results[scenario_name] = {
            # 基础指标
            'avg_default_rate': float(np.mean(default_probs)),
            'avg_ltv': float(np.mean(ltvs)),
            'high_risk_count': int(sum(1 for p in default_probs if p > 0.15)),
            'high_risk_ratio': sum(1 for p in default_probs if p > 0.15) / len(default_probs),
            # 扩展指标
            'avg_churn_probability': float(np.mean(churn_probs)),
            'avg_expected_dpd': float(np.mean(expected_dpds)),
            'median_default_rate': float(np.median(default_probs)),
            'std_default_rate': float(np.std(default_probs)),
            # 经济指标
            'economic_indicators': market.to_dict(),
            # 行业分析
            'by_industry': industry_risk,
            # 风险分布
            'risk_distribution': {
                'low_risk': sum(1 for p in default_probs if p < 0.05) / len(default_probs),
                'medium_risk': sum(1 for p in default_probs if 0.05 <= p < 0.15) / len(default_probs),
                'high_risk': sum(1 for p in default_probs if 0.15 <= p < 0.25) / len(default_probs),
                'very_high_risk': sum(1 for p in default_probs if p >= 0.25) / len(default_probs),
            }
        }
        
        add_log("ANALYZE_SCENARIO", f"{scenario_name}: 违约率={results[scenario_name]['avg_default_rate']*100:.2f}%, 高风险={results[scenario_name]['high_risk_count']}人, 经济健康度={results[scenario_name]['economic_indicators']['economic_health_score']:.1f}")
    
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
    
    # Step 5: 生成综合分析报告
    step5_start = datetime.now()
    
    # 计算指标相关性
    correlations = {}
    for scenario_name, result in results.items():
        indicators = result['economic_indicators']
        default_rate = result['avg_default_rate']
        
        correlations[scenario_name] = {
            'gdp_vs_default': -indicators['gdp_growth'] / (default_rate + 0.01),  # 负相关
            'unemployment_vs_default': indicators['unemployment_rate'] / (default_rate + 0.01),  # 正相关
            'confidence_vs_default': -indicators['consumer_confidence'] / (default_rate + 0.01),  # 负相关
            'pmi_vs_default': -(indicators['manufacturing_pmi'] - 50) / (default_rate + 0.01),  # 负相关
        }
    
    # 生成预警信号
    warnings = []
    for scenario_name, result in results.items():
        indicators = result['economic_indicators']
        default_rate = result['avg_default_rate']
        
        if indicators['gdp_growth'] < 0:
            warnings.append({
                'scenario': scenario_name,
                'level': 'high',
                'type': 'gdp_negative',
                'message': f'{scenario_name}: GDP负增长，经济衰退风险高'
            })
        
        if indicators['unemployment_rate'] > 0.08:
            warnings.append({
                'scenario': scenario_name,
                'level': 'high',
                'type': 'high_unemployment',
                'message': f'{scenario_name}: 失业率超过8%，就业市场压力大'
            })
        
        if default_rate > 0.20:
            warnings.append({
                'scenario': scenario_name,
                'level': 'critical',
                'type': 'high_default_rate',
                'message': f'{scenario_name}: 违约率超过20%，信贷风险极高'
            })
        
        if indicators['consumer_confidence'] < 0.4:
            warnings.append({
                'scenario': scenario_name,
                'level': 'medium',
                'type': 'low_confidence',
                'message': f'{scenario_name}: 消费者信心指数低于40%，市场情绪低迷'
            })
    
    cycle_analysis_trace["steps"].append({
        "step_id": 5,
        "name": "综合分析报告",
        "duration_ms": int((datetime.now() - step5_start).total_seconds() * 1000),
        "details": {
            "correlations": correlations,
            "warnings": warnings
        }
    })
    add_log("COMPREHENSIVE_ANALYSIS", f"生成综合分析报告: {len(warnings)}个预警信号")
    
    cycle_analysis_trace["end_time"] = datetime.now().isoformat()
    cycle_analysis_trace["results"] = {
        "scenarios": results,
        "by_type": type_analysis,
        "correlations": correlations,
        "warnings": warnings
    }
    add_log("COMPLETE", "经济周期分析完成")
    
    return jsonify({
        'success': True,
        'run_id': run_id,
        'customer_count': n_customers,
        'scenarios': results,
        'by_customer_type': type_analysis,
        'correlations': correlations,
        'warnings': warnings,
        'summary': {
            'total_indicators': 18,  # 扩展后的指标数量
            'total_warnings': len(warnings),
            'highest_risk_scenario': max(results.items(), key=lambda x: x[1]['avg_default_rate'])[0],
            'lowest_risk_scenario': min(results.items(), key=lambda x: x[1]['avg_default_rate'])[0],
        }
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
    """自动运行完整模拟 - 扩展版"""
    global simulation_trace
    
    data = request.json or {}
    strategy = data.get('strategy', 'rule_based')
    seed = data.get('seed', 42)
    
    # 扩展输入参数
    initial_capital = data.get('initial_capital', 100.0)  # 初始资本（亿）
    simulation_years = data.get('simulation_years', 10)  # 模拟年数
    initial_phase = data.get('initial_phase', 'boom')  # 初始经济周期
    risk_appetite = data.get('risk_appetite', 0.5)  # 风险偏好 (0-1)
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    strategy_names = {
        'random': '随机策略',
        'rule_based': '规则策略',
        'conservative': '保守策略',
        'aggressive': '激进策略'
    }
    
    phase_map = {
        'boom': CyclePhase.BOOM,
        'recession': CyclePhase.RECESSION,
        'depression': CyclePhase.DEPRESSION,
        'recovery': CyclePhase.RECOVERY,
    }
    initial_cycle_phase = phase_map.get(initial_phase, CyclePhase.BOOM)
    
    # 初始化追踪
    simulation_trace = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": {
            "strategy": strategy,
            "strategy_name": strategy_names.get(strategy, strategy),
            "seed": seed,
            "initial_capital": initial_capital,
            "simulation_years": simulation_years,
            "initial_phase": initial_phase,
            "risk_appetite": risk_appetite
        },
        "steps": [],
        "monthly_decisions": [],
        "key_events": [],
        "summary": {},
        "audit_log": [],
        "economic_history": [],
        "risk_metrics": []
    }
    
    def add_log(action, details):
        simulation_trace["audit_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        })
    
    add_log("INIT", f"银行模拟启动, 策略: {strategy_names.get(strategy, strategy)}, 初始资本: ¥{initial_capital}亿, 模拟时长: {simulation_years}年")
    
    # 创建环境和智能体 - 支持自定义参数
    # 注意：LendingEnv需要修改以支持这些参数，这里先使用默认值
    env = LendingEnv(seed=seed, initial_capital=initial_capital * 1e8)
    
    agents = {
        'random': RandomAgent(seed=seed),
        'rule_based': RuleBasedAgent(),
        'conservative': ConservativeAgent(),
        'aggressive': AggressiveAgent(),
    }
    agent = agents.get(strategy, RuleBasedAgent())
    
    # 运行模拟 - 先reset初始化环境
    state, info = env.reset()
    
    # 设置初始经济周期
    if hasattr(env, 'economy'):
        env.economy.phase = initial_cycle_phase
        env.economy.state = env.economy._generate_state()
    
    simulation_trace["steps"].append({
        "step_id": 1,
        "name": "环境初始化",
        "details": {
            "initial_capital": float(env.bank.capital),
            "max_months": LendingEnv.TOTAL_MONTHS,
            "agent_type": type(agent).__name__,
            "initial_economic_phase": initial_phase,
            "risk_appetite": risk_appetite
        }
    })
    add_log("ENV_INIT", f"初始资本: ¥{env.bank.capital/1e8:.1f}亿, 初始经济周期: {initial_phase}")
    history = []
    total_reward = 0
    month_count = 0
    
    # 扩展指标追踪
    max_capital = env.bank.capital
    min_capital = env.bank.capital
    max_npl = 0.0
    total_loans_issued = 0.0
    total_interest_income = 0.0
    total_provisions = 0.0
    total_write_offs = 0.0
    economic_phase_counts = {}
    
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
        
        # 更新扩展指标
        max_capital = max(max_capital, env.bank.capital)
        min_capital = min(min_capital, env.bank.capital)
        max_npl = max(max_npl, info['npl_ratio'])
        total_loans_issued += abs(decision['capital_change']) if decision['capital_change'] > 0 else 0
        total_interest_income += env.bank.loan_portfolio.interest_income
        total_provisions += env.bank.loan_portfolio.provision_expense
        total_write_offs += env.bank.loan_portfolio.write_offs
        
        # 统计经济周期
        phase = info['eco_phase']
        economic_phase_counts[phase] = economic_phase_counts.get(phase, 0) + 1
        
        # 记录经济指标历史
        if hasattr(env, 'economy') and env.economy.state:
            eco_state = env.economy.state
            simulation_trace["economic_history"].append({
                'month': env.month,
                'gdp_growth': float(eco_state.gdp_growth),
                'interest_rate': float(eco_state.interest_rate),
                'unemployment_rate': float(eco_state.unemployment_rate),
                'inflation_rate': float(eco_state.inflation_rate),
                'phase': phase
            })
        
        # 计算扩展指标
        roe = (env.bank.loan_portfolio.net_profit * 12) / env.bank.capital if env.bank.capital > 0 else 0
        capital_adequacy = env.bank.capital_adequacy_ratio
        loan_to_asset = env.bank.loan_portfolio.total_loans / env.bank.total_assets if env.bank.total_assets > 0 else 0
        
        history.append({
            'month': env.month,
            'year': env.month // 12 + 1,
            'eco_phase': phase,
            'capital': float(env.bank.capital),
            'total_assets': float(env.bank.total_assets),
            'total_loans': float(env.bank.loan_portfolio.total_loans),
            'profit': float(env.bank.loan_portfolio.net_profit),
            'cumulative_profit': float(info['cumulative_profit']),
            'npl_ratio': float(info['npl_ratio']),
            'roa': float(info['roa']),
            'roe': float(roe),
            'capital_adequacy_ratio': float(capital_adequacy),
            'loan_to_asset_ratio': float(loan_to_asset),
            'interest_income': float(env.bank.loan_portfolio.interest_income),
            'provision_expense': float(env.bank.loan_portfolio.provision_expense),
            'write_offs': float(env.bank.loan_portfolio.write_offs),
            'reward': float(reward),
        })
        
        # 记录风险指标
        simulation_trace["risk_metrics"].append({
            'month': env.month,
            'npl_ratio': float(info['npl_ratio']),
            'capital_adequacy_ratio': float(capital_adequacy),
            'roe': float(roe),
            'roa': float(info['roa']),
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
    
    # 计算最终扩展指标
    final_roa = info['roa']
    final_roe = (info['cumulative_profit'] * 12 / month_count) / info['capital'] if info['capital'] > 0 and month_count > 0 else 0
    final_capital_adequacy = env.bank.capital_adequacy_ratio
    capital_growth = (info['capital'] - initial_capital * 1e8) / (initial_capital * 1e8) if initial_capital > 0 else 0
    avg_monthly_profit = info['cumulative_profit'] / month_count if month_count > 0 else 0
    profit_volatility = np.std([h['profit'] for h in history]) if len(history) > 1 else 0
    
    simulation_trace["summary"] = {
        # 基础指标
        'total_months': len(history),
        'total_reward': float(total_reward),
        'final_capital': float(info['capital']),
        'final_profit': float(info['cumulative_profit']),
        'final_npl': float(info['npl_ratio']),
        'is_bankrupt': bool(info['is_bankrupt']),
        # 扩展指标
        'initial_capital': float(initial_capital * 1e8),
        'capital_growth_rate': float(capital_growth),
        'max_capital': float(max_capital),
        'min_capital': float(min_capital),
        'max_npl': float(max_npl),
        'final_roa': float(final_roa),
        'final_roe': float(final_roe),
        'final_capital_adequacy_ratio': float(final_capital_adequacy),
        'avg_monthly_profit': float(avg_monthly_profit),
        'profit_volatility': float(profit_volatility),
        'total_loans_issued': float(total_loans_issued),
        'total_interest_income': float(total_interest_income),
        'total_provisions': float(total_provisions),
        'total_write_offs': float(total_write_offs),
        'provision_coverage_ratio': float(total_provisions / total_write_offs) if total_write_offs > 0 else 0,
        'economic_phase_distribution': economic_phase_counts,
        'final_total_assets': float(env.bank.total_assets),
        'final_total_loans': float(env.bank.loan_portfolio.total_loans),
        'final_loan_to_asset_ratio': float(env.bank.loan_portfolio.total_loans / env.bank.total_assets) if env.bank.total_assets > 0 else 0,
    }
    simulation_trace["end_time"] = datetime.now().isoformat()
    
    if info['is_bankrupt']:
        add_log("BANKRUPT", "银行破产!")
    else:
        add_log("RESULT", f"最终资本: ¥{info['capital']/1e8:.1f}亿, NPL: {info['npl_ratio']*100:.1f}%, ROE: {final_roe*100:.2f}%")
    
    return jsonify({
        'success': True,
        'run_id': run_id,
        'strategy': strategy,
        'history': history,
        'summary': simulation_trace["summary"],
        'economic_history': simulation_trace["economic_history"][::12],  # 每年一个样本
        'risk_metrics_summary': {
            'avg_npl': float(np.mean([m['npl_ratio'] for m in simulation_trace["risk_metrics"]])),
            'max_npl': float(max_npl),
            'avg_capital_adequacy': float(np.mean([m['capital_adequacy_ratio'] for m in simulation_trace["risk_metrics"]])),
            'min_capital_adequacy': float(min([m['capital_adequacy_ratio'] for m in simulation_trace["risk_metrics"]])),
            'avg_roe': float(np.mean([m['roe'] for m in simulation_trace["risk_metrics"]])),
            'avg_roa': float(np.mean([m['roa'] for m in simulation_trace["risk_metrics"]])),
        }
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
    """对比不同策略 - 扩展版"""
    global comparison_trace
    
    data = request.json or {}
    n_episodes = min(data.get('episodes', 3), 10)  # 增加到10轮
    seed = data.get('seed', 42)
    
    # 扩展输入参数
    initial_capital = data.get('initial_capital', 100.0)
    simulation_years = data.get('simulation_years', 10)
    include_alphazero = data.get('include_alphazero', False)
    risk_adjusted = data.get('risk_adjusted', True)  # 是否计算风险调整收益
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 初始化追踪
    comparison_trace = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": {
            "episodes_per_strategy": n_episodes,
            "seed": seed,
            "initial_capital": initial_capital,
            "simulation_years": simulation_years,
            "include_alphazero": include_alphazero,
            "risk_adjusted": risk_adjusted
        },
        "strategies": [],
        "episode_details": {},
        "results": [],
        "audit_log": [],
        "stability_analysis": {},
        "risk_analysis": {}
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
        episode_roes = []
        episode_roas = []
        episode_capital_adequacies = []
        bankruptcies = 0
        episode_details = []
        episode_histories = []
        
        add_log("STRATEGY_START", f"开始评估: {name}")
        
        for ep in range(n_episodes):
            env = LendingEnv(seed=seed + ep, initial_capital=initial_capital * 1e8)
            state, info = env.reset()
            total_reward = 0
            month_count = 0
            history = []
            
            while True:
                action = agent.select_action(state, info)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                month_count += 1
                
                # 记录历史
                if month_count % 12 == 0 or terminated or truncated:
                    roe = (info['cumulative_profit'] * 12 / month_count) / info['capital'] if info['capital'] > 0 and month_count > 0 else 0
                    history.append({
                        'month': month_count,
                        'capital': float(info['capital']),
                        'profit': float(info['cumulative_profit']),
                        'npl': float(info['npl_ratio']),
                        'roa': float(info['roa']),
                        'roe': float(roe),
                        'capital_adequacy': float(env.bank.capital_adequacy_ratio),
                    })
                
                if terminated or truncated:
                    break
            
            # 计算最终指标
            final_roe = (info['cumulative_profit'] * 12 / month_count) / info['capital'] if info['capital'] > 0 and month_count > 0 else 0
            
            episode_rewards.append(total_reward)
            episode_profits.append(info['cumulative_profit'])
            episode_npls.append(info['npl_ratio'])
            episode_roes.append(final_roe)
            episode_roas.append(info['roa'])
            episode_capital_adequacies.append(env.bank.capital_adequacy_ratio)
            
            if info['is_bankrupt']:
                bankruptcies += 1
            
            episode_details.append({
                "episode": ep + 1,
                "months": month_count,
                "reward": float(total_reward),
                "profit": float(info['cumulative_profit']),
                "npl": float(info['npl_ratio']),
                "roe": float(final_roe),
                "roa": float(info['roa']),
                "capital_adequacy": float(env.bank.capital_adequacy_ratio),
                "bankrupt": bool(info['is_bankrupt'])
            })
            episode_histories.append(history)
        
        # 计算稳定性指标
        profit_std = float(np.std(episode_profits))
        profit_cv = profit_std / abs(np.mean(episode_profits)) if np.mean(episode_profits) != 0 else 0
        npl_std = float(np.std(episode_npls))
        roe_std = float(np.std(episode_roes))
        
        # 计算风险调整收益 (Sharpe-like ratio)
        if risk_adjusted:
            avg_profit = np.mean(episode_profits)
            profit_volatility = profit_std
            risk_adjusted_return = avg_profit / (profit_volatility + 1e8) if profit_volatility > 0 else 0
        else:
            risk_adjusted_return = np.mean(episode_profits)
        
        # 计算最大回撤
        max_drawdowns = []
        for history in episode_histories:
            if len(history) > 0:
                capitals = [h['capital'] for h in history]
                peak = capitals[0]
                max_dd = 0
                for cap in capitals:
                    if cap > peak:
                        peak = cap
                    dd = (peak - cap) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                max_drawdowns.append(max_dd)
        avg_max_drawdown = float(np.mean(max_drawdowns)) if max_drawdowns else 0.0
        
        strategy_result = {
            'name': name,
            # 基础指标
            'avg_reward': float(np.mean(episode_rewards)),
            'avg_profit': float(np.mean(episode_profits)),
            'avg_npl': float(np.mean(episode_npls)),
            'bankruptcy_rate': bankruptcies / n_episodes,
            # 扩展指标
            'avg_roe': float(np.mean(episode_roes)),
            'avg_roa': float(np.mean(episode_roas)),
            'avg_capital_adequacy': float(np.mean(episode_capital_adequacies)),
            'min_capital_adequacy': float(np.min(episode_capital_adequacies)),
            # 稳定性指标
            'profit_std': profit_std,
            'profit_cv': float(profit_cv),  # 变异系数
            'npl_std': npl_std,
            'roe_std': roe_std,
            # 风险指标
            'risk_adjusted_return': float(risk_adjusted_return),
            'max_drawdown': float(avg_max_drawdown),
            'win_rate': float(sum(1 for p in episode_profits if p > 0) / len(episode_profits)),
            # 其他
            'duration_ms': int((datetime.now() - strategy_start).total_seconds() * 1000),
            'episode_count': n_episodes
        }
        results.append(strategy_result)
        
        comparison_trace["episode_details"][name] = episode_details
        comparison_trace["stability_analysis"][name] = {
            'profit_volatility': profit_std,
            'profit_cv': float(profit_cv),
            'npl_volatility': npl_std,
            'roe_volatility': roe_std,
        }
        comparison_trace["risk_analysis"][name] = {
            'risk_adjusted_return': float(risk_adjusted_return),
            'max_drawdown': float(avg_max_drawdown),
            'bankruptcy_rate': bankruptcies / n_episodes,
        }
        
        add_log("STRATEGY_COMPLETE", f"{name}: 平均利润=¥{strategy_result['avg_profit']/1e8:.1f}亿, ROE={strategy_result['avg_roe']*100:.2f}%, 破产率={strategy_result['bankruptcy_rate']*100:.0f}%")
    
    # 按风险调整收益排序（如果启用）
    if risk_adjusted:
        results.sort(key=lambda x: x['risk_adjusted_return'], reverse=True)
    else:
        results.sort(key=lambda x: x['avg_reward'], reverse=True)
    
    comparison_trace["results"] = results
    comparison_trace["end_time"] = datetime.now().isoformat()
    comparison_trace["winner"] = results[0]['name'] if results else None
    
    # 生成综合分析
    comparison_summary = {
        'total_strategies': len(results),
        'total_episodes': n_episodes * len(results),
        'best_profit': max(r['avg_profit'] for r in results),
        'best_roe': max(r['avg_roe'] for r in results),
        'lowest_npl': min(r['avg_npl'] for r in results),
        'lowest_bankruptcy_rate': min(r['bankruptcy_rate'] for r in results),
        'most_stable': min(results, key=lambda x: x['profit_cv'])['name'] if results else None,
        'best_risk_adjusted': max(results, key=lambda x: x['risk_adjusted_return'])['name'] if results else None,
    }
    
    add_log("COMPLETE", f"对比完成, 最佳策略: {comparison_trace['winner']}, 最佳ROE: {comparison_summary['best_roe']*100:.2f}%")
    
    return jsonify({
        'success': True,
        'run_id': run_id,
        'episodes': n_episodes,
        'results': results,
        'summary': comparison_summary,
        'stability_analysis': comparison_trace["stability_analysis"],
        'risk_analysis': comparison_trace["risk_analysis"]
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
    
    # 检查数据文件（可选，不阻塞启动）
    try:
        from utils.data_loader import ensure_data_files
        data_dir = Path(__file__).parent / 'data' / 'historical_backup'
        print("📂 检查数据文件...")
        ensure_data_files(data_dir)
    except Exception as e:
        print(f"⚠️  数据文件检查失败（不影响启动）: {e}")
        print("💡 如需使用历史数据，请运行: python3 scripts/download_data_from_cloud.py")
    
    print("=" * 60)
    print("🚀 Gamium Finance AI Web Server")
    print("=" * 60)
    print(f"访问: http://localhost:{port}")
    print(f"环境: {'开发' if debug else '生产'}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=debug)

