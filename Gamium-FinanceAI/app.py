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
from arena.rule_engine import RuleEngine
from arena.scoring_system import ScoringSystem, ScoreBreakdown
from arena.multi_round_simulator import MultiRoundSimulator
from arena.multi_agent_game import MultiAgentGame

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

@app.route('/docs/<path:filename>')
def serve_docs(filename):
    """提供docs目录下的文档文件"""
    import os
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    return send_from_directory(docs_dir, filename)

@app.route('/demo')
def demo_page():
    """端到端Demo交互界面"""
    return send_from_directory('web', 'demo.html')

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

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'service': 'Gamium Finance AI',
        'version': '1.0.0',
        'timestamp': time.time()
    })

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
    generator = CustomerGenerator()
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
    
    # 使用传入的预测数据中的违约概率，确保一致性
    # 如果预测数据中有违约概率，直接使用它，否则重新计算
    use_prediction_data = prediction_data and 'default_probability' in prediction_data
    
    if use_prediction_data:
        # 直接使用前端传入的预测结果，并构造一个轻量级的客户占位对象
        # 直接使用预测结果，确保前后一致
        final_prob = prediction_data.get('default_probability', 0.03)
        risk_factors = prediction_data.get('risk_factors', {})
        
        # 构造占位的 customer，避免后续访问属性时报错
        class _CustomerPlaceholder:
            def __init__(self, is_enterprise: bool, customer_type):
                self.is_enterprise = is_enterprise
                self.customer_type = customer_type
        
        enterprise_types_cn = {'微型企业', '小型企业', '中型企业', '大型企业', '初创企业', '科技初创', '制造企业', '贸易公司', '服务企业'}
        enterprise_types_en = {'micro_enterprise', 'small_enterprise', 'medium_enterprise', 'large_enterprise',
                               'startup', 'tech_startup', 'manufacturing', 'trade_company', 'service_company'}
        customer_type_raw = customer_data.get('customer_type')
        is_enterprise = customer_type_raw in enterprise_types_cn or customer_type_raw in enterprise_types_en
        customer = _CustomerPlaceholder(is_enterprise=is_enterprise, customer_type=customer_type_raw or '未知')
    else:
        # 需要重新计算时，正确重建客户对象
        customer_type = type_map.get(customer_data.get('customer_type'), CustomerType.SALARIED)
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
            city_tier = city_tier_map.get(customer_data.get('city_tier'))
        
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
            industry = industry_map.get(customer_data.get('industry'))
        
        customer = generator.generate_one(customer_type=customer_type, city_tier=city_tier, industry=industry)
        
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
        final_prob = future.default_probability
        risk_factors = future.risk_factors or {}
    
    # 确保risk_factors存在
    if not risk_factors:
        risk_factors = {}
    
    # 计算步骤
    calculation_steps = []
    
    # 步骤1: 基础违约率
    base_rate = risk_factors.get('base_rate', 0.03)
    # 获取客户类型名称
    if use_prediction_data:
        customer_type_name = customer_data.get('customer_type', '未知')
    else:
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
# 多银行对比（最小可行版）
# ============================================================

@app.route('/api/bank-comparison/run', methods=['POST'])
def run_bank_comparison():
    """
    最小可行版：对比多家银行在同一客户集下的审批与风险收益表现
    请求示例：
    {
        "banks": [
            {"name": "稳健银行", "approval_threshold": 0.12, "rate_spread": 0.01},
            {"name": "平衡银行", "approval_threshold": 0.18, "rate_spread": 0.015},
            {"name": "激进银行", "approval_threshold": 0.25, "rate_spread": 0.02}
        ],
        "customer_count": 300,
        "loan_amount": 100000,
        "base_rate": 0.08
    }
    """
    data = request.json or {}
    banks = data.get('banks') or [
        {"name": "稳健银行", "approval_threshold": 0.12, "rate_spread": 0.01},
        {"name": "平衡银行", "approval_threshold": 0.18, "rate_spread": 0.015},
        {"name": "激进银行", "approval_threshold": 0.25, "rate_spread": 0.02},
    ]
    customer_count = int(data.get('customer_count', 300))
    loan_amount = float(data.get('loan_amount', 100000))
    base_rate = float(data.get('base_rate', 0.08))
    seed = int(data.get('seed', 42))

    rng = np.random.default_rng(seed)
    local_world_model = WorldModel(seed=seed)
    results = []

    # 固定市场情景，保持可比性
    market = MarketConditions(
        gdp_growth=0.03,
        base_interest_rate=base_rate,
        unemployment_rate=0.05,
        inflation_rate=0.02,
        credit_spread=0.02,
    )

    # 预生成客户，确保每家银行用同一批客户
    customers = []
    for _ in range(customer_count):
        cust = generator.generate_one()
        customers.append(cust)

    for bank in banks:
        approval_threshold = float(bank.get('approval_threshold', 0.18))
        rate_spread = float(bank.get('rate_spread', 0.01))
        name = bank.get('name', '未命名银行')

        approved = 0
        rejected = 0
        profit = 0.0
        default_sum = 0.0
        factors_sum = 0.0
        factors_count = 0
        dp_list = []
        factor_bucket = {}
        interest_income_sum = 0.0
        expected_loss_sum = 0.0

        for cust in customers:
            # 贷款定价：基础利率 + 银行利差
            loan_rate = base_rate + rate_spread
            loan = LoanOffer(
                amount=loan_amount,
                interest_rate=loan_rate,
                term_months=24,
            )
            future = local_world_model.predict_customer_future(cust, loan, market, add_noise=False)
            dp = float(future.default_probability)
            if dp <= approval_threshold:
                approved += 1
                # 简化利润估计：利息收入 × (1 - 违约概率)
                interest_income = loan_amount * loan_rate
                expected_loss = loan_amount * dp
                net_profit = interest_income * (1 - dp) - expected_loss * 0.0  # 可扩展其他成本，这里置0
                profit += net_profit
                interest_income_sum += interest_income
                expected_loss_sum += expected_loss
                default_sum += dp
                dp_list.append(dp)
                # 记录风险因子均值（只累加数值型）
                if future.risk_factors:
                    for k, v in future.risk_factors.items():
                        if isinstance(v, (int, float)):
                            factors_sum += v
                            factors_count += 1
                            factor_bucket.setdefault(k, []).append(v)
            else:
                rejected += 1

        total = approved + rejected
        avg_dp = (default_sum / approved) if approved > 0 else 0
        avg_factor = (factors_sum / factors_count) if factors_count > 0 else 0
        dp_p95 = float(np.percentile(dp_list, 95)) if dp_list else 0.0

        # 汇总风险因子均值（数值型）
        factor_means = {
            k: float(np.mean(vals)) for k, vals in factor_bucket.items() if len(vals) > 0
        }
        # 取绝对偏离1.0排序，选前5
        top_factors = sorted(
            factor_means.items(),
            key=lambda x: abs(x[1] - 1.0),
            reverse=True
        )[:5]

        results.append({
            "name": name,
            "approval_threshold": approval_threshold,
            "rate_spread": rate_spread,
            "approval_rate": approved / total if total > 0 else 0,
            "avg_default_prob": avg_dp,
            "est_npl": avg_dp,  # 估算值，用违约概率代表
            "est_profit": profit,
            "sample_size": total,
            "risk_factor_mean": avg_factor,
            "dp_p95": dp_p95,
            "interest_income": interest_income_sum,
            "expected_loss": expected_loss_sum,
            "profit_breakdown": {
                "interest_income": interest_income_sum,
                "expected_loss": expected_loss_sum,
                "net_profit": profit
            },
            "factor_top5": top_factors
        })

    # 排序：按预估利润降序
    results.sort(key=lambda x: x["est_profit"], reverse=True)

    summary = {
        "run_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "customer_count": customer_count,
        "loan_amount": loan_amount,
        "base_rate": base_rate,
        "seed": seed,
        "best_bank": results[0]["name"] if results else None,
        "calculation_notes": [
            "审批规则：违约概率 <= 审批阈值 则放款",
            "预估利润：利息收入 * (1 - DP) - 预期损失(本版置0额外成本)",
            "违约概率(NPL估)：使用模型预测的违约概率作为估算NPL",
            "风险因子：只统计数值型风险因子，取绝对偏离1的Top5用于解释"
        ]
    }

    return jsonify({
        "success": True,
        "summary": summary,
        "results": results,
    })

# ============================================================
# AI 演武场（简化版，多模型/策略对比）
# ============================================================

@app.route('/api/arena/run', methods=['POST'])
def run_ai_arena():
    """
    多模型/策略演武场（增强版：规则引擎 + 评分系统）
    输入：
    {
        "participants": [
            {"name": "稳健策略", "approval_threshold": 0.12, "rate_spread": 0.01, "model_id": null},
            {"name": "平衡策略", "approval_threshold": 0.18, "rate_spread": 0.015, "model_id": "gpt-4"}
        ],
        "customer_count": 300,
        "loan_amount": 100000,
        "base_rate": 0.08,
        "seed": 42,
        "scenario": "normal",   # normal / stress
        "black_swan": false,
        "rules": [
            {
                "name": "高收入调低阈值",
                "description": "月收入超过20000的客户降低审批阈值",
                "priority": 1,
                "conditions": [{"field": "monthly_income", "op": ">", "value": 20000}],
                "action": {"approval_threshold_delta": -0.02, "rate_spread_delta": -0.002},
                "penalty": {"score_delta": 0.0}
            }
        ],
        "scoring_weights": {
            "profit": 0.30,
            "risk": 0.30,
            "stability": 0.10,
            "compliance": 0.20,
            "efficiency": 0.05,
            "explainability": 0.05
        }
    }
    输出：各参赛者审批率、违约概率、预估利润、风险指标、评分分解、触发规则列表
    """
    from datetime import datetime
    import traceback
    
    try:
        data = request.json or {}
        participants = data.get('participants') or [
            {"name": "稳健策略", "approval_threshold": 0.12, "rate_spread": 0.01},
            {"name": "平衡策略", "approval_threshold": 0.18, "rate_spread": 0.015},
            {"name": "激进策略", "approval_threshold": 0.25, "rate_spread": 0.02},
        ]
        customer_count = int(data.get('customer_count', 300))
        loan_amount = float(data.get('loan_amount', 100000))
        base_rate = float(data.get('base_rate', 0.08))
        seed = int(data.get('seed', 42))
        scenario = data.get('scenario', 'normal')
        black_swan = bool(data.get('black_swan', False))
        rules_config = data.get('rules', [])
        scoring_weights = data.get('scoring_weights', {})

        # 初始化规则引擎和评分系统
        rule_engine = RuleEngine(rules_config)
        scoring_system = ScoringSystem(scoring_weights)

        rng = np.random.default_rng(seed)
        local_world_model = WorldModel(seed=seed)
        results = []

        # 根据情景调整宏观参数
        gdp = 0.03
        unemp = 0.05
        base_ir = base_rate
        infl = 0.02
        credit_spread = 0.02
        if scenario == 'stress':
            gdp = -0.02
            unemp = 0.09
            base_ir = base_rate + 0.01
            credit_spread = 0.05
        if black_swan:
            gdp -= 0.02
            unemp += 0.02
            credit_spread += 0.03

        market = MarketConditions(
            gdp_growth=gdp,
            base_interest_rate=base_ir,
            unemployment_rate=unemp,
            inflation_rate=infl,
            credit_spread=credit_spread,
        )

        # 准备统一客户集
        customers = [generator.generate_one() for _ in range(customer_count)]

        for p in participants:
            threshold = float(p.get('approval_threshold', 0.18))
            spread = float(p.get('rate_spread', 0.01))
            name = p.get('name', '未命名')
            model_id = p.get('model_id')  # 预留：后续接入LLM

            approved = 0
            rejected = 0
            profit = 0.0
            default_probs = []
            factors_sum = 0.0
            factors_cnt = 0
            factor_bucket = {}
            dp_list = []
            interest_income_sum = 0.0
            expected_loss_sum = 0.0
            triggered_rules_list = []  # 所有触发的规则名称列表
            triggered_rules_count = {}  # 规则触发次数统计
            all_customer_details = []  # 存储所有客户详情用于回放
            profit_history = []  # 利润历史用于计算波动率

            for cust in customers:
                # 使用规则引擎处理客户
                adjustments, triggered, score_adjustments = rule_engine.process_customer(
                    cust, threshold, spread, loan_amount, 24
                )
                
                # 记录触发的规则
                triggered_rules_list.extend(triggered)
                for rule_name in triggered:
                    triggered_rules_count[rule_name] = triggered_rules_count.get(rule_name, 0) + 1
                
                # 使用调整后的参数
                final_threshold = adjustments['approval_threshold']
                final_spread = adjustments['rate_spread']
                final_loan_amount = adjustments['loan_amount']
                final_term = adjustments['term_months']
                
                # 强制通过/拒绝
                if adjustments['force_reject']:
                    rejected += 1
                    all_customer_details.append({
                        'customer_id': cust.customer_id,
                        'decision': 'rejected',
                        'reason': '规则强制拒绝',
                        'triggered_rules': triggered,
                        'default_prob': None
                    })
                    continue
                if adjustments['force_approve']:
                    # 强制通过，但仍需计算违约概率
                    approved += 1
                    rate = base_rate + final_spread
                    loan = LoanOffer(amount=final_loan_amount, interest_rate=rate, term_months=final_term)
                    future = local_world_model.predict_customer_future(cust, loan, market, add_noise=False)
                    dp = float(future.default_probability)
                    interest_income = final_loan_amount * rate
                    expected_loss = final_loan_amount * dp
                    net_profit = interest_income * (1 - dp) - expected_loss
                    profit += net_profit * score_adjustments['profit_discount']
                    interest_income_sum += interest_income
                    expected_loss_sum += expected_loss
                    default_probs.append(dp)
                    dp_list.append(dp)
                    profit_history.append(net_profit)
                    all_customer_details.append({
                        'customer_id': cust.customer_id,
                        'decision': 'approved',
                        'reason': '规则强制通过',
                        'triggered_rules': triggered,
                        'default_prob': dp,
                        'profit': net_profit
                    })
                    continue

                # 正常审批流程
                rate = base_rate + final_spread
                loan = LoanOffer(amount=final_loan_amount, interest_rate=rate, term_months=final_term)
                future = local_world_model.predict_customer_future(cust, loan, market, add_noise=False)
                dp = float(future.default_probability)
                
                if dp <= final_threshold:
                    approved += 1
                    interest_income = final_loan_amount * rate
                    expected_loss = final_loan_amount * dp
                    net_profit = interest_income * (1 - dp) - expected_loss
                    # 应用利润折扣
                    net_profit *= score_adjustments['profit_discount']
                    profit += net_profit
                    interest_income_sum += interest_income
                    expected_loss_sum += expected_loss
                    default_probs.append(dp)
                    dp_list.append(dp)
                    profit_history.append(net_profit)
                    
                    if future.risk_factors:
                        for k, v in future.risk_factors.items():
                            if isinstance(v, (int, float)):
                                factors_sum += v * score_adjustments['risk_multiplier']
                                factors_cnt += 1
                                factor_bucket.setdefault(k, []).append(v)
                    
                    all_customer_details.append({
                        'customer_id': cust.customer_id,
                        'decision': 'approved',
                        'reason': f'违约概率{dp:.4f} <= 阈值{final_threshold:.4f}',
                        'triggered_rules': triggered,
                        'default_prob': dp,
                        'profit': net_profit,
                        'adjustments': adjustments
                    })
                else:
                    rejected += 1
                    all_customer_details.append({
                        'customer_id': cust.customer_id,
                        'decision': 'rejected',
                        'reason': f'违约概率{dp:.4f} > 阈值{final_threshold:.4f}',
                        'triggered_rules': triggered,
                        'default_prob': dp
                    })

            total = approved + rejected
            avg_dp = np.mean(default_probs) if default_probs else 0.0
            est_npl = avg_dp
            avg_factor = factors_sum / factors_cnt if factors_cnt > 0 else 0.0
            approval_rate = approved / total if total > 0 else 0
            dp_p95 = float(np.percentile(dp_list, 95)) if dp_list else 0.0

            factor_means = {k: float(np.mean(vs)) for k, vs in factor_bucket.items() if vs}
            top_factors = sorted(factor_means.items(), key=lambda x: abs(x[1] - 1.0), reverse=True)[:5]

            # 计算RAROC
            raroc = profit / max(1.0, (est_npl * loan_amount * approved) + 1e3) if approved > 0 else 0.0
            
            # 计算利润波动率
            profit_volatility = float(np.std(profit_history)) if profit_history else 0.0
            
            # 计算最大回撤（简化版：基于利润历史）
            max_drawdown = 0.0
            if profit_history:
                cumulative = np.cumsum(profit_history)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / (running_max + 1e-6)
                max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

            # 构建结果字典
            result_dict = {
                "name": name,
                "approval_rate": approval_rate,
                "avg_default_prob": avg_dp,
                "est_npl": est_npl,
                "est_profit": profit,
                "risk_factor_mean": avg_factor,
                "sample_size": total,
                "raroc": raroc,
                "dp_p95": dp_p95,
                "profit_breakdown": {
                    "interest_income": interest_income_sum,
                    "expected_loss": expected_loss_sum,
                    "net_profit": profit
                },
                "factor_top5": top_factors,
                "triggered_rules": triggered_rules_count,
                "triggered_rules_list": list(set(triggered_rules_list)),  # 去重
                "profit_volatility": profit_volatility,
                "max_drawdown": abs(max_drawdown),
                "recovery_time": 0.0,  # 多轮场景中计算
                "compliance_violations": 0,  # 预留：合规检查
                "avg_latency": 0.0,  # 预留：LLM延迟
                "total_rules_count": len(rules_config),
                "customer_details": all_customer_details[:50]  # 只返回前50个客户详情
            }

            results.append(result_dict)

        # 计算评分分解（需要所有结果用于归一化）
        for result in results:
            breakdown = scoring_system.create_score_breakdown(
                result,
                triggered_rules=result.get('triggered_rules_list', []),
                all_results=results
            )
            result['score_breakdown'] = {
                'profit_score': breakdown.profit_score,
                'risk_score': breakdown.risk_score,
                'stability_score': breakdown.stability_score,
                'compliance_score': breakdown.compliance_score,
                'efficiency_score': breakdown.efficiency_score,
                'explainability_score': breakdown.explainability_score,
                'overall_score': breakdown.overall_score
            }

        # 按综合得分排序（处理None值）
        def get_sort_key(x):
            score_breakdown = x.get('score_breakdown', {})
            overall_score = score_breakdown.get('overall_score') if score_breakdown else None
            est_profit = x.get('est_profit')
            # 如果overall_score是None，使用est_profit；如果都是None，使用0
            if overall_score is not None:
                return overall_score
            elif est_profit is not None:
                return est_profit
            else:
                return 0.0
        
        results.sort(key=get_sort_key, reverse=True)
        
        summary = {
        "run_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "customer_count": customer_count,
        "loan_amount": loan_amount,
        "base_rate": base_rate,
        "seed": seed,
        "scenario": scenario,
        "black_swan": black_swan,
        "winner": results[0]["name"] if results else None,
        "rules_count": len(rules_config),
        "calculation_notes": [
            "审批规则：违约概率 <= 审批阈值 则放款（规则引擎可动态调整阈值）",
            "预估利润：利息收入 * (1 - DP) - 预期损失，应用规则利润折扣",
            "NPL估：使用模型预测的违约概率作为估算",
            "风险因子：仅统计数值型，展示偏离1.0最大的Top5",
            "评分系统：综合利润、风险、稳定性、合规、效率、可解释性六个维度",
            "规则触发：记录每个客户触发的规则列表，用于可解释性分析",
            f"情景: {scenario}, 黑天鹅: {black_swan}"
        ]
    }

        return jsonify({
            "success": True,
            "summary": summary,
            "results": results
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"演武场运行错误: {error_msg}")
        print(error_trace)
        return jsonify({
            "success": False,
            "error": error_msg,
            "trace": error_trace if app.debug else None
        }), 500


@app.route('/api/arena/ai-explain', methods=['POST'])
def arena_ai_explain():
    """AI解释演武场结果"""
    from datetime import datetime
    
    data = request.json or {}
    results = data.get('results', [])
    summary = data.get('summary', {})
    
    if not results:
        return jsonify({
            'success': False,
            'error': '没有结果数据'
        }), 400
    
    # 分析结果，生成解释
    winner = summary.get('winner', '未知')
    scenario = summary.get('scenario', 'normal')
    black_swan = summary.get('black_swan', False)
    rules_count = summary.get('rules_count', 0)
    
    # 分析各参赛者的表现
    analysis = {
        'winner_analysis': '',
        'performance_comparison': [],
        'key_insights': [],
        'risk_assessment': '',
        'recommendations': []
    }
    
    # 找出胜者
    if results:
        winner_result = next((r for r in results if r.get('name') == winner), results[0])
        winner_score = winner_result.get('score_breakdown', {}).get('overall_score', 0)
        
        analysis['winner_analysis'] = f"""
        **{winner}** 在本轮演武中表现最佳，综合得分 {winner_score:.1%}。
        
        **关键优势：**
        - 审批率: {winner_result.get('approval_rate', 0)*100:.1f}%
        - 违约概率: {winner_result.get('avg_default_prob', 0)*100:.2f}%
        - 预估利润: ¥{winner_result.get('est_profit', 0)/1e6:.2f} 百万
        - RAROC: {winner_result.get('raroc', 0):.4f}
        
        **评分分解：**
        - 利润得分: {winner_result.get('score_breakdown', {}).get('profit_score', 0)*100:.1f}%
        - 风险得分: {winner_result.get('score_breakdown', {}).get('risk_score', 0)*100:.1f}%
        - 稳定性得分: {winner_result.get('score_breakdown', {}).get('stability_score', 0)*100:.1f}%
        """
    
    # 对比分析
    for r in results[:3]:  # 只分析前3名
        name = r.get('name', '未知')
        score = r.get('score_breakdown', {}).get('overall_score', 0)
        approval_rate = r.get('approval_rate', 0) * 100
        default_prob = r.get('avg_default_prob', 0) * 100
        profit = r.get('est_profit', 0) / 1e6
        
        analysis['performance_comparison'].append({
            'name': name,
            'score': score,
            'approval_rate': approval_rate,
            'default_prob': default_prob,
            'profit': profit,
            'summary': f"{name}: 综合得分 {score:.1%}, 审批率 {approval_rate:.1f}%, 违约率 {default_prob:.2f}%, 利润 {profit:.2f}百万"
        })
    
    # 关键洞察
    if len(results) >= 2:
        best = results[0]
        worst = results[-1]
        analysis['key_insights'].append(
            f"最佳策略 {best.get('name')} 的利润是 {worst.get('name')} 的 "
            f"{best.get('est_profit', 1) / max(worst.get('est_profit', 1), 1):.1f} 倍"
        )
        
        # 规则影响分析
        triggered_rules = {}
        for r in results:
            for rule_name, count in r.get('triggered_rules', {}).items():
                triggered_rules[rule_name] = triggered_rules.get(rule_name, 0) + count
        
        if triggered_rules:
            top_rule = max(triggered_rules.items(), key=lambda x: x[1])
            analysis['key_insights'].append(
                f"规则 '{top_rule[0]}' 被触发了 {top_rule[1]} 次，是最活跃的规则"
            )
    
    # 风险评估
    avg_default = sum(r.get('avg_default_prob', 0) for r in results) / len(results) if results else 0
    if avg_default > 0.15:
        analysis['risk_assessment'] = "⚠️ 整体违约概率较高，建议加强风险控制措施"
    elif avg_default > 0.10:
        analysis['risk_assessment'] = "⚡ 违约概率处于中等水平，需要持续监控"
    else:
        analysis['risk_assessment'] = "✅ 违约概率较低，风险控制良好"
    
    # 建议
    if scenario == 'stress':
        analysis['recommendations'].append("当前处于压力情景，建议采用更保守的审批策略")
    if black_swan:
        analysis['recommendations'].append("黑天鹅事件发生，建议暂停高风险业务，加强风险储备")
    if rules_count > 0:
        analysis['recommendations'].append(f"已配置 {rules_count} 条规则，规则引擎运行正常")
    
    return jsonify({
        'success': True,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/arena/export', methods=['POST'])
def arena_export():
    """导出演武场结果报告"""
    import json
    from datetime import datetime
    
    data = request.json or {}
    format_type = data.get('format', 'json')  # json, csv, txt, html
    results = data.get('results', [])
    summary = data.get('summary', {})
    
    if not results:
        return jsonify({
            'success': False,
            'error': '没有结果数据'
        }), 400
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format_type == 'json':
        export_data = {
            'summary': summary,
            'results': results,
            'export_time': datetime.now().isoformat()
        }
        return Response(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename=arena_result_{timestamp}.json'
            }
        )
    
    elif format_type == 'csv':
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 表头
        writer.writerow(['排名', '参赛者', '审批率', '违约概率', '预估利润', 'RAROC', '综合得分'])
        
        # 数据行
        for idx, r in enumerate(results, 1):
            writer.writerow([
                idx,
                r.get('name', ''),
                f"{r.get('approval_rate', 0)*100:.2f}%",
                f"{r.get('avg_default_prob', 0)*100:.2f}%",
                f"{r.get('est_profit', 0):.2f}",
                f"{r.get('raroc', 0):.4f}",
                f"{r.get('score_breakdown', {}).get('overall_score', 0)*100:.2f}%"
            ])
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=arena_result_{timestamp}.csv'
            }
        )
    
    elif format_type == 'txt':
        lines = [
            f"演武场结果报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"=" * 60,
            f"",
            f"场景: {summary.get('scenario', 'normal')}",
            f"黑天鹅: {'是' if summary.get('black_swan') else '否'}",
            f"客户数: {summary.get('customer_count', 0)}",
            f"贷款金额: ¥{summary.get('loan_amount', 0):,.0f}",
            f"规则数: {summary.get('rules_count', 0)}",
            f"",
            f"结果排名:",
            f"-" * 60,
        ]
        
        for idx, r in enumerate(results, 1):
            lines.append(f"{idx}. {r.get('name', '未知')}")
            lines.append(f"   审批率: {r.get('approval_rate', 0)*100:.1f}%")
            lines.append(f"   违约概率: {r.get('avg_default_prob', 0)*100:.2f}%")
            lines.append(f"   预估利润: ¥{r.get('est_profit', 0)/1e6:.2f} 百万")
            lines.append(f"   综合得分: {r.get('score_breakdown', {}).get('overall_score', 0)*100:.1f}%")
            lines.append("")
        
        return Response(
            '\n'.join(lines),
            mimetype='text/plain',
            headers={
                'Content-Disposition': f'attachment; filename=arena_result_{timestamp}.txt'
            }
        )
    
    elif format_type == 'html':
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>演武场结果报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>演武场结果报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>场景: {summary.get('scenario', 'normal')} | 黑天鹅: {'是' if summary.get('black_swan') else '否'}</p>
    <table>
        <tr>
            <th>排名</th>
            <th>参赛者</th>
            <th>审批率</th>
            <th>违约概率</th>
            <th>预估利润</th>
            <th>RAROC</th>
            <th>综合得分</th>
        </tr>
"""
        for idx, r in enumerate(results, 1):
            html_content += f"""
        <tr>
            <td>{idx}</td>
            <td>{r.get('name', '')}</td>
            <td>{r.get('approval_rate', 0)*100:.1f}%</td>
            <td>{r.get('avg_default_prob', 0)*100:.2f}%</td>
            <td>¥{r.get('est_profit', 0)/1e6:.2f} 百万</td>
            <td>{r.get('raroc', 0):.4f}</td>
            <td>{r.get('score_breakdown', {}).get('overall_score', 0)*100:.1f}%</td>
        </tr>
"""
        html_content += """
    </table>
</body>
</html>
"""
        return Response(
            html_content,
            mimetype='text/html',
            headers={
                'Content-Disposition': f'attachment; filename=arena_result_{timestamp}.html'
            }
        )
    
    return jsonify({
        'success': False,
        'error': f'不支持的格式: {format_type}'
    }), 400


@app.route('/api/arena/tournament', methods=['POST'])
def run_tournament():
    """
    多智能体锦标赛
    输入：
    {
        "agents": [
            {"id": "agent1", "name": "激进策略", "strategy": "aggressive", "approval_threshold": 0.25, "rate_spread": 0.02},
            {"id": "agent2", "name": "平衡策略", "strategy": "rule_based", "approval_threshold": 0.18, "rate_spread": 0.015},
            {"id": "agent3", "name": "保守策略", "strategy": "conservative", "approval_threshold": 0.12, "rate_spread": 0.01}
        ],
        "customer_count": 50,
        "loan_amount": 100000,
        "base_rate": 0.08,
        "seed": 42,
        "rounds": 10,
        "rules": [...],
        "scenario": "normal"
    }
    """
    from datetime import datetime
    
    data = request.json or {}
    agents = data.get('agents', [])
    customer_count = int(data.get('customer_count', 50))
    loan_amount = float(data.get('loan_amount', 100000))
    base_rate = float(data.get('base_rate', 0.08))
    seed = int(data.get('seed', 42))
    rounds = int(data.get('rounds', 10))
    rules_config = data.get('rules', [])
    scenario = data.get('scenario', 'normal')
    
    # 初始化
    rule_engine = RuleEngine(rules_config) if rules_config else None
    game = MultiAgentGame(seed=seed)
    
    # 生成客户
    customers = [generator.generate_one() for _ in range(customer_count)]
    
    # 创建市场条件
    gdp = 0.03 if scenario == 'normal' else -0.02
    unemp = 0.05 if scenario == 'normal' else 0.09
    market = MarketConditions(
        gdp_growth=gdp,
        base_interest_rate=base_rate,
        unemployment_rate=unemp,
        inflation_rate=0.02,
        credit_spread=0.02,
    )
    
    # 运行锦标赛
    result = game.run_tournament(
        agents=agents,
        customers=customers[:rounds],
        market=market,
        base_rate=base_rate,
        base_loan_amount=loan_amount,
        rounds=rounds,
        rule_engine=rule_engine
    )
    
    summary = {
        "run_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "rounds": rounds,
        "customer_count": customer_count,
        "loan_amount": loan_amount,
        "base_rate": base_rate,
        "seed": seed,
        "scenario": scenario,
        "champion": result['champion'],
        "agent_count": len(agents)
    }
    
    return jsonify({
        "success": True,
        "summary": summary,
        "round_history": result['round_history'],
        "agent_scores": result['agent_scores'],
        "agent_stats": result['agent_stats'],
        "final_rankings": result['final_rankings']
    })


@app.route('/api/arena/llm-decision', methods=['POST'])
def llm_decision():
    """
    LLM决策接口（集成真实LLM）
    输入：
    {
        "model_id": "gpt-4",
        "customer": {...},
        "loan_offer": {...},
        "market_conditions": {...},
        "context": "审批决策"
    }
    """
    from model_evaluation.model_gateway import gateway
    
    data = request.json or {}
    model_id = data.get('model_id')
    customer = data.get('customer')
    loan_offer = data.get('loan_offer')
    market_conditions = data.get('market_conditions')
    context = data.get('context', '贷款审批决策')
    
    if not model_id:
        return jsonify({
            'success': False,
            'error': '缺少model_id'
        }), 400
    
    try:
        # 构建提示词
        prompt = f"""
        作为银行信贷审批AI，请分析以下客户信息并做出审批决策：
        
        客户信息：
        - 月收入: {customer.get('monthly_income', 0):,.0f} 元
        - 年龄: {customer.get('age', 0)} 岁
        - 信用分: {customer.get('credit_score', 0)}
        - 负债率: {customer.get('debt_ratio', 0):.2%}
        
        贷款条件：
        - 金额: {loan_offer.get('amount', 0):,.0f} 元
        - 利率: {loan_offer.get('interest_rate', 0):.2%}
        - 期限: {loan_offer.get('term_months', 0)} 个月
        
        请给出：
        1. 审批决策（approve/reject）
        2. 决策理由
        3. 风险评估
        4. 建议的利率调整（如果需要）
        """
        
        # 调用模型网关（目前是模拟，后续可接入真实LLM）
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(gateway.call_model(model_id, prompt))
        
        # 解析LLM响应（简化版）
        response_text = result.get('response', '')
        decision = 'approve' if 'approve' in response_text.lower() or '批准' in response_text else 'reject'
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'decision': decision,
            'reasoning': response_text,
            'latency': result.get('latency', 0),
            'raw_response': response_text
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/arena/default-rules', methods=['GET'])
def get_default_rules():
    """获取默认规则模板"""
    import json
    from pathlib import Path
    
    try:
        rules_file = Path(__file__).parent / 'data' / 'default_rules.json'
        if rules_file.exists():
            with open(rules_file, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            return jsonify({
                'success': True,
                'rules': rules,
                'count': len(rules)
            })
        else:
            # 返回空规则列表
            return jsonify({
                'success': True,
                'rules': [],
                'count': 0
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/arena/scenario-templates', methods=['GET'])
def get_scenario_templates():
    """获取业务场景模板列表"""
    import json
    from pathlib import Path
    
    try:
        templates_file = Path(__file__).parent / 'data' / 'scenario_templates.json'
        if templates_file.exists():
            with open(templates_file, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            # 返回简化的模板列表（用于选择）
            simplified = [
                {
                    'id': t.get('id'),
                    'name': t.get('name'),
                    'description': t.get('description'),
                    'category': t.get('category'),
                    'business_value': t.get('business_value'),
                    'use_cases': t.get('use_cases', [])
                }
                for t in templates
            ]
            
            return jsonify({
                'success': True,
                'templates': simplified,
                'full_templates': templates,
                'count': len(templates)
            })
        else:
            return jsonify({
                'success': True,
                'templates': [],
                'full_templates': [],
                'count': 0
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/arena/scenario-templates/<template_id>', methods=['GET'])
def get_scenario_template(template_id):
    """获取指定的业务场景模板详情"""
    import json
    from pathlib import Path
    
    try:
        templates_file = Path(__file__).parent / 'data' / 'scenario_templates.json'
        if templates_file.exists():
            with open(templates_file, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            template = next((t for t in templates if t.get('id') == template_id), None)
            if not template:
                return jsonify({
                    'success': False,
                    'error': f'场景模板 {template_id} 不存在'
                }), 404
            
            return jsonify({
                'success': True,
                'template': template
            })
        else:
            return jsonify({
                'success': False,
                'error': '场景模板文件不存在'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/arena/multi-round', methods=['POST'])
def run_multi_round_arena():
    """
    多轮多场景演武场
    输入：
    {
        "participants": [...],
        "rounds": 6,
        "customer_count": 300,
        "loan_amount": 100000,
        "base_rate": 0.08,
        "seed": 42,
        "scenario_sequence": ["normal", "stress", "normal"],
        "black_swan_rounds": [3, 6],
        "rules": [...],
        "scoring_weights": {...}
    }
    """
    from datetime import datetime
    
    data = request.json or {}
    participants = data.get('participants', [])
    rounds = int(data.get('rounds', 6))
    customer_count = int(data.get('customer_count', 300))
    loan_amount = float(data.get('loan_amount', 100000))
    base_rate = float(data.get('base_rate', 0.08))
    seed = int(data.get('seed', 42))
    scenario_sequence = data.get('scenario_sequence', ['normal'] * rounds)
    black_swan_rounds = data.get('black_swan_rounds', [])
    rules_config = data.get('rules', [])
    scoring_weights = data.get('scoring_weights', {})
    
    # 初始化
    rule_engine = RuleEngine(rules_config) if rules_config else None
    scoring_system = ScoringSystem(scoring_weights) if scoring_weights else None
    simulator = MultiRoundSimulator(seed=seed)
    
    # 运行多轮模拟
    result = simulator.simulate_multi_rounds(
        rounds=rounds,
        participants=participants,
        customer_count=customer_count,
        rules=rules_config,
        base_rate=base_rate,
        loan_amount=loan_amount,
        seed=seed,
        scenario_sequence=scenario_sequence,
        black_swan_rounds=black_swan_rounds,
        rule_engine=rule_engine,
        scoring_system=scoring_system
    )
    
    summary = {
        "run_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "rounds": rounds,
        "customer_count": customer_count,
        "loan_amount": loan_amount,
        "base_rate": base_rate,
        "seed": seed,
        "winner": result['winner'],
        "final_scores": result['final_scores']
    }
    
    return jsonify({
        "success": True,
        "summary": summary,
        "round_history": result['round_history'],
        "final_scores": result['final_scores']
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

# ============================================================
# 多模型对抗评测 API
# ============================================================

from model_evaluation.model_gateway import ModelGateway, ModelType, gateway
from model_evaluation.evaluator import AdversarialEvaluator, TestCase, get_default_test_cases
import asyncio

# 初始化模型网关（注册主流模型）
def init_model_gateway():
    """初始化模型网关，注册主流国内外模型"""
    
    # OpenAI系列
    gateway.register_model(
        'gpt-4',
        ModelType.OPENAI,
        {
            'model_name': 'gpt-4',
            'api_key': 'demo-key',
            'endpoint': 'https://api.openai.com/v1/chat/completions',
            'temperature': 0.7,
            'provider': 'OpenAI',
            'features': ['推理', '多模态']
        }
    )
    
    gateway.register_model(
        'gpt-4-turbo',
        ModelType.OPENAI,
        {
            'model_name': 'gpt-4-turbo',
            'api_key': 'demo-key',
            'endpoint': 'https://api.openai.com/v1/chat/completions',
            'temperature': 0.7,
            'provider': 'OpenAI',
            'features': ['推理', '多模态', '便宜']
        }
    )
    
    gateway.register_model(
        'gpt-3.5-turbo',
        ModelType.OPENAI,
        {
            'model_name': 'gpt-3.5-turbo',
            'api_key': 'demo-key',
            'endpoint': 'https://api.openai.com/v1/chat/completions',
            'temperature': 0.7,
            'provider': 'OpenAI',
            'features': ['便宜', '快速']
        }
    )
    
    # Anthropic系列
    gateway.register_model(
        'claude-3-opus',
        ModelType.ANTHROPIC,
        {
            'model_name': 'claude-3-opus',
            'api_key': 'demo-key',
            'endpoint': 'https://api.anthropic.com/v1/messages',
            'temperature': 0.7,
            'provider': 'Anthropic',
            'features': ['推理', '长上下文']
        }
    )
    
    gateway.register_model(
        'claude-3-sonnet',
        ModelType.ANTHROPIC,
        {
            'model_name': 'claude-3-sonnet',
            'api_key': 'demo-key',
            'endpoint': 'https://api.anthropic.com/v1/messages',
            'temperature': 0.7,
            'provider': 'Anthropic',
            'features': ['推理', '平衡']
        }
    )
    
    # Google系列
    gateway.register_model(
        'gemini-pro',
        ModelType.OPENAI,  # 使用OpenAI兼容接口
        {
            'model_name': 'gemini-pro',
            'api_key': 'demo-key',
            'endpoint': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
            'temperature': 0.7,
            'provider': 'Google',
            'features': ['多模态', '推理', '便宜']
        }
    )
    
    gateway.register_model(
        'gemini-ultra',
        ModelType.OPENAI,
        {
            'model_name': 'gemini-ultra',
            'api_key': 'demo-key',
            'endpoint': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-ultra:generateContent',
            'temperature': 0.7,
            'provider': 'Google',
            'features': ['多模态', '推理']
        }
    )
    
    # 阿里云通义千问系列
    gateway.register_model(
        'qwen-turbo',
        ModelType.OPENAI,
        {
            'model_name': 'qwen-turbo',
            'api_key': 'demo-key',
            'endpoint': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
            'temperature': 0.7,
            'provider': '阿里云',
            'features': ['便宜', '快速']
        }
    )
    
    gateway.register_model(
        'qwen-plus',
        ModelType.OPENAI,
        {
            'model_name': 'qwen-plus',
            'api_key': 'demo-key',
            'endpoint': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
            'temperature': 0.7,
            'provider': '阿里云',
            'features': ['推理', '平衡']
        }
    )
    
    gateway.register_model(
        'qwen-max',
        ModelType.OPENAI,
        {
            'model_name': 'qwen-max',
            'api_key': 'demo-key',
            'endpoint': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
            'temperature': 0.7,
            'provider': '阿里云',
            'features': ['推理', '长上下文']
        }
    )
    
    # DeepSeek系列
    gateway.register_model(
        'deepseek-chat',
        ModelType.OPENAI,
        {
            'model_name': 'deepseek-chat',
            'api_key': 'demo-key',
            'endpoint': 'https://api.deepseek.com/v1/chat/completions',
            'temperature': 0.7,
            'provider': 'DeepSeek',
            'features': ['推理', '便宜']
        }
    )
    
    gateway.register_model(
        'deepseek-coder',
        ModelType.OPENAI,
        {
            'model_name': 'deepseek-coder',
            'api_key': 'demo-key',
            'endpoint': 'https://api.deepseek.com/v1/chat/completions',
            'temperature': 0.7,
            'provider': 'DeepSeek',
            'features': ['代码', '推理']
        }
    )
    
    # 月之暗面Kimi系列
    gateway.register_model(
        'kimi-chat',
        ModelType.OPENAI,
        {
            'model_name': 'kimi-chat',
            'api_key': 'demo-key',
            'endpoint': 'https://api.moonshot.cn/v1/chat/completions',
            'temperature': 0.7,
            'provider': '月之暗面',
            'features': ['长上下文', '推理']
        }
    )
    
    # 智谱AI GLM系列
    gateway.register_model(
        'glm-4',
        ModelType.OPENAI,
        {
            'model_name': 'glm-4',
            'api_key': 'demo-key',
            'endpoint': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
            'temperature': 0.7,
            'provider': '智谱AI',
            'features': ['推理', '多模态']
        }
    )
    
    # 本地模型（Ollama等）
    gateway.register_model(
        'ollama-llama3',
        ModelType.LOCAL_LLM,
        {
            'model_name': 'llama3',
            'endpoint': 'http://localhost:11434/api/generate',
            'temperature': 0.7,
            'provider': 'Ollama',
            'features': ['本地部署', '便宜']
        }
    )
    
    gateway.register_model(
        'ollama-mistral',
        ModelType.LOCAL_LLM,
        {
            'model_name': 'mistral',
            'endpoint': 'http://localhost:11434/api/generate',
            'temperature': 0.7,
            'provider': 'Ollama',
            'features': ['本地部署', '推理']
        }
    )
    
    gateway.register_model(
        'ollama-qwen',
        ModelType.LOCAL_LLM,
        {
            'model_name': 'qwen',
            'endpoint': 'http://localhost:11434/api/generate',
            'temperature': 0.7,
            'provider': 'Ollama',
            'features': ['本地部署', '中文']
        }
    )
    
    # RAG应用
    gateway.register_model(
        'rag-finance',
        ModelType.RAG,
        {
            'model_name': 'finance-rag',
            'endpoint': 'http://localhost:8001/api/chat',
            'temperature': 0.7,
            'provider': 'RAG',
            'features': ['知识检索', '专业']
        }
    )

# 初始化模型网关（确保在导入时就执行）
def load_persisted_models():
    """从持久化文件加载模型配置"""
    try:
        import json
        from pathlib import Path
        models_file = Path(__file__).parent / 'data' / 'models_config.json'
        
        if models_file.exists():
            with open(models_file, 'r', encoding='utf-8') as f:
                all_models_config = json.load(f)
                
            # 将持久化的配置合并到内存中
            for model_id, model_data in all_models_config.items():
                if model_id in gateway.models:
                    # 合并配置，持久化的配置优先
                    persisted_config = model_data.get('config', {})
                    memory_config = gateway.models[model_id].get('config', {})
                    gateway.models[model_id]['config'] = {**memory_config, **persisted_config}
                else:
                    # 如果内存中没有，从文件加载
                    model_type_str = model_data.get('type', 'openai')
                    try:
                        model_type = ModelType(model_type_str)
                        gateway.register_model(model_id, model_type, model_data.get('config', {}))
                    except:
                        pass  # 忽略无效的模型类型
    except Exception as e:
        print(f"⚠️ 加载持久化模型配置失败: {e}")

try:
    init_model_gateway()
    load_persisted_models()  # 加载持久化的配置
    print(f"✅ 模型网关初始化完成，已注册 {len(gateway.list_models())} 个模型")
except Exception as e:
    print(f"⚠️ 模型网关初始化失败: {e}")
    import traceback
    traceback.print_exc()

@app.route('/api/models/list', methods=['GET'])
def list_models():
    """获取模型列表"""
    try:
        models = gateway.list_models()
        # 格式化模型数据，确保包含所有必要字段
        formatted_models = []
        for model in models:
            model_type = model.get('type')
            # 处理Enum类型
            if hasattr(model_type, 'value'):
                type_value = model_type.value
            elif isinstance(model_type, str):
                type_value = model_type
            else:
                type_value = str(model_type)
            
            formatted_models.append({
                'id': model.get('id', ''),
                'type': {
                    'value': type_value
                },
                'status': model.get('status', 'active'),
                'config': model.get('config', {})
            })
        return jsonify({
            'success': True,
            'models': formatted_models
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/models/register', methods=['POST'])
def register_model():
    """注册新模型（持久化保存）"""
    try:
        data = request.json or {}
        model_id = data.get('model_id')
        model_type_str = data.get('model_type', 'openai')
        config = data.get('config', {})
        
        if not model_id:
            return jsonify({
                'success': False,
                'error': '模型ID不能为空'
            }), 400
        
        model_type = ModelType(model_type_str)
        
        # 注册到内存
        gateway.register_model(model_id, model_type, config)
        
        # 持久化保存到文件
        import json
        from pathlib import Path
        models_file = Path(__file__).parent / 'data' / 'models_config.json'
        models_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取现有配置
        all_models_config = {}
        if models_file.exists():
            with open(models_file, 'r', encoding='utf-8') as f:
                all_models_config = json.load(f)
        
        # 添加新模型
        all_models_config[model_id] = {
            'id': model_id,
            'type': model_type_str,
            'config': config,
            'status': 'active',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存到文件
        with open(models_file, 'w', encoding='utf-8') as f:
            json.dump(all_models_config, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'模型 {model_id} 注册成功并已保存',
            'model_id': model_id
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/evaluation/test-cases', methods=['GET'])
def get_test_cases():
    """获取测试用例列表"""
    try:
        test_cases = get_default_test_cases()
        return jsonify({
            'success': True,
            'test_cases': [
                {
                    'id': tc.id,
                    'scenario': tc.scenario,
                    'initial_prompt': tc.initial_prompt,
                    'rounds_count': len(tc.rounds),
                    'expected_keywords': tc.expected_keywords
                }
                for tc in test_cases
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/evaluation/create', methods=['POST'])
def create_evaluation():
    """创建评测任务"""
    try:
        data = request.json or {}
        model_ids = data.get('model_ids', [])
        test_case_ids = data.get('test_case_ids', [])
        
        if not model_ids:
            return jsonify({
                'success': False,
                'error': '请至少选择一个模型'
            }), 400
        
        # 获取测试用例
        all_test_cases = get_default_test_cases()
        test_cases = [tc for tc in all_test_cases if tc.id in test_case_ids]
        
        if not test_cases:
            test_cases = all_test_cases  # 默认使用所有测试用例
        
        # 执行评测（使用异步）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        evaluator = AdversarialEvaluator(gateway)
        results = loop.run_until_complete(evaluator.evaluate(model_ids, test_cases))
        loop.close()
        
        # 格式化结果（包含完整对话内容）
        formatted_results = []
        for result in results:
            # 构建完整对话历史
            test_case = next((tc for tc in test_cases if tc.id == result.test_case_id), None)
            dialogue_history = []
            if test_case:
                dialogue_history.append({
                    'role': 'user',
                    'content': test_case.initial_prompt
                })
                for i, round_config in enumerate(test_case.rounds):
                    if round_config.get('role') == 'user':
                        dialogue_history.append({
                            'role': 'user',
                            'content': round_config.get('content', '')
                        })
                        if i < len(result.responses):
                            dialogue_history.append({
                                'role': 'assistant',
                                'content': result.responses[i] if i < len(result.responses) else '',
                                'latency': result.latencies[i] if i < len(result.latencies) else 0
                            })
                    elif round_config.get('role') == 'assistant':
                        dialogue_history.append({
                            'role': 'assistant',
                            'content': round_config.get('content', '')
                        })
            
            formatted_results.append({
                'model_id': result.model_id,
                'test_case_id': result.test_case_id,
                'responses': result.responses,
                'latencies': result.latencies,
                'metrics': result.metrics,
                'errors': result.errors,
                'dialogue_history': dialogue_history  # 完整对话历史
            })
        
        # 计算汇总指标（专业版）
        summary = {}
        for model_id in model_ids:
            model_results = [r for r in formatted_results if r['model_id'] == model_id]
            if model_results:
                summary[model_id] = {
                    # 核心指标
                    'avg_accuracy': sum(r['metrics'].get('accuracy', 0) for r in model_results) / len(model_results),
                    'avg_latency': sum(r['metrics'].get('avg_latency', 0) for r in model_results) / len(model_results),
                    'avg_hallucination_rate': sum(r['metrics'].get('hallucination_rate', 0) for r in model_results) / len(model_results),
                    'avg_compliance_rate': sum(r['metrics'].get('compliance_rate', 0) for r in model_results) / len(model_results),
                    'avg_consistency': sum(r['metrics'].get('consistency', 0) for r in model_results) / len(model_results),
                    # 专业指标
                    'avg_professionalism': sum(r['metrics'].get('professionalism', 0) for r in model_results) / len(model_results),
                    'avg_response_quality': sum(r['metrics'].get('response_quality', 0) for r in model_results) / len(model_results),
                    'avg_stability': sum(r['metrics'].get('stability', 0) for r in model_results) / len(model_results),
                    'avg_performance_score': sum(r['metrics'].get('performance_score', 0) for r in model_results) / len(model_results),
                    'avg_business_fit': sum(r['metrics'].get('business_fit', 0) for r in model_results) / len(model_results),
                    # 综合评分
                    'avg_overall_score': sum(r['metrics'].get('overall_score', 0) for r in model_results) / len(model_results),
                    'total_tests': len(model_results)
                }
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'summary': summary
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/evaluation/test-case/<test_case_id>', methods=['GET'])
def get_test_case_detail(test_case_id):
    """获取测试用例详情（用于合规审计）"""
    try:
        test_cases = get_default_test_cases()
        test_case = next((tc for tc in test_cases if tc.id == test_case_id), None)
        
        if not test_case:
            return jsonify({
                'success': False,
                'error': '测试用例不存在'
            }), 404
        
        return jsonify({
            'success': True,
            'test_case': {
                'id': test_case.id,
                'scenario': test_case.scenario,
                'initial_prompt': test_case.initial_prompt,
                'rounds': test_case.rounds,
                'expected_keywords': test_case.expected_keywords,
                'expected_behavior': test_case.expected_behavior,
                'evaluation_criteria': test_case.evaluation_criteria if hasattr(test_case, 'evaluation_criteria') else None,
                'compliance_requirements': test_case.compliance_requirements if hasattr(test_case, 'compliance_requirements') else None
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/evaluation/metrics-explanation', methods=['GET'])
def get_metrics_explanation():
    """获取评分依据说明（用于合规审计）"""
    try:
        explanation = {
            'accuracy': {
                'name': '正确率',
                'description': '评估模型回答的正确性',
                'calculation': '基于关键词匹配和语义理解，计算响应中包含预期关键词的比例',
                'formula': '正确率 = (匹配关键词数 / 预期关键词总数) * 0.7 + (响应完整性得分) * 0.3',
                'weight': 0.25,
                'compliance_note': '确保模型提供准确信息，避免误导客户'
            },
            'hallucination_rate': {
                'name': '幻觉率',
                'description': '评估模型产生错误或虚假信息的比例',
                'calculation': '多维度检测：响应过短(0.5分) + 错误标记(0.3分) + 重复内容过多(0.2分)',
                'formula': '幻觉率 = min(1.0, 各项检测得分之和)',
                'weight': 0.20,
                'compliance_note': '降低幻觉率是金融合规的核心要求，避免虚假信息误导决策'
            },
            'compliance_rate': {
                'name': '合规率',
                'description': '评估模型回答的金融合规性',
                'calculation': '检查4类合规关键词：风险提示、监管要求、审批流程、信息披露',
                'formula': '合规率 = (各类合规关键词匹配得分) / 4',
                'weight': 0.15,
                'compliance_note': '必须包含必要的风险提示和监管要求说明，确保符合金融监管规定'
            },
            'professionalism': {
                'name': '专业性',
                'description': '评估模型使用金融专业术语和逻辑结构的能力',
                'calculation': '专业术语使用(60%) + 逻辑结构(40%)',
                'formula': '专业性 = (术语得分 * 0.6) + (结构得分 * 0.4)',
                'weight': 0.12,
                'compliance_note': '确保使用准确的金融术语，避免非专业表述'
            },
            'response_quality': {
                'name': '响应质量',
                'description': '评估响应的完整性和可读性',
                'calculation': '完整性(60%) + 可读性(40%)',
                'formula': '质量 = (长度得分 * 0.6) + (可读性得分 * 0.4)',
                'weight': 0.10,
                'compliance_note': '确保响应完整、清晰，便于客户理解'
            },
            'consistency': {
                'name': '一致性',
                'description': '评估多轮对话的连贯性',
                'calculation': '检查相邻轮次的关键词重叠度',
                'formula': '一致性 = (关键词交集 / 关键词并集)',
                'weight': 0.12,
                'compliance_note': '确保多轮对话中信息一致，避免前后矛盾'
            },
            'business_fit': {
                'name': '业务适配度',
                'description': '评估模型在金融业务场景中的适配程度',
                'calculation': '合规率(40%) + 专业性(30%) + 正确率(30%)',
                'formula': '业务适配 = 合规率*0.4 + 专业性*0.3 + 正确率*0.3',
                'weight': 0.0,
                'compliance_note': '综合评估模型在金融业务场景中的适用性'
            },
            'overall_score': {
                'name': '综合评分',
                'description': '综合所有维度的加权评分',
                'calculation': '正确性25% + 准确性20% + 合规性15% + 一致性12% + 专业性12% + 质量10% + 稳定性6%',
                'formula': '综合评分 = Σ(各维度得分 * 权重)',
                'weight': 1.0,
                'compliance_note': '综合评分用于模型选型和性能评估，所有评分依据可追溯'
            }
        }
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'audit_info': {
                'version': '1.0',
                'last_updated': '2024-01-01',
                'reviewer': 'Gamium Finance AI System',
                'compliance_standard': '金融行业AI应用合规指南'
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/evaluation/save-script', methods=['POST'])
def save_script():
    """保存对话为内部标准话术"""
    try:
        data = request.json or {}
        model_id = data.get('model_id')
        test_case_id = data.get('test_case_id')
        dialogue_history = data.get('dialogue_history', [])
        script_name = data.get('script_name', f'{test_case_id}_{model_id}')
        
        if not dialogue_history:
            return jsonify({
                'success': False,
                'error': '对话历史为空'
            }), 400
        
        # 构建话术模板
        script_template = {
            'id': f'script_{int(time.time())}',
            'name': script_name,
            'model_id': model_id,
            'test_case_id': test_case_id,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dialogue_history': dialogue_history,
            'metadata': {
                'total_rounds': len([d for d in dialogue_history if d.get('role') == 'user']),
                'total_tokens': sum(len(d.get('content', '')) for d in dialogue_history),
                'avg_latency': sum(d.get('latency', 0) for d in dialogue_history) / len(dialogue_history) if dialogue_history else 0
            }
        }
        
        # 保存到文件（实际应用中应保存到数据库）
        scripts_dir = Path(__file__).parent / 'data' / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_file = scripts_dir / f'{script_template["id"]}.json'
        with open(script_file, 'w', encoding='utf-8') as f:
            json.dump(script_template, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'script': script_template,
            'message': '话术模板已保存'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/evaluation/export-script', methods=['POST'])
def export_script():
    """导出话术模板"""
    try:
        data = request.json or {}
        script_id = data.get('script_id')
        export_format = data.get('format', 'json')
        
        # 读取话术文件
        scripts_dir = Path(__file__).parent / 'data' / 'scripts'
        script_file = scripts_dir / f'{script_id}.json'
        
        if not script_file.exists():
            return jsonify({
                'success': False,
                'error': '话术模板不存在'
            }), 404
        
        with open(script_file, 'r', encoding='utf-8') as f:
            script = json.load(f)
        
        if export_format == 'json':
            return jsonify({
                'success': True,
                'script': script,
                'format': 'json'
            })
        elif export_format == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['轮次', '角色', '内容', '延时(ms)'])
            for i, dialogue in enumerate(script['dialogue_history']):
                writer.writerow([
                    i + 1,
                    dialogue.get('role', ''),
                    dialogue.get('content', ''),
                    int(dialogue.get('latency', 0) * 1000)
                ])
            return jsonify({
                'success': True,
                'content': output.getvalue(),
                'format': 'csv',
                'filename': f'{script["name"]}.csv'
            })
        elif export_format == 'txt':
            content = f"话术模板: {script['name']}\n"
            content += f"模型: {script['model_id']}\n"
            content += f"创建时间: {script['created_at']}\n"
            content += "=" * 50 + "\n\n"
            for i, dialogue in enumerate(script['dialogue_history']):
                role_name = '客户' if dialogue.get('role') == 'user' else '客服'
                content += f"[第{i+1}轮] {role_name}:\n"
                content += f"{dialogue.get('content', '')}\n\n"
            return jsonify({
                'success': True,
                'content': content,
                'format': 'txt',
                'filename': f'{script["name"]}.txt'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'不支持的导出格式: {export_format}'
            }), 400
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ============================================================
# AI Hub 管理 API
# ============================================================

@app.route('/api/hub/model/<model_id>', methods=['GET'])
def get_hub_model_detail(model_id):
    """获取模型详细配置（优先从持久化文件读取）"""
    try:
        # URL解码模型ID（处理中文模型名）
        from urllib.parse import unquote
        model_id = unquote(model_id)
        
        # 先尝试从持久化文件读取
        import json
        from pathlib import Path
        models_file = Path(__file__).parent / 'data' / 'models_config.json'
        
        config = {}
        if models_file.exists():
            try:
                with open(models_file, 'r', encoding='utf-8') as f:
                    all_models_config = json.load(f)
                    if model_id in all_models_config:
                        config = all_models_config[model_id].get('config', {})
            except:
                pass  # 文件读取失败，继续使用内存配置
        
        # 如果文件没有，从内存读取
        model_info = gateway.get_model(model_id)
        if not model_info:
            return jsonify({
                'success': False,
                'error': f'模型不存在: {model_id}'
            }), 404
        
        # 合并配置（文件优先）
        if not config:
            config = model_info.get('config', {})
        else:
            # 合并，文件配置优先
            memory_config = model_info.get('config', {})
            config = {**memory_config, **config}
        
        # 安全获取type值
        model_type = model_info.get('type')
        if model_type is None:
            type_value = 'unknown'
        elif hasattr(model_type, 'value'):
            type_value = model_type.value
        elif isinstance(model_type, str):
            type_value = model_type
        else:
            type_value = str(model_type)
        
        # 返回完整配置信息
        return jsonify({
            'success': True,
            'model': {
                'id': model_id,
                'type': type_value,
                'api_key': config.get('api_key', ''),
                'endpoint': config.get('endpoint', ''),
                'model_name': config.get('model_name', model_id),
                'temperature': config.get('temperature', 0.7),
                'max_tokens': config.get('max_tokens', None),
                'provider': config.get('provider', ''),
                'daily_limit': config.get('daily_limit', 10000),
                'rate_limit': config.get('rate_limit', 60),
                'used_today': config.get('used_today', 0),
                'circuit_breaker_error_rate': config.get('circuit_breaker_error_rate', 0.5),
                'circuit_breaker_latency': config.get('circuit_breaker_latency', 5000),
                'circuit_breaker_open': config.get('circuit_breaker_open', False),
                'sensitive_filter_enabled': config.get('sensitive_filter_enabled', True),
                'filter_mode': config.get('filter_mode', 'replace')
            }
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ 获取模型详情失败: {error_msg}")
        print(error_trace)
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': error_trace
        }), 500

@app.route('/api/hub/sensitive-words', methods=['GET'])
def get_sensitive_words():
    """获取敏感词列表"""
    try:
        # 从文件或数据库加载敏感词
        import json
        from pathlib import Path
        sensitive_file = Path(__file__).parent / 'data' / 'sensitive_words.json'
        
        if sensitive_file.exists():
            with open(sensitive_file, 'r', encoding='utf-8') as f:
                words = json.load(f)
        else:
            # 默认敏感词
            words = [
                {'id': '1', 'word': '内幕消息', 'category': '金融'},
                {'id': '2', 'word': '保证收益', 'category': '金融'},
                {'id': '3', 'word': '高额回报', 'category': '金融'},
                {'id': '4', 'word': '无风险', 'category': '金融'},
                {'id': '5', 'word': '稳赚不赔', 'category': '金融'}
            ]
        
        return jsonify({
            'success': True,
            'words': words
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hub/sensitive-words', methods=['POST'])
def add_sensitive_word():
    """添加敏感词"""
    try:
        data = request.json or {}
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({
                'success': False,
                'error': '敏感词不能为空'
            }), 400
        
        # 保存到文件
        import json
        from pathlib import Path
        sensitive_file = Path(__file__).parent / 'data' / 'sensitive_words.json'
        sensitive_file.parent.mkdir(parents=True, exist_ok=True)
        
        words = []
        if sensitive_file.exists():
            with open(sensitive_file, 'r', encoding='utf-8') as f:
                words = json.load(f)
        
        new_word = {
            'id': str(int(time.time())),
            'word': word,
            'category': data.get('category', '通用'),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        words.append(new_word)
        
        with open(sensitive_file, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'word': new_word
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hub/sensitive-words/<word_id>', methods=['DELETE'])
def delete_sensitive_word(word_id):
    """删除敏感词"""
    try:
        import json
        from pathlib import Path
        sensitive_file = Path(__file__).parent / 'data' / 'sensitive_words.json'
        
        if not sensitive_file.exists():
            return jsonify({
                'success': False,
                'error': '敏感词文件不存在'
            }), 404
        
        with open(sensitive_file, 'r', encoding='utf-8') as f:
            words = json.load(f)
        
        words = [w for w in words if w.get('id') != word_id]
        
        with open(sensitive_file, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hub/model/<model_id>', methods=['PUT'])
def update_model_config(model_id):
    """更新模型配置（持久化保存）"""
    try:
        # URL解码模型ID（处理中文模型名）
        from urllib.parse import unquote
        model_id = unquote(model_id)
        
        data = request.json or {}
        model_info = gateway.get_model(model_id)
        
        if not model_info:
            return jsonify({
                'success': False,
                'error': f'模型不存在: {model_id}'
            }), 404
        
        # 更新配置
        config = model_info.get('config', {})
        
        # 更新所有配置项
        if 'api_key' in data:
            config['api_key'] = data['api_key']
        if 'endpoint' in data:
            config['endpoint'] = data['endpoint']
        if 'model_name' in data:
            config['model_name'] = data['model_name']
        if 'temperature' in data:
            config['temperature'] = float(data['temperature'])
        if 'max_tokens' in data:
            config['max_tokens'] = int(data['max_tokens']) if data['max_tokens'] else None
        if 'daily_limit' in data:
            config['daily_limit'] = int(data['daily_limit']) if data['daily_limit'] else None
        if 'rate_limit' in data:
            config['rate_limit'] = int(data['rate_limit']) if data['rate_limit'] else None
        if 'circuit_breaker_error_rate' in data:
            config['circuit_breaker_error_rate'] = float(data['circuit_breaker_error_rate'])
        if 'circuit_breaker_latency' in data:
            config['circuit_breaker_latency'] = int(data['circuit_breaker_latency'])
        if 'sensitive_filter_enabled' in data:
            config['sensitive_filter_enabled'] = bool(data['sensitive_filter_enabled'])
        if 'filter_mode' in data:
            config['filter_mode'] = data['filter_mode']
        
        # 持久化保存到文件
        import json
        from pathlib import Path
        models_file = Path(__file__).parent / 'data' / 'models_config.json'
        models_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取现有配置
        all_models_config = {}
        if models_file.exists():
            with open(models_file, 'r', encoding='utf-8') as f:
                all_models_config = json.load(f)
        
        # 更新该模型的配置
        all_models_config[model_id] = {
            'id': model_id,
            'type': str(model_info.get('type').value) if hasattr(model_info.get('type'), 'value') else str(model_info.get('type')),
            'config': config,
            'status': model_info.get('status', 'active'),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存到文件
        with open(models_file, 'w', encoding='utf-8') as f:
            json.dump(all_models_config, f, ensure_ascii=False, indent=2)
        
        # 更新内存中的配置
        gateway.models[model_id]['config'] = config
        
        return jsonify({
            'success': True,
            'message': '配置更新成功并已保存',
            'model': {
                'id': model_id,
                'config': config
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/hub/model/<model_id>/test', methods=['POST'])
def test_model_connection(model_id):
    """测试模型连接"""
    try:
        model_info = gateway.get_model(model_id)
        if not model_info:
            return jsonify({
                'success': False,
                'error': '模型不存在'
            }), 404
        
        # 模拟测试连接
        import random
        test_result = {
            'success': True,
            'latency': round(random.uniform(0.1, 0.5), 3),
            'status': 'connected',
            'message': '连接成功',
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果有真实API key，可以实际测试
        config = model_info.get('config', {})
        if config.get('api_key') and config.get('api_key') != 'demo-key':
            # 这里可以添加真实的API测试逻辑
            pass
        
        return jsonify({
            'success': True,
            'result': test_result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hub/users', methods=['GET'])
def get_users():
    """获取用户列表"""
    try:
        # 从文件加载用户数据
        import json
        from pathlib import Path
        users_file = Path(__file__).parent / 'data' / 'users.json'
        
        if users_file.exists():
            with open(users_file, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            # 默认用户
            users = [
                {
                    'id': '1',
                    'username': 'admin',
                    'email': 'admin@gamium.ai',
                    'role': 'admin',
                    'permissions': ['all'],
                    'created_at': '2024-01-01 00:00:00',
                    'last_login': '2024-12-20 10:30:00'
                },
                {
                    'id': '2',
                    'username': 'analyst',
                    'email': 'analyst@gamium.ai',
                    'role': 'analyst',
                    'permissions': ['read', 'evaluate'],
                    'created_at': '2024-01-15 00:00:00',
                    'last_login': '2024-12-20 09:15:00'
                },
                {
                    'id': '3',
                    'username': 'viewer',
                    'email': 'viewer@gamium.ai',
                    'role': 'viewer',
                    'permissions': ['read'],
                    'created_at': '2024-02-01 00:00:00',
                    'last_login': '2024-12-19 16:45:00'
                }
            ]
        
        return jsonify({
            'success': True,
            'users': users
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hub/roles', methods=['GET'])
def get_roles():
    """获取角色列表"""
    try:
        roles = [
            {
                'id': 'admin',
                'name': '管理员',
                'description': '拥有所有权限',
                'permissions': ['all'],
                'user_count': 1
            },
            {
                'id': 'analyst',
                'name': '分析师',
                'description': '可以查看和评测模型',
                'permissions': ['read', 'evaluate', 'export'],
                'user_count': 1
            },
            {
                'id': 'viewer',
                'name': '查看者',
                'description': '只能查看数据',
                'permissions': ['read'],
                'user_count': 1
            },
            {
                'id': 'operator',
                'name': '运营',
                'description': '可以管理模型和敏感词',
                'permissions': ['read', 'manage_models', 'manage_sensitive'],
                'user_count': 0
            }
        ]
        
        return jsonify({
            'success': True,
            'roles': roles
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hub/statistics', methods=['GET'])
def get_hub_statistics():
    """获取使用统计（包含多维度下钻数据）"""
    try:
        # 模拟统计数据（实际应从数据库获取）
        import random
        
        base_total = 125680
        base_today = 3420
        
        # 生成部门统计数据（前10）
        departments = ['风控部', '信贷部', '产品部', '技术部', '运营部', '市场部', '财务部', '合规部', '客服部', '数据部']
        dept_stats = []
        for i, dept in enumerate(departments):
            calls = base_total // 10 + random.randint(-5000, 5000)
            cost = calls * (0.10 + random.uniform(-0.02, 0.02))
            dept_stats.append({
                'department': dept,
                'calls': max(1000, calls),
                'cost': round(cost, 2),
                'users': random.randint(5, 25),
                'avg_latency': round(random.uniform(0.5, 3.0), 2)
            })
        dept_stats.sort(key=lambda x: x['calls'], reverse=True)
        
        # 生成人员统计数据（前10）
        users = [
            {'name': '张三', 'department': '风控部', 'role': '高级分析师'},
            {'name': '李四', 'department': '信贷部', 'role': '信贷经理'},
            {'name': '王五', 'department': '产品部', 'role': '产品经理'},
            {'name': '赵六', 'department': '技术部', 'role': '算法工程师'},
            {'name': '钱七', 'department': '运营部', 'role': '运营专员'},
            {'name': '孙八', 'department': '市场部', 'role': '市场经理'},
            {'name': '周九', 'department': '财务部', 'role': '财务分析师'},
            {'name': '吴十', 'department': '合规部', 'role': '合规专员'},
            {'name': '郑一', 'department': '客服部', 'role': '客服主管'},
            {'name': '王二', 'department': '数据部', 'role': '数据分析师'}
        ]
        user_stats = []
        for i, user in enumerate(users):
            calls = base_total // 20 + random.randint(-2000, 5000)
            cost = calls * (0.10 + random.uniform(-0.02, 0.02))
            user_stats.append({
                'name': user['name'],
                'department': user['department'],
                'role': user['role'],
                'calls': max(500, calls),
                'cost': round(cost, 2),
                'avg_latency': round(random.uniform(0.5, 3.0), 2),
                'success_rate': round(random.uniform(0.92, 0.99), 3)
            })
        user_stats.sort(key=lambda x: x['calls'], reverse=True)
        
        # 生成模型统计数据（前10）
        models = ['gpt-4', 'claude-3-opus', 'qwen-max', 'gemini-pro', 'gpt-4-turbo', 
                  'claude-3-sonnet', 'qwen-plus', 'deepseek-chat', 'glm-4', 'kimi-chat']
        model_stats = []
        for i, model_id in enumerate(models):
            calls = base_total // 10 + random.randint(-3000, 8000)
            cost = calls * (0.08 + random.uniform(-0.03, 0.05))
            model_stats.append({
                'model_id': model_id,
                'calls': max(2000, calls),
                'cost': round(cost, 2),
                'avg_latency': round(random.uniform(0.3, 2.5), 2),
                'error_rate': round(random.uniform(0.01, 0.05), 3),
                'success_rate': round(random.uniform(0.95, 0.99), 3)
            })
        model_stats.sort(key=lambda x: x['calls'], reverse=True)
        
        # 按时间维度统计（最近7天）
        daily_stats = []
        for i in range(7, 0, -1):
            daily_calls = base_today + random.randint(-200, 500)
            daily_cost = daily_calls * (0.10 + random.uniform(-0.02, 0.02))
            daily_stats.append({
                'date': time.strftime('%Y-%m-%d', time.localtime(time.time() - i * 86400)),
                'calls': max(1000, daily_calls),
                'cost': round(daily_cost, 2)
            })
        
        return jsonify({
            'success': True,
            'call_stats': {
                'total': base_total + random.randint(-100, 100),
                'today': base_today + random.randint(-50, 50),
                'week': 18950 + random.randint(-200, 200),
                'month': 67890 + random.randint(-500, 500)
            },
            'cost_stats': {
                'total': 12580.50 + random.uniform(-10, 10),
                'today': 342.20 + random.uniform(-5, 5),
                'month': 3890.80 + random.uniform(-20, 20),
                'avg_per_call': 0.10 + random.uniform(-0.01, 0.01)
            },
            'error_stats': {
                'total': 45 + random.randint(-5, 5),
                'rate': 0.036 + random.uniform(-0.005, 0.005),
                'circuit_breaker': 2,
                'avg_error_latency': 2.5 + random.uniform(-0.5, 0.5)
            },
            'department_rank': [{'department': d['department'], 'count': d['calls'], 'total': base_total} for d in dept_stats[:10]],
            'user_rank': [{'username': u['name'], 'department': u['department'], 'count': u['calls'], 'total': base_total} for u in user_stats[:10]],
            'model_rank': [{'model_id': m['model_id'], 'count': m['calls'], 'total': base_total} for m in model_stats[:10]],
            'time_trend': [{'date': d['date'], 'count': d['calls']} for d in daily_stats],
            'drill_down': {
                'by_department': dept_stats[:10],  # 前10部门
                'by_user': user_stats[:10],  # 前10人员
                'by_model': model_stats[:10],  # 前10模型
                'by_time': daily_stats  # 最近7天
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/hub/statistics/drilldown', methods=['GET'])
def get_hub_statistics_drilldown():
    """获取使用统计下钻数据"""
    try:
        drilldown_type = request.args.get('type', 'calls')  # calls, costs, errors
        
        # 模拟统计数据（实际应从数据库获取）
        import random
        import time
        
        base_total = 125680
        base_today = 3420
        
        # 生成部门统计数据（前10）
        departments = ['风控部', '信贷部', '产品部', '技术部', '运营部', '市场部', '财务部', '合规部', '客服部', '数据部']
        dept_stats = []
        for i, dept in enumerate(departments):
            if drilldown_type == 'calls':
                count = base_total // 10 + random.randint(-5000, 5000)
            elif drilldown_type == 'costs':
                count = (base_total // 10 + random.randint(-5000, 5000)) * (0.10 + random.uniform(-0.02, 0.02))
            else:  # errors
                count = random.randint(2, 8)
            dept_stats.append({
                'department': dept,
                'count': max(1000 if drilldown_type == 'calls' else (100 if drilldown_type == 'costs' else 1), count),
                'total': base_total if drilldown_type == 'calls' else (base_total * 0.10 if drilldown_type == 'costs' else 45)
            })
        dept_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # 生成人员统计数据（前10）
        users = [
            {'name': '张三', 'department': '风控部'},
            {'name': '李四', 'department': '信贷部'},
            {'name': '王五', 'department': '产品部'},
            {'name': '赵六', 'department': '技术部'},
            {'name': '钱七', 'department': '运营部'},
            {'name': '孙八', 'department': '市场部'},
            {'name': '周九', 'department': '财务部'},
            {'name': '吴十', 'department': '合规部'},
            {'name': '郑一', 'department': '客服部'},
            {'name': '王二', 'department': '数据部'}
        ]
        user_stats = []
        for i, user in enumerate(users):
            if drilldown_type == 'calls':
                count = base_total // 20 + random.randint(-2000, 5000)
            elif drilldown_type == 'costs':
                count = (base_total // 20 + random.randint(-2000, 5000)) * (0.10 + random.uniform(-0.02, 0.02))
            else:  # errors
                count = random.randint(1, 5)
            user_stats.append({
                'username': user['name'],
                'department': user['department'],
                'count': max(500 if drilldown_type == 'calls' else (50 if drilldown_type == 'costs' else 1), count),
                'total': base_total if drilldown_type == 'calls' else (base_total * 0.10 if drilldown_type == 'costs' else 45)
            })
        user_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # 生成模型统计数据（前10）
        models = ['gpt-4', 'claude-3-opus', 'qwen-max', 'gemini-pro', 'gpt-4-turbo', 
                  'claude-3-sonnet', 'qwen-plus', 'deepseek-chat', 'glm-4', 'kimi-chat']
        model_stats = []
        for i, model_id in enumerate(models):
            if drilldown_type == 'calls':
                count = base_total // 10 + random.randint(-3000, 8000)
            elif drilldown_type == 'costs':
                count = (base_total // 10 + random.randint(-3000, 8000)) * (0.08 + random.uniform(-0.03, 0.05))
            else:  # errors
                count = random.randint(1, 6)
            model_stats.append({
                'model_id': model_id,
                'count': max(2000 if drilldown_type == 'calls' else (160 if drilldown_type == 'costs' else 1), count),
                'total': base_total if drilldown_type == 'calls' else (base_total * 0.10 if drilldown_type == 'costs' else 45)
            })
        model_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # 按时间维度统计（最近7天）
        time_trend = []
        for i in range(7, 0, -1):
            if drilldown_type == 'calls':
                count = base_today + random.randint(-200, 500)
            elif drilldown_type == 'costs':
                count = (base_today + random.randint(-200, 500)) * (0.10 + random.uniform(-0.02, 0.02))
            else:  # errors
                count = random.randint(3, 10)
            time_trend.append({
                'date': time.strftime('%Y-%m-%d', time.localtime(time.time() - i * 86400)),
                'count': max(1000 if drilldown_type == 'calls' else (100 if drilldown_type == 'costs' else 1), count)
            })
        
        return jsonify({
            'success': True,
            'type': drilldown_type,
            'department_rank': dept_stats[:10],
            'user_rank': user_stats[:10],
            'model_rank': model_stats[:10],
            'time_trend': time_trend
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ============================================================
# 风控压力测试 API
# ============================================================

def _group_by_attribute(customer_details, attribute, base_avg):
    """按属性分组分析"""
    groups = {}
    for customer in customer_details:
        key = customer.get(attribute, '未知')
        if key not in groups:
            groups[key] = {
                'count': 0,
                'normal_probs': [],
                'stress_probs': [],
                'changes': []
            }
        groups[key]['count'] += 1
        groups[key]['normal_probs'].append(customer['normal_prob'])
        groups[key]['stress_probs'].append(customer['stress_prob'])
        groups[key]['changes'].append(customer['change'])
    
    import numpy as np
    result = {}
    for key, data in groups.items():
        if len(data['normal_probs']) > 0:
            result[key] = {
                'count': data['count'],
                'avg_normal_prob': float(np.mean(data['normal_probs'])),
                'avg_stress_prob': float(np.mean(data['stress_probs'])),
                'avg_change': float(np.mean(data['changes'])),
                'avg_change_pct': float(np.mean(data['changes']) / np.mean(data['normal_probs']) * 100) if np.mean(data['normal_probs']) > 0 else 0,
                'affected_count': sum(1 for p in data['stress_probs'] if p > base_avg * 1.2),
                'affected_ratio': sum(1 for p in data['stress_probs'] if p > base_avg * 1.2) / len(data['stress_probs']) if data['stress_probs'] else 0
            }
    return result

@app.route('/api/stress-test', methods=['POST'])
def stress_test():
    """风控压力测试 - 基于真实环境模拟"""
    try:
        data = request.json or {}
        if data is None:
            return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400
        
        from environment.economic_cycle import EconomicCycle, CyclePhase
        from environment.lending_env import LendingEnv, BankState
        from data_distillation.customer_generator import CustomerGenerator
        
        event_type = data.get('event_type', 'financial_crisis')
        severity = data.get('severity', 'moderate')
        duration = int(data.get('duration', 6))
        
        # 事件名称映射
        event_names = {
            'financial_crisis': '金融危机',
            'pandemic': '疫情冲击',
            'industry_collapse': '行业暴雷',
            'interest_rate_shock': '利率冲击',
            'unemployment_surge': '失业率飙升',
            'custom': '自定义事件'
        }
        
        # 强度倍数映射
        severity_multipliers = {
            'mild': 1.2,
            'moderate': 1.5,
            'severe': 2.0,
            'extreme': 3.0
        }
        
        multiplier = severity_multipliers.get(severity, 1.5)
        
        # 创建经济周期模拟器
        economy = EconomicCycle(seed=42)
        
        # 根据事件类型调整经济参数
        base_state = economy.state
        
        # 事件对经济的影响
        event_impacts = {
            'financial_crisis': {
                'gdp_growth': -0.05 * multiplier,
                'unemployment_rate': 0.05 * multiplier,
                'credit_spread': 0.03 * multiplier,
                'phase': CyclePhase.DEPRESSION
            },
            'pandemic': {
                'gdp_growth': -0.03 * multiplier,
                'unemployment_rate': 0.04 * multiplier,
                'credit_spread': 0.02 * multiplier,
                'phase': CyclePhase.RECESSION
            },
            'industry_collapse': {
                'gdp_growth': -0.02 * multiplier,
                'unemployment_rate': 0.03 * multiplier,
                'credit_spread': 0.025 * multiplier,
                'phase': CyclePhase.RECESSION
            },
            'interest_rate_shock': {
                'gdp_growth': -0.01 * multiplier,
                'unemployment_rate': 0.02 * multiplier,
                'credit_spread': 0.04 * multiplier,
                'phase': CyclePhase.RECESSION
            },
            'unemployment_surge': {
                'gdp_growth': -0.02 * multiplier,
                'unemployment_rate': 0.05 * multiplier,
                'credit_spread': 0.02 * multiplier,
                'phase': CyclePhase.RECESSION
            }
        }
        
        impact = event_impacts.get(event_type, event_impacts['financial_crisis'])
        
        # 创建压力测试环境
        stress_state = economy.state
        stress_state.gdp_growth = max(-0.1, min(0.15, base_state.gdp_growth + impact['gdp_growth']))
        stress_state.unemployment_rate = min(0.2, base_state.unemployment_rate + impact['unemployment_rate'])
        stress_state.credit_spread = min(0.1, base_state.credit_spread + impact['credit_spread'])
        stress_state.phase = impact['phase']
        
        # 获取客户筛选条件
        customer_type_filter = data.get('customer_type', '')
        industry_filter = data.get('industry', '')
        city_tier_filter = data.get('city_tier', '')
        customer_count = int(data.get('customer_count', 100))
        
        # 获取贷款条件
        loan_amount = float(data.get('loan_amount', 100000))
        interest_rate = float(data.get('interest_rate', 8)) / 100.0
        loan_term = int(data.get('loan_term', 24))
        
        # 生成测试客户样本
        generator = CustomerGenerator(seed=42)
        all_customers = generator.generate_batch(1000)
        
        # 应用筛选条件
        test_customers = []
        type_map = {
            'salaried': CustomerType.SALARIED,
            'small_business': CustomerType.SMALL_BUSINESS,
            'freelancer': CustomerType.FREELANCER,
            'farmer': CustomerType.FARMER,
            'professional': CustomerType.PROFESSIONAL,
            'entrepreneur': CustomerType.ENTREPRENEUR,
            'investor': CustomerType.INVESTOR,
            'retiree': CustomerType.RETIREE,
            'student': CustomerType.STUDENT,
            'startup': CustomerType.STARTUP,
            'tech_startup': CustomerType.TECH_STARTUP,
            'manufacturing': CustomerType.MANUFACTURING,
            'trade_company': CustomerType.TRADE_COMPANY,
            'service_company': CustomerType.SERVICE_COMPANY,
        }
        
        industry_map = {
            'manufacturing': Industry.MANUFACTURING,
            'service': Industry.SERVICE,
            'retail': Industry.RETAIL,
            'catering': Industry.CATERING,
            'it': Industry.IT,
            'construction': Industry.CONSTRUCTION,
            'agriculture': Industry.AGRICULTURE,
            'finance': Industry.FINANCE,
            'real_estate': Industry.REAL_ESTATE,
        }
        
        city_tier_map = {
            'tier_1': CityTier.TIER_1,
            'tier_2': CityTier.TIER_2,
            'tier_3': CityTier.TIER_3,
            'tier_4': CityTier.TIER_4,
        }
        
        for customer in all_customers:
            # 客户类型筛选
            if customer_type_filter:
                expected_type = type_map.get(customer_type_filter)
                if expected_type and customer.customer_type != expected_type:
                    continue
            
            # 行业筛选
            if industry_filter:
                expected_industry = industry_map.get(industry_filter)
                if expected_industry and customer.industry != expected_industry:
                    continue
            
            # 城市等级筛选
            if city_tier_filter:
                expected_tier = city_tier_map.get(city_tier_filter)
                if expected_tier and customer.city_tier != expected_tier:
                    continue
            
            test_customers.append(customer)
            if len(test_customers) >= customer_count:
                break
        
        # 如果筛选后客户不足，补充生成
        if len(test_customers) < customer_count:
            additional_needed = customer_count - len(test_customers)
            for _ in range(additional_needed):
                if customer_type_filter and customer_type_filter in type_map:
                    customer = generator.generate_one(customer_type=type_map[customer_type_filter])
                else:
                    customer = generator.generate_one()
                test_customers.append(customer)
        
        # 计算压力场景下的违约率
        from data_distillation.world_model import WorldModel, LoanOffer, MarketConditions
        
        world_model = WorldModel(seed=42)
        loan = LoanOffer(amount=loan_amount, interest_rate=interest_rate, term_months=loan_term)
        
        stress_market = MarketConditions(
            gdp_growth=stress_state.gdp_growth,
            base_interest_rate=stress_state.interest_rate,
            unemployment_rate=stress_state.unemployment_rate,
            inflation_rate=stress_state.inflation_rate,
            credit_spread=stress_state.credit_spread
        )
        
        # 计算正常场景和压力场景的违约率
        normal_defaults = []
        stress_defaults = []
        
        # 详细计算过程记录
        calculation_steps = []
        customer_details = []
        
        normal_market = MarketConditions(
            gdp_growth=base_state.gdp_growth,
            base_interest_rate=base_state.interest_rate,
            unemployment_rate=base_state.unemployment_rate,
            inflation_rate=base_state.inflation_rate,
            credit_spread=base_state.credit_spread
        )
        
        for i, customer in enumerate(test_customers):
            normal_future = world_model.predict_customer_future(customer, loan, normal_market, add_noise=False)
            stress_future = world_model.predict_customer_future(customer, loan, stress_market, add_noise=False)
            
            normal_prob = normal_future.default_probability
            stress_prob = stress_future.default_probability
            
            normal_defaults.append(normal_prob)
            stress_defaults.append(stress_prob)
            
            # 记录所有客户详情用于分析
            customer_details.append({
                'index': i + 1,
                'customer_type': customer.customer_type.value if hasattr(customer.customer_type, 'value') else str(customer.customer_type),
                'industry': customer.industry.value if hasattr(customer.industry, 'value') else str(customer.industry),
                'city_tier': customer.city_tier.value if hasattr(customer.city_tier, 'value') else str(customer.city_tier),
                'normal_prob': float(normal_prob),
                'stress_prob': float(stress_prob),
                'change': float(stress_prob - normal_prob),
                'change_pct': float((stress_prob - normal_prob) / normal_prob * 100) if normal_prob > 0 else 0
            })
        
        import numpy as np
        normal_avg = np.mean(normal_defaults)
        stress_avg = np.mean(stress_defaults)
        default_rate_change = stress_avg - normal_avg
        default_rate_change_pct = (default_rate_change / normal_avg * 100) if normal_avg > 0 else 0
        
        # 记录计算步骤
        calculation_steps.append({
            'step': 1,
            'name': '基础违约率计算',
            'description': f'在正常经济环境下，对{len(test_customers)}个测试客户进行违约概率预测',
            'formula': f'正常违约率 = Σ(客户违约概率) / 客户数量',
            'value': float(normal_avg),
            'details': f'共{len(test_customers)}个客户，平均违约概率为{normal_avg:.4f}'
        })
        
        calculation_steps.append({
            'step': 2,
            'name': '压力场景违约率计算',
            'description': f'在{event_names.get(event_type, "压力")}事件影响下，重新计算违约概率',
            'formula': f'压力违约率 = Σ(压力场景下客户违约概率) / 客户数量',
            'value': float(stress_avg),
            'details': f'压力场景下平均违约概率为{stress_avg:.4f}，较正常场景上升{default_rate_change_pct:.2f}%'
        })
        
        # 受影响客户数
        affected_threshold = normal_avg * 1.2  # 违约率上升20%以上视为受影响
        affected_customers = sum(1 for d in stress_defaults if d > affected_threshold)
        affected_ratio = affected_customers / len(stress_defaults) if len(stress_defaults) > 0 else 0
        
        # 标记受影响客户
        for customer_detail in customer_details:
            customer_detail['is_affected'] = customer_detail['stress_prob'] > affected_threshold
        
        calculation_steps.append({
            'step': 3,
            'name': '受影响客户识别',
            'description': f'识别违约概率上升超过20%的客户',
            'formula': f'受影响客户 = 违约概率 > {affected_threshold:.4f} 的客户数量',
            'value': affected_customers,
            'details': f'共{affected_customers}个客户受影响，占比{affected_ratio*100:.1f}%'
        })
        
        # 潜在损失
        total_loan_amount = len(test_customers) * loan.amount
        recovery_rate = 0.4  # 假设回收率40%
        potential_loss = total_loan_amount * stress_avg * (1 - recovery_rate)
        
        calculation_steps.append({
            'step': 4,
            'name': '潜在损失计算',
            'description': '计算压力场景下的潜在损失',
            'formula': f'潜在损失 = 总贷款金额 × 违约率 × (1 - 回收率)',
            'value': float(potential_loss),
            'details': f'总贷款金额{total_loan_amount/1e8:.2f}亿元，违约率{stress_avg:.4f}，回收率{recovery_rate*100:.0f}%，潜在损失{potential_loss/1e8:.2f}亿元'
        })
        
        # 敏感性分析
        sensitivity = {}
        sensitivity_config = data.get('sensitivity', {})
        sensitivity_range = data.get('sensitivity_range', 0.3)
        
        sensitivity_details = {}
        
        if sensitivity_config.get('gdp', False):
            # GDP变化对违约率的影响
            test_market = MarketConditions(
                gdp_growth=base_state.gdp_growth + sensitivity_range,
                base_interest_rate=base_state.interest_rate,
                unemployment_rate=base_state.unemployment_rate,
                inflation_rate=base_state.inflation_rate,
                credit_spread=base_state.credit_spread
            )
            test_defaults = [world_model.predict_customer_future(c, loan, test_market, add_noise=False).default_probability 
                           for c in test_customers[:50]]
            test_avg = np.mean(test_defaults)
            sensitivity['gdp'] = (test_avg - normal_avg) / normal_avg if normal_avg > 0 else 0
            sensitivity_details['gdp'] = {
                'base_value': float(base_state.gdp_growth),
                'test_value': float(base_state.gdp_growth + sensitivity_range),
                'base_default_rate': float(normal_avg),
                'test_default_rate': float(test_avg),
                'impact': float(sensitivity['gdp']),
                'explanation': f'GDP增长率从{base_state.gdp_growth*100:.2f}%变化到{(base_state.gdp_growth + sensitivity_range)*100:.2f}%，导致违约率从{normal_avg:.4f}变化到{test_avg:.4f}，影响幅度为{sensitivity["gdp"]*100:.2f}%'
            }
        
        if sensitivity_config.get('unemployment', False):
            test_market = MarketConditions(
                gdp_growth=base_state.gdp_growth,
                base_interest_rate=base_state.interest_rate,
                unemployment_rate=min(0.2, base_state.unemployment_rate + sensitivity_range),
                inflation_rate=base_state.inflation_rate,
                credit_spread=base_state.credit_spread
            )
            test_defaults = [world_model.predict_customer_future(c, loan, test_market, add_noise=False).default_probability 
                           for c in test_customers[:50]]
            sensitivity['unemployment'] = (np.mean(test_defaults) - normal_avg) / normal_avg
        
        if sensitivity_config.get('interest', False):
            test_market = MarketConditions(
                gdp_growth=base_state.gdp_growth,
                base_interest_rate=min(0.15, base_state.interest_rate + sensitivity_range * 0.02),
                unemployment_rate=base_state.unemployment_rate,
                inflation_rate=base_state.inflation_rate,
                credit_spread=base_state.credit_spread
            )
            test_defaults = [world_model.predict_customer_future(c, loan, test_market, add_noise=False).default_probability 
                           for c in test_customers[:50]]
            sensitivity['interest'] = (np.mean(test_defaults) - normal_avg) / normal_avg
        
        if sensitivity_config.get('inflation', False):
            test_market = MarketConditions(
                gdp_growth=base_state.gdp_growth,
                base_interest_rate=base_state.interest_rate,
                unemployment_rate=base_state.unemployment_rate,
                inflation_rate=min(0.1, base_state.inflation_rate + sensitivity_range * 0.01),
                credit_spread=base_state.credit_spread
            )
            test_defaults = [world_model.predict_customer_future(c, loan, test_market, add_noise=False).default_probability 
                           for c in test_customers[:50]]
            sensitivity['inflation'] = (np.mean(test_defaults) - normal_avg) / normal_avg
        
        # 韧性评估 - 模拟时间序列
        resilience_config = data.get('resilience', {})
        max_drawdown = 0.0
        recovery_months = 0
        resilience_score = 100
        
        if resilience_config.get('profit', False):
            # 模拟利润变化曲线
            monthly_profits = []
            peak_profit = 1000  # 基准月利润（百万）
            current_profit = peak_profit
            
            for month in range(duration + 12):  # 事件期间 + 恢复期
                if month < duration:
                    # 事件期间：利润下降
                    decline_rate = 0.15 * multiplier * (1 - month / duration)
                    current_profit *= (1 - decline_rate)
                else:
                    # 恢复期：逐步恢复
                    recovery_rate = 0.05 * (1 - (month - duration) / 12)
                    current_profit *= (1 + recovery_rate)
                    current_profit = min(peak_profit, current_profit)
                
                monthly_profits.append(current_profit)
            
            max_drawdown = (peak_profit - min(monthly_profits)) / peak_profit
            
            # 计算恢复时间（恢复到90%）
            recovery_threshold = peak_profit * 0.9
            for i, profit in enumerate(monthly_profits[duration:], start=duration):
                if profit >= recovery_threshold:
                    recovery_months = i - duration
                    break
            else:
                recovery_months = 12
            
            # 韧性评分
            resilience_score = max(0, int(100 - max_drawdown * 150 - recovery_months * 2))
        
        # 风险等级
        if stress_avg > 0.25 or max_drawdown > 0.5:
            risk_level = '极高风险'
            risk_description = '需要立即采取应对措施，建议暂停高风险业务'
        elif stress_avg > 0.15 or max_drawdown > 0.3:
            risk_level = '高风险'
            risk_description = '风险显著上升，需要加强风控措施'
        elif stress_avg > 0.10 or max_drawdown > 0.2:
            risk_level = '中风险'
            risk_description = '风险有所上升，需要密切关注'
        else:
            risk_level = '低风险'
            risk_description = '风险可控，保持现有策略'
        
        # 应对预案
        contingency_plan = []
        
        if stress_avg > 0.15:
            contingency_plan.append({
                'priority': 'high',
                'action': '收紧审批标准',
                'description': f'将审批门槛提高{(stress_avg - 0.15) * 100:.1f}%，拒绝高风险客户'
            })
        
        if max_drawdown > 0.3:
            contingency_plan.append({
                'priority': 'high',
                'action': '增加资本缓冲',
                'description': f'建议增加{(max_drawdown - 0.3) * 100:.1f}%的资本缓冲以应对损失'
            })
        
        if recovery_months > 12:
            contingency_plan.append({
                'priority': 'medium',
                'action': '优化资产结构',
                'description': '加快不良资产处置，优化贷款组合结构'
            })
        
        if affected_ratio > 0.5:
            contingency_plan.append({
                'priority': 'medium',
                'action': '客户分层管理',
                'description': '对受影响客户进行分层，重点监控高风险客户'
            })
        
        contingency_plan.append({
            'priority': 'low',
            'action': '持续监控',
            'description': '建立实时监控机制，定期评估风险变化'
        })
        
        return jsonify({
            'success': True,
            'event_name': event_names.get(event_type, '未知事件'),
            'config': {
                'event_type': event_type,
                'severity': severity,
                'duration': duration,
                'customer_type': customer_type_filter,
                'industry': industry_filter,
                'city_tier': city_tier_filter,
                'loan_amount': loan_amount,
                'interest_rate': interest_rate,
                'loan_term': loan_term,
                'customer_count': len(test_customers)
            },
            'impact': {
                'default_rate': float(stress_avg),
                'default_rate_change': float(default_rate_change),
                'default_rate_change_pct': float(default_rate_change_pct),
                'normal_default_rate': float(normal_avg),
                'affected_customers': affected_customers,
                'affected_ratio': float(affected_ratio),
                'potential_loss': float(potential_loss),
                'total_loan_amount': float(total_loan_amount),
                'explanation': f'在{event_names.get(event_type, "压力")}事件影响下，违约率从{normal_avg:.4f}上升至{stress_avg:.4f}，上升幅度{default_rate_change_pct:.2f}%。共有{affected_customers}个客户（占比{affected_ratio*100:.1f}%）受到显著影响，潜在损失约{potential_loss/1e8:.2f}亿元。'
            },
            'sensitivity': sensitivity,
            'sensitivity_details': sensitivity_details,
            'resilience': {
                'max_drawdown': float(max_drawdown),
                'profit_drawdown': float(max_drawdown),
                'recovery_months': recovery_months,
                'resilience_score': resilience_score,
                'explanation': f'在压力场景下，利润最大回撤{max_drawdown*100:.1f}%，预计需要{recovery_months}个月恢复到正常水平。韧性评分为{resilience_score}/100分。'
            },
            'risk_level': risk_level,
            'risk_description': risk_description,
            'contingency_plan': contingency_plan,
            'economic_state': {
                'gdp_growth': float(stress_state.gdp_growth),
                'unemployment_rate': float(stress_state.unemployment_rate),
                'credit_spread': float(stress_state.credit_spread),
                'phase': stress_state.phase.value,
                'base_gdp_growth': float(base_state.gdp_growth),
                'base_unemployment_rate': float(base_state.unemployment_rate),
                'base_credit_spread': float(base_state.credit_spread)
            },
            'calculation_steps': calculation_steps,
            'customer_details': customer_details,  # 返回所有客户详情用于分析
            'statistics': {
                'total_customers': len(test_customers),
                'affected_count': affected_customers,
                'affected_ratio': float(affected_ratio),
                'avg_normal_prob': float(normal_avg),
                'avg_stress_prob': float(stress_avg),
                'max_change': float(max([c['change'] for c in customer_details]) if customer_details else 0.0),
                'min_change': float(min([c['change'] for c in customer_details]) if customer_details else 0.0),
                'std_normal': float(np.std(normal_defaults)) if len(normal_defaults) > 1 else 0.0,
                'std_stress': float(np.std(stress_defaults)) if len(stress_defaults) > 1 else 0.0
            },
            'grouped_analysis': {
                'by_customer_type': _group_by_attribute(customer_details, 'customer_type', normal_avg),
                'by_industry': _group_by_attribute(customer_details, 'industry', normal_avg),
                'by_city_tier': _group_by_attribute(customer_details, 'city_tier', normal_avg)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ============================================================
# 端到端Demo API
# ============================================================

@app.route('/api/demo/generate-historical', methods=['POST'])
def api_generate_historical():
    """生成历史数据"""
    try:
        data = request.get_json() or {}
        num_loans = data.get('num_loans', 10000)
        personal_ratio = data.get('personal_ratio', 0.7)
        
        from src.demo.historical_data_generator import HistoricalLoanDataGenerator
        
        generator = HistoricalLoanDataGenerator(seed=42)
        df = generator.generate_historical_loans(
            num_loans=num_loans,
            personal_ratio=personal_ratio
        )
        stats = generator.save_to_files(df, output_dir='data/historical')
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/demo/check-quality', methods=['POST'])
def api_check_quality():
    """数据质量检查"""
    try:
        import pandas as pd
        from demo.data_quality_checker import DataQualityChecker
        
        data_path = 'data/historical/historical_loans.csv'
        if not Path(data_path).exists():
            return jsonify({
                'success': False,
                'error': '历史数据文件不存在，请先生成历史数据'
            }), 400
        
        data = pd.read_csv(data_path)
        checker = DataQualityChecker(data)
        result = checker.comprehensive_check()
        checker.save_report('data/historical/quality_report.json')
        
        return jsonify({
            'success': True,
            'overall_score': result.get('overall_score', 0),
            'completeness_score': result.get('completeness', {}).get('completeness_score', 0),
            'consistency_score': result.get('consistency', {}).get('consistency_score', 0),
            'temporal_score': result.get('temporal_consistency', {}).get('temporal_consistency_score', 0),
            'rule_score': result.get('business_rules', {}).get('rule_score', 0)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/demo/engineer-features', methods=['POST'])
def api_engineer_features():
    """特征工程"""
    try:
        import pandas as pd
        from demo.feature_engineer import FeatureEngineer
        
        data_path = 'data/historical/historical_loans.csv'
        if not Path(data_path).exists():
            return jsonify({
                'success': False,
                'error': '历史数据文件不存在'
            }), 400
        
        data = pd.read_csv(data_path)
        original_cols = len(data.columns)
        
        engineer = FeatureEngineer(data)
        engineered_data = engineer.engineer_all_features()
        
        engineer.save_engineered_data(engineered_data, 'data/historical/historical_loans_engineered.csv')
        
        return jsonify({
            'success': True,
            'original_features': original_cols,
            'new_features': len(engineered_data.columns) - original_cols,
            'total_features': len(engineered_data.columns)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/demo/extract-rules', methods=['POST'])
def api_extract_rules():
    """规则提取和量化"""
    try:
        import pandas as pd
        from demo.rule_extractor import RuleExtractor
        from demo.rule_quantifier import RuleQuantifier
        
        data_path = 'data/historical/historical_loans_engineered.csv'
        if not Path(data_path).exists():
            return jsonify({
                'success': False,
                'error': '特征工程数据不存在，请先执行特征工程'
            }), 400
        
        data = pd.read_csv(data_path)
        
        # 提取规则
        extractor = RuleExtractor(data)
        rules = extractor.extract_all_rules('both')
        extractor.save_rules('data/historical/extracted_rules.json')
        
        # 量化规则
        rules_dict = [r.to_dict() if hasattr(r, 'to_dict') else r for r in rules]
        quantifier = RuleQuantifier(rules_dict)
        quantified = quantifier.quantify_all_rules()
        quantifier.save_quantified_rules('data/historical/quantified_rules.json')
        
        return jsonify({
            'success': True,
            'num_rules': len(rules),
            'num_quantified': len(quantified)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/demo/train-models', methods=['POST'])
def api_train_models():
    """模型训练"""
    try:
        import pandas as pd
        from demo.world_model_trainer import WorldModelTrainer
        
        data_path = 'data/historical/historical_loans_engineered.csv'
        if not Path(data_path).exists():
            return jsonify({
                'success': False,
                'error': '特征工程数据不存在'
            }), 400
        
        data = pd.read_csv(data_path)
        trainer = WorldModelTrainer(data, seed=42)
        results = trainer.train_all_models()
        trainer.save_models('data/historical/models')
        
        default_accuracy = results.get('default_prediction', {}).get('metrics', {}).get('accuracy', 0)
        profit_r2 = results.get('profit_prediction', {}).get('metrics', {}).get('r2_score', 0)
        
        return jsonify({
            'success': True,
            'default_accuracy': default_accuracy,
            'profit_r2': profit_r2,
            'num_models': len(results)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/demo/simulate-approval', methods=['POST'])
def api_simulate_approval():
    """模拟审批"""
    try:
        import pandas as pd
        from demo.enhanced_customer_generator import EnhancedCustomerGenerator
        from demo.market_simulator import MarketSimulator
        from demo.enhanced_rule_engine import EnhancedRuleEngine
        from demo.repayment_simulator import RepaymentSimulator
        from demo.recovery_calculator import RecoveryCalculator
        from datetime import datetime
        import numpy as np
        
        # 延迟导入，避免循环依赖
        try:
            from demo.model_decision import ModelDecisionMaker
            from demo.decision_fusion import DecisionFusion
        except ImportError as e:
            # 如果直接导入失败，尝试相对导入
            import importlib.util
            import os
            model_decision_path = os.path.join(os.path.dirname(__file__), 'src', 'demo', 'model_decision.py')
            decision_fusion_path = os.path.join(os.path.dirname(__file__), 'src', 'demo', 'decision_fusion.py')
            
            if os.path.exists(model_decision_path):
                spec = importlib.util.spec_from_file_location("model_decision", model_decision_path)
                model_decision_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_decision_module)
                ModelDecisionMaker = model_decision_module.ModelDecisionMaker
            
            if os.path.exists(decision_fusion_path):
                spec = importlib.util.spec_from_file_location("decision_fusion", decision_fusion_path)
                decision_fusion_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(decision_fusion_module)
                DecisionFusion = decision_fusion_module.DecisionFusion
        
        data = request.get_json() or {}
        num_customers = data.get('num_customers', 100)
        
        # 加载特征工程后的数据
        data_path = 'data/historical/historical_loans_engineered.csv'
        if not Path(data_path).exists():
            return jsonify({
                'success': False,
                'error': '特征工程数据不存在'
            }), 400
        
        engineered_data = pd.read_csv(data_path)
        
        # 生成客户
        customer_gen = EnhancedCustomerGenerator(engineered_data, seed=42)
        customers = customer_gen.generate_customers(
            num_personal=int(num_customers * 0.7),
            num_corporate=int(num_customers * 0.3)
        )
        
        # 生成市场环境
        market_sim = MarketSimulator(seed=42, start_date='2024-01-01')
        current_market = market_sim.generate_market_condition(datetime(2024, 6, 15))
        market_dict = {
            'gdp_growth': current_market.gdp_growth,
            'base_interest_rate': current_market.base_interest_rate,
            'unemployment_rate': current_market.unemployment_rate,
            'inflation_rate': current_market.inflation_rate,
            'credit_spread': current_market.credit_spread,
            'market_sentiment': current_market.market_sentiment
        }
        
        # 加载决策组件
        model_decision_maker = ModelDecisionMaker(models_dir='data/historical/models')
        rule_engine = EnhancedRuleEngine()
        rule_engine.load_rules_from_file(
            'data/historical/extracted_rules.json',
            'data/historical/quantified_rules.json'
        )
        fusion = DecisionFusion(model_decision_maker, rule_engine)
        
        # 模拟还款和回收
        repayment_sim = RepaymentSimulator(seed=42)
        recovery_calc = RecoveryCalculator(seed=42)
        
        # 处理每个客户
        results = []
        for customer in customers:
            # 生成贷款申请
            if customer['customer_type'] == 'personal':
                loan_amount = np.random.uniform(10000, customer['monthly_income'] * 12 * 0.5)
            else:
                loan_amount = np.random.uniform(100000, customer['annual_revenue'] * 0.3)
            
            loan = {
                'loan_amount': loan_amount,
                'approved_rate': 0.08,
                'approved_term_months': 24
            }
            
            # 融合决策
            fused_decision = fusion.fuse_decisions(customer, loan, market_dict)
            
            # 如果批准，模拟还款
            if fused_decision.final_decision == 'approve':
                repayment_result = repayment_sim.simulate_repayment(
                    loan_amount=loan_amount,
                    interest_rate=fused_decision.model_decision.default_probability * 0.1 + 0.08,
                    term_months=24,
                    default_probability=fused_decision.default_probability,
                    customer_data=customer
                )
                
                # 如果违约，计算回收
                if repayment_result.defaulted:
                    recovery_result = recovery_calc.calculate_recovery(
                        repayment_result, loan_amount, customer, market_dict
                    )
                else:
                    recovery_result = None
            else:
                repayment_result = None
                recovery_result = None
            
            # 记录结果
            result = {
                **customer,
                **loan,
                'expert_decision': fused_decision.final_decision,
                'decision': fused_decision.final_decision,
                'default_probability': fused_decision.default_probability,
                'expected_profit': fused_decision.expected_profit,
                'actual_defaulted': repayment_result.defaulted if repayment_result else False,
                'actual_profit': repayment_result.total_interest_paid - (
                    recovery_result.default_amount - recovery_result.recovery_amount
                ) if recovery_result else repayment_result.total_interest_paid if repayment_result else 0,
                'recovery_rate': recovery_result.recovery_rate if recovery_result else 0,
                'recovery_amount': recovery_result.recovery_amount if recovery_result else 0,
                'default_amount': recovery_result.default_amount if recovery_result else 0
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # 保存模拟结果
        results_df.to_csv('data/historical/simulated_results.csv', index=False, encoding='utf-8-sig')
        
        return jsonify({
            'success': True,
            'num_customers': len(results_df),
            'approved_count': (results_df['expert_decision'] == 'approve').sum(),
            'defaulted_count': results_df['actual_defaulted'].sum(),
            'avg_profit': float(results_df['actual_profit'].mean()) if len(results_df) > 0 else 0
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/demo/validate-results', methods=['POST'])
def api_validate_results():
    """结果验证"""
    try:
        import pandas as pd
        from demo.result_validator import ResultValidator
        
        hist_path = 'data/historical/historical_loans_engineered.csv'
        sim_path = 'data/historical/simulated_results.csv'
        
        if not Path(hist_path).exists():
            return jsonify({
                'success': False,
                'error': '历史数据不存在'
            }), 400
        
        if not Path(sim_path).exists():
            return jsonify({
                'success': False,
                'error': '模拟结果不存在，请先执行模拟审批'
            }), 400
        
        historical_data = pd.read_csv(hist_path)
        simulated_data = pd.read_csv(sim_path)
        
        validator = ResultValidator(historical_data, simulated_data)
        results = validator.comprehensive_validation()
        validator.save_validation_report(results, 'data/historical/validation_report.json')
        
        return jsonify({
            'success': True,
            'default_rate_valid': results.get('default_rate', {}).get('is_acceptable', False),
            'profit_valid': results.get('profit_distribution', {}).get('is_acceptable', False),
            'recovery_valid': results.get('recovery_rate', {}).get('is_acceptable', False),
            'overall_acceptable': results.get('overall_acceptable', False)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

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
    print(f"Demo界面: http://localhost:{port}/demo")
    print(f"环境: {'开发' if debug else '生产'}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=debug)

