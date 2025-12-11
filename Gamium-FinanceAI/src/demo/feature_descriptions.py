"""
特征说明字典
为每个特征提供详细的中文说明
"""

FEATURE_DESCRIPTIONS = {
    # 衍生特征
    'loan_to_annual_income_ratio': {
        'name': '贷款年收入比',
        'description': '贷款金额与年收入的比值，衡量客户的还款能力。值越小表示还款压力越小。',
        'category': '衍生特征',
        'calculation': '贷款金额 / (月收入 × 12)'
    },
    'monthly_payment_to_income_ratio': {
        'name': '月供收入比',
        'description': '月还款额与月收入的比值，反映客户每月还款压力。通常建议不超过50%。',
        'category': '衍生特征',
        'calculation': '月还款额 / 月收入'
    },
    'debt_ratio': {
        'name': '负债率',
        'description': '总负债与总资产的比值，衡量客户的财务杠杆水平。值越高风险越大。',
        'category': '衍生特征',
        'calculation': '总负债 / 总资产'
    },
    'debt_to_asset_ratio': {
        'name': '资产负债率',
        'description': '企业总负债与总资产的比值，反映企业财务风险。通常建议不超过70%。',
        'category': '衍生特征',
        'calculation': '总负债 / 总资产'
    },
    'current_ratio': {
        'name': '流动比率',
        'description': '流动资产与流动负债的比值，衡量企业短期偿债能力。通常建议大于1.5。',
        'category': '衍生特征',
        'calculation': '流动资产 / 流动负债'
    },
    'loan_to_revenue_ratio': {
        'name': '贷款营收比',
        'description': '贷款金额与年营收的比值，衡量企业还款能力。值越小表示还款压力越小。',
        'category': '衍生特征',
        'calculation': '贷款金额 / 年营收'
    },
    'comprehensive_risk_score': {
        'name': '综合风险评分',
        'description': '综合考虑信用分、负债率、收入稳定性等因素的综合风险评分。值越高风险越大。',
        'category': '衍生特征',
        'calculation': '加权综合评分'
    },
    'debt_ratio_category': {
        'name': '负债率分类',
        'description': '将负债率分为低、中、高三个等级，用于风险分层。',
        'category': '衍生特征',
        'calculation': '基于负债率的分段'
    },
    'current_ratio_category': {
        'name': '流动比率分类',
        'description': '将流动比率分为健康、一般、危险三个等级。',
        'category': '衍生特征',
        'calculation': '基于流动比率的分段'
    },
    'revenue_size_category': {
        'name': '营收规模分类',
        'description': '将企业年营收分为小型、中型、大型三个等级。',
        'category': '衍生特征',
        'calculation': '基于年营收的分段'
    },
    'profit_category': {
        'name': '利润分类',
        'description': '将预期利润分为亏损、低利润、中利润、高利润四个等级。',
        'category': '衍生特征',
        'calculation': '基于预期利润的分段'
    },
    'roi_category': {
        'name': '投资回报率分类',
        'description': '将投资回报率分为低、中、高三个等级。',
        'category': '衍生特征',
        'calculation': '基于ROI的分段'
    },
    'credit_score_category': {
        'name': '信用分分类',
        'description': '将信用分分为很差、较差、一般、良好、优秀五个等级，用于风险分层。',
        'category': '衍生特征',
        'calculation': '基于信用分的分段（0-550:很差, 550-650:较差, 650-700:一般, 700-750:良好, 750-850:优秀）'
    },
    'annual_payment_to_revenue_': {
        'name': '年还款额营收比',
        'description': '年还款额与年营收的比值，衡量企业的还款压力。值越小表示还款压力越小。',
        'category': '衍生特征',
        'calculation': '年还款额 / 年营收'
    },
    
    # 时间特征
    'application_year': {
        'name': '申请年份',
        'description': '贷款申请的年份，用于分析不同年份的贷款表现差异。',
        'category': '时间特征',
        'calculation': '从申请日期提取年份'
    },
    'application_month': {
        'name': '申请月份',
        'description': '贷款申请的月份（1-12），用于分析季节性因素对贷款表现的影响。',
        'category': '时间特征',
        'calculation': '从申请日期提取月份'
    },
    'application_quarter': {
        'name': '申请季度',
        'description': '贷款申请的季度（1-4），用于分析季度性趋势。',
        'category': '时间特征',
        'calculation': '从申请日期计算季度'
    },
    'application_day_of_week': {
        'name': '申请星期',
        'description': '贷款申请是星期几（0-6），可能反映客户申请行为的规律性。',
        'category': '时间特征',
        'calculation': '从申请日期提取星期'
    },
    'application_is_month_start': {
        'name': '是否月初',
        'description': '申请日期是否在月初（前3天），可能反映客户资金需求的时间规律。',
        'category': '时间特征',
        'calculation': '判断日期是否在月初'
    },
    'application_is_month_end': {
        'name': '是否月末',
        'description': '申请日期是否在月末（后3天），可能反映客户资金需求的时间规律。',
        'category': '时间特征',
        'calculation': '判断日期是否在月末'
    },
    'application_is_quarter_end': {
        'name': '是否季末',
        'description': '申请日期是否在季度末，可能与企业财务周期相关。',
        'category': '时间特征',
        'calculation': '判断日期是否在季度末'
    },
    'application_is_weekend': {
        'name': '是否周末',
        'description': '申请日期是否在周末，可能反映客户申请渠道的偏好。',
        'category': '时间特征',
        'calculation': '判断日期是否在周末'
    },
    'approval_delay_days': {
        'name': '审批延迟天数',
        'description': '从申请到审批的天数，可能反映审批流程效率或风险审查的严格程度。',
        'category': '时间特征',
        'calculation': '审批日期 - 申请日期'
    },
    'days_to_default': {
        'name': '违约天数',
        'description': '从放款到违约的天数，用于分析违约发生的时间规律。',
        'category': '时间特征',
        'calculation': '违约日期 - 放款日期'
    },
    'default_month': {
        'name': '违约月份',
        'description': '发生违约的月份，用于分析违约的季节性特征。',
        'category': '时间特征',
        'calculation': '从违约日期提取月份'
    },
    'years_revenue_interaction': {
        'name': '经营年限与营收交互',
        'description': '经营年限与年营收的交互项，反映企业成熟度与规模的综合影响。',
        'category': '时间特征',
        'calculation': '经营年限 × 年营收'
    },
    
    # 交互特征
    'credit_debt_interaction': {
        'name': '信用分与负债率交互',
        'description': '信用分与负债率的交互项，反映信用状况与财务杠杆的综合影响。',
        'category': '交互特征',
        'calculation': '信用分 × 负债率'
    },
    'income_stability_interaction': {
        'name': '收入稳定性交互',
        'description': '收入水平与工作稳定性的交互项，反映收入质量的综合指标。',
        'category': '交互特征',
        'calculation': '月收入 × 工作年限'
    },
    'age_income_interaction': {
        'name': '年龄与收入交互',
        'description': '年龄与月收入的交互项，反映不同年龄段收入水平的差异。',
        'category': '交互特征',
        'calculation': '年龄 × 月收入'
    },
    'debt_liquidity_interaction': {
        'name': '负债与流动性交互',
        'description': '负债率与流动比率的交互项，反映企业财务结构的综合风险。',
        'category': '交互特征',
        'calculation': '负债率 × 流动比率'
    },
    'loan_interest_interaction': {
        'name': '贷款与利率交互',
        'description': '贷款金额与利率的交互项，反映贷款成本对风险的影响。',
        'category': '交互特征',
        'calculation': '贷款金额 × 利率'
    },
    'loan_term_interaction': {
        'name': '贷款与期限交互',
        'description': '贷款金额与期限的交互项，反映贷款规模与期限的综合影响。',
        'category': '交互特征',
        'calculation': '贷款金额 × 期限'
    },
    
    # 稳定性特征
    'job_stability': {
        'name': '工作稳定性',
        'description': '工作年限与年龄的比值，反映工作稳定性。值越高表示工作越稳定。',
        'category': '稳定性特征',
        'calculation': '工作年限 / (年龄 - 18)'
    },
    'business_maturity': {
        'name': '企业经营成熟度',
        'description': '经营年限与行业平均经营年限的比值，反映企业经营的成熟程度。',
        'category': '稳定性特征',
        'calculation': '经营年限 / 行业平均经营年限'
    },
    
    # 目标特征
    'is_approved': {
        'name': '是否审批通过',
        'description': '专家决策结果，1表示通过，0表示拒绝。用于模型训练的目标变量。',
        'category': '目标特征',
        'calculation': '专家决策编码'
    },
    'is_defaulted': {
        'name': '是否违约',
        'description': '实际违约情况，1表示违约，0表示正常还款。用于模型训练的目标变量。',
        'category': '目标特征',
        'calculation': '实际违约情况编码'
    }
}


def get_feature_description(feature_name: str) -> dict:
    """
    获取特征的详细说明
    
    Args:
        feature_name: 特征名称
    
    Returns:
        特征说明字典，如果不存在则返回默认说明
    """
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]
    
    # 根据特征名称推断说明
    desc = {
        'name': feature_name,
        'description': '特征工程生成的特征',
        'category': '其他',
        'calculation': 'N/A'
    }
    
    # 根据特征名称模式推断
    if 'ratio' in feature_name.lower():
        desc['description'] = '比率类特征，用于衡量两个指标之间的关系。'
        desc['category'] = '衍生特征'
    elif 'category' in feature_name.lower():
        desc['description'] = '分类特征，将连续变量转换为分类变量。'
        desc['category'] = '衍生特征'
    elif 'interaction' in feature_name.lower() or '_x_' in feature_name.lower():
        desc['description'] = '交互特征，反映两个或多个特征之间的交互作用。'
        desc['category'] = '交互特征'
    elif 'year' in feature_name.lower() or 'month' in feature_name.lower() or 'day' in feature_name.lower() or 'quarter' in feature_name.lower():
        desc['description'] = '时间特征，从日期中提取的时间相关信息。'
        desc['category'] = '时间特征'
    elif 'score' in feature_name.lower():
        desc['description'] = '评分特征，综合多个指标计算的风险或质量评分。'
        desc['category'] = '衍生特征'
    elif 'stability' in feature_name.lower() or 'maturity' in feature_name.lower():
        desc['description'] = '稳定性特征，反映客户或企业的稳定性指标。'
        desc['category'] = '稳定性特征'
    elif feature_name.startswith('is_'):
        desc['description'] = '布尔特征，表示某种状态是否存在。'
        desc['category'] = '目标特征' if feature_name in ['is_approved', 'is_defaulted'] else '衍生特征'
    
    return desc


def get_all_feature_descriptions(feature_list: list) -> dict:
    """
    获取所有特征的说明
    
    Args:
        feature_list: 特征名称列表
    
    Returns:
        特征名称到说明的字典
    """
    return {feat: get_feature_description(feat) for feat in feature_list}

