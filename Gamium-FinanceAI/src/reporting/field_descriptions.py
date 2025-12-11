"""
报表字段详细说明
提供每个报表字段的详细说明、监管要求、计算公式等信息
"""

FIELD_DESCRIPTIONS = {
    # G01 资本充足率报表字段
    'core_tier1_capital': {
        'description': '核心一级资本是指银行在持续经营条件下无条件用来吸收损失的资本工具。主要包括：实收资本或普通股、资本公积、盈余公积、一般风险准备、未分配利润、少数股东资本可计入部分等。',
        'unit': '元',
        'regulatory_requirement': '根据《商业银行资本管理办法（试行）》，核心一级资本充足率不得低于5%。核心一级资本是银行资本结构中最核心的部分，用于吸收银行经营中的损失。',
        'calculation_method': '核心一级资本 = 实收资本 + 资本公积 + 盈余公积 + 一般风险准备 + 未分配利润 + 少数股东资本可计入部分 - 核心一级资本扣减项',
        'data_source': '从银行资产负债表、资本构成明细表等财务系统中提取'
    },
    'other_tier1_capital': {
        'description': '其他一级资本是指除核心一级资本外的其他一级资本工具及其溢价。主要包括：其他一级资本工具及其溢价、少数股东资本可计入部分等。',
        'unit': '元',
        'regulatory_requirement': '其他一级资本与核心一级资本合计构成一级资本，一级资本充足率不得低于6%。',
        'calculation_method': '其他一级资本 = 其他一级资本工具及其溢价 + 少数股东资本可计入部分 - 其他一级资本扣减项',
        'data_source': '从银行资本构成明细表中提取'
    },
    'tier2_capital': {
        'description': '二级资本是指在破产清算条件下可以用来吸收损失的资本工具。主要包括：二级资本工具及其溢价、超额贷款损失准备等。',
        'unit': '元',
        'regulatory_requirement': '二级资本与一级资本合计构成总资本，资本充足率不得低于8%。',
        'calculation_method': '二级资本 = 二级资本工具及其溢价 + 超额贷款损失准备 - 二级资本扣减项',
        'data_source': '从银行资本构成明细表中提取'
    },
    'total_capital': {
        'description': '总资本是指银行可用于吸收损失的各类资本工具的总和，包括核心一级资本、其他一级资本和二级资本。',
        'unit': '元',
        'regulatory_requirement': '总资本用于计算资本充足率，资本充足率 = 总资本 / 风险加权资产，不得低于8%。',
        'calculation_method': '总资本 = 核心一级资本 + 其他一级资本 + 二级资本',
        'data_source': '通过公式计算得出'
    },
    'credit_risk_rwa': {
        'description': '信用风险加权资产是指银行因信用风险而需要计提资本的风险加权资产。包括表内信用风险加权资产和表外信用风险加权资产。',
        'unit': '元',
        'regulatory_requirement': '信用风险加权资产是计算资本充足率的分母，需要按照监管要求进行风险加权。',
        'calculation_method': '信用风险加权资产 = Σ(资产余额 × 风险权重)，风险权重根据资产类型、评级、担保情况等因素确定',
        'data_source': '从银行风险管理系统、资产分类系统中提取，按照《商业银行资本管理办法》规定的风险权重计算'
    },
    'market_risk_rwa': {
        'description': '市场风险加权资产是指银行因市场风险（利率风险、汇率风险、商品价格风险等）而需要计提资本的风险加权资产。',
        'unit': '元',
        'regulatory_requirement': '市场风险加权资产需要按照监管要求进行计量，通常使用内部模型法或标准法。',
        'calculation_method': '市场风险加权资产 = 市场风险资本要求 × 12.5（或使用内部模型法计算）',
        'data_source': '从银行市场风险管理系统、交易系统中提取'
    },
    'operational_risk_rwa': {
        'description': '操作风险加权资产是指银行因操作风险而需要计提资本的风险加权资产。操作风险包括内部欺诈、外部欺诈、系统故障、流程缺陷等。',
        'unit': '元',
        'regulatory_requirement': '操作风险加权资产需要按照监管要求进行计量，通常使用基本指标法、标准法或高级计量法。',
        'calculation_method': '操作风险加权资产 = 操作风险资本要求 × 12.5（或使用高级计量法计算）',
        'data_source': '从银行操作风险管理系统、损失数据库中提取'
    },
    'total_rwa': {
        'description': '风险加权资产合计是指银行所有风险加权资产的总和，包括信用风险、市场风险和操作风险加权资产。',
        'unit': '元',
        'regulatory_requirement': '风险加权资产合计是计算资本充足率的分母，资本充足率 = 总资本 / 风险加权资产合计，不得低于8%。',
        'calculation_method': '风险加权资产合计 = 信用风险加权资产 + 市场风险加权资产 + 操作风险加权资产',
        'data_source': '通过公式计算得出'
    },
    'capital_adequacy_ratio': {
        'description': '资本充足率是指银行总资本与风险加权资产的比率，反映银行资本对风险的覆盖程度。',
        'unit': '%',
        'regulatory_requirement': '根据《商业银行资本管理办法（试行）》，资本充足率不得低于8%，其中核心一级资本充足率不得低于5%，一级资本充足率不得低于6%。',
        'calculation_method': '资本充足率 = (总资本 / 风险加权资产合计) × 100%',
        'data_source': '通过公式计算得出'
    },
    'tier1_capital_ratio': {
        'description': '一级资本充足率是指银行一级资本（核心一级资本 + 其他一级资本）与风险加权资产的比率。',
        'unit': '%',
        'regulatory_requirement': '根据《商业银行资本管理办法（试行）》，一级资本充足率不得低于6%。',
        'calculation_method': '一级资本充足率 = ((核心一级资本 + 其他一级资本) / 风险加权资产合计) × 100%',
        'data_source': '通过公式计算得出'
    },
    'core_tier1_ratio': {
        'description': '核心一级资本充足率是指银行核心一级资本与风险加权资产的比率，是衡量银行资本充足性的核心指标。',
        'unit': '%',
        'regulatory_requirement': '根据《商业银行资本管理办法（试行）》，核心一级资本充足率不得低于5%。',
        'calculation_method': '核心一级资本充足率 = (核心一级资本 / 风险加权资产合计) × 100%',
        'data_source': '通过公式计算得出'
    },
    
    # G11 资产质量报表字段
    'normal_loans': {
        'description': '正常类贷款是指借款人能够履行合同，没有足够理由怀疑贷款本息不能按时足额偿还的贷款。',
        'unit': '元',
        'regulatory_requirement': '正常类贷款是银行资产质量良好的表现，正常类贷款占比越高，说明银行资产质量越好。',
        'calculation_method': '正常类贷款 = 所有正常类贷款的余额合计',
        'data_source': '从银行信贷管理系统、贷款分类系统中提取，按照《贷款风险分类指引》进行分类'
    },
    'special_mention_loans': {
        'description': '关注类贷款是指尽管借款人目前有能力偿还贷款本息，但存在一些可能对偿还产生不利影响因素的贷款。',
        'unit': '元',
        'regulatory_requirement': '关注类贷款虽然尚未逾期，但需要密切关注，及时采取措施防范风险。',
        'calculation_method': '关注类贷款 = 所有关注类贷款的余额合计',
        'data_source': '从银行信贷管理系统、贷款分类系统中提取'
    },
    'substandard_loans': {
        'description': '次级类贷款是指借款人的还款能力出现明显问题，完全依靠其正常营业收入无法足额偿还贷款本息，即使执行担保，也可能会造成一定损失的贷款。',
        'unit': '元',
        'regulatory_requirement': '次级类贷款属于不良贷款，需要计提专项准备，计提比例不低于25%。',
        'calculation_method': '次级类贷款 = 所有次级类贷款的余额合计',
        'data_source': '从银行信贷管理系统、不良贷款管理系统中提取'
    },
    'doubtful_loans': {
        'description': '可疑类贷款是指借款人无法足额偿还贷款本息，即使执行担保，也肯定要造成较大损失的贷款。',
        'unit': '元',
        'regulatory_requirement': '可疑类贷款属于不良贷款，需要计提专项准备，计提比例不低于50%。',
        'calculation_method': '可疑类贷款 = 所有可疑类贷款的余额合计',
        'data_source': '从银行信贷管理系统、不良贷款管理系统中提取'
    },
    'loss_loans': {
        'description': '损失类贷款是指在采取所有可能的措施或一切必要的法律程序之后，本息仍然无法收回，或只能收回极少部分的贷款。',
        'unit': '元',
        'regulatory_requirement': '损失类贷款属于不良贷款，需要计提专项准备，计提比例为100%。',
        'calculation_method': '损失类贷款 = 所有损失类贷款的余额合计',
        'data_source': '从银行信贷管理系统、不良贷款管理系统中提取'
    },
    'total_loans': {
        'description': '贷款总额是指银行所有贷款（包括正常类、关注类、次级类、可疑类、损失类）的余额总和。',
        'unit': '元',
        'regulatory_requirement': '贷款总额是计算不良贷款率的分母，不良贷款率 = 不良贷款 / 贷款总额，不得超过5%。',
        'calculation_method': '贷款总额 = 正常类贷款 + 关注类贷款 + 次级类贷款 + 可疑类贷款 + 损失类贷款',
        'data_source': '通过公式计算得出'
    },
    'npl_loans': {
        'description': '不良贷款是指次级类、可疑类和损失类贷款的合计，是银行资产质量的重要指标。',
        'unit': '元',
        'regulatory_requirement': '不良贷款率 = 不良贷款 / 贷款总额，不得超过5%（监管红线）。',
        'calculation_method': '不良贷款 = 次级类贷款 + 可疑类贷款 + 损失类贷款',
        'data_source': '通过公式计算得出'
    },
    'npl_ratio': {
        'description': '不良贷款率是指不良贷款余额与贷款总额的比率，反映银行资产质量状况。',
        'unit': '%',
        'regulatory_requirement': '根据《商业银行风险监管核心指标》，不良贷款率不得超过5%，超过5%将触发监管措施。',
        'calculation_method': '不良贷款率 = (不良贷款 / 贷款总额) × 100%',
        'data_source': '通过公式计算得出'
    },
    
    # G21 流动性报表字段
    'high_quality_liquid_assets': {
        'description': '优质流动性资产是指在压力情景下能够通过出售或抵（质）押方式，在无损失或极小损失的情况下快速变现的各类资产。主要包括：现金、存放央行款项、国债、央行票据、政策性金融债等。',
        'unit': '元',
        'regulatory_requirement': '优质流动性资产是计算流动性覆盖率（LCR）的分子，LCR = 优质流动性资产 / 未来30天现金净流出，不得低于100%。',
        'calculation_method': '优质流动性资产 = 一级资产 + 二级资产（按监管规定的折算系数计算）',
        'data_source': '从银行资产负债表、资金管理系统中提取，按照《商业银行流动性风险管理办法》规定的标准进行分类和折算'
    },
    'net_cash_outflow_30d': {
        'description': '未来30天现金净流出是指在压力情景下，未来30天内预期现金流出总量减去预期现金流入总量。',
        'unit': '元',
        'regulatory_requirement': '未来30天现金净流出是计算流动性覆盖率（LCR）的分母，LCR = 优质流动性资产 / 未来30天现金净流出，不得低于100%。',
        'calculation_method': '未来30天现金净流出 = 预期现金流出 - 预期现金流入（按监管规定的折算系数计算）',
        'data_source': '从银行资金管理系统、资产负债管理系统中提取，按照《商业银行流动性风险管理办法》规定的标准进行计算'
    },
    'lcr': {
        'description': '流动性覆盖率（LCR）是指优质流动性资产与未来30天现金净流出的比率，用于衡量银行短期流动性风险。',
        'unit': '%',
        'regulatory_requirement': '根据《商业银行流动性风险管理办法》，流动性覆盖率不得低于100%。',
        'calculation_method': '流动性覆盖率 = (优质流动性资产 / 未来30天现金净流出) × 100%',
        'data_source': '通过公式计算得出'
    },
    
    # 通用字段
    'report_period': {
        'description': '报表期间是指报表所反映的时间范围，通常为季度、月度等。',
        'unit': '',
        'regulatory_requirement': '报表期间必须准确填写，用于监管部门了解银行在特定时期的经营状况。',
        'calculation_method': '根据报表类型和报送频率确定，如季度报表为"YYYY年第X季度"，月度报表为"YYYY年X月"',
        'data_source': '系统自动生成'
    },
    'report_date': {
        'description': '报表日期是指报表生成的日期，通常为报表期间结束后的某个日期。',
        'unit': '',
        'regulatory_requirement': '报表日期必须准确填写，用于记录报表的生成时间。',
        'calculation_method': '系统自动生成当前日期',
        'data_source': '系统自动生成'
    },
    'bank_name': {
        'description': '银行名称是指报送报表的银行机构的完整名称。',
        'unit': '',
        'regulatory_requirement': '银行名称必须准确填写，用于监管部门识别报送机构。',
        'calculation_method': '从银行基本信息系统中提取',
        'data_source': '从银行基本信息系统中提取'
    }
}

def get_field_description(field_id: str) -> Dict[str, str]:
    """获取字段详细说明"""
    return FIELD_DESCRIPTIONS.get(field_id, {
        'description': '暂无说明',
        'unit': '',
        'regulatory_requirement': '',
        'calculation_method': '',
        'data_source': ''
    })

