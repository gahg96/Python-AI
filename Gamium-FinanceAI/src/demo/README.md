# 端到端贷款审批Demo - 使用指南

## 查看已完成模块的效果

### 方法1：运行展示脚本（推荐）

```bash
cd Gamium-FinanceAI
python src/demo/demo_showcase.py
```

这会展示所有已完成模块的统计信息和效果。

### 方法2：查看生成的数据文件

所有生成的数据文件都在 `data/historical/` 目录下：

1. **历史贷款数据**
   - `historical_loans.csv` - 原始历史数据（10,000条记录）
   - `historical_loans_engineered.csv` - 特征工程后的数据（80个特征）

2. **质量检查报告**
   - `quality_report.json` - 数据质量检查报告

3. **提取的规则**
   - `extracted_rules.json` - 从历史数据中提取的业务规则
   - `quantified_rules.json` - 量化后的规则元数据

4. **市场环境数据**
   - `market_conditions.csv` - 市场环境时间序列数据

5. **学习到的分布**
   - `learned_distributions.json` - 客户特征分布参数

### 方法3：运行各个模块的测试脚本

```bash
# 1. 历史数据生成器
python src/demo/historical_data_generator.py

# 2. 数据质量检查
python src/demo/data_quality_checker.py

# 3. 特征工程
python src/demo/feature_engineer.py

# 4. 规则提取
python src/demo/rule_extractor.py

# 5. 规则量化
python src/demo/rule_quantifier.py

# 6. 增强版客户生成器
python src/demo/enhanced_customer_generator.py

# 7. 市场环境模拟器
python src/demo/market_simulator.py
```

## 已完成模块列表

✅ **已完成（7/16）**

1. ✅ 历史数据生成器 - 生成10,000条历史贷款数据
2. ✅ 数据质量检查模块 - 完整性、一致性、时间一致性检查
3. ✅ 特征工程模块 - 创建34个新特征
4. ✅ 业务规则提取模块 - 提取4条业务规则
5. ✅ 规则量化模块 - 将规则转化为可执行函数
6. ✅ 增强版客户生成器 - 学习历史分布生成真实客户
7. ✅ 市场环境模拟器 - GDP、利率、失业率等市场因子

⏳ **进行中（1/16）**

8. 🔄 世界模型训练 - 训练违约预测模型和还款行为模型

⏸️ **待完成（8/16）**

9. ⏸️ 增强版规则引擎
10. ⏸️ 模型决策模块
11. ⏸️ 决策融合模块
12. ⏸️ 还款行为模拟器
13. ⏸️ 回收率计算器
14. ⏸️ 结果验证模块
15. ⏸️ 端到端集成
16. ⏸️ 可视化报告

## 数据文件说明

### historical_loans.csv
包含10,000条历史贷款记录，字段包括：
- 客户信息：customer_id, customer_type, age, monthly_income, credit_score等
- 贷款信息：loan_amount, loan_purpose, requested_term_months等
- 审批信息：expert_decision, approved_amount, approved_rate等
- 结果信息：actual_defaulted, actual_profit, recovery_amount等
- 市场环境：gdp_growth, base_interest_rate, unemployment_rate等

### historical_loans_engineered.csv
在原始数据基础上增加了34个新特征：
- 衍生特征：loan_to_annual_income_ratio, comprehensive_risk_score等
- 时间特征：application_year, application_month, approval_delay_days等
- 交互特征：credit_debt_interaction, income_stability_interaction等
- 目标特征：is_approved, is_defaulted, profit_category等

### extracted_rules.json
从历史数据中提取的业务规则，包括：
- 规则类型：threshold（阈值）、range（范围）、ratio（比例）、composite（复合）
- 规则信息：字段、操作符、值、置信度、支持度等

## 下一步

继续实现剩余模块，最终完成端到端的贷款审批Demo系统。

