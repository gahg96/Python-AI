# Gamium Finance AI - API 接口文档

> **版本**: v1.0  
> **基础URL**: `http://localhost:5000`  
> **格式**: JSON  
> **编码**: UTF-8

---

## 目录

- [1. 页面路由](#1-页面路由)
- [2. 客户生成与管理](#2-客户生成与管理)
- [3. 预测分析](#3-预测分析)
- [4. 经济周期分析](#4-经济周期分析)
- [5. 银行模拟](#5-银行模拟)
- [6. 策略对比](#6-策略对比)
- [7. AI演武场](#7-ai演武场)
- [8. 数据蒸馏](#8-数据蒸馏)
- [9. 数据浏览](#9-数据浏览)
- [10. 实时监控](#10-实时监控)
- [11. 模型管理](#11-模型管理)
- [12. 评估测试](#12-评估测试)
- [13. AI Hub](#13-ai-hub)
- [14. 压力测试](#14-压力测试)

---

## 1. 页面路由

### 1.1 首页（大屏展示）

**GET** `/`

返回大屏展示页面（dashboard.html）

**响应**: HTML页面

---

### 1.2 操作控制台

**GET** `/control` 或 `/ops`

返回操作控制台页面（index.html）

**响应**: HTML页面

---

### 1.3 项目介绍

**GET** `/story`

返回Gamium AI故事展示页面

**响应**: HTML页面

---

### 1.4 数据生成流程

**GET** `/data-generation`

返回数据生成流程详解页面

**响应**: HTML页面

---

### 1.5 银行系统架构

**GET** `/banking-architecture`

返回银行系统架构与数据提取详解页面

**响应**: HTML页面

---

### 1.6 模型评估术语

**GET** `/model-terms`

返回模型评估术语详解页面

**响应**: HTML页面

---

### 1.7 文档服务

**GET** `/docs/<path:filename>`

提供文档文件服务

**参数**:
- `filename` (path): 文档文件路径

**响应**: 文件内容

---

### 1.8 图片服务

**GET** `/images/<path:filename>`

提供图片文件服务

**参数**:
- `filename` (path): 图片文件路径

**响应**: 图片文件

---

## 2. 客户生成与管理

### 2.1 生成单个客户

**POST** `/api/customer/generate`

生成一个模拟客户数据

**请求体**:
```json
{
  "customer_type": "salaried",  // 可选: salaried, small_business, freelancer, farmer, professional, entrepreneur, investor, retiree, student, micro_enterprise, small_enterprise, medium_enterprise, large_enterprise, startup, tech_startup, manufacturing, trade_company, service_company
  "risk_profile": "medium"       // 可选: low, medium, high
}
```

**响应**:
```json
{
  "success": true,
  "customer": {
    "customer_id": "CUST_xxx",
    "customer_type": "工薪阶层",
    "age": 35,
    "monthly_income": 15000,
    "credit_score": 720,
    "employment_status": "employed",
    "city_tier": "一线城市",
    "industry": "IT/互联网",
    // ... 更多客户属性
  }
}
```

---

### 2.2 批量生成客户

**POST** `/api/customer/batch`

批量生成多个客户

**请求体**:
```json
{
  "count": 100,                  // 生成数量
  "customer_type": "salaried",   // 可选
  "risk_profile": "medium"        // 可选
}
```

**响应**:
```json
{
  "success": true,
  "customers": [
    { /* 客户对象 */ },
    // ...
  ],
  "count": 100
}
```

---

### 2.3 获取违约概率详细计算

**POST** `/api/customer/default-probability-detail`

获取违约概率的详细计算过程和解释

**请求体**:
```json
{
  "customer": { /* 客户对象 */ },
  "prediction": { /* 预测结果 */ },
  "loan": {
    "amount": 100000,
    "interest_rate": 0.08,
    "term_months": 36
  },
  "market": { /* 市场条件 */ }
}
```

**响应**:
```json
{
  "success": true,
  "default_probability": 0.15,
  "calculation_steps": [
    {
      "step": "基础违约概率",
      "value": 0.12,
      "explanation": "基于客户信用评分计算"
    },
    // ... 更多步骤
  ],
  "factors": {
    "positive": ["高收入", "稳定工作"],
    "negative": ["高负债率", "短期信用记录"]
  }
}
```

---

### 2.4 获取流失概率详细计算

**POST** `/api/customer/churn-probability-detail`

获取客户流失概率的详细计算过程

**请求体**: 同违约概率接口

**响应**:
```json
{
  "success": true,
  "churn_probability": 0.08,
  "calculation_steps": [ /* 计算步骤 */ ],
  "factors": { /* 影响因素 */ }
}
```

---

### 2.5 获取风险评分详细计算

**POST** `/api/customer/risk-score-detail`

获取风险评分的详细计算过程

**请求体**: 同违约概率接口

**响应**:
```json
{
  "success": true,
  "risk_score": 65,
  "calculation_steps": [ /* 计算步骤 */ ],
  "breakdown": {
    "credit_score": 30,
    "income_stability": 20,
    "debt_ratio": 15
  }
}
```

---

### 2.6 获取贷款历史

**POST** `/api/customer/loan-history`

获取客户的贷款历史记录

**请求体**:
```json
{
  "customer": { /* 客户对象 */ },
  "max_history": 10  // 可选，最多返回记录数
}
```

**响应**:
```json
{
  "success": true,
  "loan_history": [
    {
      "loan_id": "LOAN_xxx",
      "amount": 50000,
      "interest_rate": 0.075,
      "term_months": 24,
      "status": "completed",
      "defaulted": false,
      "start_date": "2022-01-01",
      "end_date": "2024-01-01"
    },
    // ...
  ]
}
```

---

## 3. 预测分析

### 3.1 综合预测

**POST** `/api/predict`

对客户进行综合预测分析（违约概率、流失概率、风险评分等）

**请求体**:
```json
{
  "customer": { /* 客户对象 */ },
  "loan": {
    "amount": 100000,
    "interest_rate": 0.08,
    "term_months": 36
  },
  "market": { /* 市场条件，可选 */ }
}
```

**响应**:
```json
{
  "success": true,
  "customer": { /* 客户对象 */ },
  "loan": { /* 贷款信息 */ },
  "market": { /* 市场条件 */ },
  "prediction": {
    "default_probability": 0.15,
    "churn_probability": 0.08,
    "risk_score": 65,
    "ltv": 85000,
    "expected_profit": 12000
  }
}
```

---

## 4. 经济周期分析

### 4.1 经济周期影响分析

**POST** `/api/analysis/economic-cycle`

分析经济周期对贷款业务的影响

**请求体**:
```json
{
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 10000000,
  "strategy": {
    "approval_threshold": 0.15,
    "rate_spread": 0.02
  },
  "scenarios": ["boom", "recession", "depression", "recovery"]  // 可选
}
```

**响应**:
```json
{
  "success": true,
  "run_id": "CYCLE_xxx",
  "config": { /* 配置信息 */ },
  "results": {
    "boom": {
      "total_profit": 2000000,
      "default_rate": 0.05,
      "customer_count": 500
    },
    "recession": { /* ... */ },
    // ...
  },
  "summary": {
    "best_phase": "boom",
    "worst_phase": "depression",
    "recommendations": [ /* 建议 */ ]
  }
}
```

---

### 4.2 获取分析追踪

**GET** `/api/analysis/trace`

获取经济周期分析的追踪数据

**查询参数**:
- `run_id` (可选): 分析运行ID

**响应**:
```json
{
  "success": true,
  "trace": {
    "run_id": "CYCLE_xxx",
    "config": { /* 配置 */ },
    "steps": [ /* 分析步骤 */ ],
    "results": { /* 结果 */ }
  }
}
```

---

## 5. 银行模拟

### 5.1 启动模拟

**POST** `/api/simulation/start`

启动银行经营模拟

**请求体**:
```json
{
  "initial_capital": 10000000,
  "strategy": {
    "approval_threshold": 0.15,
    "rate_spread": 0.02
  },
  "duration_months": 12
}
```

**响应**:
```json
{
  "success": true,
  "simulation_id": "SIM_xxx",
  "status": "running"
}
```

---

### 5.2 单步执行

**POST** `/api/simulation/step`

执行模拟的一个时间步

**请求体**:
```json
{
  "simulation_id": "SIM_xxx"
}
```

**响应**:
```json
{
  "success": true,
  "month": 1,
  "capital": 10200000,
  "profit": 200000,
  "customers": 50,
  "defaults": 2
}
```

---

### 5.3 自动运行

**POST** `/api/simulation/auto-run`

自动运行完整模拟

**请求体**:
```json
{
  "initial_capital": 10000000,
  "strategy": {
    "approval_threshold": 0.15,
    "rate_spread": 0.02
  },
  "duration_months": 12,
  "seed": 42
}
```

**响应**:
```json
{
  "success": true,
  "simulation_id": "SIM_xxx",
  "summary": {
    "final_capital": 12000000,
    "total_profit": 2000000,
    "total_customers": 600,
    "default_rate": 0.08,
    "monthly_results": [ /* 每月结果 */ ]
  }
}
```

---

### 5.4 获取模拟追踪

**GET** `/api/simulation/trace`

获取银行模拟的追踪数据

**查询参数**:
- `simulation_id` (可选): 模拟ID

**响应**:
```json
{
  "success": true,
  "trace": {
    "simulation_id": "SIM_xxx",
    "config": { /* 配置 */ },
    "steps": [ /* 模拟步骤 */ ],
    "summary": { /* 汇总 */ }
  }
}
```

---

## 6. 策略对比

### 6.1 策略对比分析

**POST** `/api/comparison/strategies`

对比多个策略的表现

**请求体**:
```json
{
  "strategies": [
    {
      "name": "稳健策略",
      "approval_threshold": 0.12,
      "rate_spread": 0.01
    },
    {
      "name": "平衡策略",
      "approval_threshold": 0.18,
      "rate_spread": 0.015
    }
  ],
  "customer_count": 1000,
  "duration_months": 12,
  "seed": 42
}
```

**响应**:
```json
{
  "success": true,
  "run_id": "COMP_xxx",
  "results": [
    {
      "strategy": "稳健策略",
      "total_profit": 1500000,
      "default_rate": 0.05,
      "customer_count": 400,
      "roi": 0.15
    },
    // ...
  ],
  "winner": "稳健策略",
  "recommendations": [ /* 建议 */ ]
}
```

---

### 6.2 获取对比追踪

**GET** `/api/comparison/trace`

获取策略对比的追踪数据

**查询参数**:
- `run_id` (可选): 对比运行ID

**响应**: 同策略对比分析

---

### 6.3 银行对比运行

**POST** `/api/bank-comparison/run`

运行多银行对比模拟

**请求体**:
```json
{
  "banks": [
    {
      "name": "银行A",
      "initial_capital": 10000000,
      "strategy": { /* 策略配置 */ }
    },
    // ...
  ],
  "duration_months": 12,
  "seed": 42
}
```

**响应**:
```json
{
  "success": true,
  "results": [
    {
      "bank": "银行A",
      "final_capital": 12000000,
      "profit": 2000000,
      "market_share": 0.35
    },
    // ...
  ],
  "winner": "银行A"
}
```

---

## 7. AI演武场

### 7.1 运行演武场

**POST** `/api/arena/run`

运行多模型/策略演武场（支持规则引擎和评分系统）

**请求体**:
```json
{
  "participants": [
    {
      "name": "稳健策略",
      "approval_threshold": 0.12,
      "rate_spread": 0.01,
      "model_id": null  // 可选，指定使用的模型ID
    },
    {
      "name": "平衡策略",
      "approval_threshold": 0.18,
      "rate_spread": 0.015,
      "model_id": "gpt-4"  // 可选
    }
  ],
  "customer_count": 300,
  "loan_amount": 100000,
  "base_rate": 0.08,
  "seed": 42,
  "scenario": "normal",  // normal / stress
  "black_swan": false,
  "rules": [
    {
      "name": "高收入调低阈值",
      "description": "月收入超过20000的客户降低审批阈值",
      "priority": 1,
      "enabled": true,
      "category": "risk",
      "conditions": [
        {
          "field": "monthly_income",
          "operator": ">",
          "value": 20000,
          "logical_op": "and"
        }
      ],
      "action": {
        "approval_threshold_delta": -0.02,
        "rate_spread_delta": -0.002
      },
      "penalty": {
        "score_delta": 0.0
      }
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
```

**响应**:
```json
{
  "success": true,
  "run_id": "ARENA_xxx",
  "participants": [
    {
      "name": "稳健策略",
      "approval_rate": 0.45,
      "default_rate": 0.05,
      "expected_profit": 1200000,
      "risk_score": 85,
      "score_breakdown": {
        "profit_score": 90,
        "risk_score": 85,
        "stability_score": 80,
        "compliance_score": 90,
        "efficiency_score": 75,
        "explainability_score": 80,
        "overall_score": 85.5
      },
      "triggered_rules": ["高收入调低阈值", "良好信用记录"]
    },
    // ...
  ],
  "summary": {
    "winner": "稳健策略",
    "total_customers": 300,
    "rules_count": 7,
    "scenario": "normal",
    "black_swan": false,
    "calculation_notes": "计算说明..."
  },
  "customer_details": [ /* 客户详情 */ ]
}
```

---

### 7.2 AI解释

**POST** `/api/arena/ai-explain`

获取演武场结果的AI解释

**请求体**:
```json
{
  "arena_result": { /* 演武场结果 */ }
}
```

**响应**:
```json
{
  "success": true,
  "explanation": {
    "winner": "稳健策略",
    "winner_analysis": "稳健策略在风险控制和利润平衡方面表现最佳...",
    "performance_comparison": "各策略对比分析...",
    "key_insights": ["洞察1", "洞察2"],
    "risk_assessment": "风险评估...",
    "recommendations": ["建议1", "建议2"]
  }
}
```

---

### 7.3 导出结果

**POST** `/api/arena/export`

导出演武场结果

**请求体**:
```json
{
  "arena_result": { /* 演武场结果 */ },
  "format": "json"  // json / csv / txt
}
```

**响应**: 文件下载或JSON数据

---

### 7.4 锦标赛

**POST** `/api/arena/tournament`

运行锦标赛模式（多轮淘汰）

**请求体**:
```json
{
  "participants": [ /* 参赛者列表 */ ],
  "rounds": 3,
  "customers_per_round": 100
}
```

**响应**:
```json
{
  "success": true,
  "tournament_id": "TOUR_xxx",
  "rounds": [
    {
      "round": 1,
      "winners": ["策略A", "策略B"],
      "eliminated": ["策略C"]
    },
    // ...
  ],
  "champion": "策略A"
}
```

---

### 7.5 LLM决策

**POST** `/api/arena/llm-decision`

使用LLM进行贷款决策

**请求体**:
```json
{
  "customer": { /* 客户对象 */ },
  "model_id": "gpt-4",
  "loan": {
    "amount": 100000,
    "base_rate": 0.08
  }
}
```

**响应**:
```json
{
  "success": true,
  "decision": {
    "approved": true,
    "confidence": 0.85,
    "reasoning": "客户信用良好，收入稳定...",
    "adjusted_rate": 0.082,
    "adjusted_amount": 100000
  }
}
```

---

### 7.6 获取默认规则

**GET** `/api/arena/default-rules`

获取默认的规则配置

**响应**:
```json
{
  "success": true,
  "rules": [
    {
      "name": "高收入优惠",
      "description": "月收入超过20000的客户给予利率优惠",
      "priority": 1,
      "enabled": true,
      "category": "promotion",
      "conditions": [ /* 条件列表 */ ],
      "action": { /* 动作配置 */ },
      "penalty": { /* 惩罚配置 */ }
    },
    // ...
  ]
}
```

---

### 7.7 多轮演武场

**POST** `/api/arena/multi-round`

运行多轮演武场（支持宏观经济扰动和压力情景）

**请求体**:
```json
{
  "participants": [ /* 参赛者列表 */ ],
  "rounds": 5,
  "customers_per_round": 100,
  "macro_perturbations": true,
  "stress_scenarios": ["recession", "black_swan"],
  "cumulative_scoring": true
}
```

**响应**:
```json
{
  "success": true,
  "run_id": "MULTI_xxx",
  "rounds": [
    {
      "round": 1,
      "market_conditions": { /* 市场条件 */ },
      "results": [ /* 各参赛者结果 */ ]
    },
    // ...
  ],
  "cumulative_scores": {
    "策略A": 425.5,
    "策略B": 410.2
  },
  "final_winner": "策略A"
}
```

---

## 8. 数据蒸馏

### 8.1 获取数据源

**GET** `/api/distillation/sources`

获取可用的数据源列表

**响应**:
```json
{
  "success": true,
  "sources": [
    {
      "name": "test_data",
      "path": "/path/to/test_data",
      "files": [
        {
          "name": "customers.parquet",
          "size": 1024000,
          "size_mb": 1.0
        }
      ],
      "summary": { /* 数据摘要 */ }
    }
  ]
}
```

---

### 8.2 运行数据蒸馏

**POST** `/api/distillation/run`

运行数据蒸馏流程

**请求体**:
```json
{
  "source": "test_data",
  "target_size": 10000,
  "preserve_distribution": true,
  "features": ["age", "income", "credit_score"]
}
```

**响应**:
```json
{
  "success": true,
  "run_id": "DIST_xxx",
  "config": { /* 配置信息 */ },
  "steps": [ /* 蒸馏步骤 */ ],
  "results": {
    "original_size": 100000,
    "distilled_size": 10000,
    "compression_ratio": 0.1,
    "quality_score": 0.95
  }
}
```

---

### 8.3 获取蒸馏追踪

**GET** `/api/distillation/trace`

获取数据蒸馏的追踪数据

**查询参数**:
- `run_id` (可选): 蒸馏运行ID

**响应**:
```json
{
  "success": true,
  "trace": {
    "run_id": "DIST_xxx",
    "config": { /* 配置 */ },
    "steps": [ /* 步骤 */ ],
    "results": { /* 结果 */ }
  }
}
```

---

### 8.4 获取审计日志

**GET** `/api/distillation/audit-log`

获取数据蒸馏的审计日志

**查询参数**:
- `run_id` (可选): 蒸馏运行ID

**响应**:
```json
{
  "success": true,
  "audit_log": [
    {
      "timestamp": "2024-01-01T10:00:00",
      "action": "start_distillation",
      "details": { /* 详情 */ }
    },
    // ...
  ]
}
```

---

### 8.5 获取蒸馏状态

**GET** `/api/distillation/status`

获取数据蒸馏的当前状态

**响应**:
```json
{
  "success": true,
  "status": "running",  // running / completed / failed
  "progress": 0.65,
  "current_step": "feature_selection"
}
```

---

## 9. 数据浏览

### 9.1 获取数据源列表

**GET** `/api/data/sources`

获取可用的数据源列表

**响应**:
```json
{
  "success": true,
  "sources": [
    {
      "name": "test_data",
      "path": "/path/to/test_data",
      "files": [ /* 文件列表 */ ],
      "summary": { /* 摘要 */ }
    }
  ]
}
```

---

### 9.2 获取数据表结构

**POST** `/api/data/schema`

获取数据表的字段结构

**请求体**:
```json
{
  "source": "test_data",
  "table": "customers.parquet"
}
```

**响应**:
```json
{
  "success": true,
  "schema": {
    "columns": [
      {
        "name": "customer_id",
        "type": "string",
        "nullable": false
      },
      // ...
    ],
    "row_count": 100000
  }
}
```

---

### 9.3 浏览数据

**POST** `/api/data/browse`

浏览数据表内容

**请求体**:
```json
{
  "source": "test_data",
  "table": "customers.parquet",
  "page": 1,
  "page_size": 100,
  "filters": { /* 可选过滤条件 */ }
}
```

**响应**:
```json
{
  "success": true,
  "data": [
    { /* 数据行 */ },
    // ...
  ],
  "total": 100000,
  "page": 1,
  "page_size": 100
}
```

---

### 9.4 获取单条记录

**GET** `/api/data/record/<source>/<table>/<record_id>`

获取单条数据记录的详细信息

**路径参数**:
- `source`: 数据源名称
- `table`: 表名
- `record_id`: 记录ID

**响应**:
```json
{
  "success": true,
  "record": { /* 完整记录数据 */ }
}
```

---

### 9.5 获取数据统计

**POST** `/api/data/stats`

获取数据表的统计信息

**请求体**:
```json
{
  "source": "test_data",
  "table": "customers.parquet",
  "columns": ["age", "income", "credit_score"]  // 可选，指定要统计的列
}
```

**响应**:
```json
{
  "success": true,
  "stats": {
    "age": {
      "min": 18,
      "max": 80,
      "mean": 42.5,
      "std": 12.3,
      "percentiles": { "25": 32, "50": 42, "75": 53 }
    },
    // ...
  }
}
```

---

## 10. 实时监控

### 10.1 实时数据Tick

**POST** `/api/realtime/tick`

生成一个实时数据tick（用于大屏展示）

**响应**:
```json
{
  "success": true,
  "transaction": {
    "customer_id": "CUST_xxx",
    "loan_amount": 100000,
    "decision": "approved",
    "timestamp": "2024-01-01T10:00:00"
  },
  "stats": {
    "capital": 10000000,
    "profit": 200000,
    "npl": 0.05,
    "customers": 500,
    "approved_today": 50,
    "rejected_today": 10,
    "total_processed": 60,
    "approval_rate": 0.83
  },
  "explanation": {
    "customer_analysis": { /* 客户分析 */ },
    "risk_factors": [ /* 风险因素 */ ],
    "positive_factors": [ /* 正面因素 */ ],
    "rules_triggered": [ /* 触发的规则 */ ],
    "decision": { /* 决策详情 */ }
  },
  "stress_test": { /* 压力测试结果 */ },
  "ai_hub": { /* AI Hub统计 */ }
}
```

---

### 10.2 获取实时统计

**GET** `/api/realtime/stats`

获取当前的实时统计数据

**响应**:
```json
{
  "success": true,
  "stats": {
    "capital": 10000000,
    "profit": 200000,
    "npl": 0.05,
    "customers": 500,
    "approved_today": 50,
    "rejected_today": 10,
    "total_processed": 60,
    "approval_rate": 0.83
  }
}
```

---

### 10.3 重置实时数据

**POST** `/api/realtime/reset`

重置实时监控数据

**响应**:
```json
{
  "success": true,
  "message": "实时数据已重置"
}
```

---

## 11. 模型管理

### 11.1 获取模型列表

**GET** `/api/models/list`

获取所有已注册的模型列表

**响应**:
```json
{
  "success": true,
  "models": [
    {
      "model_id": "gpt-4",
      "name": "GPT-4",
      "provider": "openai",
      "type": "llm",
      "status": "active",
      "config": { /* 模型配置 */ }
    },
    // ...
  ]
}
```

---

### 11.2 注册模型

**POST** `/api/models/register`

注册一个新的模型

**请求体**:
```json
{
  "model_id": "gpt-4",
  "name": "GPT-4",
  "provider": "openai",
  "type": "llm",
  "config": {
    "api_key": "xxx",
    "endpoint": "https://api.openai.com/v1/chat/completions"
  }
}
```

**响应**:
```json
{
  "success": true,
  "model": { /* 模型信息 */ }
}
```

---

## 12. 评估测试

### 12.1 获取测试用例

**GET** `/api/evaluation/test-cases`

获取所有测试用例

**响应**:
```json
{
  "success": true,
  "test_cases": [
    {
      "test_case_id": "TC_xxx",
      "name": "高收入客户测试",
      "customer": { /* 客户数据 */ },
      "expected_result": { /* 期望结果 */ }
    },
    // ...
  ]
}
```

---

### 12.2 创建评估

**POST** `/api/evaluation/create`

创建一个新的评估任务

**请求体**:
```json
{
  "name": "模型A评估",
  "model_id": "gpt-4",
  "test_cases": ["TC_xxx", "TC_yyy"],
  "metrics": ["accuracy", "precision", "recall"]
}
```

**响应**:
```json
{
  "success": true,
  "evaluation_id": "EVAL_xxx",
  "results": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.88
  }
}
```

---

### 12.3 获取测试用例详情

**GET** `/api/evaluation/test-case/<test_case_id>`

获取测试用例的详细信息

**路径参数**:
- `test_case_id`: 测试用例ID

**响应**:
```json
{
  "success": true,
  "test_case": { /* 完整测试用例数据 */ }
}
```

---

### 12.4 获取指标解释

**GET** `/api/evaluation/metrics-explanation`

获取评估指标的解释说明

**查询参数**:
- `metric` (可选): 指标名称

**响应**:
```json
{
  "success": true,
  "explanations": {
    "accuracy": "准确率：正确预测的比例...",
    "precision": "精确率：预测为正例中实际为正例的比例...",
    // ...
  }
}
```

---

### 12.5 保存脚本

**POST** `/api/evaluation/save-script`

保存评估脚本

**请求体**:
```json
{
  "script_name": "my_evaluation",
  "script_content": "def evaluate(...): ..."
}
```

**响应**:
```json
{
  "success": true,
  "script_id": "SCRIPT_xxx"
}
```

---

### 12.6 导出脚本

**POST** `/api/evaluation/export-script`

导出评估脚本

**请求体**:
```json
{
  "script_id": "SCRIPT_xxx",
  "format": "python"  // python / json
}
```

**响应**: 文件下载或脚本内容

---

## 13. AI Hub

### 13.1 获取模型信息

**GET** `/api/hub/model/<model_id>`

获取指定模型的详细信息

**路径参数**:
- `model_id`: 模型ID

**响应**:
```json
{
  "success": true,
  "model": {
    "model_id": "gpt-4",
    "name": "GPT-4",
    "provider": "openai",
    "type": "llm",
    "status": "active",
    "config": { /* 配置 */ },
    "statistics": {
      "total_calls": 1000,
      "success_rate": 0.98,
      "avg_latency": 250
    }
  }
}
```

---

### 13.2 更新模型

**PUT** `/api/hub/model/<model_id>`

更新模型配置

**路径参数**:
- `model_id`: 模型ID

**请求体**:
```json
{
  "name": "GPT-4 Updated",
  "config": { /* 新配置 */ }
}
```

**响应**:
```json
{
  "success": true,
  "model": { /* 更新后的模型信息 */ }
}
```

---

### 13.3 测试模型

**POST** `/api/hub/model/<model_id>/test`

测试模型连接和响应

**路径参数**:
- `model_id`: 模型ID

**请求体**:
```json
{
  "test_input": "测试输入"
}
```

**响应**:
```json
{
  "success": true,
  "test_result": {
    "connected": true,
    "response_time": 250,
    "response": "测试响应"
  }
}
```

---

### 13.4 获取敏感词列表

**GET** `/api/hub/sensitive-words`

获取所有敏感词

**响应**:
```json
{
  "success": true,
  "sensitive_words": [
    {
      "word_id": "SW_xxx",
      "word": "敏感词1",
      "category": "risk",
      "created_at": "2024-01-01T10:00:00"
    },
    // ...
  ]
}
```

---

### 13.5 添加敏感词

**POST** `/api/hub/sensitive-words`

添加新的敏感词

**请求体**:
```json
{
  "word": "敏感词",
  "category": "risk"
}
```

**响应**:
```json
{
  "success": true,
  "sensitive_word": { /* 敏感词信息 */ }
}
```

---

### 13.6 删除敏感词

**DELETE** `/api/hub/sensitive-words/<word_id>`

删除敏感词

**路径参数**:
- `word_id`: 敏感词ID

**响应**:
```json
{
  "success": true,
  "message": "敏感词已删除"
}
```

---

### 13.7 获取用户列表

**GET** `/api/hub/users`

获取所有用户

**响应**:
```json
{
  "success": true,
  "users": [
    {
      "user_id": "USER_xxx",
      "username": "admin",
      "role": "admin",
      "created_at": "2024-01-01T10:00:00"
    },
    // ...
  ]
}
```

---

### 13.8 获取角色列表

**GET** `/api/hub/roles`

获取所有角色

**响应**:
```json
{
  "success": true,
  "roles": [
    {
      "role_id": "ROLE_xxx",
      "name": "admin",
      "permissions": ["read", "write", "delete"]
    },
    // ...
  ]
}
```

---

### 13.9 获取统计信息

**GET** `/api/hub/statistics`

获取AI Hub的统计信息

**响应**:
```json
{
  "success": true,
  "statistics": {
    "total_models": 10,
    "active_models": 8,
    "total_calls": 100000,
    "total_cost": 5000,
    "avg_latency": 250,
    "error_rate": 0.02
  }
}
```

---

### 13.10 获取统计下钻

**GET** `/api/hub/statistics/drilldown`

获取统计信息的详细下钻数据

**查询参数**:
- `metric` (可选): 指标名称
- `time_range` (可选): 时间范围

**响应**:
```json
{
  "success": true,
  "drilldown": {
    "by_model": [
      {
        "model_id": "gpt-4",
        "calls": 50000,
        "cost": 2500
      },
      // ...
    ],
    "by_time": [
      {
        "date": "2024-01-01",
        "calls": 1000,
        "cost": 50
      },
      // ...
    ]
  }
}
```

---

## 14. 压力测试

### 14.1 运行压力测试

**POST** `/api/stress-test`

运行压力测试

**请求体**:
```json
{
  "scenario": "recession",  // recession / black_swan / market_crash
  "initial_capital": 10000000,
  "strategy": {
    "approval_threshold": 0.15,
    "rate_spread": 0.02
  },
  "duration_months": 12,
  "stress_level": "high"  // low / medium / high / extreme
}
```

**响应**:
```json
{
  "success": true,
  "test_id": "STRESS_xxx",
  "results": {
    "normal_rate": 0.05,
    "stress_rate": 0.15,
    "resilience_score": 75,
    "profit_drawdown": 0.30,
    "recovery_time": 6,
    "risk_level": "medium",
    "last_event": "经济衰退"
  },
  "recommendations": [ /* 建议 */ ]
}
```

---

## 错误响应格式

所有API在发生错误时返回以下格式：

```json
{
  "success": false,
  "error": "错误描述",
  "error_code": "ERROR_CODE",  // 可选
  "traceback": "..."  // 仅在debug模式下返回
}
```

**HTTP状态码**:
- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

---

## 认证与授权

当前版本API无需认证，未来版本可能添加：
- API Key认证
- JWT Token认证
- OAuth2.0授权

---

## 限流与配额

当前版本无限制，未来版本可能添加：
- 请求频率限制（如：100请求/分钟）
- 配额管理
- 优先级队列

---

## 版本历史

- **v1.0** (2024-01-01): 初始版本，包含所有核心功能

---

## 联系方式

- **项目地址**: https://github.com/your-repo/gamium-finance-ai
- **文档地址**: http://localhost:5000/docs
- **问题反馈**: 通过GitHub Issues提交

---

**文档最后更新**: 2024-01-01



