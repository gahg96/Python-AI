# 端到端Demo网页界面使用指南

## 访问地址

启动服务后，访问：
```
http://localhost:5000/demo
```

## 功能说明

### 界面特点

1. **7个步骤面板**
   - 每个步骤可独立执行
   - 可折叠展开查看详情
   - 实时显示执行状态

2. **进度跟踪**
   - 总体进度条显示完成百分比
   - 每个步骤显示执行状态（待执行/执行中/完成/失败）

3. **实时日志**
   - 每个步骤都有日志输出窗口
   - 显示执行过程和结果

4. **结果展示**
   - 执行完成后显示关键指标
   - 统计信息一目了然

### 使用方式

#### 方式1：一键运行完整流程

点击页面底部的 **"🚀 一键运行完整流程"** 按钮，系统会自动依次执行所有7个步骤。

#### 方式2：逐步执行

1. **步骤1：历史数据准备**
   - 设置历史贷款数量（默认10000）
   - 设置对私贷款比例（默认0.7）
   - 点击"生成历史数据"

2. **步骤2：数据质量检查**
   - 点击"执行质量检查"
   - 查看完整性、一致性、时间一致性等得分

3. **步骤3：特征工程**
   - 点击"执行特征工程"
   - 查看新增特征数量

4. **步骤4：规则提取和量化**
   - 点击"提取规则"
   - 查看提取的规则数量

5. **步骤5：模型训练**
   - 点击"训练模型"
   - 查看模型性能指标

6. **步骤6：模拟审批**
   - 设置模拟客户数量（默认100）
   - 点击"开始模拟"
   - 查看审批结果统计

7. **步骤7：结果验证**
   - 点击"执行验证"
   - 查看验证结果（违约率、利润分布、回收率）

## API端点

### 1. 生成历史数据
**POST** `/api/demo/generate-historical`

请求体：
```json
{
  "num_loans": 10000,
  "personal_ratio": 0.7
}
```

### 2. 数据质量检查
**POST** `/api/demo/check-quality`

### 3. 特征工程
**POST** `/api/demo/engineer-features`

### 4. 规则提取
**POST** `/api/demo/extract-rules`

### 5. 模型训练
**POST** `/api/demo/train-models`

### 6. 模拟审批
**POST** `/api/demo/simulate-approval`

请求体：
```json
{
  "num_customers": 100
}
```

### 7. 结果验证
**POST** `/api/demo/validate-results`

## 注意事项

1. **执行顺序**：建议按照步骤顺序执行，因为后续步骤依赖前面的结果
2. **执行时间**：完整流程可能需要几分钟时间，请耐心等待
3. **数据文件**：所有生成的数据文件保存在 `data/historical/` 目录
4. **错误处理**：如果某个步骤失败，会显示错误信息，可以查看日志了解详情

## 数据文件位置

- 历史数据：`data/historical/historical_loans.csv`
- 特征工程后数据：`data/historical/historical_loans_engineered.csv`
- 质量报告：`data/historical/quality_report.json`
- 提取的规则：`data/historical/extracted_rules.json`
- 量化规则：`data/historical/quantified_rules.json`
- 训练模型：`data/historical/models/`
- 模拟结果：`data/historical/simulated_results.csv`
- 验证报告：`data/historical/validation_report.json`

## 常见问题

### Q: 页面无法访问？
A: 确保Flask服务已启动，访问 `http://localhost:5000/demo`

### Q: API返回错误？
A: 检查：
1. 是否按顺序执行了前面的步骤
2. 数据文件是否存在
3. 查看浏览器控制台的错误信息

### Q: 执行时间很长？
A: 这是正常的，特别是：
- 历史数据生成（10000条记录）
- 模型训练（需要训练多个模型）
- 模拟审批（需要处理每个客户）

### Q: 如何查看详细结果？
A: 可以：
1. 在网页界面查看结果摘要
2. 直接查看生成的数据文件（CSV/JSON格式）
3. 运行 `python src/demo/demo_showcase.py` 查看完整统计

