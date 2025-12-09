#!/bin/bash
# 提交代码到GitHub

cd /Users/carrot/Python-AI/Gamium-FinanceAI

echo "📦 准备提交代码到GitHub..."
echo ""

# 添加所有更改
echo "1. 添加文件..."
git add -A

# 显示状态
echo ""
echo "2. 文件状态:"
git status --short

# 提交
echo ""
echo "3. 提交更改..."
git commit -m "feat: 完善模型评估和风险因子说明功能

- 添加模型评估术语详解页面（HTML和Markdown）
- 添加风险因子确定方法详解文档
- 添加LTV生命周期价值详解文档
- 在客户预测界面添加模型评估指标说明
- 在客户画像中添加LTV详细说明弹窗
- 修复术语解释页面文字颜色对比度问题
- 添加客户信用评分预测脚本
- 添加模型评估脚本
- 添加数据提取和特征工程脚本
- 添加训练模型脚本
- 添加示例特征文件生成脚本
- 添加Parquet文件查看工具
- 更新Web界面，添加系统架构和术语解释链接
- 优化数据生成脚本，支持分块合并避免内存溢出"

# 推送到GitHub
echo ""
echo "4. 推送到GitHub..."
git push origin main

echo ""
echo "✅ 代码已同步到GitHub！"


