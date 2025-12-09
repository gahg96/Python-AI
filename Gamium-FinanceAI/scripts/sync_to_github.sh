#!/bin/bash
# 一键提交代码到GitHub

set -e  # 遇到错误立即退出

echo "🚀 开始同步代码到GitHub..."
echo ""

# 进入项目目录
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
echo "📁 项目目录: $PROJECT_DIR"
echo ""

# 检查是否在git仓库中
if [ ! -d ".git" ]; then
    echo "❌ 错误: 当前目录不是git仓库"
    echo "💡 请先初始化git仓库:"
    echo "   git init"
    echo "   git remote add origin https://github.com/你的用户名/仓库名.git"
    exit 1
fi

# 检查是否有远程仓库
if ! git remote | grep -q origin; then
    echo "❌ 错误: 未找到远程仓库 'origin'"
    echo "💡 请添加远程仓库:"
    echo "   git remote add origin https://github.com/你的用户名/仓库名.git"
    exit 1
fi

echo "📊 检查git状态..."
git status --short
echo ""

# 询问是否继续
read -p "是否继续提交? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 1
fi

echo ""
echo "📦 添加所有更改..."
git add -A

echo ""
echo "📝 提交更改..."
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
- 优化数据生成脚本，支持分块合并避免内存溢出
- 添加数据状态检查脚本"

echo ""
echo "📤 推送到GitHub..."
git push origin main || git push origin master

echo ""
echo "✅ 完成！代码已同步到GitHub"
echo ""
echo "📋 查看提交记录:"
git log --oneline -5

