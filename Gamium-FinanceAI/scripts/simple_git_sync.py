#!/usr/bin/env python3
import subprocess
import os
import tempfile
from pathlib import Path

project_dir = Path(__file__).parent.parent
os.chdir(project_dir)

print("📁 项目目录:", project_dir)
print()

# 检查git状态
print("📊 检查git状态...")
subprocess.run(['git', 'status', '--short'], cwd=project_dir)
print()

# 添加所有更改
print("📦 添加所有更改...")
subprocess.run(['git', 'add', '-A'], cwd=project_dir)
print()

# 提交
print("📝 提交更改...")
commit_msg = """feat: 完善模型评估和风险因子说明功能

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
- 添加数据状态检查脚本"""

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
    f.write(commit_msg)
    temp_file = f.name

try:
    subprocess.run(['git', 'commit', '-F', temp_file], cwd=project_dir)
finally:
    os.unlink(temp_file)
print()

# 推送
print("📤 推送到GitHub...")
result = subprocess.run(['git', 'push', 'origin', 'main'], cwd=project_dir)
if result.returncode != 0:
    print("💡 尝试master分支...")
    subprocess.run(['git', 'push', 'origin', 'master'], cwd=project_dir)

print()
print("✅ 完成！")

