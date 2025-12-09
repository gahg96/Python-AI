#!/usr/bin/env python3
"""只同步Gamium-FinanceAI目录的代码到GitHub"""

import subprocess
import os
import tempfile
from pathlib import Path

# Gamium-FinanceAI目录
gamium_dir = Path(__file__).parent.parent.absolute()
parent_dir = gamium_dir.parent

print("📁 Gamium-FinanceAI目录:", gamium_dir)
print("📁 Git仓库根目录:", parent_dir)
print()

# 检查git仓库位置
if not (parent_dir / ".git").exists():
    print("❌ 错误: 未找到git仓库")
    print("💡 请先初始化git仓库")
    exit(1)

print("✅ 检测到git仓库在上级目录")
print("📦 只提交Gamium-FinanceAI子目录的文件")
print()

# 检查git状态（只显示Gamium-FinanceAI目录）
print("📊 检查git状态（仅Gamium-FinanceAI目录）...")
result = subprocess.run(
    ['git', 'status', '--short', 'Gamium-FinanceAI/'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)
if result.stdout.strip():
    print(result.stdout)
else:
    print("   (无更改)")
print()

# 只添加Gamium-FinanceAI目录下的所有更改
print("📦 添加Gamium-FinanceAI目录下的所有更改...")
print("   (不会添加上级目录的文件，如 .DS_Store, .vscode/ 等)")
subprocess.run(['git', 'add', 'Gamium-FinanceAI/'], cwd=parent_dir)
print()

# 检查是否有更改
result = subprocess.run(
    ['git', 'diff', '--cached', '--quiet'],
    cwd=parent_dir
)
if result.returncode == 0:
    print("✅ 没有需要提交的更改")
    exit(0)

# 显示将要提交的文件（只显示Gamium-FinanceAI相关的）
print("📋 将要提交的文件（仅Gamium-FinanceAI）:")
result = subprocess.run(
    ['git', 'status', '--short'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)
for line in result.stdout.split('\n'):
    if line.strip() and 'Gamium-FinanceAI/' in line:
        print(f"   {line}")
print()

# 确认没有添加上级目录的文件
print("🔍 确认没有添加上级目录的文件...")
result = subprocess.run(
    ['git', 'status', '--short'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)
has_parent_files = False
for line in result.stdout.split('\n'):
    if line.strip() and not line.startswith(' ') and 'Gamium-FinanceAI/' not in line:
        if not line.startswith('?? ../'):
            has_parent_files = True
            print(f"   ⚠️  警告: 发现非Gamium-FinanceAI文件: {line}")

if not has_parent_files:
    print("   ✅ 确认：只包含Gamium-FinanceAI目录的文件")
print()

# 提交更改
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
    result = subprocess.run(
        ['git', 'commit', '-F', temp_file],
        cwd=parent_dir
    )
    if result.returncode != 0:
        print("❌ 提交失败")
        exit(1)
    print("✅ 提交成功")
finally:
    os.unlink(temp_file)
print()

# 推送到GitHub
print("📤 推送到GitHub...")
result = subprocess.run(
    ['git', 'push', 'origin', 'main'],
    cwd=parent_dir
)
if result.returncode != 0:
    print("💡 尝试master分支...")
    result = subprocess.run(
        ['git', 'push', 'origin', 'master'],
        cwd=parent_dir
    )
    if result.returncode != 0:
        print("❌ 推送失败")
        print("💡 请检查:")
        print("   1. 是否已设置远程仓库: git remote -v")
        print("   2. 是否有推送权限")
        print("   3. 分支名称是否正确")
        exit(1)

print()
print("✅ 完成！Gamium-FinanceAI代码已同步到GitHub")
print()

# 显示最近5条提交记录
print("📋 最近提交记录:")
subprocess.run(['git', 'log', '--oneline', '-5'], cwd=parent_dir)
