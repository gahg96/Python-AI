#!/usr/bin/env python3
"""只提交小文件，排除大文件"""

import subprocess
import os
import tempfile
from pathlib import Path

# Gamium-FinanceAI目录
gamium_dir = Path(__file__).parent.parent.absolute()
parent_dir = gamium_dir.parent

print("=" * 60)
print("🚀 只提交小文件（排除大文件）")
print("=" * 60)
print()

# 步骤1: 移除已暂存的大文件
print("📋 [1/6] 检查并移除已暂存的大文件...")
result = subprocess.run(
    ['git', 'status', '--short', '--cached'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

large_files_removed = []
for line in result.stdout.split('\n'):
    if 'Gamium-FinanceAI/data/historical_backup/' in line:
        file_path = line.split()[-1] if line.strip() else None
        if file_path:
            print(f"   移除大文件: {file_path}")
            subprocess.run(['git', 'reset', 'HEAD', file_path], cwd=parent_dir)
            large_files_removed.append(file_path)

if large_files_removed:
    print(f"   ✅ 已移除 {len(large_files_removed)} 个大文件")
else:
    print("   ✅ 没有大文件在暂存区")
print()

# 步骤2: 添加小文件（排除大文件目录）
print("📦 [2/6] 添加小文件（排除data/historical_backup/）...")
print("   正在执行 git add Gamium-FinanceAI/ ...")

# 先添加所有文件
subprocess.run(['git', 'add', 'Gamium-FinanceAI/'], cwd=parent_dir)

# 然后移除大文件目录
subprocess.run(['git', 'reset', 'HEAD', 'Gamium-FinanceAI/data/historical_backup/'], cwd=parent_dir)

print("   ✅ 添加完成")
print()

# 步骤3: 确认要提交的文件
print("📋 [3/6] 确认要提交的文件...")
result = subprocess.run(
    ['git', 'status', '--short', '--cached'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

files_to_commit = []
total_size = 0
for line in result.stdout.split('\n'):
    if line.strip() and 'Gamium-FinanceAI/' in line:
        files_to_commit.append(line)
        # 尝试获取文件大小
        parts = line.split()
        if len(parts) >= 2:
            file_path = parent_dir / parts[-1]
            if file_path.exists():
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass

print(f"   ✅ 将提交 {len(files_to_commit)} 个文件")
print(f"   ✅ 总大小: {total_size/1024/1024:.2f} MB")

# 显示前10个文件
print("\n   文件列表（前10个）:")
for line in files_to_commit[:10]:
    print(f"      {line}")
if len(files_to_commit) > 10:
    print(f"      ... 还有 {len(files_to_commit) - 10} 个文件")

# 确认没有大文件
has_large = False
for line in files_to_commit:
    if 'historical_backup' in line or 'historical_large' in line:
        has_large = True
        print(f"      ⚠️  警告: 发现大文件目录: {line}")

if not has_large:
    print("\n   ✅ 确认：没有大文件")
print()

# 步骤4: 检查是否有更改
result = subprocess.run(
    ['git', 'diff', '--cached', '--quiet'],
    cwd=parent_dir
)
if result.returncode == 0:
    print("✅ 没有需要提交的更改")
    exit(0)

# 步骤5: 提交
print("📝 [4/6] 提交更改...")
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
- 添加数据状态检查脚本
- 更新.gitignore排除大文件目录"""

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
    f.write(commit_msg)
    temp_file = f.name

try:
    result = subprocess.run(
        ['git', 'commit', '-F', temp_file],
        cwd=parent_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"   ❌ 提交失败: {result.stderr}")
        exit(1)
    print("   ✅ 提交成功")
    if result.stdout:
        print(f"   {result.stdout.strip()}")
finally:
    os.unlink(temp_file)
print()

# 步骤6: 推送
print("📤 [5/6] 推送到GitHub...")
result = subprocess.run(
    ['git', 'push', 'origin', 'main'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)
if result.returncode != 0:
    print("   💡 尝试master分支...")
    result = subprocess.run(
        ['git', 'push', 'origin', 'master'],
        cwd=parent_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"   ❌ 推送失败: {result.stderr}")
        exit(1)

print("   ✅ 推送成功")
print()

print("=" * 60)
print("✅ 完成！小文件已提交到GitHub")
print("💡 大文件已排除，后续可以单独处理")
print("=" * 60)


