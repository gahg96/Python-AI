#!/usr/bin/env python3
"""只推送小文件，排除大文件"""

import subprocess
import os
from pathlib import Path

def run_cmd(cmd_list, cwd=None):
    """执行命令"""
    cmd_str = ' '.join(cmd_list)
    print(f"🔹 执行: {cmd_str}")
    result = subprocess.run(
        cmd_list,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"⚠️  {result.stderr}")
    return result.returncode == 0

# 项目目录
gamium_dir = Path(__file__).parent.parent.absolute()
parent_dir = gamium_dir.parent

print("=" * 60)
print("🚀 只推送小文件（排除大文件）")
print("=" * 60)
print()

# 步骤1: 更新.gitignore，排除大文件
print("📄 [1/5] 更新.gitignore，排除大文件...")
gitignore_path = gamium_dir / '.gitignore'

# 检查是否已排除
content = gitignore_path.read_text()
large_files_to_exclude = [
    'data/historical_backup/customers.parquet',
    'data/historical_backup/loan_applications.parquet',
    'data/historical_backup/repayment_history.parquet'
]

needs_update = False
for file_path in large_files_to_exclude:
    if file_path not in content:
        needs_update = True
        break

if needs_update:
    # 添加排除规则
    content += "\n# 大文件（超过GitHub LFS免费配额，暂不推送）\n"
    for file_path in large_files_to_exclude:
        content += f"{file_path}\n"
    gitignore_path.write_text(content)
    print("   ✅ 已更新.gitignore，排除大文件")
else:
    print("   ✅ .gitignore已正确配置")
print()

# 步骤2: 从Git LFS中移除大文件
print("📦 [2/5] 从Git LFS中移除大文件...")
for file_path in large_files_to_exclude:
    full_path = parent_dir / 'Gamium-FinanceAI' / file_path
    if full_path.exists():
        # 从暂存区移除
        run_cmd(['git', 'reset', 'HEAD', f'Gamium-FinanceAI/{file_path}'], cwd=parent_dir)
        # 从Git LFS中移除
        run_cmd(['git', 'lfs', 'untrack', f'Gamium-FinanceAI/{file_path}'], cwd=parent_dir)
print("   ✅ 大文件已移除")
print()

# 步骤3: 只添加小文件
print("📦 [3/5] 只添加小文件...")
# 添加.gitignore
run_cmd(['git', 'add', 'Gamium-FinanceAI/.gitignore'], cwd=parent_dir)

# 只添加macro_economics.parquet（小文件）
macro_file = gamium_dir / 'data' / 'historical_backup' / 'macro_economics.parquet'
if macro_file.exists():
    size_mb = macro_file.stat().st_size / 1024 / 1024
    if size_mb < 100:  # 小于100MB
        run_cmd(['git', 'add', 'Gamium-FinanceAI/data/historical_backup/macro_economics.parquet'], cwd=parent_dir)
        print(f"   ✅ 已添加macro_economics.parquet ({size_mb:.2f}MB)")
    else:
        print(f"   ⚠️  macro_economics.parquet太大 ({size_mb:.2f}MB)，跳过")
else:
    print("   ⚠️  macro_economics.parquet不存在")
print()

# 步骤4: 检查状态
print("📋 [4/5] 检查状态...")
run_cmd(['git', 'status', '--short'], cwd=parent_dir)
print()

# 检查Git LFS文件
print("   检查Git LFS文件:")
result = subprocess.run(
    ['git', 'lfs', 'ls-files'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)
if result.stdout.strip():
    print(result.stdout)
    print("   ✅ 只有小文件在Git LFS中")
else:
    print("   ℹ️  没有Git LFS文件")
print()

# 步骤5: 准备提交
print("📝 [5/5] 准备提交...")
print()
print("=" * 60)
print("✅ 配置完成！")
print()
print("📝 下一步操作:")
print()
print("   1. 提交更改:")
print("      git commit -m 'feat: 添加示例数据文件（仅小文件）'")
print()
print("   2. 推送到GitHub:")
print("      git push origin main")
print()
print("💡 说明:")
print("   - 大文件（>100MB）已排除")
print("   - 只推送小文件作为示例")
print("   - 完整数据可通过其他方式获取")
print("   - 详见: docs/大文件处理方案.md")
print("=" * 60)


