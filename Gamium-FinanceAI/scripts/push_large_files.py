#!/usr/bin/env python3
"""推送大文件到GitHub（使用Git LFS）"""

import subprocess
import os
import tempfile
from pathlib import Path

def run_cmd(cmd_list, cwd=None, show_output=True):
    """执行命令"""
    cmd_str = ' '.join(cmd_list)
    if show_output:
        print(f"🔹 执行: {cmd_str}")
    result = subprocess.run(
        cmd_list,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if show_output and result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        if show_output:
            print(f"⚠️  {result.stderr}")
        return False, result.stderr
    return result.returncode == 0, result.stdout

# 项目目录
gamium_dir = Path(__file__).parent.parent.absolute()
parent_dir = gamium_dir.parent

print("=" * 60)
print("🚀 推送大文件到GitHub（使用Git LFS）")
print("=" * 60)
print()

# 步骤1: 检查Git LFS
print("📦 [1/7] 检查Git LFS...")
result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
if result.returncode != 0:
    print("   ❌ Git LFS未安装")
    print("   💡 请先安装:")
    print("      macOS: brew install git-lfs")
    print("      Linux: sudo apt install git-lfs")
    exit(1)
print(f"   ✅ {result.stdout.strip()}")
print()

# 步骤2: 初始化Git LFS
print("🔧 [2/7] 初始化Git LFS...")
run_cmd(['git', 'lfs', 'install'], cwd=parent_dir)
print()

# 步骤3: 检查.gitattributes
print("📄 [3/7] 检查.gitattributes...")
gitattributes = gamium_dir / '.gitattributes'
if not gitattributes.exists():
    print("   ⚠️  .gitattributes不存在，正在创建...")
    gitattributes.write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    print("   ✅ 已创建.gitattributes")
else:
    print("   ✅ .gitattributes已存在")
print()

# 步骤4: 添加.gitattributes
print("📦 [4/7] 添加.gitattributes...")
run_cmd(['git', 'add', 'Gamium-FinanceAI/.gitattributes'], cwd=parent_dir)
print()

# 步骤5: 添加大文件
print("📦 [5/7] 添加大文件（使用Git LFS）...")
print("   注意：这可能需要一些时间...")

# 先添加.gitignore的更改
run_cmd(['git', 'add', 'Gamium-FinanceAI/.gitignore'], cwd=parent_dir)

# 添加大文件目录
success, output = run_cmd(['git', 'add', 'Gamium-FinanceAI/data/historical_backup/'], cwd=parent_dir)
if success:
    print("   ✅ 大文件已添加到Git LFS")
else:
    print("   ⚠️  添加文件时出现问题")
print()

# 步骤6: 检查状态
print("📋 [6/7] 检查状态...")
run_cmd(['git', 'status', '--short'], cwd=parent_dir)
print()

# 检查LFS文件
print("   检查Git LFS文件:")
run_cmd(['git', 'lfs', 'ls-files'], cwd=parent_dir)
print()

# 步骤7: 提交和推送
print("📝 [7/7] 准备提交...")
print()
print("💡 接下来需要手动执行:")
print()
print("   1. 提交更改:")
print("      git commit -m 'feat: 添加大文件数据（使用Git LFS）'")
print()
print("   2. 推送到GitHub:")
print("      git push origin main")
print()
print("⚠️  重要提示:")
print("   - 大文件上传可能需要很长时间（取决于文件大小和网络速度）")
print("   - GitHub免费账户Git LFS配额: 1GB存储 + 1GB带宽/月")
print("   - 如果超过配额，需要升级账户或使用其他存储方案")
print("   - 上传过程中请保持网络连接稳定")
print()
print("=" * 60)


