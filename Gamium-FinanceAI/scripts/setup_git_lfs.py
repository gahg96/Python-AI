#!/usr/bin/env python3
"""设置Git LFS并推送大文件"""

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
print("🚀 设置Git LFS并推送大文件")
print("=" * 60)
print()

# 步骤1: 检查Git LFS是否已安装
print("📦 [1/6] 检查Git LFS...")
result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
if result.returncode != 0:
    print("   ❌ Git LFS未安装")
    print("   💡 请先安装Git LFS:")
    print("      macOS: brew install git-lfs")
    print("      Linux: sudo apt install git-lfs")
    print("      Windows: 下载 https://git-lfs.github.com/")
    exit(1)
print(f"   ✅ Git LFS已安装: {result.stdout.strip()}")
print()

# 步骤2: 初始化Git LFS
print("🔧 [2/6] 初始化Git LFS...")
if not run_cmd(['git', 'lfs', 'install'], cwd=parent_dir):
    print("   ⚠️  初始化失败，可能已经初始化过")
print()

# 步骤3: 跟踪.parquet文件
print("📝 [3/6] 设置跟踪.parquet文件...")
# 检查是否已跟踪
result = subprocess.run(
    ['git', 'lfs', 'track'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

if '*.parquet' not in result.stdout:
    run_cmd(['git', 'lfs', 'track', '*.parquet'], cwd=parent_dir)
    run_cmd(['git', 'lfs', 'track', 'data/historical_backup/*.parquet'], cwd=parent_dir)
    print("   ✅ 已设置跟踪.parquet文件")
else:
    print("   ✅ .parquet文件已在跟踪列表中")
print()

# 步骤4: 更新.gitignore（移除historical_backup的排除）
print("📄 [4/6] 更新.gitignore...")
gitignore_path = gamium_dir / '.gitignore'
if gitignore_path.exists():
    content = gitignore_path.read_text()
    # 移除data/historical_backup/的排除
    new_content = content.replace('data/historical_backup/', '# data/historical_backup/  # 使用Git LFS管理')
    if new_content != content:
        gitignore_path.write_text(new_content)
        print("   ✅ 已更新.gitignore，允许推送historical_backup目录")
    else:
        print("   ✅ .gitignore已正确配置")
else:
    print("   ⚠️  .gitignore不存在")
print()

# 步骤5: 添加文件
print("📦 [5/6] 添加大文件...")
print("   注意：大文件会使用Git LFS，上传可能需要一些时间")
run_cmd(['git', 'add', 'Gamium-FinanceAI/.gitattributes'], cwd=parent_dir)
run_cmd(['git', 'add', 'Gamium-FinanceAI/.gitignore'], cwd=parent_dir)
run_cmd(['git', 'add', 'Gamium-FinanceAI/data/historical_backup/'], cwd=parent_dir)
print()

# 步骤6: 检查状态
print("📋 [6/6] 检查状态...")
run_cmd(['git', 'status', '--short'], cwd=parent_dir)
print()

print("=" * 60)
print("✅ Git LFS设置完成！")
print()
print("📝 下一步操作:")
print("   1. 检查状态: git status")
print("   2. 提交更改: git commit -m 'feat: 添加大文件数据（使用Git LFS）'")
print("   3. 推送到GitHub: git push origin main")
print()
print("💡 注意:")
print("   - 大文件上传可能需要较长时间")
print("   - 确保GitHub账户有足够的LFS配额（免费账户1GB）")
print("   - 如果上传失败，可能需要增加LFS配额或使用其他存储方案")
print("=" * 60)


