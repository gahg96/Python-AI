#!/usr/bin/env python3
"""只同步Gamium-FinanceAI目录的代码到GitHub（带进度显示）"""

import subprocess
import os
import tempfile
import time
from pathlib import Path

# Gamium-FinanceAI目录
gamium_dir = Path(__file__).parent.parent.absolute()
parent_dir = gamium_dir.parent

print("=" * 60)
print("🚀 同步Gamium-FinanceAI代码到GitHub（带进度显示）")
print("=" * 60)
print()
print(f"📁 Gamium-FinanceAI目录: {gamium_dir}")
print(f"📁 Git仓库根目录: {parent_dir}")
print()

# 检查git仓库位置
if not (parent_dir / ".git").exists():
    print("❌ 错误: 未找到git仓库")
    exit(1)

print("✅ 检测到git仓库在上级目录")
print()

# 步骤1: 检查当前状态
print("📊 [1/5] 检查当前状态...")
result = subprocess.run(
    ['git', 'status', '--short', 'Gamium-FinanceAI/'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)
if result.stdout.strip():
    lines = result.stdout.strip().split('\n')
    print(f"   发现 {len(lines)} 个文件有更改")
    # 显示前10个文件
    for line in lines[:10]:
        print(f"   {line}")
    if len(lines) > 10:
        print(f"   ... 还有 {len(lines) - 10} 个文件")
else:
    print("   (无更改)")
print()

# 步骤2: 检查要添加的文件大小（排除.gitignore的文件）
print("📏 [2/5] 检查要添加的文件大小...")
print("   正在扫描文件（排除.gitignore中的文件）...")

# 使用git check-ignore来检查哪些文件会被忽略
result = subprocess.run(
    ['git', 'check-ignore', '-v', 'Gamium-FinanceAI/'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

# 计算实际要添加的文件大小
total_size = 0
file_count = 0
large_files = []

for root, dirs, files in os.walk(gamium_dir):
    # 跳过.git目录和temp目录
    dirs[:] = [d for d in dirs if d != '.git' and d != 'temp' and not d.startswith('__pycache__')]
    
    for file in files:
        file_path = Path(root) / file
        rel_path = file_path.relative_to(parent_dir)
        
        # 检查是否被gitignore忽略
        check_result = subprocess.run(
            ['git', 'check-ignore', str(rel_path)],
            cwd=parent_dir,
            capture_output=True
        )
        
        if check_result.returncode != 0:  # 不被忽略
            try:
                size = file_path.stat().st_size
                total_size += size
                file_count += 1
                if size > 10 * 1024 * 1024:  # 大于10MB
                    large_files.append((str(rel_path), size))
            except:
                pass

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

print(f"   ✅ 将添加 {file_count} 个文件")
print(f"   ✅ 总大小: {format_size(total_size)} ({total_size/1024/1024:.2f} MB)")
if large_files:
    print(f"   ⚠️  发现 {len(large_files)} 个大文件 (>10MB):")
    for path, size in large_files[:5]:
        print(f"      {path}: {format_size(size)}")
    if len(large_files) > 5:
        print(f"      ... 还有 {len(large_files) - 5} 个大文件")
print()

# 步骤3: 添加文件
print("📦 [3/5] 添加Gamium-FinanceAI目录下的所有更改...")
print("   正在执行 git add Gamium-FinanceAI/ ...")
start_time = time.time()

result = subprocess.run(
    ['git', 'add', 'Gamium-FinanceAI/'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

elapsed = time.time() - start_time
if result.returncode == 0:
    print(f"   ✅ 添加完成（耗时 {elapsed:.2f} 秒）")
else:
    print(f"   ❌ 添加失败: {result.stderr}")
    exit(1)
print()

# 步骤4: 确认要提交的文件
print("📋 [4/5] 确认要提交的文件...")
result = subprocess.run(
    ['git', 'status', '--short'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

gamium_files = []
other_files = []
for line in result.stdout.split('\n'):
    if line.strip():
        if 'Gamium-FinanceAI/' in line:
            gamium_files.append(line)
        elif not line.startswith('?? ../'):
            other_files.append(line)

print(f"   ✅ Gamium-FinanceAI文件: {len(gamium_files)} 个")
if other_files:
    print(f"   ⚠️  其他文件: {len(other_files)} 个（这些不会被提交）")
    for line in other_files[:5]:
        print(f"      {line}")
else:
    print("   ✅ 确认：只包含Gamium-FinanceAI目录的文件")
print()

# 检查是否有更改
result = subprocess.run(
    ['git', 'diff', '--cached', '--quiet'],
    cwd=parent_dir
)
if result.returncode == 0:
    print("✅ 没有需要提交的更改")
    exit(0)

# 步骤5: 提交
print("📝 [5/5] 提交更改...")
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
    print("   正在提交...")
    start_time = time.time()
    result = subprocess.run(
        ['git', 'commit', '-F', temp_file],
        cwd=parent_dir,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"   ❌ 提交失败: {result.stderr}")
        exit(1)
    print(f"   ✅ 提交成功（耗时 {elapsed:.2f} 秒）")
    if result.stdout:
        print(f"   {result.stdout.strip()}")
finally:
    os.unlink(temp_file)
print()

# 步骤6: 推送
print("📤 [6/6] 推送到GitHub...")
print("   正在推送...")
start_time = time.time()

result = subprocess.run(
    ['git', 'push', 'origin', 'main'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

elapsed = time.time() - start_time

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
        print("   💡 请检查:")
        print("      1. 是否已设置远程仓库: git remote -v")
        print("      2. 是否有推送权限")
        print("      3. 分支名称是否正确")
        exit(1)

print(f"   ✅ 推送成功（耗时 {elapsed:.2f} 秒）")
if result.stdout:
    print(f"   {result.stdout.strip()}")
print()

print("=" * 60)
print("✅ 完成！Gamium-FinanceAI代码已同步到GitHub")
print("=" * 60)
print()

# 显示最近5条提交记录
print("📋 最近提交记录:")
subprocess.run(['git', 'log', '--oneline', '-5'], cwd=parent_dir)


