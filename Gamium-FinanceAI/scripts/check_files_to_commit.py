#!/usr/bin/env python3
"""检查将要提交的文件和大小"""

import subprocess
import os
from pathlib import Path

gamium_dir = Path(__file__).parent.parent.absolute()
parent_dir = gamium_dir.parent

print("=" * 60)
print("📊 检查将要提交的文件")
print("=" * 60)
print()

# 检查git状态
print("1. 检查git状态...")
result = subprocess.run(
    ['git', 'status', '--short', 'Gamium-FinanceAI/'],
    cwd=parent_dir,
    capture_output=True,
    text=True
)

if result.stdout.strip():
    lines = result.stdout.strip().split('\n')
    print(f"   发现 {len(lines)} 个文件有更改")
else:
    print("   (无更改)")
    exit(0)
print()

# 计算文件大小
print("2. 计算文件大小...")
total_size = 0
file_count = 0
large_files = []

for line in result.stdout.strip().split('\n'):
    if not line.strip():
        continue
    
    # 解析git status输出
    status = line[:2].strip()
    file_path_str = line[3:].strip()
    
    if file_path_str.startswith('"') and file_path_str.endswith('"'):
        file_path_str = file_path_str[1:-1]
    
    file_path = parent_dir / file_path_str
    
    if file_path.exists():
        try:
            size = file_path.stat().st_size
            total_size += size
            file_count += 1
            if size > 10 * 1024 * 1024:  # 大于10MB
                large_files.append((file_path_str, size))
        except:
            pass

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

print(f"   ✅ 文件数: {file_count}")
print(f"   ✅ 总大小: {format_size(total_size)} ({total_size/1024/1024:.2f} MB)")

if large_files:
    print(f"\n   ⚠️  发现 {len(large_files)} 个大文件 (>10MB):")
    for path, size in large_files[:10]:
        print(f"      {format_size(size):>10} - {path}")
    if len(large_files) > 10:
        print(f"      ... 还有 {len(large_files) - 10} 个大文件")
else:
    print("   ✅ 没有大文件，提交应该很快")

print()
print("=" * 60)
print("💡 如果文件很大，提交可能会慢")
print("   建议检查.gitignore是否正确排除了数据文件")
print("=" * 60)


