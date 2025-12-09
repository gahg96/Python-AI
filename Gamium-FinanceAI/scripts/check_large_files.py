#!/usr/bin/env python3
"""检查大文件大小"""

import os
from pathlib import Path

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

# 检查historical_backup目录
backup_dir = Path('data/historical_backup')
if backup_dir.exists():
    print("📊 检查 data/historical_backup/ 目录:")
    print()
    
    large_files = []
    total_size = 0
    
    for file in backup_dir.glob('*.parquet'):
        size = file.stat().st_size
        total_size += size
        large_files.append((file.name, size))
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   文件数: {len(large_files)}")
    print(f"   总大小: {format_size(total_size)}")
    print()
    print("   文件列表:")
    for name, size in large_files:
        print(f"      {format_size(size):>10} - {name}")
        if size > 100 * 1024 * 1024:  # 大于100MB
            print(f"         ⚠️  超过GitHub 100MB限制，需要使用Git LFS")
    print()
    
    if total_size > 0:
        print("💡 GitHub限制:")
        print("   - 单个文件最大: 100MB")
        print("   - 仓库总大小: 建议不超过1GB（免费账户）")
        print("   - 大文件需要使用: Git LFS (Large File Storage)")
        print()
        print("📦 建议:")
        if any(size > 100 * 1024 * 1024 for _, size in large_files):
            print("   1. 安装Git LFS: brew install git-lfs")
            print("   2. 初始化Git LFS: git lfs install")
            print("   3. 跟踪.parquet文件: git lfs track '*.parquet'")
            print("   4. 添加并提交")
        else:
            print("   ✅ 所有文件都在100MB以下，可以直接推送")
else:
    print("❌ data/historical_backup/ 目录不存在")


