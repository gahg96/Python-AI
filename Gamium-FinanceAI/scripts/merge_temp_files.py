#!/usr/bin/env python3
"""
分块合并临时文件，避免内存溢出
"""

import pandas as pd
from pathlib import Path
import sys

def merge_files_chunked(file_pattern: str, output_file: Path, chunk_size: int = 50, desc: str = ""):
    """分块合并文件，避免内存溢出"""
    files = sorted(Path('data/historical/temp').glob(file_pattern))
    if not files:
        print(f"  ⚠️  未找到文件: {file_pattern}")
        return 0
    
    print(f"  {desc} ({len(files)} 个文件，分块大小: {chunk_size})...")
    
    chunks = []
    for i in range(0, len(files), chunk_size):
        chunk_files = files[i:i+chunk_size]
        chunk = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        chunks.append(chunk)
        if (i // chunk_size + 1) % 5 == 0 or i + chunk_size >= len(files):
            print(f"    进度: {min(i+len(chunk_files), len(files))}/{len(files)}")
    
    # 最终合并
    print(f"    最终合并中...")
    result = pd.concat(chunks, ignore_index=True)
    n_records = len(result)
    
    # 保存
    result.to_parquet(output_file, index=False)
    del chunks, result  # 释放内存
    
    return n_records

if __name__ == '__main__':
    output_dir = Path('data/historical')
    temp_dir = output_dir / 'temp'
    
    if not temp_dir.exists():
        print("❌ 未找到 temp 目录")
        sys.exit(1)
    
    print("🔄 开始分块合并临时文件...")
    print("=" * 60)
    
    # 合并客户数据
    n_customers = merge_files_chunked(
        'customers_*.parquet',
        output_dir / 'customers.parquet',
        chunk_size=50,
        desc='合并客户数据'
    )
    print(f"    ✅ 完成: {n_customers:,} 客户\n")
    
    # 合并贷款数据
    n_loans = merge_files_chunked(
        'loans_*.parquet',
        output_dir / 'loan_applications.parquet',
        chunk_size=50,
        desc='合并贷款数据'
    )
    print(f"    ✅ 完成: {n_loans:,} 贷款申请\n")
    
    # 合并还款数据（使用更小的分块）
    n_repayments = merge_files_chunked(
        'repayments_*.parquet',
        output_dir / 'repayment_history.parquet',
        chunk_size=30,  # 还款文件更大，使用更小的分块
        desc='合并还款数据'
    )
    print(f"    ✅ 完成: {n_repayments:,} 还款记录\n")
    
    print("=" * 60)
    print("✅ 合并完成！")
    print(f"   客户: {n_customers:,}")
    print(f"   贷款: {n_loans:,}")
    print(f"   还款: {n_repayments:,}")

