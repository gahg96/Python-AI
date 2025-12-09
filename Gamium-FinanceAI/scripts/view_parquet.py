#!/usr/bin/env python3
"""
Parquet文件快速查看工具

用法:
    python3 view_parquet.py <文件路径> [选项]

选项:
    --head N        显示前N行 (默认: 10)
    --schema        只显示schema
    --stats         显示统计信息
    --info          显示文件信息
    --columns       只显示列名
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def show_file_info(file_path):
    """显示文件基本信息"""
    file = Path(file_path)
    if not file.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    size = file.stat().st_size
    print(f"📄 文件: {file.name}")
    print(f"   路径: {file.absolute()}")
    print(f"   大小: {format_size(size)}")
    return True

def show_schema(file_path):
    """显示schema"""
    try:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema
        print("\n📋 Schema:")
        print("=" * 60)
        for i, field in enumerate(schema):
            field_type = str(field.physical_type) if hasattr(field, 'physical_type') else str(type(field))
            print(f"  {i+1}. {field.name:30s} {field_type}")
        print("=" * 60)
    except Exception as e:
        # 如果pyarrow读取失败，尝试用pandas
        try:
            df = pd.read_parquet(file_path, nrows=0)  # 只读schema，不读数据
            print("\n📋 Schema (通过pandas):")
            print("=" * 60)
            for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
                print(f"  {i}. {col:30s} {str(dtype)}")
            print("=" * 60)
        except Exception as e2:
            print(f"❌ 读取schema失败: {e}")

def show_metadata(file_path):
    """显示元数据"""
    try:
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        num_rows = metadata.num_rows
        num_columns = len(parquet_file.schema)
        num_row_groups = metadata.num_row_groups
        
        print(f"\n📊 文件元数据:")
        print("=" * 60)
        print(f"  行数: {num_rows:,}")
        print(f"  列数: {num_columns}")
        print(f"  Row Groups: {num_row_groups}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 读取元数据失败: {e}")

def show_head(file_path, n=10):
    """显示前N行"""
    try:
        df = pd.read_parquet(file_path)
        print(f"\n📋 前 {n} 行数据:")
        print("=" * 60)
        print(df.head(n).to_string())
        print("=" * 60)
        print(f"\n总行数: {len(df):,}")
        print(f"总列数: {len(df.columns)}")
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")

def show_stats(file_path):
    """显示统计信息"""
    try:
        df = pd.read_parquet(file_path)
        print(f"\n📊 统计信息:")
        print("=" * 60)
        print(df.describe())
        print("=" * 60)
    except Exception as e:
        print(f"❌ 生成统计信息失败: {e}")

def show_columns(file_path):
    """显示列名"""
    try:
        df = pd.read_parquet(file_path)
        print(f"\n📋 列名 ({len(df.columns)} 列):")
        print("=" * 60)
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            print(f"  {i:3d}. {col:30s} ({dtype})")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 读取列信息失败: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Parquet文件快速查看工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('file', help='Parquet文件路径')
    parser.add_argument('--head', type=int, default=10, help='显示前N行 (默认: 10)')
    parser.add_argument('--schema', action='store_true', help='只显示schema')
    parser.add_argument('--stats', action='store_true', help='显示统计信息')
    parser.add_argument('--info', action='store_true', help='显示文件信息')
    parser.add_argument('--columns', action='store_true', help='只显示列名')
    
    args = parser.parse_args()
    
    # 显示文件信息
    if not show_file_info(args.file):
        sys.exit(1)
    
    # 根据选项显示内容
    if args.schema:
        show_schema(args.file)
        show_metadata(args.file)
    elif args.stats:
        show_stats(args.file)
    elif args.columns:
        show_columns(args.file)
    elif args.info:
        show_schema(args.file)
        show_metadata(args.file)
        show_columns(args.file)
    else:
        # 默认显示：schema + 前N行
        show_schema(args.file)
        show_metadata(args.file)
        show_head(args.file, args.head)

if __name__ == '__main__':
    main()

