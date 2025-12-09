#!/usr/bin/env python3
"""简单检查数据状态"""

import os
import json
from pathlib import Path

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    print("=" * 60)
    print("📊 数据生成状态检查")
    print("=" * 60)
    print()
    
    data_dir = Path('data/historical')
    temp_dir = data_dir / 'temp'
    
    # 检查最终文件
    print("1. 最终数据文件:")
    final_files = {
        'customers.parquet': data_dir / 'customers.parquet',
        'loan_applications.parquet': data_dir / 'loan_applications.parquet',
        'repayment_history.parquet': data_dir / 'repayment_history.parquet',
        'macro_economics.parquet': data_dir / 'macro_economics.parquet',
    }
    
    total_size = 0
    all_exist = True
    
    for name, path in final_files.items():
        if path.exists():
            size = path.stat().st_size
            total_size += size
            print(f"   ✅ {name}: {format_size(size)}")
        else:
            print(f"   ❌ {name}: 不存在")
            all_exist = False
    
    print(f"\n   总大小: {format_size(total_size)} ({total_size/1024/1024/1024:.2f} GB)")
    print()
    
    # 检查临时文件
    print("2. 临时文件:")
    if temp_dir.exists():
        temp_files = list(temp_dir.glob('*.parquet'))
        if temp_files:
            temp_size = sum(f.stat().st_size for f in temp_files)
            customer_files = len(list(temp_dir.glob('customers_*.parquet')))
            loan_files = len(list(temp_dir.glob('loans_*.parquet')))
            repayment_files = len(list(temp_dir.glob('repayments_*.parquet')))
            
            print(f"   ⚠️  临时文件: {len(temp_files)} 个")
            print(f"      客户: {customer_files} 个")
            print(f"      贷款: {loan_files} 个")
            print(f"      还款: {repayment_files} 个")
            print(f"      大小: {format_size(temp_size)} ({temp_size/1024/1024/1024:.2f} GB)")
            print()
            print("   💡 如果生成已完成，可以运行合并:")
            print("      python3 scripts/merge_temp_files.py")
        else:
            print("   ✅ 无临时文件")
    else:
        print("   ✅ 无临时目录")
    print()
    
    # 检查summary
    print("3. 数据摘要:")
    summary_file = data_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"   ✅ 客户数: {summary.get('total_customers', 0):,}")
        print(f"   ✅ 贷款数: {summary.get('total_loans', 0):,}")
        print(f"   ✅ 还款数: {summary.get('total_repayments', 0):,}")
        print(f"   ✅ 数据大小: {summary.get('total_size_gb', 0):.2f} GB")
    else:
        print("   ❌ summary.json 不存在")
    print()
    
    # 总结
    print("=" * 60)
    print("📋 状态总结:")
    print("=" * 60)
    
    if all_exist:
        if temp_dir.exists() and list(temp_dir.glob('*.parquet')):
            print("   ⚠️  最终文件已生成，但临时文件未清理")
            print("   💡 建议运行合并脚本清理临时文件")
        else:
            print("   ✅ 数据生成已完成！")
            print(f"   📦 总大小: {format_size(total_size)} ({total_size/1024/1024/1024:.2f} GB)")
            if total_size/1024/1024/1024 < 9:
                print("   ⚠️  数据大小未达到10GB目标（当前约4.85GB）")
                print("   💡 如需更多数据，可以重新运行生成脚本")
            print("   ✅ 可以开始使用数据进行训练")
    else:
        print("   ❌ 数据生成未完成")
        print("   💡 请检查生成进程或重新启动生成")
    
    print("=" * 60)

if __name__ == '__main__':
    main()


