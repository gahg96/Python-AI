#!/usr/bin/env python3
"""检查数据生成状态"""

import os
import subprocess
from pathlib import Path

def get_size(path):
    """获取文件或目录大小"""
    if not os.path.exists(path):
        return 0, 0
    if os.path.isfile(path):
        size = os.path.getsize(path)
        return size, 1
    else:
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        return total_size, file_count

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def check_process():
    """检查生成进程"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if 'generate_dataset.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    return True, pid
        return False, None
    except:
        return False, None

def main():
    print("=" * 60)
    print("📊 数据生成状态检查")
    print("=" * 60)
    print()
    
    # 检查进程
    print("1. 检查生成进程:")
    is_running, pid = check_process()
    if is_running:
        print(f"   ✅ 生成进程运行中 (PID: {pid})")
    else:
        print("   ❌ 生成进程未运行")
    print()
    
    # 检查数据目录
    data_dir = Path('data/historical')
    temp_dir = data_dir / 'temp'
    
    print("2. 检查数据文件:")
    
    # 检查最终文件
    final_files = {
        'customers.parquet': data_dir / 'customers.parquet',
        'loan_applications.parquet': data_dir / 'loan_applications.parquet',
        'repayment_history.parquet': data_dir / 'repayment_history.parquet',
        'macro_economics.parquet': data_dir / 'macro_economics.parquet',
    }
    
    total_final_size = 0
    all_final_exist = True
    
    for name, path in final_files.items():
        if path.exists():
            size, _ = get_size(path)
            total_final_size += size
            print(f"   ✅ {name}: {format_size(size)}")
        else:
            print(f"   ❌ {name}: 不存在")
            all_final_exist = False
    
    print()
    print(f"   最终文件总大小: {format_size(total_final_size)}")
    print()
    
    # 检查临时文件
    print("3. 检查临时文件:")
    if temp_dir.exists():
        temp_size, temp_count = get_size(temp_dir)
        if temp_count > 0:
            print(f"   ⚠️  临时文件: {temp_count} 个文件, {format_size(temp_size)}")
            print(f"   💡 如果生成已完成，可以运行清理脚本: python3 scripts/merge_temp_files.py")
        else:
            print("   ✅ 无临时文件")
    else:
        print("   ✅ 无临时目录")
    print()
    
    # 检查summary.json
    print("4. 检查数据摘要:")
    summary_file = data_dir / 'summary.json'
    if summary_file.exists():
        import json
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"   ✅ 客户数: {summary.get('total_customers', 0):,}")
        print(f"   ✅ 贷款数: {summary.get('total_loans', 0):,}")
        print(f"   ✅ 数据大小: {summary.get('total_size_gb', 0):.2f} GB")
    else:
        print("   ❌ summary.json 不存在")
    print()
    
    # 总结
    print("=" * 60)
    print("📋 状态总结:")
    print("=" * 60)
    
    if is_running:
        print("   ⏳ 数据生成进行中...")
        print("   💡 请等待生成完成")
    elif all_final_exist and temp_count == 0:
        print("   ✅ 数据生成已完成！")
        print(f"   📦 总大小: {format_size(total_final_size)}")
        print("   💡 可以开始使用数据进行训练")
    elif all_final_exist and temp_count > 0:
        print("   ⚠️  最终文件已生成，但临时文件未清理")
        print("   💡 可以运行合并脚本: python3 scripts/merge_temp_files.py")
    else:
        print("   ❌ 数据生成未完成")
        print("   💡 请检查生成进程或重新启动生成")
    
    print("=" * 60)

if __name__ == '__main__':
    main()


