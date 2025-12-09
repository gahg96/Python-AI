#!/usr/bin/env python3
"""测试云存储数据访问"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cloud_storage import DataLoader, load_data

print("=" * 60)
print("🧪 测试云存储数据访问")
print("=" * 60)
print()

# 测试1: 检查配置
print("📋 [1/3] 检查配置...")
try:
    loader = DataLoader()
    print("   ✅ 数据加载器初始化成功")
    print(f"   📁 配置: {loader.config if hasattr(loader, 'config') else '默认配置'}")
except Exception as e:
    print(f"   ❌ 初始化失败: {e}")
    print("   💡 请检查 config/data_sources.yaml 配置")
print()

# 测试2: 测试加载小文件（macro_economics）
print("📦 [2/3] 测试加载宏观数据（小文件）...")
try:
    macro = load_data('macro_economics', use_cache=True)
    print(f"   ✅ 加载成功: {len(macro)} 条记录")
    print(f"   📊 列: {list(macro.columns)[:5]}...")
except Exception as e:
    print(f"   ⚠️  加载失败: {e}")
    print("   💡 这是正常的，如果本地文件不存在且未配置云存储")
print()

# 测试3: 测试数据加载器集成
print("📚 [3/3] 测试数据加载器集成...")
try:
    from src.data_distillation.data_loader import load_historical_data
    
    print("   尝试从本地加载...")
    loader = load_historical_data("data/historical_backup", use_cloud_storage=True)
    loader.load(sample_size=100)  # 只加载100条作为测试
    
    if loader.customers is not None:
        print(f"   ✅ 客户数据: {len(loader.customers)} 条")
    if loader.loans is not None:
        print(f"   ✅ 贷款数据: {len(loader.loans)} 条")
    if loader.macro is not None:
        print(f"   ✅ 宏观数据: {len(loader.macro)} 条")
        
except Exception as e:
    print(f"   ⚠️  测试失败: {e}")
    print("   💡 这是正常的，如果数据文件不存在")
print()

print("=" * 60)
print("✅ 测试完成")
print()
print("💡 使用说明:")
print("   1. 配置 config/data_sources.yaml")
print("   2. 上传大文件到云存储（Google Drive/Dropbox）")
print("   3. 更新配置文件中的文件ID或链接")
print("   4. 代码会自动下载并缓存文件")
print()
print("📖 详细文档: docs/云存储数据访问指南.md")
print("=" * 60)

