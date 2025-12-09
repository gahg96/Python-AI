"""数据加载工具，支持从云存储自动下载"""

import os
from pathlib import Path
import subprocess
import sys

def ensure_data_files(data_dir=None):
    """确保数据文件存在，如果不存在则尝试下载"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'historical_backup'
    else:
        data_dir = Path(data_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查必需的文件
    required_files = [
        'customers.parquet',
        'loan_applications.parquet',
        'repayment_history.parquet',
        'macro_economics.parquet'
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = data_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"⚠️  缺少数据文件: {', '.join(missing_files)}")
        print("💡 正在尝试从云存储下载...")
        
        # 运行下载脚本
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'download_data_from_cloud.py'
        if script_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ 数据文件下载完成")
                else:
                    print(f"❌ 下载失败: {result.stderr}")
                    print("\n💡 请手动运行下载脚本:")
                    print(f"   python3 {script_path}")
                    return False
            except Exception as e:
                print(f"❌ 下载脚本执行失败: {e}")
                return False
        else:
            print("❌ 下载脚本不存在")
            print("💡 请配置数据来源并运行下载脚本")
            return False
    
    return True

def load_historical_data(data_dir=None, file_type='customers'):
    """加载历史数据文件"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'historical_backup'
    else:
        data_dir = Path(data_dir)
    
    # 确保文件存在
    if not ensure_data_files(data_dir):
        raise FileNotFoundError("数据文件不存在且下载失败")
    
    # 加载文件
    try:
        import pandas as pd
        
        file_map = {
            'customers': 'customers.parquet',
            'loans': 'loan_applications.parquet',
            'repayments': 'repayment_history.parquet',
            'macro': 'macro_economics.parquet'
        }
        
        filename = file_map.get(file_type, file_type)
        file_path = data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        print(f"📂 加载数据: {filename}")
        df = pd.read_parquet(file_path)
        print(f"   ✅ 加载完成: {len(df)} 行")
        return df
        
    except ImportError:
        raise ImportError("需要安装pandas和pyarrow: pip install pandas pyarrow")
    except Exception as e:
        raise Exception(f"加载数据失败: {e}")

if __name__ == '__main__':
    # 测试
    ensure_data_files()
    print("✅ 数据文件检查完成")


