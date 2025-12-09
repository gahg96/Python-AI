"""云存储数据访问模块"""

import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

class CloudStorageDownloader:
    """从云存储下载数据的工具类"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        初始化下载器
        
        Args:
            cache_dir: 本地缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, filename: str, force: bool = False) -> Path:
        """
        从URL下载文件
        
        Args:
            url: 文件下载链接
            filename: 本地文件名
            force: 是否强制重新下载
            
        Returns:
            下载后的文件路径
        """
        file_path = self.cache_dir / filename
        
        # 如果文件已存在且不强制下载，直接返回
        if file_path.exists() and not force:
            print(f"✅ 使用缓存文件: {file_path}")
            return file_path
        
        print(f"📥 正在下载: {filename}")
        print(f"   来源: {url}")
        
        try:
            # 下载文件（支持大文件）
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r   进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
            
            print(f"\n✅ 下载完成: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            if file_path.exists():
                file_path.unlink()
            raise
    
    def download_google_drive(self, file_id: str, filename: str, force: bool = False) -> Path:
        """
        从Google Drive下载文件
        
        Args:
            file_id: Google Drive文件ID
            filename: 本地文件名
            force: 是否强制重新下载
            
        Returns:
            下载后的文件路径
        """
        # Google Drive直接下载链接格式
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return self.download_file(url, filename, force)
    
    def download_dropbox(self, share_link: str, filename: str, force: bool = False) -> Path:
        """
        从Dropbox下载文件
        
        Args:
            share_link: Dropbox分享链接
            filename: 本地文件名
            force: 是否强制重新下载
            
        Returns:
            下载后的文件路径
        """
        # 将分享链接转换为直接下载链接
        if '?dl=0' in share_link:
            url = share_link.replace('?dl=0', '?dl=1')
        else:
            url = share_link + ('&' if '?' in share_link else '?') + 'dl=1'
        
        return self.download_file(url, filename, force)
    
    def load_parquet(self, file_path: Path) -> pd.DataFrame:
        """
        加载Parquet文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            DataFrame
        """
        print(f"📖 正在加载: {file_path}")
        return pd.read_parquet(file_path)


class DataLoader:
    """数据加载器，支持从云存储或本地加载"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据加载器
        
        Args:
            config: 数据源配置
        """
        self.config = config or self._load_default_config()
        self.downloader = CloudStorageDownloader()
    
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        config_path = Path("config/data_sources.yaml")
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                print("⚠️  警告: PyYAML未安装，无法加载配置文件")
                return {}
        return {}
    
    def get_customers(self, use_cache: bool = True) -> pd.DataFrame:
        """获取客户数据"""
        return self._load_data('customers', use_cache)
    
    def get_loan_applications(self, use_cache: bool = True) -> pd.DataFrame:
        """获取贷款申请数据"""
        return self._load_data('loan_applications', use_cache)
    
    def get_repayment_history(self, use_cache: bool = True) -> pd.DataFrame:
        """获取还款历史数据"""
        return self._load_data('repayment_history', use_cache)
    
    def get_macro_economics(self, use_cache: bool = True) -> pd.DataFrame:
        """获取宏观经济数据"""
        return self._load_data('macro_economics', use_cache)
    
    def _load_data(self, data_name: str, use_cache: bool = True) -> pd.DataFrame:
        """
        加载数据（从云存储或本地）
        
        Args:
            data_name: 数据名称
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame
        """
        # 1. 先检查本地文件
        local_path = Path(f"data/historical_backup/{data_name}.parquet")
        if local_path.exists():
            print(f"✅ 使用本地文件: {local_path}")
            return self.downloader.load_parquet(local_path)
        
        # 2. 检查缓存
        cache_path = self.downloader.cache_dir / f"{data_name}.parquet"
        if cache_path.exists() and use_cache:
            print(f"✅ 使用缓存文件: {cache_path}")
            return self.downloader.load_parquet(cache_path)
        
        # 3. 从云存储下载
        if data_name in self.config:
            source = self.config[data_name]
            source_type = source.get('type', 'url')
            
            if source_type == 'google_drive':
                file_id = source['file_id']
                file_path = self.downloader.download_google_drive(
                    file_id, f"{data_name}.parquet", force=not use_cache
                )
            elif source_type == 'dropbox':
                share_link = source['share_link']
                file_path = self.downloader.download_dropbox(
                    share_link, f"{data_name}.parquet", force=not use_cache
                )
            elif source_type == 'url':
                url = source['url']
                file_path = self.downloader.download_file(
                    url, f"{data_name}.parquet", force=not use_cache
                )
            else:
                raise ValueError(f"不支持的数据源类型: {source_type}")
            
            return self.downloader.load_parquet(file_path)
        
        # 4. 如果都没有，抛出错误
        raise FileNotFoundError(
            f"无法找到数据文件: {data_name}.parquet\n"
            f"请检查:\n"
            f"  1. 本地文件: {local_path}\n"
            f"  2. 配置文件: config/data_sources.yaml\n"
            f"  3. 或使用脚本生成数据: python3 scripts/generate_dataset.py"
        )


# 便捷函数
def load_data(data_name: str, use_cache: bool = True) -> pd.DataFrame:
    """
    便捷函数：加载数据
    
    Args:
        data_name: 数据名称 (customers, loan_applications, repayment_history, macro_economics)
        use_cache: 是否使用缓存
        
    Returns:
        DataFrame
    """
    loader = DataLoader()
    method_map = {
        'customers': loader.get_customers,
        'loan_applications': loader.get_loan_applications,
        'repayment_history': loader.get_repayment_history,
        'macro_economics': loader.get_macro_economics,
    }
    
    if data_name not in method_map:
        raise ValueError(f"未知的数据名称: {data_name}")
    
    return method_map[data_name](use_cache)

