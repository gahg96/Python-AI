#!/usr/bin/env python3
"""从云存储下载大文件数据"""

import os
import sys
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib

def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, output_path, chunk_size=8192, show_progress=True):
    """下载文件，支持断点续传"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在
    if output_path.exists():
        print(f"   ✅ 文件已存在: {output_path}")
        return True
    
    try:
        # 支持断点续传
        headers = {}
        if output_path.exists():
            headers['Range'] = f'bytes={output_path.stat().st_size}-'
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))
        if 'content-range' in response.headers:
            # 断点续传
            range_info = response.headers['content-range']
            total_size = int(range_info.split('/')[-1])
            downloaded = output_path.stat().st_size
        else:
            downloaded = 0
        
        # 下载文件
        mode = 'ab' if downloaded > 0 else 'wb'
        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        if show_progress:
            print()  # 换行
        print(f"   ✅ 下载完成: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        if output_path.exists():
            output_path.unlink()  # 删除不完整的文件
        return False

def download_from_google_drive(file_id, output_path):
    """从Google Drive下载文件（需要公开链接）"""
    # Google Drive直接下载链接格式
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # 先获取确认页面
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # 检查是否需要确认（大文件）
    if 'virus scan warning' in response.text.lower():
        # 提取确认token
        import re
        confirm_token = re.search(r'confirm=([^&]+)', response.text)
        if confirm_token:
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token.group(1)}"
    
    return download_file(url, output_path)

def download_from_dropbox(share_link, output_path):
    """从Dropbox下载文件"""
    # 将分享链接转换为直接下载链接
    # 格式: https://www.dropbox.com/s/xxxxx/file.parquet?dl=0
    # 转换为: https://www.dropbox.com/s/xxxxx/file.parquet?dl=1
    if '?dl=0' in share_link:
        download_url = share_link.replace('?dl=0', '?dl=1')
    elif '?dl=1' not in share_link:
        download_url = share_link + '?dl=1'
    else:
        download_url = share_link
    
    return download_file(download_url, output_path)

def download_from_url(url, output_path):
    """通用URL下载"""
    return download_file(url, output_path)

def main():
    """主函数"""
    print("=" * 60)
    print("📥 从云存储下载大文件数据")
    print("=" * 60)
    print()
    
    # 数据目录
    data_dir = Path(__file__).parent.parent / 'data' / 'historical_backup'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置文件路径
    config_file = Path(__file__).parent.parent / 'config' / 'data_sources.yaml'
    
    if not config_file.exists():
        print("❌ 配置文件不存在: config/data_sources.yaml")
        print("💡 请先创建配置文件，参考: config/data_sources.yaml.example")
        return 1
    
    # 读取配置
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
    except ImportError:
        print("❌ 需要安装PyYAML: pip install pyyaml")
        return 1
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")
        return 1
    
    # 下载文件
    files_to_download = config.get('files', [])
    if not files_to_download:
        print("⚠️  配置文件中没有文件列表")
        return 0
    
    print(f"📋 找到 {len(files_to_download)} 个文件需要下载")
    print()
    
    success_count = 0
    for file_info in files_to_download:
        filename = file_info.get('filename')
        source_type = file_info.get('source_type', 'url')  # url, google_drive, dropbox
        source = file_info.get('source')
        expected_hash = file_info.get('hash')  # 可选：用于验证
        
        if not filename or not source:
            print(f"⚠️  跳过无效配置: {file_info}")
            continue
        
        output_path = data_dir / filename
        
        print(f"📥 下载: {filename}")
        print(f"   来源: {source_type} - {source}")
        
        # 根据来源类型下载
        success = False
        if source_type == 'google_drive':
            success = download_from_google_drive(source, output_path)
        elif source_type == 'dropbox':
            success = download_from_dropbox(source, output_path)
        else:  # url
            success = download_from_url(source, output_path)
        
        # 验证文件（如果提供了哈希值）
        if success and expected_hash and output_path.exists():
            actual_hash = calculate_file_hash(output_path)
            if actual_hash != expected_hash:
                print(f"   ⚠️  文件哈希值不匹配！")
                print(f"      期望: {expected_hash}")
                print(f"      实际: {actual_hash}")
                output_path.unlink()  # 删除损坏的文件
                success = False
            else:
                print(f"   ✅ 文件验证通过")
        
        if success:
            success_count += 1
        
        print()
    
    print("=" * 60)
    print(f"✅ 完成！成功下载 {success_count}/{len(files_to_download)} 个文件")
    print("=" * 60)
    
    return 0 if success_count == len(files_to_download) else 1

if __name__ == '__main__':
    sys.exit(main())


