#!/usr/bin/env python3
"""提交代码到GitHub"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """执行命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return False

def main():
    repo_dir = Path(__file__).parent.parent
    
    print("📦 准备提交代码到GitHub...")
    print()
    
    # 添加所有更改
    print("1. 添加文件...")
    if not run_cmd("git add -A", cwd=repo_dir):
        print("❌ 添加文件失败")
        return
    
    # 显示状态
    print()
    print("2. 文件状态:")
    run_cmd("git status --short", cwd=repo_dir)
    
    # 提交
    print()
    print("3. 提交更改...")
    commit_msg = """feat: 完善模型评估和风险因子说明功能

- 添加模型评估术语详解页面（HTML和Markdown）
- 添加风险因子确定方法详解文档
- 添加LTV生命周期价值详解文档
- 在客户预测界面添加模型评估指标说明
- 在客户画像中添加LTV详细说明弹窗
- 修复术语解释页面文字颜色对比度问题
- 添加客户信用评分预测脚本
- 添加模型评估脚本
- 添加数据提取和特征工程脚本
- 添加训练模型脚本
- 添加示例特征文件生成脚本
- 添加Parquet文件查看工具
- 更新Web界面，添加系统架构和术语解释链接
- 优化数据生成脚本，支持分块合并避免内存溢出"""
    
    if not run_cmd(f'git commit -m "{commit_msg}"', cwd=repo_dir):
        print("❌ 提交失败")
        return
    
    # 推送到GitHub
    print()
    print("4. 推送到GitHub...")
    if not run_cmd("git push origin main", cwd=repo_dir):
        print("❌ 推送失败")
        return
    
    print()
    print("✅ 代码已同步到GitHub！")

if __name__ == '__main__':
    main()

