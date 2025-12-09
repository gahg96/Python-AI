#!/usr/bin/env python3
"""同步代码到GitHub"""

import subprocess
import os
import sys
import tempfile
from pathlib import Path

def run_cmd(cmd_list, cwd=None):
    """执行命令（使用列表而不是shell）"""
    cmd_str = ' '.join(cmd_list)
    print(f"🔹 执行: {cmd_str}")
    try:
        result = subprocess.run(
            cmd_list,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"⚠️  警告: {result.stderr}", file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        return False

def main():
    # 进入项目目录
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    print(f"📁 项目目录: {project_dir}")
    print()
    
    # 检查是否在git仓库中
    if not (project_dir / ".git").exists():
        print("❌ 错误: 当前目录不是git仓库")
        print("💡 请先初始化git仓库")
        return 1
    
    # 检查git状态
    print("📊 检查git状态...")
    run_cmd(['git', 'status', '--short'], cwd=project_dir)
    print()
    
    # 添加所有更改
    print("📦 添加所有更改...")
    if not run_cmd(['git', 'add', '-A'], cwd=project_dir):
        print("❌ 添加文件失败")
        return 1
    print()
    
    # 检查是否有更改
    result = subprocess.run(
        ['git', 'diff', '--cached', '--quiet'],
        cwd=project_dir,
        capture_output=True
    )
    if result.returncode == 0:
        print("✅ 没有需要提交的更改")
        return 0
    
    # 提交更改
    print("📝 提交更改...")
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
- 优化数据生成脚本，支持分块合并避免内存溢出
- 添加数据状态检查脚本"""
    
    # 使用临时文件提交
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write(commit_msg)
        temp_file = f.name
    
    try:
        if not run_cmd(['git', 'commit', '-F', temp_file], cwd=project_dir):
            print("❌ 提交失败")
            return 1
    finally:
        os.unlink(temp_file)
    print()
    
    # 推送到GitHub
    print("📤 推送到GitHub...")
    # 先尝试main分支
    if not run_cmd(['git', 'push', 'origin', 'main'], cwd=project_dir):
        # 如果main失败，尝试master
        print("💡 尝试master分支...")
        if not run_cmd(['git', 'push', 'origin', 'master'], cwd=project_dir):
            print("❌ 推送失败")
            print("💡 请手动执行: git push origin main")
            return 1
    
    print()
    print("✅ 完成！代码已同步到GitHub")
    print()
    
    # 显示最近5条提交记录
    print("📋 最近提交记录:")
    run_cmd(['git', 'log', '--oneline', '-5'], cwd=project_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
