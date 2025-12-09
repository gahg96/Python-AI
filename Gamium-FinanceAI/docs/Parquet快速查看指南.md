# Parquet 文件快速查看指南

## 🚀 使用我们提供的脚本（最简单）

### 基本用法
```bash
# 查看文件基本信息 + 前10行
python3 scripts/view_parquet.py data/historical/customers.parquet

# 查看前20行
python3 scripts/view_parquet.py data/historical/customers.parquet --head 20

# 只查看schema（列结构）
python3 scripts/view_parquet.py data/historical/customers.parquet --schema

# 查看统计信息
python3 scripts/view_parquet.py data/historical/customers.parquet --stats

# 查看所有列名
python3 scripts/view_parquet.py data/historical/customers.parquet --columns

# 查看完整信息（schema + 元数据 + 列名）
python3 scripts/view_parquet.py data/historical/customers.parquet --info
```

## 📊 其他常用工具

### 1. Python + pandas（最灵活）
```python
import pandas as pd

# 读取文件
df = pd.read_parquet('data/historical/customers.parquet')

# 查看前几行
df.head()

# 查看基本信息
df.info()
df.describe()
```

### 2. VS Code 扩展
- 安装扩展：`Parquet Viewer`
- 直接在VS Code中双击打开.parquet文件

### 3. DBeaver（图形界面，推荐）
- 下载：https://dbeaver.io/
- 安装Parquet插件
- 图形化查看和查询

### 4. 命令行工具
```bash
# 安装parquet-tools
pip install parquet-tools

# 查看schema
parquet-tools schema data/historical/customers.parquet

# 查看前10行
parquet-tools head -n 10 data/historical/customers.parquet
```

## 💡 推荐方案

- **快速查看**：使用我们的脚本 `scripts/view_parquet.py`
- **数据分析**：Python + pandas
- **图形界面**：DBeaver
- **VS Code用户**：安装 Parquet Viewer 扩展

更多详细信息请查看：`docs/Parquet文件查看工具.md`
