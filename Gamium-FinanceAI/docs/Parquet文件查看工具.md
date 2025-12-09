# Parquet 文件查看工具指南

## 📋 目录
1. [Python 工具](#python-工具)
2. [命令行工具](#命令行工具)
3. [图形界面工具](#图形界面工具)
4. [VS Code 扩展](#vs-code-扩展)
5. [在线工具](#在线工具)
6. [快速查看脚本](#快速查看脚本)

---

## 🐍 Python 工具

### 1. **pandas** (最常用)
```python
import pandas as pd

# 读取整个文件
df = pd.read_parquet('data/historical/customers.parquet')

# 查看前几行
print(df.head())

# 查看基本信息
print(df.info())
print(df.describe())
```

### 2. **pyarrow** (更底层，性能更好)
```python
import pyarrow.parquet as pq

# 读取元数据
parquet_file = pq.ParquetFile('data/historical/customers.parquet')
print(parquet_file.metadata)
print(parquet_file.schema)

# 读取数据
table = parquet_file.read()
df = table.to_pandas()
```

### 3. **fastparquet** (另一个选择)
```python
import fastparquet

# 读取文件
pf = fastparquet.ParquetFile('data/historical/customers.parquet')
df = pf.to_pandas()
```

---

## 💻 命令行工具

### 1. **parquet-tools** (推荐)
```bash
# 安装
pip install parquet-tools

# 查看schema
parquet-tools schema data/historical/customers.parquet

# 查看前N行
parquet-tools head -n 10 data/historical/customers.parquet

# 查看元数据
parquet-tools meta data/historical/customers.parquet

# 查看行数
parquet-tools rowcount data/historical/customers.parquet
```

### 2. **parquet-cli**
```bash
# 安装
pip install parquet-cli

# 查看schema
parquet schema data/historical/customers.parquet

# 查看数据
parquet cat data/historical/customers.parquet | head -20
```

### 3. **duckdb** (SQL查询)
```bash
# 安装
pip install duckdb

# 使用SQL查询
duckdb -c "SELECT * FROM 'data/historical/customers.parquet' LIMIT 10"
```

---

## 🖥️ 图形界面工具

### 1. **DBeaver** (免费，推荐)
- **下载**: https://dbeaver.io/
- **特点**: 
  - 支持多种数据库格式
  - 可以安装 Parquet 插件
  - 图形化界面，易于使用
  - 支持数据导出

### 2. **DataGrip** (JetBrains，付费)
- **下载**: https://www.jetbrains.com/datagrip/
- **特点**:
  - 强大的SQL编辑器
  - 支持Parquet文件
  - 智能代码补全

### 3. **Apache Drill** (免费)
- **下载**: https://drill.apache.org/
- **特点**:
  - 专门用于查询Parquet等格式
  - 支持SQL查询
  - 需要配置

### 4. **DuckDB** (命令行 + 图形界面)
- **下载**: https://duckdb.org/
- **特点**:
  - 轻量级
  - 支持Parquet直接查询
  - 有Web界面版本

---

## 📝 VS Code 扩展

### 1. **Parquet Viewer**
- **扩展名**: `parquet-viewer`
- **功能**: 直接在VS Code中查看Parquet文件
- **安装**: VS Code扩展市场搜索 "parquet-viewer"

### 2. **Jupyter Notebook**
- 在VS Code中使用Jupyter，用pandas读取Parquet文件
- 支持交互式查看和可视化

---

## 🌐 在线工具

### 1. **Parquet Viewer Online**
- **网址**: https://parquet-viewer-online.com/
- **特点**: 上传文件在线查看（注意数据安全）

### 2. **Apache Arrow Flight SQL**
- 需要搭建服务，适合企业内部使用

---

## 🚀 快速查看脚本

我已经为您创建了一个便捷的查看脚本！

### 使用方法
```bash
# 查看文件基本信息
python3 scripts/view_parquet.py data/historical/customers.parquet

# 查看前N行
python3 scripts/view_parquet.py data/historical/customers.parquet --head 20

# 查看schema
python3 scripts/view_parquet.py data/historical/customers.parquet --schema

# 查看统计信息
python3 scripts/view_parquet.py data/historical/customers.parquet --stats
```

---

## 💡 推荐方案

### 日常使用
1. **Python + pandas** - 最灵活，适合数据分析
2. **VS Code + Parquet Viewer扩展** - 快速查看
3. **DBeaver** - 图形界面，适合非程序员

### 命令行快速查看
```bash
# 使用我们提供的脚本
python3 scripts/view_parquet.py <文件路径>
```

### 大数据查询
- **DuckDB** - 轻量级，性能好
- **Apache Drill** - 功能强大

---

## ⚠️ 注意事项

1. **大文件**: Parquet文件可能很大，建议：
   - 使用分块读取（`chunksize`参数）
   - 只读取需要的列
   - 使用条件过滤

2. **内存**: 读取整个大文件可能占用大量内存

3. **性能**: 
   - `pyarrow` 通常比 `fastparquet` 更快
   - 列式存储，只读取需要的列会更快

---

## 📚 相关资源

- Parquet格式文档: https://parquet.apache.org/
- PyArrow文档: https://arrow.apache.org/docs/python/
- Pandas文档: https://pandas.pydata.org/docs/

