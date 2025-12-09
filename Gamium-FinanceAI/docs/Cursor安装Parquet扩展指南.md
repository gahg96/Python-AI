# 在 Cursor 中安装 Parquet Viewer 扩展

## 方法一：通过扩展市场安装（推荐）

### 步骤：

1. **打开扩展面板**
   - 快捷键：`Cmd + Shift + X` (Mac) 或 `Ctrl + Shift + X` (Windows/Linux)
   - 或者点击左侧边栏的扩展图标（四个方块）

2. **搜索扩展**
   - 在搜索框中输入：`Parquet Viewer`
   - 或者搜索：`parquet`

3. **安装扩展**
   - 找到 "Parquet Viewer" 扩展（作者通常是 `RandomFractalsInc` 或类似）
   - 点击 "Install" 按钮

4. **验证安装**
   - 安装完成后，尝试打开一个 `.parquet` 文件
   - 应该能看到格式化的表格视图

## 方法二：通过命令面板安装

1. **打开命令面板**
   - 快捷键：`Cmd + Shift + P` (Mac) 或 `Ctrl + Shift + P` (Windows/Linux)

2. **输入命令**
   - 输入：`Extensions: Install Extensions`
   - 回车

3. **搜索并安装**
   - 在搜索框中输入：`Parquet Viewer`
   - 选择并安装

## 推荐的 Parquet 扩展

### 1. **Parquet Viewer** (推荐)
- **扩展ID**: `RandomFractalsInc.vscode-parquet-viewer`
- **功能**: 
  - 表格视图
  - Schema 查看
  - 数据导出
  - 支持大文件

### 2. **Parquet** (备选)
- **扩展ID**: `dvirgiln.vscode-parquet`
- **功能**: 轻量级 Parquet 查看器

## 使用方法

安装完成后：

1. **直接打开文件**
   - 在 Cursor 中双击 `.parquet` 文件
   - 会自动以表格形式显示

2. **右键菜单**
   - 右键点击 `.parquet` 文件
   - 选择 "Open With..." → "Parquet Viewer"

## 验证安装

运行以下命令测试：

```bash
# 在项目目录中
python3 scripts/view_parquet.py data/historical/temp/customers_0000.parquet --head 5
```

然后在 Cursor 中打开 `data/historical/temp/customers_0000.parquet` 文件，应该能看到格式化的表格。

## 如果遇到问题

1. **扩展未显示**
   - 确保 Cursor 已更新到最新版本
   - 尝试重启 Cursor

2. **无法打开文件**
   - 检查文件路径是否正确
   - 确保文件格式是 `.parquet`

3. **性能问题**
   - 大文件可能加载较慢
   - 建议使用我们的脚本查看大文件

## 替代方案

如果扩展安装有问题，可以使用：

1. **我们的查看脚本**（已提供）
   ```bash
   python3 scripts/view_parquet.py <文件路径>
   ```

2. **Python + pandas**（最灵活）
   ```python
   import pandas as pd
   df = pd.read_parquet('文件路径')
   df.head()
   ```

3. **DBeaver**（图形界面工具）
   - 下载：https://dbeaver.io/

