# Cursor 快速安装 Parquet Viewer 扩展

## 🚀 最简单的方法（3步）

### 步骤 1：打开扩展面板
- 按快捷键：`Cmd + Shift + X` (Mac)
- 或点击左侧边栏的 **扩展图标**（四个方块）

### 步骤 2：搜索扩展
在搜索框中输入：
```
Parquet Viewer
```

### 步骤 3：安装
找到 **"Parquet Viewer"** 扩展（作者：RandomFractalsInc），点击 **"Install"** 按钮

---

## ✅ 验证安装

安装完成后，在 Cursor 中打开一个 `.parquet` 文件：
```bash
# 在终端中打开文件
open data/historical/temp/customers_0000.parquet
```

或者在 Cursor 中：
1. 按 `Cmd + O` 打开文件
2. 选择 `data/historical/temp/customers_0000.parquet`
3. 应该能看到格式化的表格视图

---

## 📋 扩展信息

- **扩展名称**: Parquet Viewer
- **扩展ID**: `RandomFractalsInc.vscode-parquet-viewer`
- **功能**: 
  - ✅ 表格视图显示 Parquet 文件
  - ✅ 查看 Schema
  - ✅ 支持大文件
  - ✅ 数据导出

---

## 🔧 如果安装遇到问题

### 方法 1：通过命令面板
1. 按 `Cmd + Shift + P` 打开命令面板
2. 输入：`Extensions: Install Extensions`
3. 搜索：`Parquet Viewer`
4. 安装

### 方法 2：使用我们的脚本（无需扩展）
```bash
python3 scripts/view_parquet.py data/historical/temp/customers_0000.parquet
```

---

## 💡 提示

- 安装后可能需要重启 Cursor
- 大文件（>100MB）可能加载较慢，建议使用我们的脚本查看
- 如果扩展市场中没有找到，可以尝试搜索 `parquet` 关键字

