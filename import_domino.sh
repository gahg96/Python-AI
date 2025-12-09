#!/bin/bash

# DominoDB数据库导入脚本
# 将导出的SQL文件导入到本地OceanBase数据库

set -e

echo "=========================================="
echo "DominoDB数据库导入工具"
echo "=========================================="

# 配置区域
LOCAL_HOST="${OCEANBASE_HOST:-127.0.0.1}"
LOCAL_PORT="${OCEANBASE_PORT:-2881}"
DB_USER="${OCEANBASE_USER:-root@sys}"
DB_NAME="DominoDB"

# 检查mysql命令
if ! command -v mysql &> /dev/null; then
    echo "❌ 错误: 未找到mysql命令"
    echo "   请安装MySQL客户端工具"
    exit 1
fi

# 检查SQL文件
if [ -z "$1" ]; then
    echo "用法: $0 <SQL文件路径>"
    echo ""
    echo "示例:"
    echo "  $0 backups/domino_backup_20231123.sql"
    echo "  $0 backups/domino_backup_20231123.sql.gz"
    exit 1
fi

SQL_FILE="$1"

if [ ! -f "$SQL_FILE" ]; then
    echo "❌ 错误: 文件不存在: $SQL_FILE"
    exit 1
fi

echo ""
echo "📋 导入配置:"
echo "   数据库: $LOCAL_HOST:$LOCAL_PORT"
echo "   数据库名: $DB_NAME"
echo "   用户: $DB_USER"
echo "   SQL文件: $SQL_FILE"
echo ""

# 提示输入密码
read -sp "请输入数据库密码: " DB_PASS
echo ""

# 检查数据库是否存在
echo ""
echo "🔍 检查数据库连接..."
mysql -h $LOCAL_HOST -P $LOCAL_PORT -u $DB_USER -p$DB_PASS -e "SELECT 1;" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ 数据库连接失败，请检查配置"
    exit 1
fi

echo "✅ 数据库连接成功"
echo ""

# 询问是否创建数据库
DB_EXISTS=$(mysql -h $LOCAL_HOST -P $LOCAL_PORT -u $DB_USER -p$DB_PASS -e "SHOW DATABASES LIKE '$DB_NAME';" 2>/dev/null | grep -c "$DB_NAME" || true)

if [ "$DB_EXISTS" -eq 0 ]; then
    echo "⚠️  数据库 $DB_NAME 不存在"
    read -p "是否创建数据库? (y/n) [y]: " create_db
    create_db=${create_db:-y}
    
    if [ "$create_db" = "y" ]; then
        echo "📦 创建数据库..."
        mysql -h $LOCAL_HOST -P $LOCAL_PORT -u $DB_USER -p$DB_PASS -e "CREATE DATABASE IF NOT EXISTS $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;" 2>/dev/null
        echo "✅ 数据库创建成功"
    else
        echo "❌ 取消导入"
        exit 1
    fi
else
    echo "⚠️  数据库 $DB_NAME 已存在"
    read -p "是否清空现有数据? (y/n) [n]: " clear_db
    clear_db=${clear_db:-n}
    
    if [ "$clear_db" = "y" ]; then
        echo "🗑️  清空现有数据..."
        mysql -h $LOCAL_HOST -P $LOCAL_PORT -u $DB_USER -p$DB_PASS -e "DROP DATABASE IF EXISTS $DB_NAME; CREATE DATABASE $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;" 2>/dev/null
        echo "✅ 数据已清空"
    fi
fi

echo ""
echo "📥 开始导入数据..."

# 判断文件类型并导入
if [[ $SQL_FILE == *.gz ]]; then
    echo "   检测到压缩文件，正在解压并导入..."
    gunzip -c $SQL_FILE | mysql -h $LOCAL_HOST -P $LOCAL_PORT -u $DB_USER -p$DB_PASS --default-character-set=utf8mb4 2>/dev/null
else
    echo "   正在导入SQL文件..."
    mysql -h $LOCAL_HOST -P $LOCAL_PORT -u $DB_USER -p$DB_PASS --default-character-set=utf8mb4 < $SQL_FILE 2>/dev/null
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 导入成功!"
    echo ""
    
    # 验证数据
    echo "📊 数据统计:"
    mysql -h $LOCAL_HOST -P $LOCAL_PORT -u $DB_USER -p$DB_PASS $DB_NAME -e "
        SELECT 
            (SELECT COUNT(*) FROM T_NCB_RULES) as '规章制度数量',
            (SELECT COUNT(*) FROM T_ATTACHMENT_REF) as '附件数量',
            (SELECT COUNT(*) FROM T_MIGRATION_LOG) as '迁移记录数';
    " 2>/dev/null
    
    echo ""
    echo "=========================================="
    echo "✅ 导入完成！"
    echo ""
    echo "现在可以:"
    echo "  1. 使用DBeaver连接到本地数据库查看数据"
    echo "  2. 启动前端展示系统进行测试"
    echo "  3. 访问 http://localhost:3000 查看界面"
    echo "=========================================="
else
    echo ""
    echo "❌ 导入失败，请检查:"
    echo "   1. SQL文件是否完整"
    echo "   2. 数据库权限是否足够"
    echo "   3. 字符编码是否正确"
    exit 1
fi

