#!/bin/bash

# DominoDB数据库导出脚本
# 从服务器导出OceanBase数据库到本地

set -e

echo "=========================================="
echo "DominoDB数据库导出工具"
echo "=========================================="

# 配置区域（请修改为你的实际配置）
SERVER_HOST="${OCEANBASE_HOST:-你的服务器IP}"
SERVER_PORT="${OCEANBASE_PORT:-2881}"
DB_USER="${OCEANBASE_USER:-root@sys}"
DB_NAME="DominoDB"
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 检查mysqldump是否安装
if ! command -v mysqldump &> /dev/null; then
    echo "❌ 错误: 未找到mysqldump命令"
    echo "   请安装MySQL客户端工具"
    echo "   macOS: brew install mysql-client"
    echo "   Ubuntu: sudo apt-get install mysql-client"
    exit 1
fi

# 检查服务器配置
if [ "$SERVER_HOST" = "你的服务器IP" ]; then
    echo "⚠️  请先配置服务器地址"
    echo ""
    echo "方式1: 修改脚本中的 SERVER_HOST 变量"
    echo "方式2: 使用环境变量:"
    echo "      export OCEANBASE_HOST=你的服务器IP"
    echo "      export OCEANBASE_PORT=2881"
    echo "      export OCEANBASE_USER=root@sys"
    echo ""
    read -p "请输入服务器IP: " SERVER_HOST
    read -p "请输入端口 [2881]: " input_port
    SERVER_PORT=${input_port:-2881}
    read -p "请输入用户名 [root@sys]: " input_user
    DB_USER=${input_user:-root@sys}
fi

echo ""
echo "📋 导出配置:"
echo "   服务器: $SERVER_HOST:$SERVER_PORT"
echo "   数据库: $DB_NAME"
echo "   用户: $DB_USER"
echo "   备份目录: $BACKUP_DIR"
echo ""

# 提示输入密码
read -sp "请输入数据库密码: " DB_PASS
echo ""

# 导出选项
echo ""
echo "请选择导出方式:"
echo "  1) 完整导出（结构+数据）"
echo "  2) 仅导出表结构"
echo "  3) 仅导出数据"
echo "  4) 导出特定表"
read -p "请选择 [1]: " export_type
export_type=${export_type:-1}

BACKUP_FILE="$BACKUP_DIR/domino_backup_$DATE.sql"

case $export_type in
    1)
        echo ""
        echo "📦 开始完整导出..."
        mysqldump -h $SERVER_HOST -P $SERVER_PORT -u $DB_USER -p$DB_PASS \
            --databases $DB_NAME \
            --single-transaction \
            --routines \
            --triggers \
            --events \
            --quick \
            --lock-tables=false \
            --default-character-set=utf8mb4 \
            > $BACKUP_FILE
        ;;
    2)
        echo ""
        echo "📦 开始导出表结构..."
        mysqldump -h $SERVER_HOST -P $SERVER_PORT -u $DB_USER -p$DB_PASS \
            --databases $DB_NAME \
            --no-data \
            --routines \
            --triggers \
            --default-character-set=utf8mb4 \
            > $BACKUP_FILE
        ;;
    3)
        echo ""
        echo "📦 开始导出数据..."
        mysqldump -h $SERVER_HOST -P $SERVER_PORT -u $DB_USER -p$DB_PASS \
            --databases $DB_NAME \
            --no-create-info \
            --quick \
            --lock-tables=false \
            --default-character-set=utf8mb4 \
            > $BACKUP_FILE
        ;;
    4)
        echo ""
        echo "可用的表:"
        mysql -h $SERVER_HOST -P $SERVER_PORT -u $DB_USER -p$DB_PASS $DB_NAME -e "SHOW TABLES;" 2>/dev/null | tail -n +2
        echo ""
        read -p "请输入要导出的表名（用空格分隔）: " tables
        echo ""
        echo "📦 开始导出指定表..."
        mysqldump -h $SERVER_HOST -P $SERVER_PORT -u $DB_USER -p$DB_PASS \
            $DB_NAME $tables \
            --quick \
            --lock-tables=false \
            --default-character-set=utf8mb4 \
            > $BACKUP_FILE
        ;;
    *)
        echo "❌ 无效的选择"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    # 获取文件大小
    FILE_SIZE=$(du -h $BACKUP_FILE | cut -f1)
    
    echo ""
    echo "✅ 导出成功!"
    echo "   文件: $BACKUP_FILE"
    echo "   大小: $FILE_SIZE"
    echo ""
    
    # 询问是否压缩
    read -p "是否压缩备份文件? (y/n) [y]: " compress
    compress=${compress:-y}
    
    if [ "$compress" = "y" ]; then
        echo "📦 正在压缩..."
        gzip $BACKUP_FILE
        COMPRESSED_FILE="${BACKUP_FILE}.gz"
        COMPRESSED_SIZE=$(du -h $COMPRESSED_FILE | cut -f1)
        echo "✅ 压缩完成!"
        echo "   文件: $COMPRESSED_FILE"
        echo "   大小: $COMPRESSED_SIZE"
        BACKUP_FILE=$COMPRESSED_FILE
    fi
    
    echo ""
    echo "=========================================="
    echo "📥 导入到本地数据库:"
    echo ""
    echo "方式1: 使用mysql命令"
    if [[ $BACKUP_FILE == *.gz ]]; then
        echo "   gunzip < $BACKUP_FILE | mysql -h 127.0.0.1 -P 2881 -u root@sys -p DominoDB"
    else
        echo "   mysql -h 127.0.0.1 -P 2881 -u root@sys -p DominoDB < $BACKUP_FILE"
    fi
    echo ""
    echo "方式2: 使用DBeaver"
    echo "   1. 连接到本地数据库"
    echo "   2. 右键数据库 → 工具 → 执行脚本"
    echo "   3. 选择文件: $BACKUP_FILE"
    echo "=========================================="
else
    echo ""
    echo "❌ 导出失败，请检查:"
    echo "   1. 服务器地址和端口是否正确"
    echo "   2. 用户名和密码是否正确"
    echo "   3. 网络连接是否正常"
    echo "   4. 数据库是否存在"
    exit 1
fi

