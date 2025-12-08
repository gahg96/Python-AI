#!/bin/bash
# 清理已完成的临时文件

echo "🧹 Gamium 临时文件清理工具"
echo "================================"
echo ""

# 检查是否有正在运行的生成进程
if pgrep -f "generate_dataset.py" > /dev/null; then
    PID=$(pgrep -f "generate_dataset.py" | head -1)
    OUTPUT=$(ps -p $PID -o args= | grep -oP '--output \K[^\s]+' || echo "unknown")
    echo "⚠️  检测到正在运行的数据生成进程 (PID: $PID)"
    echo "   输出目录: $OUTPUT"
    echo "   该目录的 temp 文件正在使用中，不能清理"
    echo ""
fi

# 检查各个数据目录
for data_dir in data/historical data/historical_large; do
    if [ -d "$data_dir/temp" ]; then
        temp_size=$(du -sh "$data_dir/temp" 2>/dev/null | awk '{print $1}')
        file_count=$(ls -1 "$data_dir/temp"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
        
        # 检查是否有最终文件（说明生成已完成）
        has_final=$(ls "$data_dir"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
        
        echo "📁 $data_dir/temp"
        echo "   大小: $temp_size"
        echo "   文件数: $file_count"
        
        if [ "$has_final" -gt 0 ]; then
            echo "   ✅ 检测到最终文件，生成已完成"
            echo "   🗑️  可以安全清理"
        else
            # 检查是否正在使用
            if pgrep -f "generate_dataset.py.*$data_dir" > /dev/null; then
                echo "   ⚠️  正在使用中，不能清理"
            else
                echo "   ⚠️  未检测到最终文件，但无运行进程"
                echo "   💡 可能是未完成的生成，建议手动检查"
            fi
        fi
        echo ""
    fi
done

# 交互式清理
echo "请选择要清理的目录："
echo "1) data/historical_large/temp (已完成，12GB)"
echo "2) 清理所有已完成的 temp 目录"
echo "3) 取消"
echo ""
read -p "请输入选项 (1-3): " choice

case $choice in
    1)
        if [ -d "data/historical_large/temp" ]; then
            read -p "确认删除 data/historical_large/temp? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                echo "正在清理..."
                rm -rf data/historical_large/temp
                echo "✅ 已清理 data/historical_large/temp"
            else
                echo "已取消"
            fi
        fi
        ;;
    2)
        echo "正在清理所有已完成的 temp 目录..."
        cleaned=0
        for data_dir in data/historical data/historical_large; do
            if [ -d "$data_dir/temp" ]; then
                # 检查是否有最终文件且无运行进程
                has_final=$(ls "$data_dir"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
                is_running=$(pgrep -f "generate_dataset.py.*$data_dir" > /dev/null && echo "yes" || echo "no")
                
                if [ "$has_final" -gt 0 ] && [ "$is_running" = "no" ]; then
                    echo "  清理 $data_dir/temp..."
                    rm -rf "$data_dir/temp"
                    cleaned=$((cleaned + 1))
                fi
            fi
        done
        if [ $cleaned -gt 0 ]; then
            echo "✅ 已清理 $cleaned 个目录"
        else
            echo "ℹ️  没有可清理的目录（都在使用中或未完成）"
        fi
        ;;
    3)
        echo "已取消"
        ;;
    *)
        echo "无效选项"
        ;;
esac

