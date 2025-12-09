#!/bin/bash
# 数据生成进度监控脚本

echo "📊 Gamium 数据生成进度监控"
echo "================================"

# 检查进程
if pgrep -f "generate_dataset.py" > /dev/null; then
    PID=$(pgrep -f "generate_dataset.py" | head -1)
    echo "✅ 生成进程: 运行中 (PID: $PID)"
    
    # 获取进程资源使用
    if command -v ps > /dev/null; then
        PS_OUTPUT=$(ps -p $PID -o %cpu,%mem,etime 2>/dev/null | tail -1)
        if [ -n "$PS_OUTPUT" ]; then
            CPU=$(echo $PS_OUTPUT | awk '{print $1}')
            MEM=$(echo $PS_OUTPUT | awk '{print $2}')
            TIME=$(echo $PS_OUTPUT | awk '{print $3}')
            echo "   CPU: ${CPU}%, MEM: ${MEM}%, 运行时间: ${TIME}"
        fi
    fi
else
    echo "❌ 生成进程: 未运行"
fi

echo ""

# 检查临时文件
TEMP_DIR="data/historical/temp"
if [ -d "$TEMP_DIR" ]; then
    CUSTOMER_FILES=$(ls -1 "$TEMP_DIR"/customers_*.parquet 2>/dev/null | wc -l | tr -d ' ')
    LOAN_FILES=$(ls -1 "$TEMP_DIR"/loans_*.parquet 2>/dev/null | wc -l | tr -d ' ')
    REPAYMENT_FILES=$(ls -1 "$TEMP_DIR"/repayments_*.parquet 2>/dev/null | wc -l | tr -d ' ')
    
    # 从日志中获取目标批次数
    TARGET_BATCHES=$(grep "总批次数:" data_generation.log 2>/dev/null | tail -1 | awk '{print $2}' || echo "153")
    
    echo "📁 临时文件统计:"
    echo "   客户文件: $CUSTOMER_FILES / $TARGET_BATCHES"
    if [ "$CUSTOMER_FILES" -gt 0 ] && [ "$TARGET_BATCHES" -gt 0 ]; then
        PROGRESS=$((CUSTOMER_FILES * 100 / TARGET_BATCHES))
        echo "   进度: ${PROGRESS}%"
    fi
    echo "   贷款文件: $LOAN_FILES"
    echo "   还款文件: $REPAYMENT_FILES"
    
    TEMP_SIZE=$(du -sh "$TEMP_DIR" 2>/dev/null | awk '{print $1}')
    echo "   临时数据大小: $TEMP_SIZE"
    
    # 估算剩余时间（简单估算）
    if [ "$CUSTOMER_FILES" -gt 10 ] && [ "$TARGET_BATCHES" -gt 0 ]; then
        REMAINING=$((TARGET_BATCHES - CUSTOMER_FILES))
        echo "   剩余批次: $REMAINING"
    fi
else
    echo "⏳ 临时目录不存在（可能已完成生成）"
fi

echo ""

# 检查最终文件
if [ -f "data/historical/customers.parquet" ]; then
    echo "📦 最终文件大小:"
    ls -lh data/historical/*.parquet 2>/dev/null | awk '{print "   " $5 " - " $9}'
    echo ""
    TOTAL_SIZE=$(du -sh data/historical 2>/dev/null | awk '{print $1}')
    echo "   总大小: $TOTAL_SIZE"
    
    # 检查是否有temp文件（说明合并未完成）
    if [ -d "$TEMP_DIR" ] && [ "$(ls -1 "$TEMP_DIR"/*.parquet 2>/dev/null | wc -l)" -gt 0 ]; then
        echo ""
        echo "⚠️  检测到未合并的临时文件，可以运行合并脚本:"
        echo "   python3 scripts/merge_temp_files.py"
    fi
else
    echo "⏳ 最终文件尚未生成（仍在生成临时文件阶段）"
fi

echo ""
echo "📝 查看详细日志:"
echo "   tail -f data_generation.log"
echo ""
echo "💡 提示: 如果生成中断，可以使用以下命令合并已生成的批次:"
echo "   python3 scripts/merge_temp_files.py"

