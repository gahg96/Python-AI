#!/bin/bash
# 数据生成进度监控脚本

echo "📊 Gamium 数据生成进度监控"
echo "================================"

# 检查进程
if pgrep -f "generate_dataset.py" > /dev/null; then
    echo "✅ 生成进程: 运行中"
    PID=$(pgrep -f "generate_dataset.py" | head -1)
    echo "   PID: $PID"
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
    
    echo "📁 临时文件统计:"
    echo "   客户文件: $CUSTOMER_FILES"
    echo "   贷款文件: $LOAN_FILES"
    echo "   还款文件: $REPAYMENT_FILES"
    
    TEMP_SIZE=$(du -sh "$TEMP_DIR" 2>/dev/null | awk '{print $1}')
    echo "   临时数据大小: $TEMP_SIZE"
else
    echo "❌ 临时目录不存在"
fi

echo ""

# 检查最终文件
if [ -f "data/historical/customers.parquet" ]; then
    FINAL_SIZE=$(du -sh data/historical/*.parquet 2>/dev/null | awk '{sum+=$1} END {print sum}')
    echo "📦 最终文件大小:"
    ls -lh data/historical/*.parquet 2>/dev/null | awk '{print "   " $5 " - " $9}'
    echo ""
    TOTAL_SIZE=$(du -sh data/historical 2>/dev/null | awk '{print $1}')
    echo "   总大小: $TOTAL_SIZE"
else
    echo "⏳ 最终文件尚未生成（仍在生成临时文件阶段）"
fi

echo ""
echo "📝 查看详细日志:"
echo "   tail -f data_generation.log"

