#!/bin/bash
# 检查服务状态脚本

echo "🔍 检查服务状态..."
echo ""

# 检查进程
echo "1. 检查Python进程:"
ps aux | grep "[p]ython.*app.py" && echo "   ✅ 服务进程运行中" || echo "   ❌ 未找到服务进程"

echo ""

# 检查端口
echo "2. 检查端口5000:"
if lsof -i:5000 > /dev/null 2>&1; then
    echo "   ✅ 端口5000已被占用"
    lsof -i:5000 | head -3
else
    echo "   ❌ 端口5000未被占用"
fi

echo ""

# 测试HTTP连接
echo "3. 测试HTTP连接:"
if curl -s http://localhost:5000/ > /dev/null 2>&1; then
    echo "   ✅ 服务可以访问"
    echo "   🌐 访问地址: http://localhost:5000"
    echo "   📊 大屏监控: http://localhost:5000/"
    echo "   🎮 操作控制台: http://localhost:5000/control"
else
    echo "   ❌ 服务无法访问"
    echo "   💡 提示: 服务可能还在启动中，请稍等几秒后重试"
fi

echo ""
echo "4. 如果服务未运行，可以使用以下命令启动:"
echo "   cd /Users/carrot/Python-AI/Gamium-FinanceAI"
echo "   python3 app.py"


