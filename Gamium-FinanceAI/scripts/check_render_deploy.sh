#!/bin/bash
# 检查Render自动部署配置

echo "🔍 Render自动部署检查"
echo "===================="
echo ""

# 检查当前分支
echo "📋 当前Git信息:"
echo "   分支: $(git branch --show-current)"
echo "   最新提交: $(git log -1 --oneline)"
echo "   远程仓库: $(git remote get-url origin 2>/dev/null || echo '未设置')"
echo ""

# 检查render.yaml
echo "📄 检查render.yaml配置:"
if [ -f "render.yaml" ]; then
    echo "   ✅ render.yaml存在"
    if grep -q "autoDeploy: true" render.yaml; then
        echo "   ✅ autoDeploy已启用"
    else
        echo "   ⚠️  autoDeploy未设置为true"
    fi
else
    echo "   ⚠️  render.yaml不存在"
fi
echo ""

# 检查Procfile
echo "📄 检查Procfile:"
if [ -f "Procfile" ]; then
    echo "   ✅ Procfile存在"
    cat Procfile | sed 's/^/      /'
else
    echo "   ⚠️  Procfile不存在"
fi
echo ""

echo "💡 如果Render没有自动部署，请："
echo "   1. 在Render Dashboard中检查Auto-Deploy设置"
echo "   2. 确认分支设置为 'main'"
echo "   3. 点击 'Manual Deploy' 手动触发部署"
echo "   4. 检查GitHub Webhook是否正常"
echo ""
echo "📖 详细说明请查看: docs/Render自动部署问题排查.md"


