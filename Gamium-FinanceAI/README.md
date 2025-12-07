# Gamium Finance AI 🎮💰

**基于 AlphaZero 的金融行业经营决策系统**

利用类似 AlphaZero 的强化学习方法，构建金融机构（银行、保险等）的智能经营决策系统。通过"数字孪生 + AI 博弈"的方式，让 AI 在虚拟环境中模拟数百万次经营决策，找到穿越经济周期的最优策略。

## 🎯 项目愿景

> "如果 AlphaZero 能在围棋中战胜人类冠军，那么它能否帮助银行在经济周期中做出更好的决策？"

本项目受 [Gamium 金融行业最佳实践](docs/) 启发，旨在：

1. **构建金融数字孪生** - 模拟真实的银行经营环境（信贷、风控、客户行为、经济周期）
2. **训练 AlphaZero 智能体** - 通过自我对弈学习最优经营策略
3. **支持决策验证** - 在 AI 验证的策略空间中探索"what-if"场景

## 🌐 在线体验

**🔗 在线Demo**: [https://gamium-finance-ai.onrender.com](https://gamium-finance-ai.onrender.com)

- **大屏监控**: 实时银行经营数据可视化
- **客户预测**: AI 风险评估与审批建议
- **经济周期分析**: 不同周期下的违约率变化
- **银行模拟**: 10年经营模拟体验
- **策略对比**: 4种策略表现对比
- **数据蒸馏**: 从历史数据提炼商业规律
- **数据浏览**: 查看客户详细字段信息

## 🚀 快速开始

### 本地运行

```bash
cd Gamium-FinanceAI

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动 Web 服务
python app.py
# 访问 http://localhost:5000
```

### 命令行演示

```bash
# 快速演示 - 观看 AI 运营银行 10 年
python demo.py --mode quick

# 策略对比 - 比较不同策略的表现
python demo.py --mode compare

# 交互模式 - 自己当行长做决策
python demo.py --mode interactive

# 数据蒸馏演示 - 展示从历史数据学习规律的过程
python demo_distillation.py --mode distill

# 客户预测演示 - 查看不同客户的违约预测
python demo_distillation.py --mode quick

# 经济周期影响分析
python demo_distillation.py --mode batch
```

### 训练 AlphaZero

```bash
# 快速训练 (约 5-10 分钟)
python train.py --iterations 30 --games 5

# 完整训练 (约 1 小时)
python train.py --iterations 100 --games 20

# 训练完成后，模型保存在 experiments/ 目录
```

## 📁 项目结构

```
Gamium-FinanceAI/
├── src/
│   ├── environment/           # 🌍 模拟环境
│   │   ├── lending_env.py     # 信贷策略环境 (Gym兼容)
│   │   └── economic_cycle.py  # 经济周期模拟器
│   │
│   ├── agents/                # 🤖 智能体
│   │   ├── network.py         # 神经网络 (策略+价值双头)
│   │   ├── mcts.py            # 蒙特卡洛树搜索
│   │   ├── alphazero_agent.py # AlphaZero 完整实现
│   │   └── baseline_agents.py # 基线策略
│   │
│   ├── data_distillation/     # 🔬 数据蒸馏 (新!)
│   │   ├── customer_generator.py   # 客户生成器
│   │   ├── world_model.py          # 世界模型 (蒸馏出的物理定律)
│   │   └── distillation_pipeline.py # 完整蒸馏管道
│   │
│   └── utils/                 # 🛠 工具
│       ├── visualization.py   # 可视化
│       └── logger.py          # 日志
│
├── docs/                      # 📚 文档
├── experiments/               # 💾 训练结果
├── train.py                   # 🚀 AlphaZero 训练脚本
├── demo.py                    # 🎮 银行经营演示
├── demo_distillation.py       # 🔮 数据蒸馏演示 (新!)
└── requirements.txt           # 📦 依赖
```

## 🔬 数据蒸馏流程

从历史数据中提炼"商业物理定律"的五步流程：

```
第一步: 目标定义与数据准备    → 收集并清洗历史贷款记录
第二步: 特征工程              → 提取静态/行为/信贷/环境特征  
第三步: 规律建模              → 训练模型学习隐藏规律
第四步: 函数封装              → 封装为 predict_customer_future() API
第五步: 验证与校准            → 使用保留数据回测验证
```

**蒸馏后的世界模型学到的关键规律：**
- 📈 小微企业主在经济下行时违约率急剧上升 (3x)
- ⚠️ 负债率超过60%是高风险信号 (2.5x)
- 🍽️ 餐饮业客户风险系数最高 (1.4x)
- 📊 历史逾期>90天的客户风险提高3倍

**经济周期影响分析示例:**
| 经济周期 | 平均违约率 | 高风险客户占比 |
|---------|-----------|--------------|
| 繁荣期 | 11.28% | 23% |
| 正常期 | 11.28% | 22% |
| 衰退期 | 14.72% | 28% |
| 萧条期 | **26.95%** | **70%** |

## 🎮 环境设计

### 信贷策略环境 (LendingEnv)

模拟银行 10 年的信贷经营，AI 需要在不同经济周期下做出决策：

**状态空间 (22 维)**:
- 宏观经济: GDP 增长率、利率、失业率、通胀率、信用利差、周期阶段
- 银行状态: 资本、资产、贷款、不良率、资本充足率、客群分布、利润

**动作空间 (5 维)**:
- 利率调整: [-2%, +2%]
- 审批通过率: [30%, 90%]
- 客群分配: 优质/次优/次级客户的投放权重

**奖励函数**:
- ✅ 月度利润 (正向激励)
- ⚠️ NPL 超标惩罚
- 🚨 资本充足率过低惩罚
- 💀 破产大额惩罚

### 经济周期

模拟四阶段经济周期：繁荣 → 衰退 → 萧条 → 复苏

```
繁荣期: GDP↑ 失业↓ 违约↓ → 可适度扩张
衰退期: GDP→ 失业↑ 违约↑ → 开始收缩
萧条期: GDP↓ 失业↑↑ 违约↑↑ → 防守为主  
复苏期: GDP↑ 失业↓ 违约↓ → 把握机会
```

## 🧠 AlphaZero 架构

### 神经网络

```
输入(22维) → 残差块x4 → 策略头(120动作概率)
                      → 价值头(状态估值)
```

### 训练循环

```
1. 自我对弈 → 生成 (状态, 动作概率, 奖励) 样本
2. 样本入库 → 经验回放缓冲区
3. 批量训练 → 策略损失 + 价值损失
4. 重复迭代 → 策略逐步提升
```

## 📊 性能对比

经过训练后，AlphaZero 通常能够：

| 策略 | 平均奖励 | 累计利润 | 破产率 |
|------|---------|---------|-------|
| 随机策略 | ~-50 | -200亿 | 60% |
| 规则策略 | ~20 | +80亿 | 15% |
| 保守策略 | ~15 | +60亿 | 5% |
| 激进策略 | ~-10 | -50亿 | 40% |
| **AlphaZero** | **~40** | **+150亿** | **<5%** |

*具体数值因训练时长和随机种子而异*

## 🔮 下一步计划

- [ ] **更真实的环境** - 添加资金成本、监管约束、市场竞争
- [ ] **多智能体博弈** - 模拟多家银行竞争
- [ ] **迁移学习** - 用真实数据微调模型
- [ ] **策略可解释性** - 理解 AI 决策背后的逻辑
- [ ] **WebUI** - 可视化决策界面

## 📚 参考资料

- [AlphaZero 论文](https://arxiv.org/abs/1712.01815)
- [Gamium 金融最佳实践](docs/Gamium%20金融行业最佳实践：开启商业决策的"未来演武场".pdf)
- [OpenAI Gym](https://gymnasium.farama.org/)

## 🚀 部署指南

### 一键部署到 Render (推荐)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Fork 本仓库到您的 GitHub
2. 登录 [Render](https://render.com)
3. 点击 "New" → "Web Service"
4. 连接您的 GitHub 仓库
5. 选择 `Gamium-FinanceAI` 目录
6. 配置:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
7. 点击 "Create Web Service"

### Docker 部署

```bash
# 构建镜像
docker build -t gamium-finance-ai .

# 运行容器
docker run -p 5000:5000 gamium-finance-ai
```

### Railway 部署

1. 登录 [Railway](https://railway.app)
2. 点击 "New Project" → "Deploy from GitHub repo"
3. 选择仓库，Railway 会自动检测并部署

### Heroku 部署

```bash
heroku login
heroku create gamium-finance-ai
git push heroku main
```

## 📝 License

MIT License

---

**Made with ❤️ for Financial AI Innovation**
