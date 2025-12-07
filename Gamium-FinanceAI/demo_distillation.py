#!/usr/bin/env python3
"""
数据蒸馏交互式演示

展示完整的数据蒸馏流程和实时预测效果

用法:
    python demo_distillation.py                 # 交互式演示
    python demo_distillation.py --mode quick    # 快速演示
    python demo_distillation.py --mode batch    # 批量对比
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_distillation.customer_generator import (
    CustomerGenerator, CustomerProfile, CustomerType, CityTier, Industry
)
from data_distillation.world_model import (
    WorldModel, LoanOffer, MarketConditions, CustomerFuture
)
from data_distillation.distillation_pipeline import (
    DistillationPipeline, DistillationConfig, ValidationResult
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, FloatPrompt, IntPrompt, Confirm
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("提示: 安装 rich 库可获得更好的体验: pip install rich")


def print_header(title: str):
    """打印标题"""
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold yellow]{title:^60}[/bold yellow]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    else:
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}\n")


def print_customer(customer: CustomerProfile):
    """打印客户信息"""
    if RICH_AVAILABLE:
        table = Table(title="📋 客户画像", box=box.ROUNDED, show_header=False)
        table.add_column("属性", style="cyan")
        table.add_column("值", style="white")
        
        table.add_row("客户ID", customer.customer_id)
        table.add_row("年龄", f"{customer.age} 岁 ({customer.age_group})")
        table.add_row("城市等级", customer.city_tier.value)
        table.add_row("客户类型", customer.customer_type.value)
        table.add_row("所属行业", customer.industry.value)
        table.add_row("从业年限", f"{customer.years_in_business:.1f} 年")
        table.add_row("─"*15, "─"*20)
        table.add_row("月收入", f"¥{customer.monthly_income:,.0f}")
        table.add_row("收入波动率", f"{customer.income_volatility:.1%}")
        table.add_row("总资产", f"¥{customer.total_assets:,.0f}")
        table.add_row("总负债", f"¥{customer.total_liabilities:,.0f}")
        table.add_row("负债率", f"{customer.debt_ratio:.1%}")
        table.add_row("─"*15, "─"*20)
        table.add_row("存款余额", f"¥{customer.deposit_balance:,.0f}")
        table.add_row("成为客户", f"{customer.months_as_customer} 个月")
        table.add_row("历史贷款", f"{customer.previous_loans} 次")
        table.add_row("最大逾期", f"{customer.max_historical_dpd} 天")
        table.add_row("─"*15, "─"*20)
        
        risk_color = "green" if customer.risk_score < 0.3 else "yellow" if customer.risk_score < 0.6 else "red"
        table.add_row("风险评分", f"[{risk_color}]{customer.risk_score:.2f}[/{risk_color}]")
        
        console.print(table)
    else:
        print(f"\n客户画像: {customer.customer_id}")
        print(f"  类型: {customer.customer_type.value}")
        print(f"  行业: {customer.industry.value}")
        print(f"  月收入: ¥{customer.monthly_income:,.0f}")
        print(f"  负债率: {customer.debt_ratio:.1%}")
        print(f"  风险评分: {customer.risk_score:.2f}")


def print_prediction(future: CustomerFuture, model: WorldModel):
    """打印预测结果"""
    if RICH_AVAILABLE:
        # 风险颜色
        risk_color = "green" if future.default_probability < 0.05 else \
                     "yellow" if future.default_probability < 0.15 else "red"
        
        table = Table(title="🔮 预测结果", box=box.ROUNDED, show_header=False)
        table.add_column("指标", style="cyan")
        table.add_column("值", style="white")
        
        table.add_row("违约概率", f"[{risk_color}]{future.default_probability:.2%}[/{risk_color}]")
        
        ltv_color = "green" if future.expected_ltv > 0 else "red"
        table.add_row("预期LTV", f"[{ltv_color}]¥{future.expected_ltv:,.0f}[/{ltv_color}]")
        
        table.add_row("流失概率", f"{future.churn_probability:.2%}")
        table.add_row("预期逾期", f"{future.expected_dpd:.0f} 天")
        table.add_row("置信度", f"{future.confidence:.0%}")
        
        console.print(table)
        
        # 风险因素
        if future.risk_factors:
            console.print("\n[bold]风险因素分解:[/bold]")
            for factor, value in future.risk_factors.items():
                if value > 1.2:
                    console.print(f"  ⚠️  {factor}: [red]{value:.2f}x[/red]")
                elif value < 0.9:
                    console.print(f"  ✅ {factor}: [green]{value:.2f}x[/green]")
                else:
                    console.print(f"  ➡️  {factor}: {value:.2f}x")
    else:
        print(f"\n预测结果:")
        print(f"  违约概率: {future.default_probability:.2%}")
        print(f"  预期LTV: ¥{future.expected_ltv:,.0f}")
        print(f"  流失概率: {future.churn_probability:.2%}")


def create_customer_interactively(generator: CustomerGenerator) -> CustomerProfile:
    """交互式创建客户"""
    if RICH_AVAILABLE:
        console.print("\n[bold]创建客户画像[/bold]")
        console.print("(直接回车使用随机值)\n")
        
        # 客户类型
        console.print("客户类型: 1=工薪 2=小微企业主 3=自由职业 4=农户")
        choice = Prompt.ask("选择", default="0")
        if choice == "0":
            customer_type = None
        else:
            types = [CustomerType.SALARIED, CustomerType.SMALL_BUSINESS, 
                    CustomerType.FREELANCER, CustomerType.FARMER]
            customer_type = types[int(choice)-1] if choice in "1234" else None
        
        # 风险偏好
        console.print("\n风险偏好: 1=低风险 2=中等 3=高风险")
        risk_choice = Prompt.ask("选择", default="2")
        risk_map = {"1": "low", "2": "medium", "3": "high"}
        risk_profile = risk_map.get(risk_choice, "medium")
        
        customer = generator.generate_one(
            customer_type=customer_type,
            risk_profile=risk_profile
        )
        
        # 允许修改关键参数
        if Confirm.ask("\n是否修改详细参数?", default=False):
            income = FloatPrompt.ask("月收入", default=customer.monthly_income)
            customer.monthly_income = income
            
            debt_ratio = FloatPrompt.ask("负债率 (0-1)", default=customer.debt_ratio)
            customer.total_liabilities = customer.total_assets * debt_ratio
            
            dpd = IntPrompt.ask("历史最大逾期天数", default=customer.max_historical_dpd)
            customer.max_historical_dpd = dpd
        
        return customer
    else:
        return generator.generate_one()


def create_loan_interactively() -> LoanOffer:
    """交互式创建贷款条件"""
    if RICH_AVAILABLE:
        console.print("\n[bold]设置贷款条件[/bold]\n")
        
        amount = FloatPrompt.ask("贷款金额 (元)", default=100000.0)
        rate = FloatPrompt.ask("年利率", default=0.08)
        term = IntPrompt.ask("期限 (月)", default=24)
        
        return LoanOffer(
            amount=amount,
            interest_rate=rate,
            term_months=term,
        )
    else:
        return LoanOffer(amount=100000, interest_rate=0.08, term_months=24)


def create_market_interactively() -> MarketConditions:
    """交互式创建市场环境"""
    if RICH_AVAILABLE:
        console.print("\n[bold]设置宏观环境[/bold]")
        console.print("预设场景: 1=繁荣期 2=正常 3=衰退 4=萧条 5=自定义")
        
        choice = Prompt.ask("选择", default="2")
        
        presets = {
            "1": MarketConditions(0.06, 0.05, 0.04, 0.02, 0.02),  # 繁荣
            "2": MarketConditions(0.03, 0.04, 0.05, 0.02, 0.02),  # 正常
            "3": MarketConditions(0.01, 0.03, 0.07, 0.03, 0.03),  # 衰退
            "4": MarketConditions(-0.02, 0.02, 0.10, 0.01, 0.04), # 萧条
        }
        
        if choice in presets:
            return presets[choice]
        else:
            gdp = FloatPrompt.ask("GDP增长率", default=0.03)
            rate = FloatPrompt.ask("基准利率", default=0.04)
            unemployment = FloatPrompt.ask("失业率", default=0.05)
            inflation = FloatPrompt.ask("通胀率", default=0.02)
            spread = FloatPrompt.ask("信用利差", default=0.02)
            return MarketConditions(gdp, rate, unemployment, inflation, spread)
    else:
        return MarketConditions(0.03, 0.04, 0.05, 0.02, 0.02)


def interactive_demo():
    """交互式演示"""
    print_header("🎮 Gamium 数据蒸馏交互演示")
    
    # 初始化
    generator = CustomerGenerator(seed=42)
    model = WorldModel(seed=42)
    
    if RICH_AVAILABLE:
        console.print("[bold green]欢迎来到 Gamium 金融决策模拟器![/bold green]\n")
        console.print("这个演示将展示如何使用'数据蒸馏'后的世界模型")
        console.print("来预测客户在不同条件下的行为。\n")
    
    while True:
        if RICH_AVAILABLE:
            console.print("\n[bold]选择操作:[/bold]")
            console.print("  1. 生成随机客户并预测")
            console.print("  2. 自定义客户信息")
            console.print("  3. 批量对比 (不同经济周期)")
            console.print("  4. 运行完整蒸馏流程")
            console.print("  5. 退出")
            
            choice = Prompt.ask("\n请选择", choices=["1", "2", "3", "4", "5"], default="1")
        else:
            print("\n选择操作: 1=随机客户 2=自定义 3=批量对比 4=蒸馏流程 5=退出")
            choice = input("请选择: ").strip() or "1"
        
        if choice == "1":
            # 随机客户
            customer = generator.generate_one()
            loan = LoanOffer(amount=100000, interest_rate=0.08, term_months=24)
            market = MarketConditions(0.03, 0.04, 0.05, 0.02, 0.02)
            
            print_customer(customer)
            
            if RICH_AVAILABLE:
                console.print(f"\n[bold]贷款条件:[/bold] ¥{loan.amount:,.0f}, "
                            f"{loan.interest_rate:.1%}, {loan.term_months}个月")
                console.print(f"[bold]宏观环境:[/bold] GDP={market.gdp_growth:.1%}, "
                            f"失业率={market.unemployment_rate:.1%}")
            
            future = model.predict_customer_future(customer, loan, market)
            print_prediction(future, model)
            
        elif choice == "2":
            # 自定义客户
            customer = create_customer_interactively(generator)
            print_customer(customer)
            
            loan = create_loan_interactively()
            market = create_market_interactively()
            
            future = model.predict_customer_future(customer, loan, market)
            print_prediction(future, model)
            
        elif choice == "3":
            # 批量对比
            batch_comparison_demo(generator, model)
            
        elif choice == "4":
            # 蒸馏流程
            run_distillation_demo()
            
        elif choice == "5":
            if RICH_AVAILABLE:
                console.print("\n[bold green]感谢使用 Gamium![/bold green]")
            break
        
        if RICH_AVAILABLE:
            input("\n按 Enter 继续...")


def batch_comparison_demo(generator: CustomerGenerator, model: WorldModel):
    """批量对比演示 - 展示经济周期对违约率的影响"""
    print_header("📊 经济周期影响分析")
    
    # 生成一批客户
    n_customers = 100
    customers = generator.generate_batch(n_customers)
    
    # 不同经济环境
    scenarios = {
        "繁荣期": MarketConditions(0.06, 0.05, 0.04, 0.02, 0.02),
        "正常期": MarketConditions(0.03, 0.04, 0.05, 0.02, 0.02),
        "衰退期": MarketConditions(0.01, 0.03, 0.07, 0.03, 0.03),
        "萧条期": MarketConditions(-0.02, 0.02, 0.10, 0.01, 0.04),
    }
    
    # 标准贷款条件
    loan = LoanOffer(amount=100000, interest_rate=0.08, term_months=24)
    
    results = {}
    
    for scenario_name, market in scenarios.items():
        default_probs = []
        ltvs = []
        
        for customer in customers:
            future = model.predict_customer_future(customer, loan, market)
            default_probs.append(future.default_probability)
            ltvs.append(future.expected_ltv)
        
        results[scenario_name] = {
            'avg_default': np.mean(default_probs),
            'avg_ltv': np.mean(ltvs),
            'high_risk_count': sum(1 for p in default_probs if p > 0.15),
        }
    
    if RICH_AVAILABLE:
        table = Table(title=f"经济周期对 {n_customers} 位客户的影响", box=box.DOUBLE)
        table.add_column("经济周期", style="cyan")
        table.add_column("平均违约率", justify="right")
        table.add_column("平均LTV", justify="right")
        table.add_column("高风险客户", justify="right")
        
        for scenario, data in results.items():
            color = "green" if data['avg_default'] < 0.05 else \
                   "yellow" if data['avg_default'] < 0.10 else "red"
            table.add_row(
                scenario,
                f"[{color}]{data['avg_default']:.2%}[/{color}]",
                f"¥{data['avg_ltv']:,.0f}",
                str(data['high_risk_count'])
            )
        
        console.print(table)
        
        console.print("\n[bold]关键发现:[/bold]")
        boom = results["繁荣期"]['avg_default']
        recession = results["萧条期"]['avg_default']
        console.print(f"  • 从繁荣到萧条，平均违约率从 {boom:.2%} 上升到 {recession:.2%}")
        console.print(f"  • 违约率上升 {(recession/boom - 1)*100:.0f}%")
    else:
        print("\n经济周期影响:")
        for scenario, data in results.items():
            print(f"  {scenario}: 违约率={data['avg_default']:.2%}, LTV=¥{data['avg_ltv']:,.0f}")
    
    # 按客户类型细分
    if RICH_AVAILABLE:
        console.print("\n[bold]按客户类型细分 (萧条期):[/bold]")
        
        market = scenarios["萧条期"]
        by_type = {}
        
        for customer in customers:
            ctype = customer.customer_type.value
            if ctype not in by_type:
                by_type[ctype] = []
            future = model.predict_customer_future(customer, loan, market)
            by_type[ctype].append(future.default_probability)
        
        for ctype, probs in sorted(by_type.items(), key=lambda x: np.mean(x[1]), reverse=True):
            avg = np.mean(probs)
            color = "red" if avg > 0.15 else "yellow" if avg > 0.08 else "green"
            console.print(f"  {ctype}: [{color}]{avg:.2%}[/{color}]")


def run_distillation_demo(auto_confirm: bool = False):
    """运行数据蒸馏演示"""
    print_header("🔥 数据蒸馏流程演示")
    
    if RICH_AVAILABLE:
        console.print("[bold]将模拟完整的数据蒸馏流程[/bold]\n")
        console.print("这个过程将展示如何从历史数据中提炼出'商业物理定律'")
        console.print("并封装为可调用的预测函数。\n")
        
        if not auto_confirm:
            try:
                if not Confirm.ask("开始蒸馏?", default=True):
                    return
            except EOFError:
                pass  # 非交互模式直接继续
    
    # 配置
    config = DistillationConfig(
        train_years=[2019, 2020, 2021, 2022],
        test_years=[2023],
        model_type="rule_based",
    )
    
    # 运行管道
    pipeline = DistillationPipeline(config=config, seed=42)
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("蒸馏中...", total=5)
            
            pipeline.step1_prepare_data(n_synthetic=2000)
            progress.update(task, advance=1, description="特征工程...")
            
            pipeline.step2_feature_engineering()
            progress.update(task, advance=1, description="模型训练...")
            
            pipeline.step3_train_model()
            progress.update(task, advance=1, description="API封装...")
            
            pipeline.step4_create_api()
            progress.update(task, advance=1, description="验证校准...")
            
            validation = pipeline.step5_validate()
            progress.update(task, advance=1, description="完成!")
    else:
        world_model, validation = pipeline.run_full_pipeline(n_synthetic=2000)
    
    # 显示验证结果
    if RICH_AVAILABLE:
        status = "✅ 通过" if validation.passed else "❌ 未通过"
        console.print(f"\n[bold]验证结果: {status}[/bold]")
        console.print(f"  预测违约率: {validation.predicted_default_rate:.2%}")
        console.print(f"  实际违约率: {validation.actual_default_rate:.2%}")
        console.print(f"  偏差: {validation.deviation:.2%}")
        
        console.print("\n[bold green]🎉 世界模型已准备就绪![/bold green]")
        console.print("现在可以使用 predict_customer_future() 进行预测")


def quick_demo():
    """快速演示"""
    print_header("⚡ Gamium 快速演示")
    
    generator = CustomerGenerator(seed=42)
    model = WorldModel(seed=42)
    
    print("生成 3 位典型客户并预测...")
    
    # 三种风险画像
    profiles = [
        ("低风险", "low"),
        ("中等风险", "medium"),
        ("高风险", "high"),
    ]
    
    loan = LoanOffer(amount=100000, interest_rate=0.08, term_months=24)
    market = MarketConditions(0.03, 0.04, 0.05, 0.02, 0.02)
    
    for name, risk in profiles:
        customer = generator.generate_one(risk_profile=risk)
        future = model.predict_customer_future(customer, loan, market)
        
        print(f"\n{'='*50}")
        print(f"🧑 {name}客户: {customer.customer_type.value}, {customer.industry.value}")
        print(f"   月收入: ¥{customer.monthly_income:,.0f}, 负债率: {customer.debt_ratio:.1%}")
        print(f"   风险评分: {customer.risk_score:.2f}")
        print(f"\n   📊 预测结果:")
        print(f"   违约概率: {future.default_probability:.2%}")
        print(f"   预期LTV: ¥{future.expected_ltv:,.0f}")
    
    # 展示经济周期影响
    print(f"\n{'='*50}")
    print("📈 经济周期对高风险客户的影响:")
    
    customer = generator.generate_one(risk_profile="high")
    
    for scenario, market in [
        ("繁荣期", MarketConditions(0.06, 0.05, 0.04, 0.02, 0.02)),
        ("萧条期", MarketConditions(-0.02, 0.02, 0.10, 0.01, 0.04)),
    ]:
        future = model.predict_customer_future(customer, loan, market)
        print(f"   {scenario}: 违约概率 {future.default_probability:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Gamium 数据蒸馏演示")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["interactive", "quick", "batch", "distill"],
                        help="演示模式")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_demo()
    elif args.mode == "batch":
        generator = CustomerGenerator(seed=42)
        model = WorldModel(seed=42)
        batch_comparison_demo(generator, model)
    elif args.mode == "distill":
        run_distillation_demo(auto_confirm=True)
    else:
        interactive_demo()


if __name__ == "__main__":
    main()

