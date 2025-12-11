"""
Demo展示脚本
展示已完成模块的效果
"""
import pandas as pd
import json
import os
from pathlib import Path


def show_historical_data_stats():
    """展示历史数据统计"""
    print("=" * 80)
    print("1. 历史数据生成器 - 数据统计")
    print("=" * 80)
    
    stats_path = 'data/historical/statistics.json'
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print(f"总贷款数: {stats['total_loans']}")
        print(f"对私贷款: {stats['personal_loans']} ({stats['personal_loans']/stats['total_loans']:.1%})")
        print(f"对公贷款: {stats['corporate_loans']} ({stats['corporate_loans']/stats['total_loans']:.1%})")
        print(f"审批通过: {stats['approved_loans']} ({stats['approval_rate']:.2%})")
        print(f"审批拒绝: {stats['rejected_loans']} ({1-stats['approval_rate']:.2%})")
        print(f"违约数量: {stats['defaulted_loans']}")
        print(f"违约率: {stats['default_rate']:.2%}")
        print(f"平均利润: ¥{stats['avg_profit']:,.2f}")
        print(f"时间范围: {stats['date_range']['start']} 至 {stats['date_range']['end']}")
    else:
        print("❌ 统计数据文件不存在")


def show_quality_report():
    """展示数据质量报告"""
    print("\n" + "=" * 80)
    print("2. 数据质量检查模块 - 质量报告")
    print("=" * 80)
    
    report_path = 'data/historical/quality_report.json'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        overall = report.get('overall', {})
        print(f"综合得分: {overall.get('overall_score', 0):.4f}")
        print(f"总记录数: {overall.get('summary', {}).get('total_records', 0)}")
        print(f"总问题数: {overall.get('summary', {}).get('total_issues', 0)}")
        print(f"严重问题数: {overall.get('summary', {}).get('critical_issues', 0)}")
        print(f"数据质量: {'✅ 合格' if overall.get('is_acceptable', False) else '❌ 不合格'}")
        
        # 详细得分
        print("\n详细得分:")
        if 'completeness' in report:
            print(f"  完整性: {report['completeness'].get('completeness_score', 0):.4f}")
        if 'consistency' in report:
            print(f"  一致性: {report['consistency'].get('consistency_score', 0):.4f}")
        if 'temporal_consistency' in report:
            print(f"  时间一致性: {report['temporal_consistency'].get('temporal_consistency_score', 0):.4f}")
        if 'business_rules' in report:
            print(f"  业务规则: {report['business_rules'].get('rule_score', 0):.4f}")
    else:
        print("❌ 质量报告文件不存在")


def show_feature_engineering():
    """展示特征工程结果"""
    print("\n" + "=" * 80)
    print("3. 特征工程模块 - 特征统计")
    print("=" * 80)
    
    original_path = 'data/historical/historical_loans.csv'
    engineered_path = 'data/historical/historical_loans_engineered.csv'
    
    if os.path.exists(original_path) and os.path.exists(engineered_path):
        original_df = pd.read_csv(original_path)
        engineered_df = pd.read_csv(engineered_path)
        
        print(f"原始特征数: {len(original_df.columns)}")
        print(f"特征工程后: {len(engineered_df.columns)}")
        print(f"新增特征数: {len(engineered_df.columns) - len(original_df.columns)}")
        
        # 显示一些新特征
        new_features = set(engineered_df.columns) - set(original_df.columns)
        print(f"\n新增特征示例（前10个）:")
        for i, feat in enumerate(list(new_features)[:10], 1):
            print(f"  {i}. {feat}")
    else:
        print("❌ 特征工程文件不存在")


def show_extracted_rules():
    """展示提取的规则"""
    print("\n" + "=" * 80)
    print("4. 业务规则提取模块 - 提取的规则")
    print("=" * 80)
    
    rules_path = 'data/historical/extracted_rules.json'
    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        print(f"总共提取 {len(rules)} 条规则\n")
        
        for i, rule in enumerate(rules[:5], 1):  # 显示前5条
            print(f"规则 {i}: {rule['rule_name']}")
            print(f"  类型: {rule['rule_type']}")
            print(f"  客户类型: {rule['customer_type']}")
            print(f"  描述: {rule['description']}")
            print(f"  置信度: {rule['confidence']:.2%}")
            print(f"  支持度: {rule['support']:.2%}")
            print()
    else:
        print("❌ 规则文件不存在")


def show_quantified_rules():
    """展示量化的规则"""
    print("=" * 80)
    print("5. 规则量化模块 - 量化规则")
    print("=" * 80)
    
    rules_path = 'data/historical/quantified_rules.json'
    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        print(f"成功量化 {len(rules)} 条规则\n")
        
        for i, rule in enumerate(rules[:3], 1):  # 显示前3条
            print(f"规则 {i}: {rule['rule_name']}")
            print(f"  类型: {rule['rule_type']}")
            print(f"  描述: {rule['description']}")
            print(f"  权重: {rule['weight']:.4f}")
            print(f"  优先级: {rule['priority']}")
            print()
    else:
        print("❌ 量化规则文件不存在")


def show_enhanced_generator():
    """展示增强版客户生成器"""
    print("=" * 80)
    print("6. 增强版客户生成器 - 生成示例")
    print("=" * 80)
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from enhanced_customer_generator import EnhancedCustomerGenerator
    
    data_path = 'data/historical/historical_loans_engineered.csv'
    if os.path.exists(data_path):
        print("正在加载历史数据并生成示例客户...")
        data = pd.read_csv(data_path)
        generator = EnhancedCustomerGenerator(data, seed=42)
        
        # 生成示例客户
        customers = generator.generate_customers(num_personal=5, num_corporate=3)
        
        print(f"\n✅ 生成了 {len(customers)} 个示例客户\n")
        
        print("对私客户示例:")
        personal = [c for c in customers if c['customer_type'] == 'personal']
        for i, customer in enumerate(personal[:3], 1):
            print(f"  {i}. ID: {customer['customer_id']}")
            print(f"     年龄: {customer.get('age', 'N/A')}, "
                  f"月收入: ¥{customer.get('monthly_income', 0):,.0f}, "
                  f"信用分: {customer.get('credit_score', 'N/A')}")
            print(f"     负债率: {customer.get('debt_ratio', 0):.2%}, "
                  f"工作年限: {customer.get('years_in_job', 0)}年")
            print()
        
        print("对公客户示例:")
        corporate = [c for c in customers if c['customer_type'] == 'corporate']
        for i, customer in enumerate(corporate[:2], 1):
            print(f"  {i}. ID: {customer['customer_id']}")
            print(f"     注册资本: ¥{customer.get('registered_capital', 0):,.0f}, "
                  f"年营收: ¥{customer.get('annual_revenue', 0):,.0f}")
            print(f"     经营年限: {customer.get('operating_years', 0)}年, "
                  f"资产负债率: {customer.get('debt_to_asset_ratio', 0):.2%}")
            print()
    else:
        print("❌ 历史数据文件不存在")


def show_data_files():
    """展示生成的数据文件"""
    print("=" * 80)
    print("生成的数据文件")
    print("=" * 80)
    
    data_dir = Path('data/historical')
    if data_dir.exists():
        files = list(data_dir.glob('*'))
        print(f"\n数据目录: {data_dir}")
        print(f"文件数量: {len(files)}\n")
        
        for file in sorted(files):
            size = file.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  📄 {file.name}")
            print(f"     大小: {size_mb:.2f} MB")
            print()
    else:
        print("❌ 数据目录不存在")


def main():
    """主函数：展示所有模块效果"""
    print("\n" + "=" * 80)
    print("端到端贷款审批Demo - 已完成模块展示")
    print("=" * 80)
    print()
    
    # 检查数据目录
    if not os.path.exists('data/historical'):
        print("❌ 数据目录不存在，请先运行数据生成器")
        return
    
    # 展示各个模块
    show_historical_data_stats()
    show_quality_report()
    show_feature_engineering()
    show_extracted_rules()
    show_quantified_rules()
    show_enhanced_generator()
    show_data_files()
    
    print("=" * 80)
    print("展示完成！")
    print("=" * 80)
    print("\n提示：")
    print("1. 查看原始数据: data/historical/historical_loans.csv")
    print("2. 查看特征工程后数据: data/historical/historical_loans_engineered.csv")
    print("3. 查看提取的规则: data/historical/extracted_rules.json")
    print("4. 查看质量报告: data/historical/quality_report.json")
    print()


if __name__ == '__main__':
    main()

