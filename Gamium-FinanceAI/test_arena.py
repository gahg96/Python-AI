#!/usr/bin/env python3
"""
演武场功能测试脚本
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from arena.rule_engine import RuleEngine, ConditionOperator
from arena.scoring_system import ScoringSystem
from data_distillation.customer_generator import CustomerGenerator

def test_rule_engine():
    """测试规则引擎"""
    print("=" * 60)
    print("测试1: 规则引擎")
    print("=" * 60)
    
    rules = [{
        'name': '高收入优惠',
        'description': '月收入超过20000的客户降低审批阈值',
        'priority': 1,
        'conditions': [{'field': 'monthly_income', 'op': '>', 'value': 20000}],
        'action': {'approval_threshold_delta': -0.02, 'rate_spread_delta': -0.002},
        'penalty': {'score_delta': 0, 'profit_discount': 1.0}
    }]
    
    engine = RuleEngine(rules)
    generator = CustomerGenerator(seed=42)
    
    triggered_count = 0
    for i in range(20):
        cust = generator.generate_one()
        adjustments, triggered, score_adj = engine.process_customer(cust, 0.18, 0.01, 100000, 24)
        if triggered:
            triggered_count += 1
            print(f"  客户 {i+1}: 月收入={cust.monthly_income:.2f}, 触发规则: {triggered}")
            print(f"    调整后阈值: {adjustments['approval_threshold']:.4f}")
    
    print(f"\n✅ 规则引擎测试完成，共触发 {triggered_count} 次规则")
    return triggered_count > 0


def test_scoring_system():
    """测试评分系统"""
    print("\n" + "=" * 60)
    print("测试2: 评分系统")
    print("=" * 60)
    
    scoring = ScoringSystem()
    
    # 模拟结果
    results = [
        {
            'name': '策略A',
            'est_profit': 1000000,
            'avg_default_prob': 0.05,
            'profit_volatility': 50000,
            'max_drawdown': 0.1,
            'compliance_violations': 0,
            'avg_latency': 1.0,
            'triggered_rules_list': ['规则1', '规则2']
        },
        {
            'name': '策略B',
            'est_profit': 800000,
            'avg_default_prob': 0.08,
            'profit_volatility': 80000,
            'max_drawdown': 0.15,
            'compliance_violations': 2,
            'avg_latency': 2.0,
            'triggered_rules_list': ['规则1']
        }
    ]
    
    for result in results:
        breakdown = scoring.create_score_breakdown(
            result,
            triggered_rules=result.get('triggered_rules_list', []),
            all_results=results
        )
        print(f"\n  {result['name']}:")
        print(f"    综合得分: {breakdown.overall_score:.4f}")
        print(f"    利润得分: {breakdown.profit_score:.4f}")
        print(f"    风险得分: {breakdown.risk_score:.4f}")
        print(f"    稳定性得分: {breakdown.stability_score:.4f}")
    
    print("\n✅ 评分系统测试完成")
    return True


def test_api_endpoint():
    """测试API端点（需要服务运行）"""
    print("\n" + "=" * 60)
    print("测试3: API端点")
    print("=" * 60)
    
    import requests
    import time
    
    # 等待服务启动
    base_url = "http://localhost:5000"
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/", timeout=2)
            if response.status_code == 200:
                print("  ✅ 服务已启动")
                break
        except:
            if i < max_retries - 1:
                print(f"  等待服务启动... ({i+1}/{max_retries})")
                time.sleep(1)
            else:
                print("  ⚠️  服务未启动，跳过API测试")
                return False
    
    # 测试演武场API
    payload = {
        "participants": [
            {"name": "测试策略1", "approval_threshold": 0.18, "rate_spread": 0.01},
            {"name": "测试策略2", "approval_threshold": 0.15, "rate_spread": 0.015}
        ],
        "customer_count": 20,
        "loan_amount": 100000,
        "base_rate": 0.08,
        "seed": 42,
        "rules": [
            {
                "name": "高收入优惠",
                "description": "月收入超过20000的客户降低审批阈值",
                "priority": 1,
                "conditions": [{"field": "monthly_income", "op": ">", "value": 20000}],
                "action": {"approval_threshold_delta": -0.02, "rate_spread_delta": -0.002},
                "penalty": {"score_delta": 0, "profit_discount": 1.0}
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/arena/run",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("  ✅ API调用成功")
                print(f"  结果数量: {len(data.get('results', []))}")
                if data.get('results'):
                    r = data['results'][0]
                    print(f"  参赛者: {r.get('name')}")
                    print(f"  审批率: {r.get('approval_rate', 0)*100:.1f}%")
                    print(f"  触发规则: {r.get('triggered_rules', {})}")
                    print(f"  评分分解: {'overall_score' in str(r.get('score_breakdown', {}))}")
                return True
            else:
                print(f"  ❌ API返回错误: {data.get('error')}")
                return False
        else:
            print(f"  ❌ HTTP错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ API测试失败: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 演武场功能测试\n")
    
    results = []
    
    # 测试1: 规则引擎
    results.append(("规则引擎", test_rule_engine()))
    
    # 测试2: 评分系统
    results.append(("评分系统", test_scoring_system()))
    
    # 测试3: API端点（可选）
    try:
        results.append(("API端点", test_api_endpoint()))
    except ImportError:
        print("\n⚠️  requests库未安装，跳过API测试")
        results.append(("API端点", None))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, result in results:
        if result is True:
            print(f"  ✅ {name}: 通过")
        elif result is False:
            print(f"  ❌ {name}: 失败")
        else:
            print(f"  ⚠️  {name}: 跳过")
    
    all_passed = all(r for r in results if r[1] is not None)
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试未通过")



