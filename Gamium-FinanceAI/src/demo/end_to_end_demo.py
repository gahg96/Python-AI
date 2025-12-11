"""
端到端集成
整合所有模块，实现完整流程
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime

# 导入所有模块
from historical_data_generator import HistoricalLoanDataGenerator
from data_quality_checker import DataQualityChecker
from feature_engineer import FeatureEngineer
from rule_extractor import RuleExtractor
from rule_quantifier import RuleQuantifier
from enhanced_customer_generator import EnhancedCustomerGenerator
from market_simulator import MarketSimulator
from world_model_trainer import WorldModelTrainer
from enhanced_rule_engine import EnhancedRuleEngine
from model_decision import ModelDecisionMaker
from decision_fusion import DecisionFusion
from repayment_simulator import RepaymentSimulator
from recovery_calculator import RecoveryCalculator
from result_validator import ResultValidator


class EndToEndDemo:
    """端到端Demo"""
    
    def __init__(self, data_dir: str = 'data/historical'):
        """
        初始化Demo
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.historical_data = None
        self.engineered_data = None
        self.models = {}
        self.rules = {}
    
    def run_complete_pipeline(self, num_historical_loans: int = 10000,
                             num_simulated_customers: int = 100):
        """
        运行完整流程
        
        Args:
            num_historical_loans: 历史贷款数量
            num_simulated_customers: 模拟客户数量
        """
        print("=" * 80)
        print("端到端贷款审批Demo - 完整流程")
        print("=" * 80)
        print()
        
        # 阶段1：历史数据准备
        print("=" * 80)
        print("阶段1：历史数据准备")
        print("=" * 80)
        self._prepare_historical_data(num_historical_loans)
        
        # 阶段2：数据质量检查
        print("\n" + "=" * 80)
        print("阶段2：数据质量检查")
        print("=" * 80)
        self._check_data_quality()
        
        # 阶段3：特征工程
        print("\n" + "=" * 80)
        print("阶段3：特征工程")
        print("=" * 80)
        self._engineer_features()
        
        # 阶段4：规则提取和量化
        print("\n" + "=" * 80)
        print("阶段4：规则提取和量化")
        print("=" * 80)
        self._extract_and_quantify_rules()
        
        # 阶段5：模型训练
        print("\n" + "=" * 80)
        print("阶段5：模型训练")
        print("=" * 80)
        self._train_models()
        
        # 阶段6：模拟审批
        print("\n" + "=" * 80)
        print("阶段6：模拟审批")
        print("=" * 80)
        simulated_results = self._simulate_approval(num_simulated_customers)
        
        # 阶段7：结果验证
        print("\n" + "=" * 80)
        print("阶段7：结果验证")
        print("=" * 80)
        self._validate_results(simulated_results)
        
        print("\n" + "=" * 80)
        print("✅ 端到端Demo完成！")
        print("=" * 80)
        
        return simulated_results
    
    def _prepare_historical_data(self, num_loans: int):
        """准备历史数据"""
        data_path = os.path.join(self.data_dir, 'historical_loans.csv')
        
        if os.path.exists(data_path):
            print(f"✅ 历史数据已存在: {data_path}")
            self.historical_data = pd.read_csv(data_path)
        else:
            print("正在生成历史数据...")
            generator = HistoricalLoanDataGenerator(seed=42)
            self.historical_data = generator.generate_historical_loans(
                num_loans=num_loans,
                personal_ratio=0.7
            )
            generator.save_to_files(self.historical_data, self.data_dir)
        
        print(f"✅ 历史数据: {len(self.historical_data)} 条记录")
    
    def _check_data_quality(self):
        """数据质量检查"""
        checker = DataQualityChecker(self.historical_data)
        result = checker.comprehensive_check()
        checker.save_report(os.path.join(self.data_dir, 'quality_report.json'))
        print(f"✅ 数据质量得分: {result.get('overall_score', 0):.4f}")
    
    def _engineer_features(self):
        """特征工程"""
        engineered_path = os.path.join(self.data_dir, 'historical_loans_engineered.csv')
        
        if os.path.exists(engineered_path):
            print(f"✅ 特征工程数据已存在: {engineered_path}")
            self.engineered_data = pd.read_csv(engineered_path)
        else:
            print("正在进行特征工程...")
            engineer = FeatureEngineer(self.historical_data)
            self.engineered_data = engineer.engineer_all_features()
            engineer.save_engineered_data(self.engineered_data, engineered_path)
        
        print(f"✅ 特征工程完成: {len(self.engineered_data.columns)} 个特征")
    
    def _extract_and_quantify_rules(self):
        """提取和量化规则"""
        rules_path = os.path.join(self.data_dir, 'extracted_rules.json')
        quantified_path = os.path.join(self.data_dir, 'quantified_rules.json')
        
        if os.path.exists(rules_path) and os.path.exists(quantified_path):
            print(f"✅ 规则已存在")
        else:
            print("正在提取规则...")
            extractor = RuleExtractor(self.engineered_data)
            rules = extractor.extract_all_rules('both')
            extractor.save_rules(rules_path)
            
            print("正在量化规则...")
            quantifier = RuleQuantifier(rules)
            quantified = quantifier.quantify_all_rules()
            quantifier.save_quantified_rules(quantified_path)
        
        print(f"✅ 规则提取和量化完成")
    
    def _train_models(self):
        """训练模型"""
        models_dir = os.path.join(self.data_dir, 'models')
        
        if os.path.exists(os.path.join(models_dir, 'default_prediction.pkl')):
            print(f"✅ 模型已存在")
        else:
            print("正在训练模型...")
            trainer = WorldModelTrainer(self.engineered_data, seed=42)
            trainer.train_all_models()
            trainer.save_models(models_dir)
        
        print(f"✅ 模型训练完成")
    
    def _simulate_approval(self, num_customers: int) -> pd.DataFrame:
        """模拟审批"""
        print(f"正在生成 {num_customers} 个模拟客户...")
        
        # 生成客户
        customer_gen = EnhancedCustomerGenerator(self.engineered_data, seed=42)
        customers = customer_gen.generate_customers(
            num_personal=int(num_customers * 0.7),
            num_corporate=int(num_customers * 0.3)
        )
        
        # 生成市场环境
        market_sim = MarketSimulator(seed=42, start_date='2024-01-01')
        current_market = market_sim.generate_market_condition(datetime(2024, 6, 15))
        market_dict = {
            'gdp_growth': current_market.gdp_growth,
            'base_interest_rate': current_market.base_interest_rate,
            'unemployment_rate': current_market.unemployment_rate,
            'inflation_rate': current_market.inflation_rate,
            'credit_spread': current_market.credit_spread,
            'market_sentiment': current_market.market_sentiment
        }
        
        # 加载决策组件
        model_decision_maker = ModelDecisionMaker(
            models_dir=os.path.join(self.data_dir, 'models')
        )
        rule_engine = EnhancedRuleEngine()
        rule_engine.load_rules_from_file(
            os.path.join(self.data_dir, 'extracted_rules.json'),
            os.path.join(self.data_dir, 'quantified_rules.json')
        )
        fusion = DecisionFusion(model_decision_maker, rule_engine)
        
        # 模拟还款和回收
        repayment_sim = RepaymentSimulator(seed=42)
        recovery_calc = RecoveryCalculator(seed=42)
        
        # 处理每个客户
        results = []
        for customer in customers:
            # 生成贷款申请
            if customer['customer_type'] == 'personal':
                loan_amount = np.random.uniform(10000, customer['monthly_income'] * 12 * 0.5)
            else:
                loan_amount = np.random.uniform(100000, customer['annual_revenue'] * 0.3)
            
            loan = {
                'loan_amount': loan_amount,
                'approved_rate': 0.08,
                'approved_term_months': 24
            }
            
            # 融合决策
            fused_decision = fusion.fuse_decisions(customer, loan, market_dict)
            
            # 如果批准，模拟还款
            if fused_decision.final_decision == 'approve':
                repayment_result = repayment_sim.simulate_repayment(
                    loan_amount=loan_amount,
                    interest_rate=fused_decision.model_decision.default_probability * 0.1 + 0.08,
                    term_months=24,
                    default_probability=fused_decision.default_probability,
                    customer_data=customer
                )
                
                # 如果违约，计算回收
                if repayment_result.defaulted:
                    recovery_result = recovery_calc.calculate_recovery(
                        repayment_result, loan_amount, customer, market_dict
                    )
                else:
                    recovery_result = None
            else:
                repayment_result = None
                recovery_result = None
            
            # 记录结果
            result = {
                **customer,
                **loan,
                'expert_decision': fused_decision.final_decision,  # 添加expert_decision字段
                'decision': fused_decision.final_decision,
                'default_probability': fused_decision.default_probability,
                'expected_profit': fused_decision.expected_profit,
                'actual_defaulted': repayment_result.defaulted if repayment_result else False,
                'actual_profit': repayment_result.total_interest_paid - (
                    recovery_result.default_amount - recovery_result.recovery_amount
                ) if recovery_result else repayment_result.total_interest_paid if repayment_result else 0,
                'recovery_rate': recovery_result.recovery_rate if recovery_result else 0,
                'recovery_amount': recovery_result.recovery_amount if recovery_result else 0,
                'default_amount': recovery_result.default_amount if recovery_result else 0
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"✅ 模拟完成: {len(results_df)} 个客户")
        print(f"   审批通过: {(results_df['decision'] == 'approve').sum()}")
        print(f"   实际违约: {results_df['actual_defaulted'].sum()}")
        
        return results_df
    
    def _validate_results(self, simulated_results: pd.DataFrame):
        """验证结果"""
        validator = ResultValidator(self.engineered_data, simulated_results)
        validation_results = validator.comprehensive_validation()
        validator.save_validation_report(
            validation_results,
            os.path.join(self.data_dir, 'validation_report.json')
        )


def main():
    """主函数：运行端到端Demo"""
    demo = EndToEndDemo()
    results = demo.run_complete_pipeline(
        num_historical_loans=10000,
        num_simulated_customers=100
    )
    
    print(f"\n✅ Demo完成！模拟了 {len(results)} 个客户的审批流程")
    return demo, results


if __name__ == '__main__':
    main()

