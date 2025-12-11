"""
特征工程模块
用于从历史贷款数据中创建衍生特征、时间特征和交互特征
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化特征工程器
        
        Args:
            data: 历史贷款数据DataFrame
        """
        self.data = data.copy()
        self.feature_stats = {}  # 存储特征统计信息
    
    def create_derived_features(self) -> pd.DataFrame:
        """
        创建衍生特征
        
        Returns:
            添加了衍生特征的DataFrame
        """
        print("=" * 80)
        print("创建衍生特征")
        print("=" * 80)
        
        df = self.data.copy()
        
        # 1. 对私客户衍生特征
        personal_mask = df['customer_type'] == 'personal'
        
        if personal_mask.sum() > 0:
            personal_df = df[personal_mask]
            
            # 贷款金额/年收入比
            if 'loan_amount' in personal_df.columns and 'monthly_income' in personal_df.columns:
                df.loc[personal_mask, 'loan_to_annual_income_ratio'] = (
                    personal_df['loan_amount'] / (personal_df['monthly_income'] * 12 + 1e-6)
                )
                print(f"✅ 创建特征: loan_to_annual_income_ratio (对私)")
            
            # 月还款额/月收入比（偿债能力）
            if all(col in personal_df.columns for col in ['loan_amount', 'monthly_income', 'approved_rate', 'approved_term_months']):
                approved_personal = personal_df[personal_df['expert_decision'] == 'approve']
                if len(approved_personal) > 0:
                    # 计算月还款额
                    monthly_payment = self._calculate_monthly_payment(
                        approved_personal['approved_amount'],
                        approved_personal['approved_rate'],
                        approved_personal['approved_term_months']
                    )
                    df.loc[approved_personal.index, 'monthly_payment_to_income_ratio'] = (
                        monthly_payment / (approved_personal['monthly_income'] + 1e-6)
                    )
                    print(f"✅ 创建特征: monthly_payment_to_income_ratio (对私)")
            
            # 信用分等级
            if 'credit_score' in personal_df.columns:
                df.loc[personal_mask, 'credit_score_category'] = pd.cut(
                    personal_df['credit_score'],
                    bins=[0, 550, 650, 700, 750, 850],
                    labels=['很差', '较差', '一般', '良好', '优秀']
                )
                print(f"✅ 创建特征: credit_score_category (对私)")
            
            # 负债率等级
            if 'debt_ratio' in personal_df.columns:
                df.loc[personal_mask, 'debt_ratio_category'] = pd.cut(
                    personal_df['debt_ratio'],
                    bins=[0, 0.3, 0.5, 0.7, 1.0],
                    labels=['低', '中', '高', '很高']
                )
                print(f"✅ 创建特征: debt_ratio_category (对私)")
            
            # 收入稳定性（工作年限/年龄）
            if 'years_in_job' in personal_df.columns and 'age' in personal_df.columns:
                df.loc[personal_mask, 'job_stability'] = (
                    personal_df['years_in_job'] / (personal_df['age'] - 18 + 1e-6)
                )
                print(f"✅ 创建特征: job_stability (对私)")
            
            # 综合风险评分（简单加权）
            if all(col in personal_df.columns for col in ['credit_score', 'debt_ratio', 'monthly_income']):
                # 归一化到0-1
                credit_norm = (personal_df['credit_score'] - 300) / 550
                debt_norm = personal_df['debt_ratio']
                income_norm = (personal_df['monthly_income'] - 3000) / 47000
                
                # 综合评分（信用分权重0.5，负债率权重0.3，收入权重0.2）
                df.loc[personal_mask, 'comprehensive_risk_score'] = (
                    0.5 * (1 - credit_norm) +  # 信用分越低风险越高
                    0.3 * debt_norm +  # 负债率越高风险越高
                    0.2 * (1 - income_norm)  # 收入越低风险越高
                )
                print(f"✅ 创建特征: comprehensive_risk_score (对私)")
        
        # 2. 对公客户衍生特征
        corporate_mask = df['customer_type'] == 'corporate'
        
        if corporate_mask.sum() > 0:
            corporate_df = df[corporate_mask]
            
            # 贷款金额/年营收比
            if 'loan_amount' in corporate_df.columns and 'annual_revenue' in corporate_df.columns:
                df.loc[corporate_mask, 'loan_to_revenue_ratio'] = (
                    corporate_df['loan_amount'] / (corporate_df['annual_revenue'] + 1e-6)
                )
                print(f"✅ 创建特征: loan_to_revenue_ratio (对公)")
            
            # 年还款额/年营收比
            if all(col in corporate_df.columns for col in ['loan_amount', 'annual_revenue', 'approved_rate', 'approved_term_months']):
                approved_corporate = corporate_df[corporate_df['expert_decision'] == 'approve']
                if len(approved_corporate) > 0:
                    # 计算年还款额
                    monthly_payment = self._calculate_monthly_payment(
                        approved_corporate['approved_amount'],
                        approved_corporate['approved_rate'],
                        approved_corporate['approved_term_months']
                    )
                    annual_payment = monthly_payment * 12
                    df.loc[approved_corporate.index, 'annual_payment_to_revenue_ratio'] = (
                        annual_payment / (approved_corporate['annual_revenue'] + 1e-6)
                    )
                    print(f"✅ 创建特征: annual_payment_to_revenue_ratio (对公)")
            
            # 企业规模（基于年营收）
            if 'annual_revenue' in corporate_df.columns:
                df.loc[corporate_mask, 'revenue_size_category'] = pd.cut(
                    corporate_df['annual_revenue'],
                    bins=[0, 5000000, 50000000, 200000000, np.inf],
                    labels=['小型', '中型', '大型', '超大型']
                )
                print(f"✅ 创建特征: revenue_size_category (对公)")
            
            # 资产负债率等级
            if 'debt_to_asset_ratio' in corporate_df.columns:
                df.loc[corporate_mask, 'debt_to_asset_category'] = pd.cut(
                    corporate_df['debt_to_asset_ratio'],
                    bins=[0, 0.4, 0.6, 0.8, 1.0],
                    labels=['低', '中', '高', '很高']
                )
                print(f"✅ 创建特征: debt_to_asset_category (对公)")
            
            # 流动比率等级
            if 'current_ratio' in corporate_df.columns:
                df.loc[corporate_mask, 'current_ratio_category'] = pd.cut(
                    corporate_df['current_ratio'],
                    bins=[0, 1.0, 1.5, 2.0, np.inf],
                    labels=['差', '一般', '良好', '优秀']
                )
                print(f"✅ 创建特征: current_ratio_category (对公)")
            
            # 企业成熟度（经营年限/注册资本）
            if 'operating_years' in corporate_df.columns and 'registered_capital' in corporate_df.columns:
                df.loc[corporate_mask, 'business_maturity'] = (
                    corporate_df['operating_years'] * 
                    np.log1p(corporate_df['registered_capital'] / 1000000)
                )
                print(f"✅ 创建特征: business_maturity (对公)")
            
            # 综合风险评分（对公）
            if all(col in corporate_df.columns for col in ['debt_to_asset_ratio', 'current_ratio', 'operating_years']):
                debt_norm = corporate_df['debt_to_asset_ratio']
                current_norm = 1 / (corporate_df['current_ratio'] + 0.5)  # 流动比率越低风险越高
                years_norm = 1 / (corporate_df['operating_years'] + 1)  # 经营年限越短风险越高
                
                df.loc[corporate_mask, 'comprehensive_risk_score'] = (
                    0.4 * debt_norm +
                    0.3 * current_norm +
                    0.3 * years_norm
                )
                print(f"✅ 创建特征: comprehensive_risk_score (对公)")
        
        print(f"\n✅ 衍生特征创建完成，新增 {len(df.columns) - len(self.data.columns)} 个特征")
        return df
    
    def create_temporal_features(self) -> pd.DataFrame:
        """
        创建时间特征
        
        Returns:
            添加了时间特征的DataFrame
        """
        print("\n" + "=" * 80)
        print("创建时间特征")
        print("=" * 80)
        
        df = self.data.copy()
        
        # 1. 申请时间特征
        if 'application_date' in df.columns:
            app_dates = pd.to_datetime(df['application_date'], errors='coerce')
            
            # 年、月、季度
            df['application_year'] = app_dates.dt.year
            df['application_month'] = app_dates.dt.month
            df['application_quarter'] = app_dates.dt.quarter
            df['application_day_of_week'] = app_dates.dt.dayofweek
            df['application_is_weekend'] = app_dates.dt.dayofweek >= 5
            
            # 是否月初/月末
            df['application_is_month_start'] = app_dates.dt.is_month_start
            df['application_is_month_end'] = app_dates.dt.is_month_end
            
            # 是否季度末
            df['application_is_quarter_end'] = app_dates.dt.is_quarter_end
            
            print("✅ 创建申请时间特征")
        
        # 2. 审批时间特征
        if 'approval_date' in df.columns:
            approval_dates = pd.to_datetime(df['approval_date'], errors='coerce')
            
            # 审批延迟天数
            if 'application_date' in df.columns:
                app_dates = pd.to_datetime(df['application_date'], errors='coerce')
                df['approval_delay_days'] = (approval_dates - app_dates).dt.days
            
            print("✅ 创建审批时间特征")
        
        # 3. 违约时间特征（如果有违约）
        if 'default_date' in df.columns:
            default_dates = pd.to_datetime(df['default_date'], errors='coerce')
            approved_mask = df['expert_decision'] == 'approve'
            
            if approved_mask.sum() > 0:
                # 从审批到违约的天数
                if 'approval_date' in df.columns:
                    approval_dates = pd.to_datetime(df.loc[approved_mask, 'approval_date'], errors='coerce')
                    default_dates_approved = default_dates[approved_mask]
                    df.loc[approved_mask, 'days_to_default'] = (
                        (default_dates_approved - approval_dates).dt.days
                    )
                
                # 违约月份（在贷款期限中的位置）
                if all(col in df.columns for col in ['days_to_default', 'approved_term_months']):
                    approved_with_default = df[approved_mask & df['actual_defaulted'] == True]
                    if len(approved_with_default) > 0:
                        df.loc[approved_with_default.index, 'default_month'] = (
                            approved_with_default['days_to_default'] / 30
                        )
            
            print("✅ 创建违约时间特征")
        
        print(f"\n✅ 时间特征创建完成，新增 {len(df.columns) - len(self.data.columns)} 个特征")
        return df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        创建交互特征
        
        Returns:
            添加了交互特征的DataFrame
        """
        print("\n" + "=" * 80)
        print("创建交互特征")
        print("=" * 80)
        
        df = self.data.copy()
        
        # 1. 对私客户交互特征
        personal_mask = df['customer_type'] == 'personal'
        
        if personal_mask.sum() > 0:
            personal_df = df[personal_mask]
            
            # 信用分 × (1 - 负债率) - 信用质量调整
            if 'credit_score' in personal_df.columns and 'debt_ratio' in personal_df.columns:
                df.loc[personal_mask, 'credit_debt_interaction'] = (
                    personal_df['credit_score'] * (1 - personal_df['debt_ratio'])
                )
                print("✅ 创建特征: credit_debt_interaction (对私)")
            
            # 收入 × 工作稳定性
            if all(col in personal_df.columns for col in ['monthly_income', 'years_in_job']):
                df.loc[personal_mask, 'income_stability_interaction'] = (
                    personal_df['monthly_income'] * np.log1p(personal_df['years_in_job'])
                )
                print("✅ 创建特征: income_stability_interaction (对私)")
            
            # 年龄 × 收入 - 生命周期收入
            if 'age' in personal_df.columns and 'monthly_income' in personal_df.columns:
                df.loc[personal_mask, 'age_income_interaction'] = (
                    personal_df['age'] * personal_df['monthly_income']
                )
                print("✅ 创建特征: age_income_interaction (对私)")
        
        # 2. 对公客户交互特征
        corporate_mask = df['customer_type'] == 'corporate'
        
        if corporate_mask.sum() > 0:
            corporate_df = df[corporate_mask]
            
            # 经营年限 × 年营收 - 企业成长性
            if 'operating_years' in corporate_df.columns and 'annual_revenue' in corporate_df.columns:
                df.loc[corporate_mask, 'years_revenue_interaction'] = (
                    corporate_df['operating_years'] * np.log1p(corporate_df['annual_revenue'] / 1000000)
                )
                print("✅ 创建特征: years_revenue_interaction (对公)")
            
            # 资产负债率 × 流动比率 - 财务健康度
            if 'debt_to_asset_ratio' in corporate_df.columns and 'current_ratio' in corporate_df.columns:
                df.loc[corporate_mask, 'debt_liquidity_interaction'] = (
                    (1 - corporate_df['debt_to_asset_ratio']) * corporate_df['current_ratio']
                )
                print("✅ 创建特征: debt_liquidity_interaction (对公)")
        
        # 3. 通用交互特征
        # 贷款金额 × 利率 - 利息负担
        if all(col in df.columns for col in ['approved_amount', 'approved_rate']):
            approved_mask = df['expert_decision'] == 'approve'
            if approved_mask.sum() > 0:
                df.loc[approved_mask, 'loan_interest_interaction'] = (
                    df.loc[approved_mask, 'approved_amount'] * 
                    df.loc[approved_mask, 'approved_rate']
                )
                print("✅ 创建特征: loan_interest_interaction (通用)")
        
        # 贷款金额 × 期限 - 总还款负担
        if all(col in df.columns for col in ['approved_amount', 'approved_term_months']):
            approved_mask = df['expert_decision'] == 'approve'
            if approved_mask.sum() > 0:
                df.loc[approved_mask, 'loan_term_interaction'] = (
                    df.loc[approved_mask, 'approved_amount'] * 
                    df.loc[approved_mask, 'approved_term_months']
                )
                print("✅ 创建特征: loan_term_interaction (通用)")
        
        print(f"\n✅ 交互特征创建完成，新增 {len(df.columns) - len(self.data.columns)} 个特征")
        return df
    
    def create_target_features(self) -> pd.DataFrame:
        """
        创建目标特征（用于模型训练）
        
        Returns:
            添加了目标特征的DataFrame
        """
        print("\n" + "=" * 80)
        print("创建目标特征")
        print("=" * 80)
        
        df = self.data.copy()
        
        # 1. 是否批准（二分类）
        if 'expert_decision' in df.columns:
            df['is_approved'] = (df['expert_decision'] == 'approve').astype(int)
            print("✅ 创建特征: is_approved")
        
        # 2. 是否违约（二分类，只针对批准的贷款）
        if 'actual_defaulted' in df.columns:
            df['is_defaulted'] = df['actual_defaulted'].astype(int)
            print("✅ 创建特征: is_defaulted")
        
        # 3. 利润等级（多分类）
        if 'actual_profit' in df.columns:
            approved_mask = df['expert_decision'] == 'approve'
            if approved_mask.sum() > 0:
                profit = df.loc[approved_mask, 'actual_profit']
                df.loc[approved_mask, 'profit_category'] = pd.cut(
                    profit,
                    bins=[-np.inf, 0, 1000, 10000, 50000, np.inf],
                    labels=['亏损', '微利', '小利', '中利', '大利']
                )
                print("✅ 创建特征: profit_category")
        
        # 4. ROI等级
        if 'actual_roi' in df.columns:
            approved_mask = df['expert_decision'] == 'approve'
            if approved_mask.sum() > 0:
                roi = df.loc[approved_mask, 'actual_roi']
                df.loc[approved_mask, 'roi_category'] = pd.cut(
                    roi,
                    bins=[-np.inf, 0, 0.05, 0.1, 0.2, np.inf],
                    labels=['负收益', '低收益', '中收益', '高收益', '超高收益']
                )
                print("✅ 创建特征: roi_category")
        
        print(f"\n✅ 目标特征创建完成")
        return df
    
    def _calculate_monthly_payment(self, principal: pd.Series, rate: pd.Series, 
                                   term_months: pd.Series) -> pd.Series:
        """计算月还款额"""
        monthly_rate = rate / 12
        # 等额本息
        payment = principal * monthly_rate * (1 + monthly_rate) ** term_months / \
                 ((1 + monthly_rate) ** term_months - 1 + 1e-10)
        return payment
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        执行所有特征工程
        
        Returns:
            包含所有新特征的DataFrame
        """
        print("\n" + "=" * 80)
        print("特征工程 - 完整流程")
        print("=" * 80)
        print()
        
        original_cols = len(self.data.columns)
        
        # 1. 衍生特征
        df = self.create_derived_features()
        
        # 2. 时间特征
        df = FeatureEngineer(df).create_temporal_features()
        
        # 3. 交互特征
        df = FeatureEngineer(df).create_interaction_features()
        
        # 4. 目标特征
        df = FeatureEngineer(df).create_target_features()
        
        # 统计信息
        new_cols = len(df.columns) - original_cols
        print("\n" + "=" * 80)
        print("特征工程完成")
        print("=" * 80)
        print(f"原始特征数: {original_cols}")
        print(f"新增特征数: {new_cols}")
        print(f"总特征数: {len(df.columns)}")
        print("=" * 80)
        
        return df
    
    def save_engineered_data(self, df: pd.DataFrame, output_path: str = 'data/historical/historical_loans_engineered.csv'):
        """保存特征工程后的数据"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 已保存特征工程后的数据到: {output_path}")
        
        # 保存Parquet
        parquet_path = output_path.replace('.csv', '.parquet')
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"✅ 已保存Parquet格式到: {parquet_path}")


def main():
    """主函数：执行特征工程"""
    import sys
    import os
    
    # 加载数据
    data_path = 'data/historical/historical_loans.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行 historical_data_generator.py 生成数据")
        sys.exit(1)
    
    print("正在加载数据...")
    data = pd.read_csv(data_path)
    print(f"✅ 已加载 {len(data)} 条记录，{len(data.columns)} 个特征")
    
    # 执行特征工程
    engineer = FeatureEngineer(data)
    engineered_data = engineer.engineer_all_features()
    
    # 保存结果
    engineer.save_engineered_data(engineered_data)
    
    return engineered_data


if __name__ == '__main__':
    main()

