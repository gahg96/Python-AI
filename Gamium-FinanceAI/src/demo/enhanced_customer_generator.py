"""
增强版客户生成器
学习历史数据分布，生成真实的客户特征
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from scipy.stats import beta, lognorm, norm, poisson
import json
import os


class EnhancedCustomerGenerator:
    """增强版客户生成器"""
    
    def __init__(self, historical_data: pd.DataFrame, seed: int = 42):
        """
        初始化客户生成器
        
        Args:
            historical_data: 历史贷款数据
            seed: 随机种子
        """
        self.data = historical_data
        self.rng = np.random.default_rng(seed)
        
        # 学习分布
        print("正在学习历史数据分布...")
        self.personal_distributions = self._learn_distributions('personal')
        self.corporate_distributions = self._learn_distributions('corporate')
        print("✅ 分布学习完成")
    
    def _learn_distributions(self, customer_type: str) -> Dict:
        """学习历史数据的分布"""
        data = self.data[self.data['customer_type'] == customer_type]
        
        if len(data) == 0:
            return {}
        
        distributions = {}
        
        if customer_type == 'personal':
            # 年龄分布（Beta分布，映射到18-65）
            if 'age' in data.columns:
                age_data = data['age'].dropna()
                if len(age_data) > 0:
                    # 归一化到0-1
                    age_norm = (age_data - 18) / (65 - 18)
                    age_norm = np.clip(age_norm, 0.001, 0.999)  # 避免边界值
                    try:
                        a, b, loc, scale = beta.fit(age_norm, floc=0, fscale=1)
                        distributions['age'] = {
                            'type': 'beta',
                            'params': (a, b),
                            'min': 18,
                            'max': 65
                        }
                    except:
                        # 如果拟合失败，使用正态分布
                        mu, sigma = norm.fit(age_data)
                        distributions['age'] = {
                            'type': 'normal',
                            'params': (mu, sigma),
                            'min': 18,
                            'max': 65
                        }
            
            # 月收入分布（对数正态）
            if 'monthly_income' in data.columns:
                income_data = data['monthly_income'].dropna()
                income_data = income_data[income_data > 0]  # 只保留正值
                if len(income_data) > 0:
                    try:
                        # 对数正态分布
                        shape, loc, scale = lognorm.fit(income_data, floc=0)
                        distributions['monthly_income'] = {
                            'type': 'lognormal',
                            'params': (shape, loc, scale)
                        }
                    except:
                        # 如果拟合失败，使用正态分布
                        mu, sigma = norm.fit(income_data)
                        distributions['monthly_income'] = {
                            'type': 'normal',
                            'params': (mu, sigma)
                        }
            
            # 信用分分布（正态分布）
            if 'credit_score' in data.columns:
                score_data = data['credit_score'].dropna()
                if len(score_data) > 0:
                    mu, sigma = norm.fit(score_data)
                    distributions['credit_score'] = {
                        'type': 'normal',
                        'params': (mu, sigma),
                        'min': 300,
                        'max': 850
                    }
            
            # 负债率分布（Beta分布，0-1）
            if 'debt_ratio' in data.columns:
                debt_data = data['debt_ratio'].dropna()
                if len(debt_data) > 0:
                    debt_data = np.clip(debt_data, 0.001, 0.999)  # 避免边界值
                    try:
                        a, b, loc, scale = beta.fit(debt_data, floc=0, fscale=1)
                        distributions['debt_ratio'] = {
                            'type': 'beta',
                            'params': (a, b),
                            'min': 0,
                            'max': 1
                        }
                    except:
                        # 如果拟合失败，使用均匀分布
                        distributions['debt_ratio'] = {
                            'type': 'uniform',
                            'params': (0, 1)
                        }
            
            # 工作年限（与年龄相关）
            if 'years_in_job' in data.columns and 'age' in data.columns:
                years_data = data['years_in_job'].dropna()
                age_data = data['age'].dropna()
                if len(years_data) > 0:
                    # 工作年限通常不超过年龄-18
                    max_years = (age_data - 18).max()
                    # 使用Beta分布（避免边界值）
                    years_norm = years_data / (max_years + 1)
                    years_norm = np.clip(years_norm, 0.001, 0.999)  # 避免0和1
                    try:
                        a, b, loc, scale = beta.fit(years_norm, floc=0, fscale=1)
                        distributions['years_in_job'] = {
                            'type': 'beta',
                            'params': (a, b),
                            'max_ratio': max_years
                        }
                    except:
                        # 如果拟合失败，使用简单分布
                        distributions['years_in_job'] = {
                            'type': 'uniform',
                            'params': (0, max_years)
                        }
            
            # 分类特征的概率分布
            if 'employment_status' in data.columns:
                emp_dist = data['employment_status'].value_counts(normalize=True).to_dict()
                distributions['employment_status'] = {
                    'type': 'categorical',
                    'probs': emp_dist
                }
            
            if 'education_level' in data.columns:
                edu_dist = data['education_level'].value_counts(normalize=True).to_dict()
                distributions['education_level'] = {
                    'type': 'categorical',
                    'probs': edu_dist
                }
            
            if 'marital_status' in data.columns:
                marital_dist = data['marital_status'].value_counts(normalize=True).to_dict()
                distributions['marital_status'] = {
                    'type': 'categorical',
                    'probs': marital_dist
                }
            
            # 抵押物（二项分布）
            if 'has_collateral' in data.columns:
                collateral_rate = data['has_collateral'].mean()
                distributions['has_collateral'] = {
                    'type': 'bernoulli',
                    'prob': collateral_rate
                }
            
            if 'collateral_value' in data.columns:
                collateral_data = data[data['has_collateral'] == True]['collateral_value'].dropna()
                collateral_data = collateral_data[collateral_data > 0]  # 只保留正值
                if len(collateral_data) > 0:
                    try:
                        shape, loc, scale = lognorm.fit(collateral_data, floc=0)
                        distributions['collateral_value'] = {
                            'type': 'lognormal',
                            'params': (shape, loc, scale)
                        }
                    except:
                        # 如果拟合失败，使用指数分布
                        mean_val = collateral_data.mean()
                        distributions['collateral_value'] = {
                            'type': 'exponential',
                            'params': (mean_val,)
                        }
        
        elif customer_type == 'corporate':
            # 注册资本（对数正态）
            if 'registered_capital' in data.columns:
                capital_data = data['registered_capital'].dropna()
                capital_data = capital_data[capital_data > 0]  # 只保留正值
                if len(capital_data) > 0:
                    try:
                        shape, loc, scale = lognorm.fit(capital_data, floc=0)
                        distributions['registered_capital'] = {
                            'type': 'lognormal',
                            'params': (shape, loc, scale)
                        }
                    except:
                        # 如果拟合失败，使用正态分布
                        mu, sigma = norm.fit(capital_data)
                        distributions['registered_capital'] = {
                            'type': 'normal',
                            'params': (mu, sigma)
                        }
            
            # 经营年限（泊松或Beta）
            if 'operating_years' in data.columns:
                years_data = data['operating_years'].dropna()
                if len(years_data) > 0:
                    # 使用泊松分布
                    mu = years_data.mean()
                    distributions['operating_years'] = {
                        'type': 'poisson',
                        'params': (mu,)
                    }
            
            # 年营收（对数正态，与注册资本相关）
            if 'annual_revenue' in data.columns:
                revenue_data = data['annual_revenue'].dropna()
                revenue_data = revenue_data[revenue_data > 0]  # 只保留正值
                if len(revenue_data) > 0:
                    try:
                        shape, loc, scale = lognorm.fit(revenue_data, floc=0)
                        distributions['annual_revenue'] = {
                            'type': 'lognormal',
                            'params': (shape, loc, scale)
                        }
                    except:
                        # 如果拟合失败，使用正态分布
                        mu, sigma = norm.fit(revenue_data)
                        distributions['annual_revenue'] = {
                            'type': 'normal',
                            'params': (mu, sigma)
                        }
            
            # 资产负债率（Beta分布）
            if 'debt_to_asset_ratio' in data.columns:
                debt_data = data['debt_to_asset_ratio'].dropna()
                if len(debt_data) > 0:
                    debt_data = np.clip(debt_data, 0.001, 0.999)  # 避免边界值
                    try:
                        a, b, loc, scale = beta.fit(debt_data, floc=0, fscale=1)
                        distributions['debt_to_asset_ratio'] = {
                            'type': 'beta',
                            'params': (a, b),
                            'min': 0,
                            'max': 1
                        }
                    except:
                        # 如果拟合失败，使用均匀分布
                        distributions['debt_to_asset_ratio'] = {
                            'type': 'uniform',
                            'params': (0, 1)
                        }
            
            # 流动比率（正态分布）
            if 'current_ratio' in data.columns:
                ratio_data = data['current_ratio'].dropna()
                if len(ratio_data) > 0:
                    mu, sigma = norm.fit(ratio_data)
                    distributions['current_ratio'] = {
                        'type': 'normal',
                        'params': (mu, sigma)
                    }
            
            # 行业分布
            if 'industry' in data.columns:
                industry_dist = data['industry'].value_counts(normalize=True).to_dict()
                distributions['industry'] = {
                    'type': 'categorical',
                    'probs': industry_dist
                }
            
            # 公司规模（基于年营收）
            if 'company_size' in data.columns:
                size_dist = data['company_size'].value_counts(normalize=True).to_dict()
                distributions['company_size'] = {
                    'type': 'categorical',
                    'probs': size_dist
                }
        
        return distributions
    
    def _sample_from_distribution(self, dist: Dict) -> float:
        """从分布中采样"""
        dist_type = dist['type']
        
        if dist_type in ['categorical', 'bernoulli']:
            # 分类分布和伯努利分布不需要params
            pass
        else:
            params = dist['params']
        
        if dist_type == 'beta':
            a, b = params
            sample = self.rng.beta(a, b)
            # 映射到实际范围
            if 'min' in dist and 'max' in dist:
                sample = sample * (dist['max'] - dist['min']) + dist['min']
            return sample
        
        elif dist_type == 'lognormal':
            shape, loc, scale = params
            sample = lognorm.rvs(shape, loc=loc, scale=scale, random_state=self.rng)
            return max(0, sample)
        
        elif dist_type == 'exponential':
            mean_val = params[0]
            sample = self.rng.exponential(mean_val)
            return max(0, sample)
        
        elif dist_type == 'normal':
            mu, sigma = params
            sample = self.rng.normal(mu, sigma)
            # 限制范围
            if 'min' in dist:
                sample = max(sample, dist['min'])
            if 'max' in dist:
                sample = min(sample, dist['max'])
            return sample
        
        elif dist_type == 'poisson':
            mu = params[0]
            sample = self.rng.poisson(mu)
            return max(1, sample)
        
        elif dist_type == 'uniform':
            min_val, max_val = params
            sample = self.rng.uniform(min_val, max_val)
            return sample
        
        elif dist_type == 'categorical':
            probs = dist['probs']
            items = list(probs.keys())
            weights = list(probs.values())
            return self.rng.choice(items, p=weights)
        
        elif dist_type == 'bernoulli':
            prob = dist['prob']
            return self.rng.random() < prob
        
        else:
            return 0.0
    
    def generate_personal_customer(self, customer_id: Optional[str] = None) -> Dict:
        """生成对私客户"""
        if customer_id is None:
            customer_id = f"P{self.rng.integers(1000000, 9999999)}"
        
        dist = self.personal_distributions
        
        customer = {
            'customer_id': customer_id,
            'customer_type': 'personal',
        }
        
        # 生成数值特征
        if 'age' in dist:
            customer['age'] = int(self._sample_from_distribution(dist['age']))
        else:
            customer['age'] = int(self.rng.integers(22, 65))
        
        if 'monthly_income' in dist:
            customer['monthly_income'] = round(self._sample_from_distribution(dist['monthly_income']), 2)
            customer['monthly_income'] = max(3000, min(customer['monthly_income'], 50000))
        else:
            customer['monthly_income'] = round(self.rng.uniform(3000, 20000), 2)
        
        if 'credit_score' in dist:
            customer['credit_score'] = int(self._sample_from_distribution(dist['credit_score']))
        else:
            customer['credit_score'] = int(self.rng.integers(300, 850))
        
        if 'debt_ratio' in dist:
            customer['debt_ratio'] = round(self._sample_from_distribution(dist['debt_ratio']), 4)
            customer['debt_ratio'] = min(customer['debt_ratio'], 0.95)
        else:
            customer['debt_ratio'] = round(self.rng.uniform(0, 0.8), 4)
        
        if 'years_in_job' in dist:
            max_years = min(customer['age'] - 18, 30)
            years_ratio = self._sample_from_distribution(dist['years_in_job'])
            customer['years_in_job'] = int(years_ratio * max_years)
        else:
            customer['years_in_job'] = int(self.rng.integers(0, min(customer['age'] - 18, 30)))
        
        # 生成分类特征
        if 'employment_status' in dist:
            customer['employment_status'] = self._sample_from_distribution(dist['employment_status'])
        else:
            customer['employment_status'] = self.rng.choice(['在职', '自由职业', '个体经营', '退休', '待业'])
        
        if 'education_level' in dist:
            customer['education_level'] = self._sample_from_distribution(dist['education_level'])
        else:
            customer['education_level'] = self.rng.choice(['初中及以下', '高中', '大专', '本科', '硕士', '博士'])
        
        if 'marital_status' in dist:
            customer['marital_status'] = self._sample_from_distribution(dist['marital_status'])
        else:
            customer['marital_status'] = self.rng.choice(['未婚', '已婚', '离异', '丧偶'])
        
        # 抵押物
        if 'has_collateral' in dist:
            customer['has_collateral'] = self._sample_from_distribution(dist['has_collateral'])
        else:
            customer['has_collateral'] = self.rng.random() < 0.3
        
        if customer['has_collateral'] and 'collateral_value' in dist:
            customer['collateral_value'] = round(self._sample_from_distribution(dist['collateral_value']), 2)
        else:
            customer['collateral_value'] = 0.0
        
        return customer
    
    def generate_corporate_customer(self, customer_id: Optional[str] = None) -> Dict:
        """生成对公客户"""
        if customer_id is None:
            customer_id = f"C{self.rng.integers(1000000, 9999999)}"
        
        dist = self.corporate_distributions
        
        customer = {
            'customer_id': customer_id,
            'customer_type': 'corporate',
        }
        
        # 生成数值特征
        if 'registered_capital' in dist:
            customer['registered_capital'] = round(self._sample_from_distribution(dist['registered_capital']), 2)
            customer['registered_capital'] = max(100000, min(customer['registered_capital'], 100000000))
        else:
            customer['registered_capital'] = round(self.rng.uniform(100000, 10000000), 2)
        
        if 'operating_years' in dist:
            customer['operating_years'] = int(self._sample_from_distribution(dist['operating_years']))
        else:
            customer['operating_years'] = int(self.rng.integers(1, 20))
        
        if 'annual_revenue' in dist:
            customer['annual_revenue'] = round(self._sample_from_distribution(dist['annual_revenue']), 2)
            customer['annual_revenue'] = max(500000, customer['annual_revenue'])
        else:
            customer['annual_revenue'] = round(self.rng.uniform(1000000, 50000000), 2)
        
        if 'debt_to_asset_ratio' in dist:
            customer['debt_to_asset_ratio'] = round(self._sample_from_distribution(dist['debt_to_asset_ratio']), 4)
            customer['debt_to_asset_ratio'] = min(customer['debt_to_asset_ratio'], 0.9)
        else:
            customer['debt_to_asset_ratio'] = round(self.rng.uniform(0.3, 0.8), 4)
        
        if 'current_ratio' in dist:
            customer['current_ratio'] = round(self._sample_from_distribution(dist['current_ratio']), 2)
            customer['current_ratio'] = max(0.5, customer['current_ratio'])
        else:
            customer['current_ratio'] = round(self.rng.normal(1.5, 0.5), 2)
        
        # 生成分类特征
        if 'industry' in dist:
            customer['industry'] = self._sample_from_distribution(dist['industry'])
        else:
            customer['industry'] = self.rng.choice(['制造业', '零售业', '服务业', '建筑业', '房地产业'])
        
        # 公司规模（基于年营收）
        if customer['annual_revenue'] < 5000000:
            customer['company_size'] = 'small'
        elif customer['annual_revenue'] < 50000000:
            customer['company_size'] = 'medium'
        else:
            customer['company_size'] = 'large'
        
        return customer
    
    def generate_customers(self, num_personal: int = 100, num_corporate: int = 0) -> List[Dict]:
        """
        批量生成客户
        
        Args:
            num_personal: 对私客户数量
            num_corporate: 对公客户数量
        
        Returns:
            客户列表
        """
        customers = []
        
        for i in range(num_personal):
            customer = self.generate_personal_customer()
            customers.append(customer)
        
        for i in range(num_corporate):
            customer = self.generate_corporate_customer()
            customers.append(customer)
        
        return customers
    
    def save_distributions(self, output_path: str = 'data/historical/learned_distributions.json'):
        """保存学习到的分布（元数据）"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 只保存可序列化的元数据
        distributions_meta = {
            'personal': {},
            'corporate': {}
        }
        
        for field, dist in self.personal_distributions.items():
            if dist['type'] in ['categorical', 'bernoulli']:
                distributions_meta['personal'][field] = dist
        
        for field, dist in self.corporate_distributions.items():
            if dist['type'] in ['categorical', 'bernoulli']:
                distributions_meta['corporate'][field] = dist
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(distributions_meta, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存分布元数据到: {output_path}")


def main():
    """主函数：测试客户生成器"""
    import sys
    import os
    
    # 加载历史数据
    data_path = 'data/historical/historical_loans_engineered.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行 historical_data_generator.py 和 feature_engineer.py")
        sys.exit(1)
    
    print("正在加载历史数据...")
    data = pd.read_csv(data_path)
    print(f"✅ 已加载 {len(data)} 条记录")
    
    # 创建生成器
    generator = EnhancedCustomerGenerator(data, seed=42)
    
    # 生成测试客户
    print("\n生成测试客户...")
    personal_customers = generator.generate_customers(num_personal=10, num_corporate=5)
    
    print(f"\n✅ 生成了 {len(personal_customers)} 个客户")
    print("\n对私客户示例:")
    for i, customer in enumerate(personal_customers[:3], 1):
        if customer['customer_type'] == 'personal':
            print(f"  {i}. {customer['customer_id']}: "
                  f"年龄{customer['age']}, 收入¥{customer['monthly_income']:.0f}, "
                  f"信用分{customer['credit_score']}")
    
    print("\n对公客户示例:")
    for i, customer in enumerate(personal_customers, 1):
        if customer['customer_type'] == 'corporate':
            print(f"  {i}. {customer['customer_id']}: "
                  f"注册资本¥{customer['registered_capital']:,.0f}, "
                  f"年营收¥{customer['annual_revenue']:,.0f}, "
                  f"经营{customer['operating_years']}年")
            if i >= 3:
                break
    
    # 保存分布
    generator.save_distributions()
    
    return generator


if __name__ == '__main__':
    main()

