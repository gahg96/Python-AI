"""
Gamium 数据蒸馏模块

从历史数据中提炼"商业物理定律"
"""

from .customer_generator import CustomerGenerator, CustomerProfile
from .world_model import WorldModel, LoanOffer, MarketConditions, CustomerFuture
from .distillation_pipeline import DistillationPipeline
from .data_loader import HistoricalDataLoader, load_historical_data

__all__ = [
    'CustomerGenerator', 'CustomerProfile',
    'WorldModel', 'LoanOffer', 'MarketConditions', 'CustomerFuture',
    'DistillationPipeline',
    'HistoricalDataLoader', 'load_historical_data',
]

