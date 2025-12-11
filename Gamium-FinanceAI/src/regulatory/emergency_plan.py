"""
应急预案管理模块
自动触发应急预案，跟踪处置进度，评估处置效果
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class PlanStatus(Enum):
    """预案状态"""
    PENDING = "pending"  # 待执行
    EXECUTING = "executing"  # 执行中
    COMPLETED = "completed"  # 已完成
    CANCELLED = "cancelled"  # 已取消


class PlanPriority(Enum):
    """预案优先级"""
    LOW = "low"  # 低
    MEDIUM = "medium"  # 中
    HIGH = "high"  # 高
    URGENT = "urgent"  # 紧急


@dataclass
class EmergencyAction:
    """应急措施"""
    action_id: str
    name: str
    description: str
    target_indicator: str  # 目标指标
    target_value: float  # 目标值
    estimated_time: int  # 预计完成时间（小时）
    responsible_dept: str  # 责任部门
    status: str = "pending"  # pending, executing, completed, failed
    start_time: Optional[datetime] = None
    complete_time: Optional[datetime] = None
    actual_result: Optional[float] = None
    notes: str = ""


@dataclass
class EmergencyPlan:
    """应急预案"""
    plan_id: str
    name: str
    description: str
    trigger_condition: Dict[str, Any]  # 触发条件
    priority: PlanPriority
    actions: List[EmergencyAction]
    status: PlanStatus = PlanStatus.PENDING
    triggered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    effectiveness_score: Optional[float] = None  # 效果评分 0-100
    notes: str = ""


class EmergencyPlanManager:
    """应急预案管理器"""
    
    def __init__(self):
        self.plans: Dict[str, EmergencyPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self._init_default_plans()
        
    def _init_default_plans(self):
        """初始化默认应急预案"""
        
        # 预案1：资本充足率不足应急预案
        plan1 = EmergencyPlan(
            plan_id='plan_capital_adequacy',
            name='资本充足率不足应急预案',
            description='当资本充足率低于监管红线时，立即启动资本补充计划',
            trigger_condition={
                'indicator': 'capital_adequacy_ratio',
                'operator': '<',
                'threshold': 8.0,
                'duration': 0  # 立即触发
            },
            priority=PlanPriority.URGENT,
            actions=[
                EmergencyAction(
                    action_id='action1_1',
                    name='立即停止放贷',
                    description='暂停所有新增贷款业务，防止资本充足率进一步下降',
                    target_indicator='capital_adequacy_ratio',
                    target_value=8.0,
                    estimated_time=0,  # 立即执行
                    responsible_dept='信贷部'
                ),
                EmergencyAction(
                    action_id='action1_2',
                    name='启动资本补充计划',
                    description='通过增资扩股、发行资本债券等方式补充资本',
                    target_indicator='capital_adequacy_ratio',
                    target_value=10.0,
                    estimated_time=720,  # 30天
                    responsible_dept='财务部'
                ),
                EmergencyAction(
                    action_id='action1_3',
                    name='压缩风险资产',
                    description='通过资产证券化、转让等方式压缩风险加权资产',
                    target_indicator='capital_adequacy_ratio',
                    target_value=9.0,
                    estimated_time=168,  # 7天
                    responsible_dept='资产管理部'
                ),
                EmergencyAction(
                    action_id='action1_4',
                    name='提高资本充足率',
                    description='综合措施，确保资本充足率恢复到安全水平',
                    target_indicator='capital_adequacy_ratio',
                    target_value=10.0,
                    estimated_time=720,  # 30天
                    responsible_dept='风险管理部'
                )
            ]
        )
        self.plans[plan1.plan_id] = plan1
        
        # 预案2：不良贷款率超标应急预案
        plan2 = EmergencyPlan(
            plan_id='plan_npl_excess',
            name='不良贷款率超标应急预案',
            description='当不良贷款率超过监管红线时，立即启动风险处置计划',
            trigger_condition={
                'indicator': 'npl_ratio',
                'operator': '>',
                'threshold': 5.0,
                'duration': 0
            },
            priority=PlanPriority.URGENT,
            actions=[
                EmergencyAction(
                    action_id='action2_1',
                    name='立即停止高风险业务',
                    description='暂停高风险行业、高风险客户的贷款业务',
                    target_indicator='npl_ratio',
                    target_value=5.0,
                    estimated_time=0,
                    responsible_dept='信贷部'
                ),
                EmergencyAction(
                    action_id='action2_2',
                    name='加强风险控制',
                    description='提高审批门槛，加强贷后管理，降低新增不良',
                    target_indicator='npl_ratio',
                    target_value=4.0,
                    estimated_time=168,  # 7天
                    responsible_dept='风险管理部'
                ),
                EmergencyAction(
                    action_id='action2_3',
                    name='加快不良资产处置',
                    description='通过核销、转让、重组等方式加快不良资产处置',
                    target_indicator='npl_ratio',
                    target_value=3.0,
                    estimated_time=720,  # 30天
                    responsible_dept='资产管理部'
                ),
                EmergencyAction(
                    action_id='action2_4',
                    name='提高拨备覆盖率',
                    description='增加贷款损失准备，提高拨备覆盖率',
                    target_indicator='provision_coverage_ratio',
                    target_value=200.0,
                    estimated_time=168,  # 7天
                    responsible_dept='财务部'
                )
            ]
        )
        self.plans[plan2.plan_id] = plan2
        
        # 预案3：流动性不足应急预案
        plan3 = EmergencyPlan(
            plan_id='plan_liquidity_shortage',
            name='流动性不足应急预案',
            description='当流动性覆盖率低于监管红线时，立即启动流动性应急预案',
            trigger_condition={
                'indicator': 'lcr',
                'operator': '<',
                'threshold': 100.0,
                'duration': 0
            },
            priority=PlanPriority.URGENT,
            actions=[
                EmergencyAction(
                    action_id='action3_1',
                    name='启动流动性应急预案',
                    description='立即启动流动性应急预案，确保流动性安全',
                    target_indicator='lcr',
                    target_value=100.0,
                    estimated_time=0,
                    responsible_dept='资金部'
                ),
                EmergencyAction(
                    action_id='action3_2',
                    name='增加流动性资产',
                    description='通过购买国债、央行票据等增加优质流动性资产',
                    target_indicator='lcr',
                    target_value=110.0,
                    estimated_time=24,  # 1天
                    responsible_dept='资金部'
                ),
                EmergencyAction(
                    action_id='action3_3',
                    name='减少流动性负债',
                    description='通过提前还款、减少短期负债等方式减少流动性负债',
                    target_indicator='lcr',
                    target_value=120.0,
                    estimated_time=168,  # 7天
                    responsible_dept='资金部'
                ),
                EmergencyAction(
                    action_id='action3_4',
                    name='寻求外部流动性支持',
                    description='通过同业拆借、央行再贷款等方式寻求外部流动性支持',
                    target_indicator='lcr',
                    target_value=100.0,
                    estimated_time=24,  # 1天
                    responsible_dept='资金部'
                )
            ]
        )
        self.plans[plan3.plan_id] = plan3
        
        # 预案4：集中度风险过高应急预案
        plan4 = EmergencyPlan(
            plan_id='plan_concentration_risk',
            name='集中度风险过高应急预案',
            description='当集中度风险指标超过监管红线时，立即启动风险分散计划',
            trigger_condition={
                'indicator': 'single_customer_concentration',
                'operator': '>',
                'threshold': 10.0,
                'duration': 0
            },
            priority=PlanPriority.HIGH,
            actions=[
                EmergencyAction(
                    action_id='action4_1',
                    name='限制单一客户授信',
                    description='立即停止对单一客户的授信，限制集中度风险',
                    target_indicator='single_customer_concentration',
                    target_value=10.0,
                    estimated_time=0,
                    responsible_dept='信贷部'
                ),
                EmergencyAction(
                    action_id='action4_2',
                    name='限制单一行业授信',
                    description='限制对单一行业的授信，分散行业风险',
                    target_indicator='single_industry_concentration',
                    target_value=25.0,
                    estimated_time=168,  # 7天
                    responsible_dept='信贷部'
                ),
                EmergencyAction(
                    action_id='action4_3',
                    name='分散风险敞口',
                    description='通过调整授信结构，分散风险敞口',
                    target_indicator='single_customer_concentration',
                    target_value=8.0,
                    estimated_time=720,  # 30天
                    responsible_dept='风险管理部'
                ),
                EmergencyAction(
                    action_id='action4_4',
                    name='加强风险监控',
                    description='加强集中度风险监控，建立预警机制',
                    target_indicator='single_customer_concentration',
                    target_value=8.0,
                    estimated_time=168,  # 7天
                    responsible_dept='风险管理部'
                )
            ]
        )
        self.plans[plan4.plan_id] = plan4
        
        # 预案5：拨备覆盖率不足应急预案
        plan5 = EmergencyPlan(
            plan_id='plan_provision_shortage',
            name='拨备覆盖率不足应急预案',
            description='当拨备覆盖率低于监管红线时，立即启动拨备补充计划',
            trigger_condition={
                'indicator': 'provision_coverage_ratio',
                'operator': '<',
                'threshold': 150.0,
                'duration': 0
            },
            priority=PlanPriority.HIGH,
            actions=[
                EmergencyAction(
                    action_id='action5_1',
                    name='立即补充拨备',
                    description='立即计提贷款损失准备，确保拨备覆盖率达标',
                    target_indicator='provision_coverage_ratio',
                    target_value=150.0,
                    estimated_time=0,
                    responsible_dept='财务部'
                ),
                EmergencyAction(
                    action_id='action5_2',
                    name='提高拨备计提比例',
                    description='提高拨备计提比例，确保拨备充足',
                    target_indicator='provision_coverage_ratio',
                    target_value=180.0,
                    estimated_time=168,  # 7天
                    responsible_dept='财务部'
                ),
                EmergencyAction(
                    action_id='action5_3',
                    name='加快不良资产处置',
                    description='加快不良资产处置，降低不良贷款余额',
                    target_indicator='provision_coverage_ratio',
                    target_value=200.0,
                    estimated_time=720,  # 30天
                    responsible_dept='资产管理部'
                )
            ]
        )
        self.plans[plan5.plan_id] = plan5
        
    def check_and_trigger_plans(self, indicators: Dict[str, Any]) -> List[EmergencyPlan]:
        """检查指标并触发应急预案"""
        triggered_plans = []
        
        for plan_id, plan in self.plans.items():
            if plan.status in [PlanStatus.EXECUTING, PlanStatus.COMPLETED]:
                continue  # 已执行或执行中的预案不再触发
                
            condition = plan.trigger_condition
            indicator_key = condition['indicator']
            operator = condition['operator']
            threshold = condition['threshold']
            
            if indicator_key not in indicators:
                continue
                
            indicator_value = indicators[indicator_key].get('value', 0)
            should_trigger = False
            
            if operator == '<':
                should_trigger = indicator_value < threshold
            elif operator == '>':
                should_trigger = indicator_value > threshold
            elif operator == '<=':
                should_trigger = indicator_value <= threshold
            elif operator == '>=':
                should_trigger = indicator_value >= threshold
            elif operator == '==':
                should_trigger = indicator_value == threshold
                
            if should_trigger:
                plan.status = PlanStatus.EXECUTING
                plan.triggered_at = datetime.now()
                triggered_plans.append(plan)
                
                # 记录执行历史
                self.execution_history.append({
                    'plan_id': plan.plan_id,
                    'plan_name': plan.name,
                    'triggered_at': plan.triggered_at.isoformat(),
                    'trigger_condition': condition,
                    'indicator_value': indicator_value,
                    'status': plan.status.value
                })
        
        return triggered_plans
        
    def get_plan_detail(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """获取预案详情"""
        if plan_id not in self.plans:
            return None
            
        plan = self.plans[plan_id]
        return {
            'plan_id': plan.plan_id,
            'name': plan.name,
            'description': plan.description,
            'trigger_condition': plan.trigger_condition,
            'priority': plan.priority.value,
            'status': plan.status.value,
            'triggered_at': plan.triggered_at.isoformat() if plan.triggered_at else None,
            'completed_at': plan.completed_at.isoformat() if plan.completed_at else None,
            'effectiveness_score': plan.effectiveness_score,
            'notes': plan.notes,
            'actions': [
                {
                    'action_id': action.action_id,
                    'name': action.name,
                    'description': action.description,
                    'target_indicator': action.target_indicator,
                    'target_value': action.target_value,
                    'estimated_time': action.estimated_time,
                    'responsible_dept': action.responsible_dept,
                    'status': action.status,
                    'start_time': action.start_time.isoformat() if action.start_time else None,
                    'complete_time': action.complete_time.isoformat() if action.complete_time else None,
                    'actual_result': action.actual_result,
                    'notes': action.notes
                }
                for action in plan.actions
            ]
        }
        
    def get_all_plans(self) -> List[Dict[str, Any]]:
        """获取所有预案"""
        return [self.get_plan_detail(plan_id) for plan_id in self.plans.keys()]
        
    def update_action_status(self, plan_id: str, action_id: str, status: str, 
                           actual_result: Optional[float] = None, notes: str = ""):
        """更新措施状态"""
        if plan_id not in self.plans:
            return False
            
        plan = self.plans[plan_id]
        action = None
        for a in plan.actions:
            if a.action_id == action_id:
                action = a
                break
                
        if not action:
            return False
            
        action.status = status
        action.notes = notes
        
        if status == 'executing' and not action.start_time:
            action.start_time = datetime.now()
        elif status == 'completed' and not action.complete_time:
            action.complete_time = datetime.now()
            action.actual_result = actual_result
            
        # 检查所有措施是否完成
        all_completed = all(a.status == 'completed' for a in plan.actions)
        if all_completed and plan.status == PlanStatus.EXECUTING:
            plan.status = PlanStatus.COMPLETED
            plan.completed_at = datetime.now()
            
        return True
        
    def evaluate_effectiveness(self, plan_id: str, current_indicators: Dict[str, Any]) -> float:
        """评估预案效果"""
        if plan_id not in self.plans:
            return 0.0
            
        plan = self.plans[plan_id]
        if plan.status != PlanStatus.COMPLETED:
            return 0.0
            
        # 计算效果评分（0-100）
        score = 0.0
        total_weight = 0.0
        
        for action in plan.actions:
            if action.status == 'completed' and action.target_indicator in current_indicators:
                current_value = current_indicators[action.target_indicator].get('value', 0)
                target_value = action.target_value
                
                # 计算达成度
                if action.target_indicator in ['capital_adequacy_ratio', 'tier1_capital_ratio', 
                                               'provision_coverage_ratio', 'lcr', 'nsfr']:
                    # 越大越好的指标
                    if current_value >= target_value:
                        achievement = 100.0
                    else:
                        achievement = max(0, (current_value / target_value) * 100)
                else:
                    # 越小越好的指标
                    if current_value <= target_value:
                        achievement = 100.0
                    else:
                        achievement = max(0, (target_value / current_value) * 100)
                
                # 权重：紧急措施权重更高
                weight = 1.0 if action.estimated_time == 0 else 0.5
                score += achievement * weight
                total_weight += weight
                
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0.0
            
        plan.effectiveness_score = final_score
        return final_score
        
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history[-limit:]

