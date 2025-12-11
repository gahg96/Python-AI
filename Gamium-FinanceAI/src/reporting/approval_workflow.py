"""
审核流程管理模块
支持多级审核流程、审核意见记录、审核状态跟踪
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ApprovalStatus(Enum):
    """审核状态"""
    PENDING = "pending"  # 待审核
    APPROVED = "approved"  # 已通过
    REJECTED = "rejected"  # 已退回
    CANCELLED = "cancelled"  # 已取消


class ApprovalResult(Enum):
    """审核结果"""
    PASS = "pass"  # 通过
    REJECT = "reject"  # 退回
    CONDITIONAL = "conditional"  # 有条件通过


@dataclass
class ApprovalNode:
    """审核节点"""
    node_id: str
    node_name: str
    node_type: str  # sequential（顺序审核）, parallel（并行审核）, conditional（条件审核）
    approvers: List[str]  # 审核人列表
    required_approvers: int = 1  # 需要通过的审核人数（并行审核时）
    order: int = 0  # 顺序（顺序审核时）
    timeout_hours: Optional[int] = None  # 超时时间（小时）
    enabled: bool = True


@dataclass
class ApprovalRecord:
    """审核记录"""
    record_id: str
    instance_id: str
    node_id: str
    node_name: str
    approver: str
    approval_time: datetime
    result: ApprovalResult
    opinion: str  # 审核意见
    attachments: List[str] = field(default_factory=list)  # 附件列表


@dataclass
class ApprovalWorkflow:
    """审核流程"""
    workflow_id: str
    workflow_name: str
    template_id: str  # 关联的报表模板ID
    nodes: List[ApprovalNode] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ApprovalInstance:
    """审核实例"""
    instance_id: str
    workflow_id: str
    report_instance_id: str  # 关联的报表实例ID
    current_node_id: Optional[str] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    records: List[ApprovalRecord] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancelled_by: Optional[str] = None
    cancelled_reason: Optional[str] = None


class ApprovalWorkflowManager:
    """审核流程管理器"""
    
    def __init__(self):
        self.workflows: Dict[str, ApprovalWorkflow] = {}
        self.instances: Dict[str, ApprovalInstance] = {}
        self._init_default_workflows()
        
    def _init_default_workflows(self):
        """初始化默认审核流程"""
        
        # 流程1：资本充足率报表审核流程（三级审核）
        workflow1 = ApprovalWorkflow(
            workflow_id='wf_capital_adequacy',
            workflow_name='资本充足率报表审核流程',
            template_id='G01',
            nodes=[
                ApprovalNode(
                    node_id='node_1',
                    node_name='部门审核',
                    node_type='sequential',
                    approvers=['财务部经理', '风险管理部经理'],
                    required_approvers=2,
                    order=1,
                    timeout_hours=24
                ),
                ApprovalNode(
                    node_id='node_2',
                    node_name='分管领导审核',
                    node_type='sequential',
                    approvers=['分管副行长'],
                    required_approvers=1,
                    order=2,
                    timeout_hours=48
                ),
                ApprovalNode(
                    node_id='node_3',
                    node_name='行长审批',
                    node_type='sequential',
                    approvers=['行长'],
                    required_approvers=1,
                    order=3,
                    timeout_hours=72
                )
            ]
        )
        self.workflows[workflow1.workflow_id] = workflow1
        
        # 流程2：资产质量报表审核流程（二级审核）
        workflow2 = ApprovalWorkflow(
            workflow_id='wf_asset_quality',
            workflow_name='资产质量报表审核流程',
            template_id='G11',
            nodes=[
                ApprovalNode(
                    node_id='node_1',
                    node_name='部门审核',
                    node_type='sequential',
                    approvers=['信贷管理部经理', '风险管理部经理'],
                    required_approvers=2,
                    order=1,
                    timeout_hours=24
                ),
                ApprovalNode(
                    node_id='node_2',
                    node_name='分管领导审批',
                    node_type='sequential',
                    approvers=['分管副行长'],
                    required_approvers=1,
                    order=2,
                    timeout_hours=48
                )
            ]
        )
        self.workflows[workflow2.workflow_id] = workflow2
        
        # 流程3：流动性报表审核流程（二级审核）
        workflow3 = ApprovalWorkflow(
            workflow_id='wf_liquidity',
            workflow_name='流动性报表审核流程',
            template_id='G21',
            nodes=[
                ApprovalNode(
                    node_id='node_1',
                    node_name='部门审核',
                    node_type='sequential',
                    approvers=['资金管理部经理', '风险管理部经理'],
                    required_approvers=2,
                    order=1,
                    timeout_hours=24
                ),
                ApprovalNode(
                    node_id='node_2',
                    node_name='分管领导审批',
                    node_type='sequential',
                    approvers=['分管副行长'],
                    required_approvers=1,
                    order=2,
                    timeout_hours=48
                )
            ]
        )
        self.workflows[workflow3.workflow_id] = workflow3
        
    def get_workflow(self, workflow_id: str) -> Optional[ApprovalWorkflow]:
        """获取审核流程"""
        return self.workflows.get(workflow_id)
        
    def get_workflow_by_template(self, template_id: str) -> Optional[ApprovalWorkflow]:
        """根据模板ID获取审核流程"""
        for workflow in self.workflows.values():
            if workflow.template_id == template_id and workflow.enabled:
                return workflow
        return None
        
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """获取所有审核流程"""
        result = []
        for workflow in self.workflows.values():
            result.append({
                'workflow_id': workflow.workflow_id,
                'workflow_name': workflow.workflow_name,
                'template_id': workflow.template_id,
                'nodes_count': len(workflow.nodes),
                'enabled': workflow.enabled,
                'created_at': workflow.created_at.isoformat(),
                'updated_at': workflow.updated_at.isoformat()
            })
        return result
        
    def start_approval(self, workflow_id: str, report_instance_id: str) -> Optional[ApprovalInstance]:
        """
        启动审核流程
        
        Args:
            workflow_id: 流程ID
            report_instance_id: 报表实例ID
        
        Returns:
            审核实例
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow or not workflow.enabled:
            return None
            
        # 检查是否已有审核实例
        existing = None
        for inst in self.instances.values():
            if inst.report_instance_id == report_instance_id and inst.status == ApprovalStatus.PENDING:
                existing = inst
                break
                
        if existing:
            return existing
            
        # 创建新的审核实例
        instance_id = f"approval_{report_instance_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 获取第一个节点
        first_node = None
        if workflow.nodes:
            sorted_nodes = sorted(workflow.nodes, key=lambda n: n.order)
            first_node = sorted_nodes[0] if sorted_nodes else None
            
        instance = ApprovalInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            report_instance_id=report_instance_id,
            current_node_id=first_node.node_id if first_node else None,
            status=ApprovalStatus.PENDING
        )
        
        self.instances[instance_id] = instance
        return instance
        
    def approve(self, instance_id: str, node_id: str, approver: str, 
                result: ApprovalResult, opinion: str = "") -> bool:
        """
        执行审核
        
        Args:
            instance_id: 审核实例ID
            node_id: 节点ID
            approver: 审核人
            result: 审核结果
            opinion: 审核意见
        
        Returns:
            是否成功
        """
        instance = self.instances.get(instance_id)
        if not instance:
            return False
            
        if instance.status != ApprovalStatus.PENDING:
            return False
            
        workflow = self.workflows.get(instance.workflow_id)
        if not workflow:
            return False
            
        # 查找当前节点
        current_node = None
        for node in workflow.nodes:
            if node.node_id == node_id:
                current_node = node
                break
                
        if not current_node or current_node.node_id != instance.current_node_id:
            return False
            
        # 检查审核人是否在节点审核人列表中
        if approver not in current_node.approvers:
            return False
            
        # 创建审核记录
        record = ApprovalRecord(
            record_id=f"record_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            instance_id=instance_id,
            node_id=node_id,
            node_name=current_node.node_name,
            approver=approver,
            approval_time=datetime.now(),
            result=result,
            opinion=opinion
        )
        instance.records.append(record)
        
        # 处理审核结果
        if result == ApprovalResult.REJECT:
            # 退回
            instance.status = ApprovalStatus.REJECTED
            instance.completed_at = datetime.now()
            return True
        elif result == ApprovalResult.PASS:
            # 通过，检查是否需要更多审核人
            if current_node.node_type == 'parallel':
                # 并行审核，需要统计通过人数
                approved_count = sum(1 for r in instance.records 
                                   if r.node_id == node_id and r.result == ApprovalResult.PASS)
                if approved_count < current_node.required_approvers:
                    # 还需要更多审核人
                    return True
            # 顺序审核或并行审核已完成，进入下一节点
            return self._move_to_next_node(instance, workflow)
        elif result == ApprovalResult.CONDITIONAL:
            # 有条件通过，需要修改后重新审核
            return True
            
        return False
        
    def _move_to_next_node(self, instance: ApprovalInstance, workflow: ApprovalWorkflow) -> bool:
        """移动到下一个节点"""
        if not workflow.nodes:
            # 没有更多节点，审核完成
            instance.status = ApprovalStatus.APPROVED
            instance.completed_at = datetime.now()
            return True
            
        # 获取当前节点
        current_node = None
        for node in workflow.nodes:
            if node.node_id == instance.current_node_id:
                current_node = node
                break
                
        if not current_node:
            return False
            
        # 查找下一个节点
        sorted_nodes = sorted(workflow.nodes, key=lambda n: n.order)
        current_index = -1
        for i, node in enumerate(sorted_nodes):
            if node.node_id == instance.current_node_id:
                current_index = i
                break
                
        if current_index < 0 or current_index >= len(sorted_nodes) - 1:
            # 没有下一个节点，审核完成
            instance.status = ApprovalStatus.APPROVED
            instance.completed_at = datetime.now()
            return True
            
        # 移动到下一个节点
        next_node = sorted_nodes[current_index + 1]
        instance.current_node_id = next_node.node_id
        return True
        
    def cancel_approval(self, instance_id: str, cancelled_by: str, reason: str = "") -> bool:
        """取消审核"""
        instance = self.instances.get(instance_id)
        if not instance:
            return False
            
        if instance.status != ApprovalStatus.PENDING:
            return False
            
        instance.status = ApprovalStatus.CANCELLED
        instance.cancelled_at = datetime.now()
        instance.cancelled_by = cancelled_by
        instance.cancelled_reason = reason
        return True
        
    def get_instance(self, instance_id: str) -> Optional[ApprovalInstance]:
        """获取审核实例"""
        return self.instances.get(instance_id)
        
    def get_instance_by_report(self, report_instance_id: str) -> Optional[ApprovalInstance]:
        """根据报表实例ID获取审核实例"""
        for instance in self.instances.values():
            if instance.report_instance_id == report_instance_id:
                return instance
        return None
        
    def get_all_instances(self) -> List[Dict[str, Any]]:
        """获取所有审核实例"""
        result = []
        for instance in self.instances.values():
            workflow = self.workflows.get(instance.workflow_id)
            current_node = None
            if instance.current_node_id and workflow:
                for node in workflow.nodes:
                    if node.node_id == instance.current_node_id:
                        current_node = node
                        break
                        
            result.append({
                'instance_id': instance.instance_id,
                'workflow_id': instance.workflow_id,
                'workflow_name': workflow.workflow_name if workflow else '',
                'report_instance_id': instance.report_instance_id,
                'current_node_id': instance.current_node_id,
                'current_node_name': current_node.node_name if current_node else '',
                'status': instance.status.value,
                'records_count': len(instance.records),
                'started_at': instance.started_at.isoformat(),
                'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
                'cancelled_at': instance.cancelled_at.isoformat() if instance.cancelled_at else None
            })
        return result
        
    def get_instance_detail(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """获取审核实例详情"""
        instance = self.instances.get(instance_id)
        if not instance:
            return None
            
        workflow = self.workflows.get(instance.workflow_id)
        if not workflow:
            return None
            
        # 获取当前节点信息
        current_node = None
        if instance.current_node_id:
            for node in workflow.nodes:
                if node.node_id == instance.current_node_id:
                    current_node = node
                    break
                    
        # 获取节点进度
        nodes_progress = []
        sorted_nodes = sorted(workflow.nodes, key=lambda n: n.order)
        for node in sorted_nodes:
            node_records = [r for r in instance.records if r.node_id == node.node_id]
            is_completed = any(r.result == ApprovalResult.PASS for r in node_records)
            is_current = node.node_id == instance.current_node_id
            is_pending = not is_completed and not is_current
            
            nodes_progress.append({
                'node_id': node.node_id,
                'node_name': node.node_name,
                'node_type': node.node_type,
                'order': node.order,
                'approvers': node.approvers,
                'required_approvers': node.required_approvers,
                'is_current': is_current,
                'is_completed': is_completed,
                'is_pending': is_pending,
                'records': [
                    {
                        'approver': r.approver,
                        'approval_time': r.approval_time.isoformat(),
                        'result': r.result.value,
                        'opinion': r.opinion
                    }
                    for r in node_records
                ]
            })
            
        return {
            'instance_id': instance.instance_id,
            'workflow_id': instance.workflow_id,
            'workflow_name': workflow.workflow_name,
            'report_instance_id': instance.report_instance_id,
            'current_node_id': instance.current_node_id,
            'current_node_name': current_node.node_name if current_node else '',
            'status': instance.status.value,
            'nodes_progress': nodes_progress,
            'records': [
                {
                    'record_id': r.record_id,
                    'node_id': r.node_id,
                    'node_name': r.node_name,
                    'approver': r.approver,
                    'approval_time': r.approval_time.isoformat(),
                    'result': r.result.value,
                    'opinion': r.opinion
                }
                for r in instance.records
            ],
            'started_at': instance.started_at.isoformat(),
            'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
            'cancelled_at': instance.cancelled_at.isoformat() if instance.cancelled_at else None,
            'cancelled_by': instance.cancelled_by,
            'cancelled_reason': instance.cancelled_reason
        }

