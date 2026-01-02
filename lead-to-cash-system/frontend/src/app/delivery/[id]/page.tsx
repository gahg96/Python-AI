'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { useI18n } from "@/lib/i18n/I18nContext";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2, ArrowLeft, Plus, Trash2, Save, FileText, AlertTriangle, Edit, Upload, Download, TrendingUp, DollarSign, X } from 'lucide-react';

export default function ProjectDetailPage() {
    const { id } = useParams();
    const router = useRouter();
    const { t } = useI18n();
    const [project, setProject] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [users, setUsers] = useState<any[]>([]);

    // UI Logic
    const [isEditing, setIsEditing] = useState(false);
    const [editData, setEditData] = useState<any>({});
    const [activeTab, setActiveTab] = useState('overview');
    const [validationErrors, setValidationErrors] = useState<{ [key: string]: boolean }>({});
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [uploadingMeetingId, setUploadingMeetingId] = useState<string | null>(null);

    // Sub-forms state
    const [newResource, setNewResource] = useState({ userId: '', role: '', allocationPct: '100' });
    const [newMeeting, setNewMeeting] = useState({ title: '', type: 'Weekly', planDate: '' });
    const [newRisk, setNewRisk] = useState({ description: '', impact: 'Medium', mitigationPlan: '' });
    const [editingRiskId, setEditingRiskId] = useState<string | null>(null);
    const [editingRiskData, setEditingRiskData] = useState<any>({});

    useEffect(() => {
        fetchProject();
        fetchUsers();
    }, [id]);

    // Auto-sync emergency cost when complexity changes in edit mode
    useEffect(() => {
        if (isEditing && project?.contract?.totalContractValue) {
            const contractValue = Number(project.contract.totalContractValue);
            let rate = 0.03;
            if (editData.complexity === 'Medium') rate = 0.05;
            if (editData.complexity === 'High') rate = 0.10;
            const newCost = contractValue * rate;
            if (editData.emergencySupportCost !== newCost) {
                setEditData((prev: any) => ({ ...prev, emergencySupportCost: newCost }));
            }
        }
    }, [editData.complexity, isEditing, project?.contract?.totalContractValue, editData.emergencySupportCost]);

    const fetchProject = async () => {
        try {
            const data = await api.get(`/projects/${id}`);
            setProject(data);
            setEditData({
                status: data.status,
                description: data.description || '',
                // Default to Today if not set
                startDate: data.startDate ? data.startDate.split('T')[0] : new Date().toISOString().split('T')[0],
                // Default to 3 months later if not set
                endDate: data.endDate ? data.endDate.split('T')[0] : (() => {
                    const d = new Date();
                    d.setMonth(d.getMonth() + 3);
                    return d.toISOString().split('T')[0];
                })(),
                budget: data.budget || 0,
                targetProfitMargin: data.targetProfitMargin || 0,
                laborCost: data.laborCost || 0,
                outsourceCost: data.outsourceCost || 0,
                travelCost: data.travelCost || 0,
                emergencySupportCost: data.emergencySupportCost || 0,
                thirdPartyEquipmentCost: data.thirdPartyEquipmentCost || 0,
                softwareCost: data.softwareCost || 0,
                otherWeight: data.otherWeight || 0,
                complexity: data.complexity || 'Low',
                financialRemarks: data.financialRemarks || '',
                isDelayed: data.isDelayed || false
            });
        } catch (error) { console.error(error); }
        finally { setLoading(false); }
    };

    const fetchUsers = async () => {
        try {
            const data = await api.get('/users');
            setUsers(data);
        } catch (error) { console.error(error); }
    };

    const validateRequiredFields = () => {
        const errors: { [key: string]: boolean } = {};
        let firstErrorTab = '';

        // Overview tab required fields
        if (!editData.status) {
            errors.status = true;
            if (!firstErrorTab) firstErrorTab = 'overview';
        }
        if (!editData.startDate || editData.startDate.trim() === '') {
            errors.startDate = true;
            if (!firstErrorTab) firstErrorTab = 'overview';
        }
        if (!editData.endDate || editData.endDate.trim() === '') {
            errors.endDate = true;
            if (!firstErrorTab) firstErrorTab = 'overview';
        }

        setValidationErrors(errors);

        if (Object.keys(errors).length > 0) {
            setActiveTab(firstErrorTab);
            alert(`请填写所有必填字段（标记为红色 * 的字段）`);
            return false;
        }
        return true;
    };

    const handleSaveProject = async () => {
        // Validate required fields first
        if (!validateRequiredFields()) {
            return;
        }

        try {
            // Convert string inputs to numbers for validation
            const payload: any = {
                status: editData.status,
                description: editData.description,
                budget: editData.budget ? parseFloat(editData.budget as any) : 0,
                targetProfitMargin: editData.targetProfitMargin ? parseFloat(editData.targetProfitMargin as any) : 0,
                laborCost: editData.laborCost ? parseFloat(editData.laborCost as any) : 0,
                outsourceCost: editData.outsourceCost ? parseFloat(editData.outsourceCost as any) : 0,
                travelCost: editData.travelCost ? parseFloat(editData.travelCost as any) : 0,
                emergencySupportCost: editData.emergencySupportCost ? parseFloat(editData.emergencySupportCost as any) : 0,
                thirdPartyEquipmentCost: editData.thirdPartyEquipmentCost ? parseFloat(editData.thirdPartyEquipmentCost as any) : 0,
                softwareCost: editData.softwareCost ? parseFloat(editData.softwareCost as any) : 0,
                otherWeight: editData.otherWeight ? parseFloat(editData.otherWeight as any) : 0,
                complexity: editData.complexity,
                financialRemarks: editData.financialRemarks,
                isDelayed: editData.isDelayed,
            };

            // Only include dates if they are not empty strings
            if (editData.startDate && editData.startDate.trim() !== '') {
                payload.startDate = new Date(editData.startDate).toISOString();
            }
            if (editData.endDate && editData.endDate.trim() !== '') {
                payload.endDate = new Date(editData.endDate).toISOString();
            }

            console.log('Saving project with payload:', payload);
            await api.patch(`/projects/${id}`, payload);
            setIsEditing(false);
            setValidationErrors({});
            fetchProject();
            alert('项目保存成功！');
        } catch (error: any) {
            console.error('Save project error:', error);
            const errorMessage = error?.message || '保存失败，请查看控制台了解详情';
            alert(`保存失败: ${errorMessage}`);
        }
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !uploadingMeetingId) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            await api.upload(`/projects/meetings/${uploadingMeetingId}/upload`, formData);
            fetchProject();
            alert('会议纪要上传成功！');
        } catch (error) {
            console.error('Upload failed:', error);
            alert('上传失败，请重试');
        } finally {
            setUploadingMeetingId(null);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const handleAddResource = async () => {
        if (!newResource.userId || !newResource.role) return;
        try {
            await api.post(`/projects/${id}/resources`, newResource);
            setNewResource({ userId: '', role: '', allocationPct: '100' });
            fetchProject();
        } catch (error) { console.error(error); }
    };

    const handleAddMeeting = async () => {
        if (!newMeeting.title || !newMeeting.planDate) return;
        try {
            await api.post(`/projects/${id}/meetings`, newMeeting);
            setNewMeeting({ title: '', type: 'Weekly', planDate: '' });
            fetchProject();
        } catch (error) { console.error(error); }
    };

    const handleAddRisk = async () => {
        if (!newRisk.description) return;
        try {
            await api.post(`/projects/${id}/risks`, newRisk);
            setNewRisk({ description: '', impact: 'Medium', mitigationPlan: '' });
            fetchProject();
        } catch (error) { console.error(error); }
    };

    const handleDeleteMeeting = async (meetingId: string) => {
        if (!confirm('确定要删除这个会议吗？')) return;
        try {
            await api.delete(`/projects/meetings/${meetingId}`);
            fetchProject();
            alert('会议已删除');
        } catch (error) {
            console.error('Delete meeting error:', error);
            alert('删除失败，请重试');
        }
    };

    const handleUpdateRiskStatus = async (riskId: string, newStatus: string) => {
        try {
            await api.patch(`/projects/risks/${riskId}`, { status: newStatus });
            fetchProject();
        } catch (error) {
            console.error('Update risk status error:', error);
            alert('更新失败，请重试');
        }
    };

    const handleEditRisk = (risk: any) => {
        setEditingRiskId(risk.id);
        setEditingRiskData({
            description: risk.description,
            impact: risk.impact,
            status: risk.status,
            mitigationPlan: risk.mitigationPlan || ''
        });
    };

    const handleSaveRisk = async (riskId: string) => {
        try {
            await api.patch(`/projects/risks/${riskId}`, editingRiskData);
            setEditingRiskId(null);
            setEditingRiskData({});
            fetchProject();
            alert('风险已更新');
        } catch (error) {
            console.error('Update risk error:', error);
            alert('更新失败，请重试');
        }
    };

    const handleCancelEditRisk = () => {
        setEditingRiskId(null);
        setEditingRiskData({});
    };

    if (loading || !project) return <div className="flex justify-center p-8"><Loader2 className="animate-spin" /></div>;

    // Financial calculations
    const contractValue = Number(project.contract.totalContractValue);

    // Use editData if editing to ensure reactive updates in UI
    const currentLaborCost = Number((isEditing ? editData.laborCost : project.laborCost) || 0);
    const currentOutsourceCost = Number((isEditing ? editData.outsourceCost : project.outsourceCost) || 0);
    const currentTravelCost = Number((isEditing ? editData.travelCost : project.travelCost) || 0);
    const currentThirdPartyCost = Number((isEditing ? editData.thirdPartyEquipmentCost : project.thirdPartyEquipmentCost) || 0);
    const currentSoftwareCost = Number((isEditing ? editData.softwareCost : project.softwareCost) || 0);
    const currentOtherWeight = Number((isEditing ? editData.otherWeight : project.otherWeight) || 0);

    const currentComplexity = isEditing ? editData.complexity : project.complexity;

    // Emergency support calc logic
    let emergencySupportRate = 0.03;
    if (currentComplexity === 'Medium') emergencySupportRate = 0.05;
    if (currentComplexity === 'High') emergencySupportRate = 0.10;

    const calculatedEmergencySupport = contractValue * emergencySupportRate;
    const currentEmergencySupport = isEditing ? calculatedEmergencySupport : Number(project.emergencySupportCost || 0);

    const totalCost = currentLaborCost + currentOutsourceCost + currentTravelCost + currentThirdPartyCost + currentSoftwareCost + currentOtherWeight + currentEmergencySupport;
    const netProfit = contractValue - totalCost;
    const profitMargin = contractValue > 0 ? (netProfit / contractValue) * 100 : 0;


    return (
        <div className="container mx-auto p-6 space-y-6">
            <input
                type="file"
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileUpload}
            />

            <Button variant="ghost" onClick={() => router.back()} className="mb-4">
                <ArrowLeft className="h-4 w-4 mr-2" />
                {t("common.back")}
            </Button>

            <div className="flex justify-between items-start">
                <div className="space-y-1">
                    <h1 className="text-3xl font-bold flex items-center gap-3">
                        {project.contract.opportunity.title}
                        <Badge variant="outline">{t(`project.status.${project.status.toLowerCase()}`)}</Badge>
                        {project.isDelayed && <Badge variant="destructive" className="animate-pulse">{t("project.fields.isDelayed")}</Badge>}
                    </h1>
                    <p className="text-slate-500">{project.contract.contractNumber} - {project.contract.opportunity.customer.companyName}</p>
                </div>
                <div className="flex gap-2">
                    {!isEditing ? (
                        <Button onClick={() => setIsEditing(true)}>
                            <Edit className="h-4 w-4 mr-2" /> {t("project.actions.edit")}
                        </Button>
                    ) : (
                        <>
                            <Button variant="outline" onClick={() => setIsEditing(false)}>
                                <X className="h-4 w-4 mr-2" /> {t("project.actions.cancel")}
                            </Button>
                            <Button onClick={handleSaveProject} className="bg-green-600 hover:bg-green-700 text-white">
                                <Save className="h-4 w-4 mr-2" /> {t("project.actions.save")}
                            </Button>
                        </>
                    )}
                </div>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-6">
                    <TabsTrigger value="overview">{t("project.tabs.overview")}</TabsTrigger>
                    <TabsTrigger value="profit">{t("project.tabs.profit")}</TabsTrigger>
                    <TabsTrigger value="resources">{t("project.tabs.resources")}</TabsTrigger>
                    <TabsTrigger value="communication">{t("project.tabs.communication")}</TabsTrigger>
                    <TabsTrigger value="risks">{t("project.tabs.risks")}</TabsTrigger>
                    <TabsTrigger value="billing">{t("project.billing.milestoneTitle")}</TabsTrigger>
                </TabsList>

                <TabsContent value="overview">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                        <Card shadow-sm="true">
                            <CardHeader><CardTitle className="text-lg">{t("project.fields.financials")}</CardTitle></CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-1">
                                        <Label className="text-xs text-slate-500 uppercase tracking-wider flex items-center gap-2">
                                            {t("project.fields.budget")}
                                            <span className="text-[10px] text-blue-600 font-semibold">((自动计算))</span>
                                        </Label>
                                        <div className="text-lg font-bold text-blue-600">¥{totalCost.toLocaleString()}</div>
                                        <div className="text-[10px] text-slate-400 italic">根据利润分析中的成本明细自动汇总</div>
                                    </div>
                                    <div className="space-y-1">
                                        <Label className="text-xs text-slate-500 uppercase tracking-wider">{t("project.fields.contractValue")}</Label>
                                        <div className="text-lg font-bold text-slate-400">¥{contractValue.toLocaleString()}</div>
                                    </div>
                                </div>
                                <div className="space-y-1 pt-2">
                                    <Label className="text-xs text-slate-500 uppercase tracking-wider">{t("project.fields.description")}</Label>
                                    {isEditing ? (
                                        <Textarea rows={4} value={editData.description} onChange={e => setEditData({ ...editData, description: e.target.value })} />
                                    ) : (
                                        <div className="text-sm text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-md min-h-[100px] border border-slate-100 italic">
                                            {project.description || t("project.placeholders.noDesc")}
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                        <Card shadow-sm="true">
                            <CardHeader><CardTitle className="text-lg">{t("project.fields.timeline")}</CardTitle></CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-1">
                                        <Label className="text-xs text-slate-500 uppercase tracking-wider">
                                            {t("project.fields.startDate")}
                                            {isEditing && <span className="text-red-500 ml-1">*</span>}
                                        </Label>
                                        {isEditing ? (
                                            <Input
                                                type="date"
                                                value={editData.startDate}
                                                onChange={e => setEditData({ ...editData, startDate: e.target.value })}
                                                className={validationErrors.startDate ? 'border-red-500 focus:border-red-500' : ''}
                                            />
                                        ) : (
                                            <div className="text-lg font-medium">{project.startDate ? new Date(project.startDate).toLocaleDateString() : t("project.placeholders.notSet")}</div>
                                        )}
                                    </div>
                                    <div className="space-y-1">
                                        <Label className="text-xs text-slate-500 uppercase tracking-wider">
                                            {t("project.fields.endDate")}
                                            {isEditing && <span className="text-red-500 ml-1">*</span>}
                                        </Label>
                                        {isEditing ? (
                                            <Input
                                                type="date"
                                                value={editData.endDate}
                                                onChange={e => setEditData({ ...editData, endDate: e.target.value })}
                                                className={validationErrors.endDate ? 'border-red-500 focus:border-red-500' : ''}
                                            />
                                        ) : (
                                            <div className="text-lg font-medium">{project.endDate ? new Date(project.endDate).toLocaleDateString() : t("project.placeholders.notSet")}</div>
                                        )}
                                    </div>
                                </div>
                                <div className="flex items-center space-x-3 pt-6 p-4 bg-slate-50 rounded-lg border border-dashed border-slate-200">
                                    <Checkbox
                                        id="delayed"
                                        checked={isEditing ? editData.isDelayed : project.isDelayed}
                                        disabled={!isEditing}
                                        onCheckedChange={(checked) => setEditData({ ...editData, isDelayed: checked })}
                                        className="h-5 w-5"
                                    />
                                    <Label htmlFor="delayed" className={`cursor-pointer font-semibold ${project.isDelayed ? 'text-red-600' : 'text-slate-700'}`}>
                                        {t("project.fields.delayedLabel")}
                                    </Label>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="profit">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                        <Card className="md:col-span-1 border-l-4 border-l-blue-500">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <DollarSign className="h-5 w-5 text-blue-500" />
                                    {t("project.fields.financialAnalysis")}
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="space-y-1">
                                        <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.laborCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" size={1} className="h-8 text-sm" value={editData.laborCost} onChange={e => setEditData({ ...editData, laborCost: e.target.value })} />
                                        ) : (
                                            <div className="font-bold">¥{currentLaborCost.toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div className="space-y-1">
                                        <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.outsourceCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" size={1} className="h-8 text-sm" value={editData.outsourceCost} onChange={e => setEditData({ ...editData, outsourceCost: e.target.value })} />
                                        ) : (
                                            <div className="font-bold">¥{currentOutsourceCost.toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div className="space-y-1">
                                        <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.travelCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" size={1} className="h-8 text-sm" value={editData.travelCost} onChange={e => setEditData({ ...editData, travelCost: e.target.value })} />
                                        ) : (
                                            <div className="font-bold">¥{currentTravelCost.toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div className="space-y-1">
                                        <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.softwareCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" size={1} className="h-8 text-sm" value={editData.softwareCost} onChange={e => setEditData({ ...editData, softwareCost: e.target.value })} />
                                        ) : (
                                            <div className="font-bold">¥{currentSoftwareCost.toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div className="space-y-1">
                                        <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.thirdPartyEquipmentCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" size={1} className="h-8 text-sm" value={editData.thirdPartyEquipmentCost} onChange={e => setEditData({ ...editData, thirdPartyEquipmentCost: e.target.value })} />
                                        ) : (
                                            <div className="font-bold">¥{currentThirdPartyCost.toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div className="space-y-1">
                                        <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.otherWeight")}</Label>
                                        {isEditing ? (
                                            <Input type="number" size={1} className="h-8 text-sm" value={editData.otherWeight} onChange={e => setEditData({ ...editData, otherWeight: e.target.value })} />
                                        ) : (
                                            <div className="font-bold">¥{currentOtherWeight.toLocaleString()}</div>
                                        )}
                                    </div>
                                </div>

                                <div className="space-y-2 pt-2 border-t border-dashed">
                                    <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.complexity")} & {t("project.fields.emergencySupportCost")}</Label>
                                    <div className="flex gap-2">
                                        <div className="flex-1">
                                            {isEditing ? (
                                                <Select value={editData.complexity} onValueChange={(val) => setEditData({ ...editData, complexity: val })}>
                                                    <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="Low">Low (3%)</SelectItem>
                                                        <SelectItem value="Medium">Medium (5%)</SelectItem>
                                                        <SelectItem value="High">High (10%)</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            ) : (
                                                <Badge variant="secondary">{project.complexity}</Badge>
                                            )}
                                        </div>
                                        <div className="flex-1 text-right">
                                            <div className="text-sm font-bold">¥{currentEmergencySupport.toLocaleString()}</div>
                                            <div className="text-[10px] text-slate-400">{(emergencySupportRate * 100).toFixed(0)}% of Value</div>
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-dashed">
                                    <div className="flex justify-between items-center bg-red-50 p-2 rounded-lg text-red-700">
                                        <span className="text-xs font-semibold">{t("project.fields.totalCost")}</span>
                                        <span className="text-base font-black">¥{totalCost.toLocaleString()}</span>
                                    </div>
                                </div>

                                <div className="space-y-1 pt-2">
                                    <Label className="text-[10px] text-slate-500 uppercase tracking-wider">{t("project.fields.financialRemarks")}</Label>
                                    {isEditing ? (
                                        <Textarea rows={3} className="text-xs" value={editData.financialRemarks} onChange={e => setEditData({ ...editData, financialRemarks: e.target.value })} placeholder="Add dynamic notes here..." />
                                    ) : (
                                        <div className="text-[11px] text-slate-600 bg-slate-50 p-2 rounded border border-slate-100 italic min-h-[60px]">
                                            {project.financialRemarks || "No remarks added."}
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>

                        <div className="md:col-span-2 grid grid-cols-1 gap-6">
                            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 text-white overflow-hidden relative border-0">
                                <div className="absolute top-0 right-0 p-8 opacity-10">
                                    <TrendingUp className="h-32 w-32" />
                                </div>
                                <CardContent className="p-10 flex items-center justify-between relative z-10">
                                    <div className="space-y-2">
                                        <div className="flex items-center gap-2 text-slate-400 font-medium tracking-wide">
                                            <TrendingUp className="h-5 w-5" />
                                            <span>{t("project.fields.profitMargin")}</span>
                                        </div>
                                        <div className={`text-7xl font-black transition-colors duration-1000 ${profitMargin > 20 ? 'text-green-400' : 'text-yellow-400'}`}>
                                            {profitMargin.toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="text-right space-y-3">
                                        <div className="flex items-center gap-2 text-slate-400 justify-end font-medium tracking-wide">
                                            <DollarSign className="h-5 w-5" />
                                            <span>{t("project.fields.netProfit")}</span>
                                        </div>
                                        <div className="text-5xl font-black text-white">
                                            ¥{netProfit.toLocaleString()}
                                        </div>
                                        <div className="text-xs text-slate-400">Calculated from total contract value</div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card shadow-sm="true">
                                <CardHeader>
                                    <CardTitle className="text-sm text-slate-500">目标利润率设置</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    <div className="space-y-2">
                                        <Label className="text-sm font-semibold text-slate-700">目标利润率 (%)</Label>
                                        {isEditing ? (
                                            <Input
                                                type="number"
                                                min="0"
                                                max="100"
                                                step="0.1"
                                                value={editData.targetProfitMargin}
                                                onChange={e => setEditData({ ...editData, targetProfitMargin: e.target.value })}
                                                className="text-lg font-bold"
                                            />
                                        ) : (
                                            <div className="text-2xl font-bold text-blue-600">{project.targetProfitMargin}%</div>
                                        )}
                                        <p className="text-xs text-slate-400 italic">设置项目的目标利润率，用于与实际利润率对比</p>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card shadow-sm="true">
                                <CardHeader><CardTitle className="text-sm text-slate-500">{t("project.fields.profitMargin")} vs Target</CardTitle></CardHeader>
                                <CardContent className="h-56 flex items-end gap-12 px-12 pb-10">
                                    <div className="flex-1 flex flex-col items-center gap-3">
                                        <div className="w-full bg-slate-100 rounded-t-xl relative border border-slate-200" style={{ height: '140px' }}>
                                            <div className="absolute bottom-0 w-full bg-slate-400 rounded-t-xl transition-all duration-700 shadow-inner" style={{ height: `${project.targetProfitMargin}%` }}></div>
                                            <div className="absolute -top-6 w-full text-center text-xs font-bold text-slate-500">{project.targetProfitMargin}%</div>
                                        </div>
                                        <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Target</span>
                                    </div>
                                    <div className="flex-1 flex flex-col items-center gap-3">
                                        <div className="w-full bg-slate-100 rounded-t-xl relative border border-slate-200" style={{ height: '140px' }}>
                                            <div className={`absolute bottom-0 w-full rounded-t-xl transition-all duration-1000 shadow-lg ${profitMargin >= project.targetProfitMargin ? 'bg-green-500' : 'bg-orange-500'}`} style={{ height: `${Math.min(profitMargin, 100)}%` }}></div>
                                            <div className={`absolute -top-6 w-full text-center text-xs font-black ${profitMargin >= project.targetProfitMargin ? 'text-green-600' : 'text-orange-600'}`}>{profitMargin.toFixed(1)}%</div>
                                        </div>
                                        <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Actual</span>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </TabsContent>

                <TabsContent value="billing">
                    <div className="grid grid-cols-1 gap-6 mt-6">
                        {/* 1. Contract Milestones (Single Source of Truth) */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <TrendingUp className="h-5 w-5 text-blue-600" />
                                    {t("project.billing.milestoneTitle")}
                                </CardTitle>
                                <CardDescription>{t("project.billing.milestoneDesc")} ({project.contract.contractNumber})</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="rounded-md border">
                                    <table className="w-full text-sm">
                                        <thead className="bg-slate-50 border-b">
                                            <tr>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.milestone")}</th>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.amount")}</th>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.status")}</th>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.dueDate")}</th>
                                                <th className="p-3 text-right font-medium text-slate-500">{t("project.billing.action")}</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {project.contract.milestones?.map((m: any) => (
                                                <tr key={m.id} className="border-b last:border-0 hover:bg-slate-50/50">
                                                    <td className="p-3 font-medium">{m.name}</td>
                                                    <td className="p-3">¥{Number(m.amount).toLocaleString()}</td>
                                                    <td className="p-3">
                                                        <Badge variant={
                                                            m.status === 'Paid' ? 'default' :
                                                                m.status === 'Invoiced' ? 'secondary' :
                                                                    m.status === 'Verified' ? 'outline' : 'secondary'
                                                        } className={m.status === 'Verified' ? 'bg-green-50 text-green-700 border-green-200' : ''}>
                                                            {m.status}
                                                        </Badge>
                                                    </td>
                                                    <td className="p-3 text-slate-500">{m.dueDate ? new Date(m.dueDate).toLocaleDateString() : '-'}</td>
                                                    <td className="p-3 text-right flex items-center justify-end gap-2">
                                                        {/* Verify Action */}
                                                        {m.status !== 'Invoiced' && m.status !== 'Paid' && (
                                                            m.status === 'Verified' ? (
                                                                <Button size="sm" variant="ghost" className="h-7 text-xs text-slate-400 hover:text-slate-600"
                                                                    onClick={async () => {
                                                                        if (!confirm(t("project.billing.confirmUnverify"))) return;
                                                                        await api.patch(`/contracts/milestones/${m.id}`, { status: 'WIP' });
                                                                        fetchProject();
                                                                    }}>
                                                                    {t("project.billing.unverify")}
                                                                </Button>
                                                            ) : (
                                                                <Button size="sm" variant="outline" className="h-7 text-xs border-green-200 text-green-600 hover:bg-green-50"
                                                                    onClick={async () => {
                                                                        await api.patch(`/contracts/milestones/${m.id}`, { status: 'Verified' });
                                                                        fetchProject();
                                                                    }}>
                                                                    {t("project.billing.verify")}
                                                                </Button>
                                                            )
                                                        )}

                                                        {(m.status === 'Verified' || m.status === 'Ready_to_Invoice') && !m.invoiceDate && (
                                                            <Button size="sm" variant="outline" className="h-7 text-xs border-blue-200 text-blue-600 hover:bg-blue-50"
                                                                onClick={() => {
                                                                    router.push(`/finance/invoices/new?milestoneId=${m.id}`);
                                                                }}>
                                                                {t("project.billing.generateInvoice")}
                                                            </Button>
                                                        )}
                                                        {m.status === 'Invoiced' && <span className="text-xs text-slate-400">{t("project.billing.invoiceCreated")}</span>}
                                                        {m.status === 'Paid' && <span className="text-xs text-green-600 font-bold">{t("project.billing.paid")}</span>}
                                                    </td>
                                                </tr>
                                            ))}
                                            {(!project.contract.milestones || project.contract.milestones.length === 0) && (
                                                <tr><td colSpan={5} className="p-8 text-center text-slate-400">{t("project.billing.noMilestones")}</td></tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </CardContent>
                        </Card>

                        {/* 2. Project Invoices */}
                        <Card>
                            <CardHeader><CardTitle className="text-base">{t("project.billing.relatedInvoices")}</CardTitle></CardHeader>
                            <CardContent>
                                <div className="rounded-md border">
                                    <table className="w-full text-sm">
                                        <thead className="bg-slate-50 border-b">
                                            <tr>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.invoiceNumber")}</th>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.amount")}</th>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.date")}</th>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.status")}</th>
                                                <th className="p-3 text-left font-medium text-slate-500">{t("project.billing.milestone")}</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {project.invoices?.map((inv: any) => (
                                                <tr key={inv.id} className="border-b last:border-0">
                                                    <td className="p-3 font-medium text-blue-600">{inv.invoiceNumber}</td>
                                                    <td className="p-3">¥{inv.totalAmount.toLocaleString()}</td>
                                                    <td className="p-3 text-slate-500">{new Date(inv.invoiceDate).toLocaleDateString()}</td>
                                                    <td className="p-3">
                                                        <Badge variant={inv.status === 'Paid' ? 'default' : 'outline'}>{inv.status}</Badge>
                                                    </td>
                                                    <td className="p-3 text-slate-500">{project.contract.milestones?.find((m: any) => m.id === inv.milestoneId)?.name || '-'}</td>
                                                </tr>
                                            ))}
                                            {(!project.invoices || project.invoices.length === 0) && (
                                                <tr><td colSpan={5} className="p-8 text-center text-slate-400">{t("project.billing.noInvoices")}</td></tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="resources">
                    <Card className="mt-6">
                        <CardHeader>
                            <CardTitle>{t("project.tabs.resources")}</CardTitle>
                            <CardDescription>{t("project.descriptions.resourceAlloc")}</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <div className="flex flex-col md:flex-row gap-4 items-end bg-slate-50 p-6 rounded-xl border border-slate-100">
                                <div className="flex-1 w-full space-y-2">
                                    <Label className="text-slate-600 font-semibold">{t("project.fields.member")}</Label>
                                    <Select value={newResource.userId} onValueChange={(val) => setNewResource({ ...newResource, userId: val })}>
                                        <SelectTrigger className="bg-white"><SelectValue placeholder={t("project.placeholders.selectUser")} /></SelectTrigger>
                                        <SelectContent>
                                            {users.map(u => <SelectItem key={u.id} value={u.id}>{u.displayName}</SelectItem>)}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="flex-1 w-full space-y-2">
                                    <Label className="text-slate-600 font-semibold">{t("project.fields.role")}</Label>
                                    <Input className="bg-white" placeholder={t("project.placeholders.rolePlaceholder")} value={newResource.role} onChange={e => setNewResource({ ...newResource, role: e.target.value })} />
                                </div>
                                <div className="w-full md:w-32 space-y-2">
                                    <Label className="text-slate-600 font-semibold">{t("project.fields.allocation")}</Label>
                                    <Input className="bg-white" type="number" value={newResource.allocationPct} onChange={e => setNewResource({ ...newResource, allocationPct: e.target.value })} />
                                </div>
                                <Button onClick={handleAddResource} className="w-full md:w-auto bg-slate-900 text-white">
                                    <Plus className="h-4 w-4 mr-1" /> {t("project.actions.add")}
                                </Button>
                            </div>

                            <div className="border rounded-xl overflow-hidden shadow-sm">
                                <table className="w-full text-sm">
                                    <thead className="bg-slate-50 border-b">
                                        <tr>
                                            <th className="px-6 py-4 text-left font-bold text-slate-600">{t("project.fields.member")}</th>
                                            <th className="px-6 py-4 text-left font-bold text-slate-600">{t("project.fields.role")}</th>
                                            <th className="px-6 py-4 text-left font-bold text-slate-600">{t("project.fields.allocation")}</th>
                                            <th className="px-6 py-4 text-right font-bold text-slate-600">{t("project.fields.action")}</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-100">
                                        {!project.resources || project.resources.length === 0 ? (
                                            <tr><td colSpan={4} className="px-6 py-10 text-center text-slate-400 italic">No resources assigned yet.</td></tr>
                                        ) : project.resources.map((r: any) => (
                                            <tr key={r.id} className="hover:bg-slate-50/50 transition-colors">
                                                <td className="px-6 py-4">
                                                    <div className="font-bold text-slate-900">{r.user?.displayName || "Unknown User"}</div>
                                                    <div className="text-xs text-slate-400 font-mono uppercase">{r.user?.role || "USER"}</div>
                                                </td>
                                                <td className="px-6 py-4 text-slate-600 font-medium">{r.role}</td>
                                                <td className="px-6 py-4">
                                                    <Badge variant="secondary" className="bg-blue-50 text-blue-700 font-bold">{r.allocationPct}%</Badge>
                                                </td>
                                                <td className="px-6 py-4 text-right">
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        className="text-slate-400 hover:text-red-500 hover:bg-red-50 transition-all"
                                                        onClick={async () => {
                                                            try {
                                                                await api.delete(`/projects/resources/${r.id}`);
                                                                fetchProject();
                                                            } catch (error) { console.error(error); }
                                                        }}
                                                    >
                                                        <Trash2 className="h-4 w-4" />
                                                    </Button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="communication">
                    <Card className="mt-6">
                        <CardHeader>
                            <CardTitle>{t("project.tabs.communication")}</CardTitle>
                            <CardDescription>{t("project.descriptions.commPlan")}</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end bg-slate-50 p-6 rounded-xl border border-slate-100">
                                <div className="space-y-2">
                                    <Label className="text-slate-600 font-semibold">{t("project.fields.meetingTitle")}</Label>
                                    <Input className="bg-white" placeholder={t("project.placeholders.meetingTitle")} value={newMeeting.title} onChange={e => setNewMeeting({ ...newMeeting, title: e.target.value })} />
                                </div>
                                <div className="space-y-2">
                                    <Label className="text-slate-600 font-semibold">{t("project.fields.meetingType")}</Label>
                                    <Select value={newMeeting.type} onValueChange={(val: any) => setNewMeeting({ ...newMeeting, type: val })}>
                                        <SelectTrigger className="bg-white"><SelectValue placeholder="Weekly" /></SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="Kickoff">Kickoff</SelectItem>
                                            <SelectItem value="Weekly">Weekly</SelectItem>
                                            <SelectItem value="Technical">Technical</SelectItem>
                                            <SelectItem value="Review">Review</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label className="text-slate-600 font-semibold">{t("project.fields.planDate")}</Label>
                                    <Input className="bg-white" type="date" value={newMeeting.planDate} onChange={e => setNewMeeting({ ...newMeeting, planDate: e.target.value })} />
                                </div>
                                <Button onClick={handleAddMeeting} className="bg-slate-900 text-white">
                                    <Plus className="h-4 w-4 mr-1" /> {t("project.actions.schedule")}
                                </Button>
                            </div>

                            <div className="grid grid-cols-1 gap-3">
                                {project.meetings.length === 0 ? (
                                    <div className="text-center py-12 text-slate-400 border border-dashed rounded-xl italic">No meetings scheduled yet.</div>
                                ) : project.meetings.map((m: any) => (
                                    <div key={m.id} className="flex flex-col md:flex-row items-center justify-between p-5 border rounded-xl hover:shadow-md hover:border-blue-200 transition-all bg-white group">
                                        <div className="flex items-center gap-5 w-full">
                                            <div className="h-12 w-12 bg-blue-50 text-blue-500 rounded-xl flex items-center justify-center group-hover:bg-blue-500 group-hover:text-white transition-colors duration-300">
                                                <FileText className="h-6 w-6" />
                                            </div>
                                            <div className="space-y-1">
                                                <div className="font-bold text-slate-900 flex items-center gap-2">
                                                    {m.title}
                                                    <Badge variant="outline" className="text-[10px] uppercase font-bold tracking-tighter h-5">{m.type}</Badge>
                                                </div>
                                                <div className="text-sm text-slate-400 flex items-center gap-4 font-medium">
                                                    <span className="flex items-center gap-1"><FileText className="h-3 w-3" /> {new Date(m.planDate).toLocaleDateString()}</span>
                                                    {m.filename && <span className="flex items-center gap-1 text-green-600"><Save className="h-3 w-3" /> {m.filename}</span>}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex gap-2 mt-4 md:mt-0 w-full md:w-auto">
                                            {m.filename ? (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="flex-1 md:flex-none border-green-200 text-green-700 hover:bg-green-50"
                                                    onClick={() => window.open(`http://localhost:3000/${m.filepath}`)}
                                                >
                                                    <Download className="h-4 w-4 mr-2" /> {t("project.actions.downloadMinutes")}
                                                </Button>
                                            ) : (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="flex-1 md:flex-none border-blue-200 text-blue-700 hover:bg-blue-50"
                                                    disabled={uploadingMeetingId === m.id}
                                                    onClick={() => { setUploadingMeetingId(m.id); fileInputRef.current?.click(); }}
                                                >
                                                    {uploadingMeetingId === m.id ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Upload className="h-4 w-4 mr-2" />}
                                                    {t("project.actions.uploadMinutes")}
                                                </Button>
                                            )}
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                className="text-slate-300 hover:text-red-500 transition-colors"
                                                onClick={() => handleDeleteMeeting(m.id)}
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="risks">
                    <Card className="mt-6">
                        <CardHeader>
                            <CardTitle>{t("project.tabs.risks")}</CardTitle>
                            <CardDescription>{t("project.descriptions.riskTrack")}</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-8">
                            <div className="space-y-5 bg-slate-50 p-6 rounded-xl border border-slate-100">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <Label className="text-slate-600 font-semibold">{t("project.fields.riskDesc")}</Label>
                                        <Input className="bg-white" placeholder={t("project.placeholders.riskDesc")} value={newRisk.description} onChange={e => setNewRisk({ ...newRisk, description: e.target.value })} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label className="text-slate-600 font-semibold">{t("project.fields.impactLevel")}</Label>
                                        <Select onValueChange={(val) => setNewRisk({ ...newRisk, impact: val as any })}>
                                            <SelectTrigger className="bg-white"><SelectValue placeholder="Medium" /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="Low">Low</SelectItem>
                                                <SelectItem value="Medium">Medium</SelectItem>
                                                <SelectItem value="High">High</SelectItem>
                                                <SelectItem value="Critical">Critical</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <Label className="text-slate-600 font-semibold">{t("project.fields.mitigation")}</Label>
                                    <Textarea className="bg-white" rows={3} placeholder={t("project.placeholders.mitigation")} value={newRisk.mitigationPlan} onChange={e => setNewRisk({ ...newRisk, mitigationPlan: e.target.value })} />
                                </div>
                                <Button onClick={handleAddRisk} className="bg-slate-900 text-white w-full md:w-auto">
                                    <AlertTriangle className="h-4 w-4 mr-1" /> {t("project.actions.logRisk")}
                                </Button>
                            </div>

                            <div className="grid grid-cols-1 gap-4">
                                {project.risks.length === 0 ? (
                                    <div className="text-center py-12 text-slate-400 border border-dashed rounded-xl italic">No risks identified. Good job PM!</div>
                                ) : project.risks.map((risk: any) => (
                                    <div key={risk.id} className="p-6 border rounded-xl hover:shadow-md transition-all border-l-4" style={{ borderLeftColor: risk.impact === 'Critical' || risk.impact === 'High' ? '#ef4444' : risk.impact === 'Medium' ? '#f59e0b' : '#3b82f6' }}>
                                        {editingRiskId === risk.id ? (
                                            // Edit Mode
                                            <div className="space-y-4">
                                                <div className="space-y-2">
                                                    <Label className="text-sm font-semibold text-slate-700">{t("project.fields.riskDesc")}</Label>
                                                    <Input
                                                        value={editingRiskData.description}
                                                        onChange={e => setEditingRiskData({ ...editingRiskData, description: e.target.value })}
                                                        className="font-medium"
                                                    />
                                                </div>
                                                <div className="grid grid-cols-2 gap-4">
                                                    <div className="space-y-2">
                                                        <Label className="text-sm font-semibold text-slate-700">{t("project.fields.impactLevel")}</Label>
                                                        <Select value={editingRiskData.impact} onValueChange={(val) => setEditingRiskData({ ...editingRiskData, impact: val })}>
                                                            <SelectTrigger><SelectValue /></SelectTrigger>
                                                            <SelectContent>
                                                                <SelectItem value="Low">Low</SelectItem>
                                                                <SelectItem value="Medium">Medium</SelectItem>
                                                                <SelectItem value="High">High</SelectItem>
                                                            </SelectContent>
                                                        </Select>
                                                    </div>
                                                    <div className="space-y-2">
                                                        <Label className="text-sm font-semibold text-slate-700">状态</Label>
                                                        <Select value={editingRiskData.status} onValueChange={(val) => setEditingRiskData({ ...editingRiskData, status: val })}>
                                                            <SelectTrigger><SelectValue /></SelectTrigger>
                                                            <SelectContent>
                                                                <SelectItem value="Open">OPEN</SelectItem>
                                                                <SelectItem value="Mitigated">MITIGATED</SelectItem>
                                                                <SelectItem value="Closed">CLOSED</SelectItem>
                                                            </SelectContent>
                                                        </Select>
                                                    </div>
                                                </div>
                                                <div className="space-y-2">
                                                    <Label className="text-sm font-semibold text-slate-700">缓解策略</Label>
                                                    <Textarea
                                                        rows={3}
                                                        value={editingRiskData.mitigationPlan}
                                                        onChange={e => setEditingRiskData({ ...editingRiskData, mitigationPlan: e.target.value })}
                                                        placeholder="如何应对这个风险？"
                                                    />
                                                </div>
                                                <div className="flex gap-2 justify-end">
                                                    <Button variant="outline" size="sm" onClick={handleCancelEditRisk}>
                                                        <X className="h-4 w-4 mr-1" /> 取消
                                                    </Button>
                                                    <Button size="sm" onClick={() => handleSaveRisk(risk.id)} className="bg-green-600 hover:bg-green-700">
                                                        <Save className="h-4 w-4 mr-1" /> 保存
                                                    </Button>
                                                </div>
                                            </div>
                                        ) : (
                                            // View Mode
                                            <>
                                                <div className="flex justify-between items-start mb-4">
                                                    <div className="flex items-center gap-3">
                                                        <Badge className={risk.impact === 'High' || risk.impact === 'Critical' ? 'bg-red-50 text-red-600 border-red-100 uppercase' : risk.impact === 'Medium' ? 'bg-amber-50 text-amber-600 border-amber-100 uppercase' : 'bg-blue-50 text-blue-600 border-blue-100 uppercase'} variant="outline">
                                                            {risk.impact}
                                                        </Badge>
                                                        <h3 className="font-bold text-slate-900">{risk.description}</h3>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <Badge variant="secondary" className="font-bold text-[10px] tracking-wide bg-slate-100 text-slate-700 uppercase">
                                                            {risk.status}
                                                        </Badge>
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => handleEditRisk(risk)}
                                                            className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                                                        >
                                                            <Edit className="h-4 w-4" />
                                                        </Button>
                                                    </div>
                                                </div>
                                                {risk.mitigationPlan && (
                                                    <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg font-medium leading-relaxed">
                                                        <div className="text-xs font-bold text-slate-400 uppercase mb-1 flex items-center gap-1"><Save className="h-3 w-3" /> Mitigation Strategy</div>
                                                        {risk.mitigationPlan}
                                                    </div>
                                                )}
                                            </>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}
