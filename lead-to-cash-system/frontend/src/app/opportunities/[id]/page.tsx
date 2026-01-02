"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Building2, Users, Briefcase, FileText, Upload, X, FileIcon, DollarSign, Save, Plus, Pencil, Eye, Clock, Calendar, Target, Paperclip, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n/I18nContext";
import { RichTextEditor } from "@/components/RichTextEditor";

interface Opportunity {
    id: string;
    opportunityNumber: string;
    title: string;
    status: string;
    estimatedValue: number;
    probability: number;
    source: string;
    expectedCloseDate: string;
    salesStage: string;
    competitors: string;
    decisionMakers: string;
    salesOwner: string;
    dealType: string;
    deliveryModel: string;
    estimatedEffort: number;
    richDescription: string;
    projectBudget: number;
    businessCost: number;
    laborCost: number;
    otherCost: number;
    grossProfit: number;
    profitMargin: number;
    createdAt: string;
    customer: {
        id: string;
        companyName: string;
        industry: string;
        companySize: string;
        city: string;
        contactName: string;
        contactTitle: string;
        contactPhone: string;
        contactEmail: string;
    };
    followUps: Array<{
        id: string;
        content: string;
        createdBy: string;
        createdAt: string;
    }>;
    attachments: Array<{
        id: string;
        filename: string;
        filepath: string;
        mimetype: string;
        size: number;
        createdAt: string;
    }>;
}

interface BiddingTask {
    id: string;
    name: string;
    category: string;
    isCompleted: boolean;
    assignee: string | null;
}

interface ProcurementDocument {
    id: string;
    filename: string;
    size: number;
    docType: string;
}

interface Procurement {
    id: string;
    procurementNumber: string;
    type: string;
    status: string;
    commercialOwner: string | null;
    technicalOwner: string | null;
    customerBudget: number | null;
    ourQuote: number | null;
    submissionDeadline: string | null;
    notificationDate: string | null;
    bidLocation: string | null;
    depositAmount: number | null;
    notes: string | null;
    resultNote?: string;
    tasks: BiddingTask[];
    documents: ProcurementDocument[];
}

export default function OpportunityDetailPage() {
    const { t } = useI18n();
    const params = useParams();
    const router = useRouter();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [opportunity, setOpportunity] = useState<Opportunity | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isEditing, setIsEditing] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    // Edit form state
    const [form, setForm] = useState({
        title: "",
        status: "",
        estimatedValue: 0,
        probability: 0,
        source: "",
        expectedCloseDate: "",
        salesStage: "",
        competitors: "",
        decisionMakers: "",
        salesOwner: "",
        dealType: "",
        deliveryModel: "",
        estimatedEffort: 0,
        richDescription: "",
        projectBudget: 0,
        businessCost: 0,
        laborCost: 0,
        otherCost: 0,
        grossProfit: 0,
        profitMargin: 0,
    });

    // Follow-up state
    const [showFollowUpEditor, setShowFollowUpEditor] = useState(false);
    const [followUpContent, setFollowUpContent] = useState("");
    const [isSubmittingFollowUp, setIsSubmittingFollowUp] = useState(false);

    // Procurement state
    const [procurements, setProcurements] = useState<Procurement[]>([]);
    const [showNewProcurement, setShowNewProcurement] = useState(false);
    const [newProcType, setNewProcType] = useState("");

    const opportunityId = params.id as string;

    useEffect(() => {
        if (opportunityId) {
            fetchOpportunity();
            fetchProcurements();
        }
    }, [opportunityId]);

    const fetchProcurements = async () => {
        try {
            const data = await api.get(`/procurements?opportunityId=${opportunityId}`);
            setProcurements(data);
        } catch (error) {
            console.error("Failed to fetch procurements", error);
        }
    };

    const handleCreateProcurement = async () => {
        if (!newProcType) return;
        try {
            await api.post("/procurements", {
                opportunityId,
                type: newProcType,
            });
            setShowNewProcurement(false);
            setNewProcType("");
            fetchProcurements();
        } catch (error) {
            console.error("Failed to create procurement", error);
        }
    };

    const handleUpdate = async (field: keyof Opportunity, value: any) => {
        if (!opportunity) return;
        try {
            await api.patch(`/opportunities/${params.id}`, { [field]: value });
            setOpportunity({ ...opportunity, [field]: value });
        } catch (error) {
            console.error("Failed to update opportunity:", error);
        }
    };

    const handleCustomerUpdate = async (field: keyof Opportunity['customer'], value: any) => {
        if (!opportunity?.customer) return;
        try {
            setOpportunity({
                ...opportunity,
                customer: {
                    ...opportunity.customer,
                    [field]: value
                }
            });
            await api.patch(`/customers/${opportunity.customer.id}`, { [field]: value });
        } catch (error) {
            console.error("Failed to update customer:", error);
            // Revert could be implemented here if needed, but for now we rely on re-fetch or user retry
        }
    };

    const handleToggleTask = async (taskId: string, currentStatus: boolean) => {
        try {
            await api.patch(`/procurements/tasks/${taskId}`, {
                isCompleted: !currentStatus,
            });
            fetchProcurements();
        } catch (error) {
            console.error("Failed to update task", error);
        }
    };

    const fetchOpportunity = async () => {
        try {
            const data = await api.get(`/opportunities/${opportunityId}`);
            setOpportunity(data);
            // Initialize form with opportunity data
            setForm({
                title: data.title || "",
                status: data.status || "New",
                estimatedValue: data.estimatedValue || 0,
                probability: data.probability || 0,
                source: data.source || "",
                expectedCloseDate: data.expectedCloseDate ? data.expectedCloseDate.split('T')[0] : "",
                salesStage: data.salesStage || "",
                competitors: data.competitors || "",
                decisionMakers: data.decisionMakers || "",
                salesOwner: data.salesOwner || "",
                dealType: data.dealType || "",
                deliveryModel: data.deliveryModel || "",
                estimatedEffort: data.estimatedEffort || 0,
                richDescription: data.richDescription || "",
                projectBudget: data.projectBudget || 0,
                businessCost: data.businessCost || 0,
                laborCost: data.laborCost || 0,
                otherCost: data.otherCost || 0,
                grossProfit: data.grossProfit || 0,
                profitMargin: data.profitMargin || 0,
            });
        } catch (error) {
            console.error("Failed to fetch opportunity", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSave = async () => {
        setIsSaving(true);
        try {
            await api.patch(`/opportunities/${opportunityId}`, {
                title: form.title,
                status: form.status,
                estimatedValue: Number(form.estimatedValue),
                probability: Number(form.probability),
                source: form.source || undefined,
                expectedCloseDate: form.expectedCloseDate || undefined,
                salesStage: form.salesStage || undefined,
                competitors: form.competitors || undefined,
                decisionMakers: form.decisionMakers || undefined,
                salesOwner: form.salesOwner || undefined,
                dealType: form.dealType || undefined,
                deliveryModel: form.deliveryModel || undefined,
                estimatedEffort: Number(form.estimatedEffort) || undefined,
                richDescription: form.richDescription || undefined,
                projectBudget: Number(form.projectBudget) || undefined,
                businessCost: Number(form.businessCost) || undefined,
                laborCost: Number(form.laborCost) || undefined,
                otherCost: Number(form.otherCost) || undefined,
                grossProfit: Number(form.grossProfit) || undefined,
                profitMargin: Number(form.profitMargin) || undefined,
            });
            setIsEditing(false);
            fetchOpportunity();
        } catch (error) {
            console.error("Failed to update opportunity", error);
        } finally {
            setIsSaving(false);
        }
    };

    const handleCancelEdit = () => {
        if (opportunity) {
            setForm({
                title: opportunity.title || "",
                status: opportunity.status || "New",
                estimatedValue: opportunity.estimatedValue || 0,
                probability: opportunity.probability || 0,
                source: opportunity.source || "",
                expectedCloseDate: opportunity.expectedCloseDate ? opportunity.expectedCloseDate.split('T')[0] : "",
                salesStage: opportunity.salesStage || "",
                competitors: opportunity.competitors || "",
                decisionMakers: opportunity.decisionMakers || "",
                salesOwner: opportunity.salesOwner || "",
                dealType: opportunity.dealType || "",
                deliveryModel: opportunity.deliveryModel || "",
                estimatedEffort: opportunity.estimatedEffort || 0,
                richDescription: opportunity.richDescription || "",
                projectBudget: opportunity.projectBudget || 0,
                businessCost: opportunity.businessCost || 0,
                laborCost: opportunity.laborCost || 0,
                otherCost: opportunity.otherCost || 0,
                grossProfit: opportunity.grossProfit || 0,
                profitMargin: opportunity.profitMargin || 0,
            });
        }
        setIsEditing(false);
    };

    const handleAddFollowUp = async () => {
        if (!followUpContent.trim()) return;
        setIsSubmittingFollowUp(true);
        try {
            await api.post(`/opportunities/${opportunityId}/follow-ups`, {
                content: followUpContent,
                createdBy: "Current User",
            });
            setFollowUpContent("");
            setShowFollowUpEditor(false);
            fetchOpportunity();
        } catch (error) {
            console.error("Failed to add follow-up", error);
        } finally {
            setIsSubmittingFollowUp(false);
        }
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            await fetch(`http://localhost:3000/opportunities/${opportunityId}/attachments`, {
                method: 'POST',
                body: formData,
            });
            fetchOpportunity();
        } catch (error) {
            console.error("Failed to upload file", error);
        }
    };

    const handleCreateContract = async () => {
        try {
            const res = await api.post("/contracts", {
                opportunityId,
                contractNumber: `CTR-${new Date().getFullYear()}-${Math.floor(Math.random() * 10000)}`,
                totalContractValue: Number(opportunity?.estimatedValue) || 0,
            });
            router.push(`/contracts/${res.id}`);
        } catch (error) {
            console.error("Failed to create contract", error);
        }
    };

    // Calculate financial fields
    const recalculateFinancials = (field: string, value: number) => {
        const newForm = { ...form, [field]: value };
        const budget = field === 'projectBudget' ? value : Number(newForm.projectBudget) || 0;
        const businessCost = field === 'businessCost' ? value : Number(newForm.businessCost) || 0;
        const laborCost = field === 'laborCost' ? value : Number(newForm.laborCost) || 0;
        const otherCost = field === 'otherCost' ? value : Number(newForm.otherCost) || 0;
        const totalCost = businessCost + laborCost + otherCost;
        const grossProfit = budget - totalCost;
        const profitMargin = budget > 0 ? Math.round((grossProfit / budget) * 100) : 0;
        setForm({ ...newForm, grossProfit, profitMargin });
    };

    if (isLoading) {
        return (
            <div className="min-h-screen bg-slate-50 p-8 flex items-center justify-center">
                <div className="text-slate-500">Loading...</div>
            </div>
        );
    }

    if (!opportunity) {
        return (
            <div className="min-h-screen bg-slate-50 p-8 flex items-center justify-center">
                <div className="text-slate-500">Opportunity not found</div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-50 p-8">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                    <Link href="/opportunities" className="flex items-center text-sm text-slate-600 hover:text-slate-900 mb-4">
                        <ArrowLeft className="h-4 w-4 mr-1" />
                        {t("nav.opportunities")}
                    </Link>
                    <div className="flex items-start justify-between">
                        <div>
                            <div className="flex items-center gap-3 mb-1">
                                <h1 className="text-3xl font-bold tracking-tight text-slate-900">{opportunity.title}</h1>
                                {opportunity.opportunityNumber && (
                                    <span className="text-sm font-mono bg-slate-100 text-slate-600 px-2 py-1 rounded">
                                        {opportunity.opportunityNumber}
                                    </span>
                                )}
                            </div>
                            <p className="text-slate-500">{opportunity.customer?.companyName}</p>
                        </div>
                        <div className="flex items-center gap-2">
                            {opportunity.status === 'Won' && (
                                <Button size="sm" variant="default" className="bg-green-600 hover:bg-green-700" onClick={handleCreateContract}>
                                    <FileText className="h-4 w-4 mr-1" />
                                    拟定合同
                                </Button>
                            )}
                            {isEditing ? (
                                <>
                                    <Button size="sm" variant="outline" onClick={handleCancelEdit}>
                                        <X className="h-4 w-4 mr-1" />
                                        取消
                                    </Button>
                                    <Button size="sm" onClick={handleSave} disabled={isSaving}>
                                        <Save className="h-4 w-4 mr-1" />
                                        {isSaving ? "保存中..." : "保存"}
                                    </Button>
                                </>
                            ) : (
                                <Button size="sm" variant="outline" onClick={() => setIsEditing(true)}>
                                    <Pencil className="h-4 w-4 mr-1" />
                                    {t("detail.edit")}
                                </Button>
                            )}
                            {opportunity.status === 'Won' && (
                                <Button size="sm" variant="default" className="bg-green-600 hover:bg-green-700" onClick={handleCreateContract}>
                                    <FileText className="h-4 w-4 mr-1" />
                                    拟定合同
                                </Button>
                            )}
                            <Badge variant={opportunity.status === 'Won' ? 'default' : opportunity.status === 'Lost' ? 'destructive' : 'secondary'}>
                                {opportunity.status}
                            </Badge>
                        </div>
                    </div>
                </div>

                {/* Tabs - Same structure as new opportunity form */}
                <Tabs defaultValue="customer" className="w-full">
                    <TabsList className="grid w-full grid-cols-7 mb-6">
                        <TabsTrigger value="customer" className="flex items-center gap-2">
                            <Building2 className="h-4 w-4" />
                            {t("form.sections.customer")}
                        </TabsTrigger>
                        <TabsTrigger value="opportunity" className="flex items-center gap-2">
                            <Briefcase className="h-4 w-4" />
                            {t("form.sections.opportunity")}
                        </TabsTrigger>
                        <TabsTrigger value="enterprise" className="flex items-center gap-2">
                            <Users className="h-4 w-4" />
                            {t("form.sections.enterprise")}
                        </TabsTrigger>
                        <TabsTrigger value="financial" className="flex items-center gap-2">
                            <DollarSign className="h-4 w-4" />
                            {t("detail.financials")}
                        </TabsTrigger>
                        <TabsTrigger value="description" className="flex items-center gap-2">
                            <FileText className="h-4 w-4" />
                            {t("form.sections.description")}
                        </TabsTrigger>
                        <TabsTrigger value="followups" className="flex items-center gap-2">
                            <Clock className="h-4 w-4" />
                            跟进记录
                        </TabsTrigger>
                        <TabsTrigger value="procurement" className="flex items-center gap-2">
                            <Target className="h-4 w-4" />
                            投标管理
                        </TabsTrigger>
                    </TabsList>

                    {/* Customer Tab */}
                    <TabsContent value="customer">
                        <Card>
                            <CardHeader>
                                <CardTitle>{t("form.sections.customer")}</CardTitle>
                                <CardDescription>{t("form.sectionDesc.customer")}</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="p-4 bg-slate-50 rounded-lg border space-y-4">
                                    <div className="flex items-center gap-2 text-lg font-semibold">
                                        <Building2 className="h-5 w-5 text-blue-500" />
                                        <Input
                                            value={opportunity.customer?.companyName || ""}
                                            onChange={(e) => handleCustomerUpdate("companyName", e.target.value)}
                                            className="max-w-md font-semibold text-lg bg-transparent border-transparent hover:border-slate-300 focus:bg-white transition-colors p-0 h-auto"
                                        />
                                    </div>
                                    <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                                        <div className="space-y-1">
                                            <Label className="text-xs text-slate-500">行业</Label>
                                            <Input
                                                value={opportunity.customer?.industry || ""}
                                                onChange={(e) => handleCustomerUpdate("industry", e.target.value)}
                                                className="bg-white h-8"
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <Label className="text-xs text-slate-500">规模</Label>
                                            <Select
                                                value={opportunity.customer?.companySize || ""}
                                                onValueChange={(val) => handleCustomerUpdate("companySize", val)}
                                            >
                                                <SelectTrigger className="bg-white h-8">
                                                    <SelectValue placeholder="Select size" />
                                                </SelectTrigger>
                                                <SelectContent>
                                                    <SelectItem value="Small">Small (1-50)</SelectItem>
                                                    <SelectItem value="Medium">Medium (51-200)</SelectItem>
                                                    <SelectItem value="Large">Large (201-1000)</SelectItem>
                                                    <SelectItem value="Enterprise">Enterprise (1000+)</SelectItem>
                                                </SelectContent>
                                            </Select>
                                        </div>
                                        <div className="space-y-1">
                                            <Label className="text-xs text-slate-500">城市</Label>
                                            <Input
                                                value={opportunity.customer?.city || ""}
                                                onChange={(e) => handleCustomerUpdate("city", e.target.value)}
                                                className="bg-white h-8"
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <Label className="text-xs text-slate-500">联系人</Label>
                                            <Input
                                                value={opportunity.customer?.contactName || ""}
                                                onChange={(e) => handleCustomerUpdate("contactName", e.target.value)}
                                                className="bg-white h-8"
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <Label className="text-xs text-slate-500">职位</Label>
                                            <Input
                                                value={opportunity.customer?.contactTitle || ""}
                                                onChange={(e) => handleCustomerUpdate("contactTitle", e.target.value)}
                                                className="bg-white h-8"
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <Label className="text-xs text-slate-500">电话</Label>
                                            <Input
                                                value={opportunity.customer?.contactPhone || ""}
                                                onChange={(e) => handleCustomerUpdate("contactPhone", e.target.value)}
                                                className="bg-white h-8"
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <Label className="text-xs text-slate-500">邮箱</Label>
                                            <Input
                                                value={opportunity.customer?.contactEmail || ""}
                                                onChange={(e) => handleCustomerUpdate("contactEmail", e.target.value)}
                                                className="bg-white h-8"
                                            />
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Opportunity Tab */}
                    <TabsContent value="opportunity">
                        <Card>
                            <CardHeader>
                                <CardTitle>{t("form.sections.opportunity")}</CardTitle>
                                <CardDescription>项目和财务基本信息</CardDescription>
                            </CardHeader>
                            <CardContent className="grid grid-cols-2 gap-4">
                                <div className="col-span-2">
                                    <Label>{t("form.projectName")} *</Label>
                                    {isEditing ? (
                                        <Input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.title}</div>
                                    )}
                                </div>
                                <div>
                                    <Label>状态</Label>
                                    {isEditing ? (
                                        <Select value={form.status} onValueChange={(v) => setForm({ ...form, status: v })}>
                                            <SelectTrigger><SelectValue /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="New">新建</SelectItem>
                                                <SelectItem value="Proposal">提案中</SelectItem>
                                                <SelectItem value="Negotiation">谈判中</SelectItem>
                                                <SelectItem value="Won">已赢单</SelectItem>
                                                <SelectItem value="Lost">已丢单</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.status}</div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.closeDate")}</Label>
                                    {isEditing ? (
                                        <Input type="date" value={form.expectedCloseDate} onChange={(e) => setForm({ ...form, expectedCloseDate: e.target.value })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">
                                            {opportunity.expectedCloseDate ? new Date(opportunity.expectedCloseDate).toLocaleDateString() : '-'}
                                        </div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.estValue")}</Label>
                                    {isEditing ? (
                                        <Input type="number" value={form.estimatedValue} onChange={(e) => setForm({ ...form, estimatedValue: Number(e.target.value) })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">
                                            ¥{(opportunity.estimatedValue || 0).toLocaleString()}
                                        </div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.probability")}</Label>
                                    {isEditing ? (
                                        <Input type="number" value={form.probability} onChange={(e) => setForm({ ...form, probability: Number(e.target.value) })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.probability || 0}%</div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.source")}</Label>
                                    {isEditing ? (
                                        <Select value={form.source} onValueChange={(v) => setForm({ ...form, source: v })}>
                                            <SelectTrigger><SelectValue placeholder="选择来源" /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="referral">客户推荐</SelectItem>
                                                <SelectItem value="website">官网</SelectItem>
                                                <SelectItem value="cold_call">主动拓展</SelectItem>
                                                <SelectItem value="event">展会/活动</SelectItem>
                                                <SelectItem value="partner">合作伙伴</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.source || '-'}</div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Enterprise Tab */}
                    <TabsContent value="enterprise">
                        <Card>
                            <CardHeader>
                                <CardTitle>{t("form.sections.enterprise")}</CardTitle>
                                <CardDescription>{t("form.sectionDesc.enterprise")}</CardDescription>
                            </CardHeader>
                            <CardContent className="grid grid-cols-2 gap-4">
                                <div>
                                    <Label>{t("form.salesStage")}</Label>
                                    {isEditing ? (
                                        <Select value={form.salesStage} onValueChange={(v) => setForm({ ...form, salesStage: v })}>
                                            <SelectTrigger><SelectValue placeholder="选择阶段" /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="initial_contact">初步接触</SelectItem>
                                                <SelectItem value="requirement">需求确认</SelectItem>
                                                <SelectItem value="proposal">方案提交</SelectItem>
                                                <SelectItem value="negotiation">商务谈判</SelectItem>
                                                <SelectItem value="contract">合同签订</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.salesStage || '-'}</div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.dealType")}</Label>
                                    {isEditing ? (
                                        <Select value={form.dealType} onValueChange={(v) => setForm({ ...form, dealType: v })}>
                                            <SelectTrigger><SelectValue placeholder="选择类型" /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="new">新签</SelectItem>
                                                <SelectItem value="renewal">续约</SelectItem>
                                                <SelectItem value="upsell">追加</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.dealType || '-'}</div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.salesOwner")}</Label>
                                    {isEditing ? (
                                        <Input value={form.salesOwner} onChange={(e) => setForm({ ...form, salesOwner: e.target.value })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.salesOwner || '-'}</div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.deliveryModel")}</Label>
                                    {isEditing ? (
                                        <Select value={form.deliveryModel} onValueChange={(v) => setForm({ ...form, deliveryModel: v })}>
                                            <SelectTrigger><SelectValue placeholder="选择模式" /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="onsite">驻场</SelectItem>
                                                <SelectItem value="remote">远程</SelectItem>
                                                <SelectItem value="hybrid">混合</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.deliveryModel || '-'}</div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.estimatedEffort")}</Label>
                                    {isEditing ? (
                                        <Input type="number" value={form.estimatedEffort} onChange={(e) => setForm({ ...form, estimatedEffort: Number(e.target.value) })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.estimatedEffort || 0} 人天</div>
                                    )}
                                </div>
                                <div>
                                    <Label>{t("form.competitors")}</Label>
                                    {isEditing ? (
                                        <Input value={form.competitors} onChange={(e) => setForm({ ...form, competitors: e.target.value })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.competitors || '-'}</div>
                                    )}
                                </div>
                                <div className="col-span-2">
                                    <Label>{t("form.decisionMakers")}</Label>
                                    {isEditing ? (
                                        <Input value={form.decisionMakers} onChange={(e) => setForm({ ...form, decisionMakers: e.target.value })} />
                                    ) : (
                                        <div className="p-2 bg-slate-50 rounded border">{opportunity.decisionMakers || '-'}</div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Financial Tab */}
                    <TabsContent value="financial">
                        <Card>
                            <CardHeader>
                                <CardTitle>{t("form.sections.financial")}</CardTitle>
                                <CardDescription>{t("form.sectionDesc.financial")}</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <Label>{t("detail.projectBudget")}</Label>
                                        {isEditing ? (
                                            <Input type="number" value={form.projectBudget} onChange={(e) => recalculateFinancials('projectBudget', Number(e.target.value))} />
                                        ) : (
                                            <div className="p-2 bg-slate-50 rounded border">¥{(opportunity.projectBudget || 0).toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div>
                                        <Label>{t("detail.businessCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" value={form.businessCost} onChange={(e) => recalculateFinancials('businessCost', Number(e.target.value))} />
                                        ) : (
                                            <div className="p-2 bg-slate-50 rounded border">¥{(opportunity.businessCost || 0).toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div>
                                        <Label>{t("detail.laborCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" value={form.laborCost} onChange={(e) => recalculateFinancials('laborCost', Number(e.target.value))} />
                                        ) : (
                                            <div className="p-2 bg-slate-50 rounded border">¥{(opportunity.laborCost || 0).toLocaleString()}</div>
                                        )}
                                    </div>
                                    <div>
                                        <Label>{t("detail.otherCost")}</Label>
                                        {isEditing ? (
                                            <Input type="number" value={form.otherCost} onChange={(e) => recalculateFinancials('otherCost', Number(e.target.value))} />
                                        ) : (
                                            <div className="p-2 bg-slate-50 rounded border">¥{(opportunity.otherCost || 0).toLocaleString()}</div>
                                        )}
                                    </div>
                                </div>

                                {/* Calculated results */}
                                <div className="border-t pt-4">
                                    <div className="text-sm text-slate-500 mb-3">计算结果</div>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                                            <Label className="text-green-700">{t("detail.grossProfit")}</Label>
                                            <div className="text-2xl font-bold text-green-600 mt-1">
                                                ¥{(isEditing ? form.grossProfit : (opportunity.grossProfit || 0)).toLocaleString()}
                                            </div>
                                        </div>
                                        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                                            <Label className="text-blue-700">{t("detail.profitMargin")}</Label>
                                            <div className="text-2xl font-bold text-blue-600 mt-1">
                                                {isEditing ? form.profitMargin : (opportunity.profitMargin || 0)}%
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Description Tab */}
                    <TabsContent value="description">
                        <Card>
                            <CardHeader>
                                <CardTitle>{t("form.sections.description")}</CardTitle>
                                <CardDescription>详细描述和附件</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                {isEditing ? (
                                    <RichTextEditor
                                        content={form.richDescription}
                                        onChange={(v) => setForm({ ...form, richDescription: v })}
                                        placeholder="输入项目详细描述..."
                                    />
                                ) : (
                                    <div>
                                        {opportunity.richDescription ? (
                                            <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: opportunity.richDescription }} />
                                        ) : (
                                            <div className="text-slate-400 italic">暂无描述</div>
                                        )}
                                    </div>
                                )}

                                {/* Attachments */}
                                <div className="border-t pt-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <Label className="text-base font-medium">附件</Label>
                                        {isEditing && (
                                            <>
                                                <Button type="button" variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
                                                    <Upload className="h-4 w-4 mr-2" />
                                                    上传附件
                                                </Button>
                                                <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileUpload} />
                                            </>
                                        )}
                                    </div>
                                    {opportunity.attachments?.length > 0 ? (
                                        <div className="space-y-2">
                                            {opportunity.attachments.map((attachment) => (
                                                <div key={attachment.id} className="flex items-center gap-3 p-3 rounded-lg border bg-slate-50">
                                                    <Paperclip className="h-5 w-5 text-blue-500" />
                                                    <div className="flex-1">
                                                        <a href={`http://localhost:3000/opportunities/attachments/download/${attachment.filename}`}
                                                            className="font-medium text-sm text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">
                                                            {attachment.filename}
                                                        </a>
                                                        <div className="text-xs text-slate-500">{(attachment.size / 1024).toFixed(1)} KB</div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="text-center py-6 text-slate-400 border-2 border-dashed rounded-lg">
                                            暂无附件
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Follow-ups Tab */}
                    <TabsContent value="followups">
                        <Card>
                            <CardHeader className="flex flex-row items-center justify-between">
                                <div>
                                    <CardTitle>跟进记录</CardTitle>
                                    <CardDescription>活动和沟通历史</CardDescription>
                                </div>
                                {isEditing && (
                                    <Button size="sm" onClick={() => setShowFollowUpEditor(true)}>
                                        <Plus className="h-4 w-4 mr-1" />
                                        添加跟进
                                    </Button>
                                )}
                            </CardHeader>
                            <CardContent>
                                {showFollowUpEditor && (
                                    <div className="mb-6 space-y-3">
                                        <RichTextEditor content={followUpContent} onChange={setFollowUpContent} placeholder="输入跟进内容..." />
                                        <div className="flex justify-end gap-2">
                                            <Button variant="outline" size="sm" onClick={() => setShowFollowUpEditor(false)}>取消</Button>
                                            <Button size="sm" onClick={handleAddFollowUp} disabled={isSubmittingFollowUp}>
                                                {isSubmittingFollowUp ? '保存中...' : '保存'}
                                            </Button>
                                        </div>
                                    </div>
                                )}

                                {opportunity.followUps?.length > 0 ? (
                                    <div className="space-y-4">
                                        {opportunity.followUps.map((followUp) => (
                                            <div key={followUp.id} className="border-l-2 border-blue-200 pl-4 py-2">
                                                <div className="flex items-center gap-2 text-sm text-slate-500 mb-2">
                                                    <span className="font-medium text-slate-700">{followUp.createdBy || 'Unknown'}</span>
                                                    <span>•</span>
                                                    <span>{new Date(followUp.createdAt).toLocaleString()}</span>
                                                </div>
                                                <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: followUp.content }} />
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-center py-8 text-slate-400">暂无跟进记录</div>
                                )}
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Procurement Tab */}
                    <TabsContent value="procurement">
                        <Card>
                            <CardHeader className="flex flex-row items-center justify-between">
                                <div>
                                    <CardTitle>投标管理</CardTitle>
                                    <CardDescription>响应客户标书/投标流程管理</CardDescription>
                                </div>
                                {isEditing && !showNewProcurement && (
                                    <Button size="sm" onClick={() => setShowNewProcurement(true)}>
                                        <Plus className="h-4 w-4 mr-1" />
                                        发起投标
                                    </Button>
                                )}
                            </CardHeader>
                            <CardContent>
                                {showNewProcurement && (
                                    <div className="mb-6 p-4 border rounded-lg bg-slate-50">
                                        <Label className="mb-2 block">选择投标类型</Label>
                                        <Select value={newProcType} onValueChange={setNewProcType}>
                                            <SelectTrigger>
                                                <SelectValue placeholder="请选择投标类型" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="DirectQuote">单一来源 - 直接报价</SelectItem>
                                                <SelectItem value="Negotiation">商务谈判</SelectItem>
                                                <SelectItem value="Comparison">比选</SelectItem>
                                                <SelectItem value="Consultation">磋商</SelectItem>
                                                <SelectItem value="PublicTender">公开招标</SelectItem>
                                            </SelectContent>
                                        </Select>
                                        <div className="flex justify-end gap-2 mt-4">
                                            <Button variant="outline" size="sm" onClick={() => setShowNewProcurement(false)}>取消</Button>
                                            <Button size="sm" onClick={handleCreateProcurement} disabled={!newProcType}>
                                                创建投标
                                            </Button>
                                        </div>
                                    </div>
                                )}

                                {procurements.length > 0 ? (
                                    <div className="space-y-6">
                                        {procurements.map((proc) => {
                                            const completedTasks = proc.tasks?.filter(t => t.isCompleted).length || 0;
                                            const totalTasks = proc.tasks?.length || 0;
                                            const progress = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;

                                            return (
                                                <div key={proc.id} className="bg-white border rounded-lg p-6 relative">
                                                    {/* Status Badge */}
                                                    <div className="absolute top-6 right-6">
                                                        <Badge variant={proc.status === 'Won' ? 'default' : proc.status === 'Lost' ? 'destructive' : 'secondary'}>
                                                            {proc.status}
                                                        </Badge>
                                                    </div>

                                                    <div className="flex items-center gap-2 mb-4">
                                                        <Briefcase className="h-5 w-5 text-blue-600" />
                                                        <h3 className="font-semibold text-lg">{proc.procurementNumber}</h3>
                                                        <span className="text-slate-500 text-sm">({proc.type})</span>
                                                    </div>

                                                    {/* Progress Bar */}
                                                    <div className="mb-6">
                                                        <div className="flex justify-between text-sm text-slate-500 mb-1">
                                                            <span>材料准备进度</span>
                                                            <span>{Math.round(progress)}% ({completedTasks}/{totalTasks})</span>
                                                        </div>
                                                        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full bg-blue-500 transition-all duration-500"
                                                                style={{ width: `${progress}%` }}
                                                            />
                                                        </div>
                                                    </div>

                                                    {/* Info Grid */}
                                                    <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
                                                        <div>
                                                            <span className="text-slate-500 block">截止日期</span>
                                                            <span>{proc.submissionDeadline ? new Date(proc.submissionDeadline).toLocaleDateString() : '-'}</span>
                                                        </div>
                                                        <div>
                                                            <span className="text-slate-500 block">开标日期</span>
                                                            <span>{proc.notificationDate ? new Date(proc.notificationDate).toLocaleDateString() : '-'}</span>
                                                        </div>
                                                        <div>
                                                            <span className="text-slate-500 block">商务负责人</span>
                                                            <span>{proc.commercialOwner || '未指定'}</span>
                                                        </div>
                                                        <div>
                                                            <span className="text-slate-500 block">技术负责人</span>
                                                            <span>{proc.technicalOwner || '未指定'}</span>
                                                        </div>
                                                    </div>

                                                    <div className="space-y-6">
                                                        {/* Tasks Section */}
                                                        <div>
                                                            <h4 className="text-sm font-medium text-slate-700 mb-3">材料清单</h4>
                                                            <div className="space-y-2">
                                                                {proc.tasks?.map((task: any) => (
                                                                    <div key={task.id} className="flex items-center gap-3 p-2 hover:bg-slate-50 rounded">
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={task.isCompleted}
                                                                            onChange={() => handleToggleTask(task.id, task.isCompleted)}
                                                                            className="rounded border-slate-300"
                                                                        />
                                                                        <span className={task.isCompleted ? "text-slate-400 line-through text-sm" : "text-sm"}>
                                                                            {task.name}
                                                                        </span>
                                                                        <Badge variant="outline" className="text-xs ml-auto">
                                                                            {task.category}
                                                                        </Badge>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>

                                                        {/* Documents Section */}
                                                        <div>
                                                            <div className="flex items-center justify-between mb-3">
                                                                <div className="text-sm font-medium text-slate-700">附件文件</div>
                                                                <label className="cursor-pointer flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700">
                                                                    <Upload className="h-4 w-4" />
                                                                    <span>上传</span>
                                                                    <input
                                                                        type="file"
                                                                        className="hidden"
                                                                        onChange={async (e) => {
                                                                            const file = e.target.files?.[0];
                                                                            if (!file) return;
                                                                            const formData = new FormData();
                                                                            formData.append('file', file);
                                                                            formData.append('docType', 'bidding');
                                                                            try {
                                                                                await api.upload(`/procurements/${proc.id}/documents`, formData);
                                                                                fetchProcurements();
                                                                                e.target.value = '';
                                                                            } catch (error) {
                                                                                console.error('Upload failed', error);
                                                                            }
                                                                        }}
                                                                    />
                                                                </label>
                                                            </div>
                                                            {proc.documents && proc.documents.length > 0 ? (
                                                                <div className="space-y-2">
                                                                    {proc.documents.map((doc: any) => (
                                                                        <div key={doc.id} className="flex items-center justify-between p-2 bg-slate-50 rounded">
                                                                            <div className="flex items-center gap-2">
                                                                                <FileIcon className="h-4 w-4 text-blue-500" />
                                                                                <span className="text-sm">{doc.filename}</span>
                                                                            </div>
                                                                            <span className="text-xs text-slate-400">
                                                                                {(doc.size / 1024).toFixed(1)} KB
                                                                            </span>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            ) : (
                                                                <div className="text-center py-4 text-slate-400 text-sm">暂无附件</div>
                                                            )}
                                                        </div>

                                                        {/* Notes Section */}
                                                        <div className="p-4 border-t">
                                                            <div className="text-sm font-medium text-slate-700 mb-3">备注信息</div>
                                                            {isEditing ? (
                                                                <RichTextEditor
                                                                    content={proc.notes || ""}
                                                                    onChange={async (newNotes) => {
                                                                        try {
                                                                            await api.patch(`/procurements/${proc.id}`, { notes: newNotes });
                                                                        } catch (error) {
                                                                            console.error("Failed to update notes", error);
                                                                        }
                                                                    }}
                                                                    placeholder="添加投标备注信息..."
                                                                />
                                                            ) : proc.notes ? (
                                                                <div className="prose prose-sm max-w-none text-slate-600" dangerouslySetInnerHTML={{ __html: proc.notes }} />
                                                            ) : (
                                                                <div className="text-slate-400 text-sm">暂无备注</div>
                                                            )}
                                                        </div>
                                                    </div>

                                                    {/* Result Registration */}
                                                    <div className="mt-8 pt-6 border-t border-slate-100">
                                                        <div className="flex items-center gap-2 mb-4">
                                                            <Target className="h-4 w-4 text-blue-600" />
                                                            <h4 className="text-sm font-medium text-slate-900">投标结果登记</h4>
                                                        </div>
                                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 p-4 rounded-lg">
                                                            <div className="space-y-4">
                                                                <div>
                                                                    <Label className="text-xs text-slate-500 mb-1.5 block">中标状态</Label>
                                                                    <Select
                                                                        value={proc.status}
                                                                        onValueChange={async (val) => {
                                                                            await api.patch(`/procurements/${proc.id}`, { status: val });
                                                                            fetchProcurements();
                                                                        }}
                                                                    >
                                                                        <SelectTrigger className="w-full bg-white h-9 text-sm">
                                                                            <SelectValue />
                                                                        </SelectTrigger>
                                                                        <SelectContent>
                                                                            <SelectItem value="Draft">Draft (草稿)</SelectItem>
                                                                            <SelectItem value="Preparing">Preparing (准备中)</SelectItem>
                                                                            <SelectItem value="Submitted">Submitted (已提交)</SelectItem>
                                                                            <SelectItem value="InProgress">InProgress (评审中)</SelectItem>
                                                                            <SelectItem value="Won">Won (中标)</SelectItem>
                                                                            <SelectItem value="Lost">Lost (未中标)</SelectItem>
                                                                        </SelectContent>
                                                                    </Select>
                                                                </div>
                                                                <div>
                                                                    <Label className="text-xs text-slate-500 mb-1.5 block">结果通知书/凭证</Label>
                                                                    <div className="space-y-2">
                                                                        <label className="cursor-pointer inline-flex items-center gap-2 px-3 py-2 bg-white border border-dashed rounded-md text-sm text-slate-500 hover:border-blue-500 hover:text-blue-500 transition-colors w-full justify-center">
                                                                            <Upload className="h-4 w-4" />
                                                                            <span>上传凭证文件</span>
                                                                            <input
                                                                                type="file"
                                                                                className="hidden"
                                                                                onChange={async (e) => {
                                                                                    const file = e.target.files?.[0];
                                                                                    if (!file) return;
                                                                                    const formData = new FormData();
                                                                                    formData.append('file', file);
                                                                                    formData.append('docType', 'result');
                                                                                    try {
                                                                                        await api.upload(`/procurements/${proc.id}/documents`, formData);
                                                                                        fetchProcurements();
                                                                                        e.target.value = '';
                                                                                    } catch (error) {
                                                                                        console.error('Upload failed', error);
                                                                                    }
                                                                                }}
                                                                            />
                                                                        </label>
                                                                        <div className="space-y-1">
                                                                            {proc.documents?.filter((d: any) => d.docType === 'result').map((d: any) => (
                                                                                <div key={d.id} className="flex items-center gap-2 text-xs bg-white px-2 py-1 rounded border">
                                                                                    <FileIcon className="h-3 w-3 text-blue-500" />
                                                                                    <span className="truncate">{d.filename}</span>
                                                                                </div>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="h-full">
                                                                    <Label className="text-xs text-slate-500 mb-1.5 block">结果说明 / 复盘总结</Label>
                                                                    <textarea
                                                                        className="w-full h-[calc(100%-24px)] min-h-[120px] p-3 text-sm bg-white border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                                                                        placeholder="请输入结果说明，如中标原因、流标分析等..."
                                                                        defaultValue={proc.resultNote || ''}
                                                                        onBlur={async (e) => {
                                                                            if (e.target.value !== proc.resultNote) {
                                                                                await api.patch(`/procurements/${proc.id}`, { resultNote: e.target.value });
                                                                            }
                                                                        }}
                                                                    />
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                ) : !showNewProcurement && (
                                    <div className="text-center py-12 text-slate-400">
                                        <Target className="h-12 w-12 mx-auto mb-3 text-slate-300" />
                                        <div>暂无投标记录</div>
                                        <div className="text-sm mt-1">点击"发起投标"开始响应客户标书</div>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>
            </div>
        </div >
    );
}
