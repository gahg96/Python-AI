'use client';

import React, { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useRouter, useParams } from 'next/navigation';
import { Loader2, ArrowLeft, CheckCircle, XCircle, FileText, AlertTriangle } from 'lucide-react';
import Link from 'next/link';
import { useI18n } from "@/lib/i18n/I18nContext";

// Types
interface Contract {
    id: string;
    contractNumber: string;
    status: string;
    totalContractValue: number;
    paymentTerms?: string;
    riskAssessment?: string;
    scope?: string;
    sla?: string;
    liability?: string;
    paymentTermsDetails?: string;
    startDate?: string;
    endDate?: string;
    createdAt: string;
    opportunity: {
        id: string;
        title: string;
        customer: { companyName: string };
    };
    drafter?: { username: string };
    approver?: { username: string };
    milestones: any[];
    documents: any[];
}

export default function ContractDetailPage() {
    const { t } = useI18n();
    const params = useParams();
    const router = useRouter();
    const [contract, setContract] = useState<Contract | null>(null);
    const [loading, setLoading] = useState(true);
    const [riskText, setRiskText] = useState("");
    const [isSaving, setIsSaving] = useState(false);

    // Edit State
    const [isEditing, setIsEditing] = useState(false);
    const [editData, setEditData] = useState<Partial<Contract>>({});

    // Milestone State
    const [newMilestone, setNewMilestone] = useState({ name: '', amount: '', dueDate: '' });
    const [showMilestoneInput, setShowMilestoneInput] = useState(false);

    const id = params.id as string;

    useEffect(() => {
        if (id) fetchContract();
    }, [id]);

    const fetchContract = async () => {
        try {
            const data = await api.get(`/contracts/${id}`);
            setContract(data);
            setEditData(data);
            setRiskText(data.riskAssessment || "");
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleSaveRisk = async () => {
        setIsSaving(true);
        try {
            await api.patch(`/contracts/${id}`, { riskAssessment: riskText });
            fetchContract();
        } catch (e) {
            console.error(e);
        } finally {
            setIsSaving(false);
        }
    };

    const handleSaveDetails = async () => {
        setIsSaving(true);
        try {
            const updatePayload = {
                scope: editData.scope,
                sla: editData.sla,
                liability: editData.liability,
                paymentTermsDetails: editData.paymentTermsDetails
            };
            await api.patch(`/contracts/${id}`, updatePayload);
            setIsEditing(false);
            fetchContract();
        } catch (e) { console.error(e); }
        finally { setIsSaving(false); }
    };

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('docType', 'contract_attachment');

        try {
            await api.upload(`/contracts/${id}/documents`, formData);
            fetchContract();
        } catch (error) { console.error(error); }
    };

    const handleAddMilestone = async () => {
        if (!newMilestone.name || !newMilestone.amount) return;
        try {
            await api.post(`/contracts/${id}/milestones`, newMilestone);
            setNewMilestone({ name: '', amount: '', dueDate: '' });
            setShowMilestoneInput(false);
            fetchContract();
        } catch (e) { console.error(e); }
    };

    const handleDeleteMilestone = async (mid: string) => {
        if (!confirm('Are you sure?')) return;
        try {
            await api.delete(`/contracts/milestones/${mid}`);
            fetchContract();
        } catch (e) { console.error(e); }
    };

    // Workflow Actions
    const handleSubmit = async () => {
        if (!confirm(t("contract.confirmSubmit"))) return;
        try {
            await api.post(`/contracts/${id}/submit`, {});
            fetchContract();
        } catch (e) { console.error(e); }
    };

    const handleApprove = async () => {
        if (!confirm(t("contract.confirmApprove"))) return;
        try {
            await api.post(`/contracts/${id}/approve`, {});
            fetchContract();
        } catch (e) { console.error(e); }
    };

    const handleReject = async () => {
        if (!confirm(t("contract.confirmReject"))) return;
        try {
            await api.post(`/contracts/${id}/reject`, {});
            fetchContract();
        } catch (e) { console.error(e); }
    };



    const handleSign = async () => {
        if (!confirm(t("contract.confirmSign"))) return;
        try {
            await api.post(`/contracts/${id}/sign`, {});
            fetchContract();
        } catch (e) { console.error(e); }
    };

    const handleInitializeProject = async () => {
        try {
            const newProj = await api.post('/projects', {
                contractId: id,
                status: 'Initialization',
                description: `Project execution for ${contract?.contractNumber}`
            });
            router.push(`/delivery/${newProj.id}`);
        } catch (e) { console.error(e); }
    };

    if (loading || !contract) return <div className="flex justify-center p-8"><Loader2 className="animate-spin" /></div>;

    const isDraft = contract.status === 'Draft';
    const isPending = contract.status === 'PendingApproval';
    const isApproved = contract.status === 'Approved';
    const isSigned = contract.status === 'Signed';

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="mb-6">
                <Link href="/contracts" className="flex items-center text-sm text-slate-600 hover:text-slate-900 mb-4">
                    <ArrowLeft className="h-4 w-4 mr-1" />
                    {t("contract.backToContracts")}
                </Link>
                <div className="flex justify-between items-start">
                    <div>
                        <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
                            {contract.contractNumber}
                            <Badge variant="outline">{contract.status}</Badge>
                        </h1>
                        <p className="text-slate-500 mt-1">
                            {contract.opportunity.customer.companyName} - {contract.opportunity.title}
                        </p>
                    </div>
                    <div className="flex gap-2">
                        {isDraft && (
                            <Button onClick={handleSubmit} className="bg-blue-600">{t("contract.submit")}</Button>
                        )}
                        {isPending && (
                            <>
                                <Button onClick={handleReject} variant="destructive">{t("contract.reject")}</Button>
                                <Button onClick={handleApprove} className="bg-green-600">{t("contract.approve")}</Button>
                            </>
                        )}
                        {isApproved && (
                            <Button onClick={handleSign} className="bg-purple-600">{t("contract.markSigned")}</Button>
                        )}
                        {isSigned && !(contract as any).project && (
                            <Button onClick={handleInitializeProject} className="bg-orange-600 hover:bg-orange-700">{t("project.actions.initialize")}</Button>
                        )}
                        {isSigned && (contract as any).project && (
                            <Link href={`/delivery/${(contract as any).project.id}`}>
                                <Button variant="outline" className="border-orange-600 text-orange-600 hover:bg-orange-50">View Project</Button>
                            </Link>
                        )}
                    </div>
                </div>
            </div>

            <Tabs defaultValue="overview">
                <TabsList>
                    <TabsTrigger value="overview">{t("contract.tabs.overview")}</TabsTrigger>
                    <TabsTrigger value="risk">{t("contract.tabs.risk")}</TabsTrigger>
                    <TabsTrigger value="milestones">{t("contract.tabs.milestones")}</TabsTrigger>
                    <TabsTrigger value="documents">Documents</TabsTrigger>
                </TabsList>

                <TabsContent value="overview">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between">
                            <CardTitle>{t("contract.details")}</CardTitle>
                            {(isDraft || isApproved) && (
                                <Button variant="outline" size="sm" onClick={() => setIsEditing(!isEditing)}>
                                    {isEditing ? t("contract.fields.cancel") : t("contract.fields.editDetails")}
                                </Button>
                            )}
                        </CardHeader>
                        <CardContent className="space-y-4">
                            {isEditing ? (
                                <div className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <Label>{t("contract.scope")}</Label>
                                            <Textarea value={editData.scope || ''} onChange={e => setEditData({ ...editData, scope: e.target.value })} />
                                        </div>
                                        <div>
                                            <Label>{t("contract.sla")}</Label>
                                            <Textarea value={editData.sla || ''} onChange={e => setEditData({ ...editData, sla: e.target.value })} />
                                        </div>
                                        <div>
                                            <Label>{t("contract.liability")}</Label>
                                            <Textarea value={editData.liability || ''} onChange={e => setEditData({ ...editData, liability: e.target.value })} />
                                        </div>
                                        <div>
                                            <Label>{t("contract.paymentTerms")}</Label>
                                            <Textarea value={editData.paymentTermsDetails || ''} onChange={e => setEditData({ ...editData, paymentTermsDetails: e.target.value })} />
                                        </div>
                                    </div>
                                    <Button onClick={handleSaveDetails}>{t("common.save")}</Button>
                                </div>
                            ) : (
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <Label>{t("contract.totalValue")}</Label>
                                        <div className="text-xl font-bold">¥{Number(contract.totalContractValue).toLocaleString()}</div>
                                    </div>
                                    <div>
                                        <Label>{t("contract.createdBy")}</Label>
                                        <div>{contract.drafter?.username || '-'}</div>
                                    </div>
                                    <div className="col-span-2">
                                        <Label>{t("contract.scope")}</Label>
                                        <div className="p-2 bg-slate-50 rounded text-sm whitespace-pre-wrap">{contract.scope || t("common.notDefined")}</div>
                                    </div>
                                    <div className="col-span-2">
                                        <Label>{t("contract.paymentTerms")}</Label>
                                        <div className="p-2 bg-slate-50 rounded text-sm whitespace-pre-wrap">{contract.paymentTermsDetails || contract.paymentTerms || t("common.notDefined")}</div>
                                    </div>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="risk">
                    <Card>
                        <CardHeader>
                            <CardTitle>{t("contract.risk.title")}</CardTitle>
                            <CardDescription>{t("contract.risk.subtitle")}</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <Textarea
                                value={riskText}
                                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setRiskText(e.target.value)}
                                placeholder={t("contract.risk.placeholder")}
                                className="min-h-[150px]"
                                disabled={isSigned}
                            />
                            {(isDraft || isPending || isApproved) && (
                                <Button onClick={handleSaveRisk} disabled={isSaving} variant="secondary">
                                    {isSaving ? t("contract.risk.saving") : t("contract.risk.save")}
                                </Button>
                            )}

                            {!isDraft && contract.approver && (
                                <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded text-green-800 flex items-center gap-2">
                                    <CheckCircle className="h-5 w-5" />
                                    <div>
                                        <div className="font-semibold">{t("contract.risk.processedBy")} {contract.approver.username}</div>
                                        <div className="text-sm">{t("table.status")}: {contract.status}</div>
                                    </div>
                                </div>
                            )}
                            {contract.status === 'Rejected' && (
                                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded text-red-800 flex items-center gap-2">
                                    <XCircle className="h-5 w-5" />
                                    <div>{t("contract.risk.rejectedBy")} {contract.approver?.username}</div>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="milestones">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between">
                            <div>
                                <CardTitle>{t("contract.milestones.title")}</CardTitle>
                                <CardDescription>{t("contract.milestones.subtitle")}</CardDescription>
                            </div>
                            {isDraft && (
                                <Button size="sm" onClick={() => setShowMilestoneInput(!showMilestoneInput)}>
                                    {showMilestoneInput ? t("contract.milestones.cancel") : t("contract.milestones.add")}
                                </Button>
                            )}
                        </CardHeader>
                        <CardContent>
                            {showMilestoneInput && (
                                <div className="mb-4 p-4 border rounded bg-slate-50 space-y-2">
                                    <Label>Name</Label>
                                    <Input value={newMilestone.name} onChange={e => setNewMilestone({ ...newMilestone, name: e.target.value })} placeholder="e.g. Initial Payment" />
                                    <div className="grid grid-cols-2 gap-2">
                                        <div><Label>Amount</Label><Input type="number" value={newMilestone.amount} onChange={e => setNewMilestone({ ...newMilestone, amount: e.target.value })} /></div>
                                        <div><Label>Due Date</Label><Input type="date" value={newMilestone.dueDate} onChange={e => setNewMilestone({ ...newMilestone, dueDate: e.target.value })} /></div>
                                    </div>
                                    <Button onClick={handleAddMilestone} size="sm">Save</Button>
                                </div>
                            )}

                            <div className="space-y-4">
                                {contract.milestones && contract.milestones.length > 0 ? (
                                    contract.milestones.map((ms: any) => (
                                        <div key={ms.id} className="flex items-center justify-between p-4 border rounded-lg">
                                            <div>
                                                <div className="font-semibold">{ms.name}</div>
                                                <div className="text-sm text-slate-500">{t("contract.milestones.dueDate")}: {ms.dueDate ? new Date(ms.dueDate).toLocaleDateString() : 'TBD'}</div>
                                            </div>
                                            <div className="text-right flex items-center gap-4">
                                                <div>
                                                    <div className="font-mono">¥{Number(ms.amount).toLocaleString()}</div>
                                                    <Badge variant="outline">{ms.status}</Badge>
                                                </div>
                                                {isDraft && <Button variant="ghost" size="sm" onClick={() => handleDeleteMilestone(ms.id)} className="text-red-500"><XCircle className="h-4 w-4" /></Button>}
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-center text-slate-500 py-8 border-dashed border-2 rounded">
                                        {t("contract.milestones.empty")}
                                    </div>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="documents">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between">
                            <CardTitle>{t("contract.documents.title")}</CardTitle>
                            <div className="relative">
                                <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={handleUpload} />
                                <Button variant="outline">{t("contract.documents.upload")}</Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-2">
                                {contract.documents && contract.documents.length > 0 ? (
                                    contract.documents.map((doc: any) => (
                                        <div key={doc.id} className="flex items-center justify-between p-3 border rounded hover:bg-slate-50">
                                            <div className="flex items-center gap-3">
                                                <FileText className="h-5 w-5 text-blue-500" />
                                                <div>
                                                    <div className="font-medium">{doc.filename}</div>
                                                    <div className="text-xs text-slate-500">
                                                        {t("contract.documents.uploadedBy")} {doc.uploadedBy?.username} on {new Date(doc.createdAt).toLocaleDateString()}
                                                    </div>
                                                </div>
                                            </div>
                                            <Button variant="ghost" size="sm" onClick={() => window.open(`http://localhost:3000/${doc.filepath}`, '_blank')}>{t("contract.documents.download")}</Button>
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-center text-slate-500 py-8">{t("contract.documents.empty")}</div>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}
