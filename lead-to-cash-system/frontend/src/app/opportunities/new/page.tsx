"use client";

import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft, Building2, Users, Briefcase, FileText, Upload, X, FileIcon, DollarSign } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { leadSchema, LeadFormValues } from "@/schemas/lead";
import { useI18n } from "@/lib/i18n/I18nContext";
import { api } from "@/lib/api";
import { RichTextEditor } from "@/components/RichTextEditor";

export default function NewOpportunityPage() {
    const { t } = useI18n();
    const router = useRouter();
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [richDescription, setRichDescription] = useState("");
    const [pendingFiles, setPendingFiles] = useState<File[]>([]);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Customer selection state
    const [customers, setCustomers] = useState<any[]>([]);
    const [selectedCustomerId, setSelectedCustomerId] = useState<string>("");
    const [isNewCustomer, setIsNewCustomer] = useState(false);

    // Fetch customers on mount
    useEffect(() => {
        api.get("/customers").then(setCustomers).catch(console.error);
    }, []);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            setPendingFiles(prev => [...prev, ...files]);
        }
    };

    const form = useForm({
        resolver: zodResolver(leadSchema),
        defaultValues: {
            companyName: "",
            industry: "",
            companySize: "",
            contactName: "",
            contactTitle: "",
            contactPhone: "",
            contactEmail: "",
            title: "",
            estimatedValue: 0,
            probability: 50,
            source: "",
            expectedCloseDate: new Date(new Date().setMonth(new Date().getMonth() + 3)).toISOString().split('T')[0],
            salesStage: "",
            competitors: "",
            decisionMakers: "",
            salesOwner: "",
            dealType: "",
            deliveryModel: "",
            estimatedEffort: 0,
            richDescription: "",
        },
    });

    const onSubmit = async (data: LeadFormValues) => {
        setIsSubmitting(true);
        try {
            let customerId: string;

            if (selectedCustomerId && !isNewCustomer) {
                // Use existing customer
                customerId = selectedCustomerId;
            } else {
                // Create new customer
                const customer = await api.post("/customers", {
                    companyName: data.companyName,
                    industry: data.industry,
                    companySize: data.companySize,
                    city: (data as any).city, // Cast as any because LeadFormValues update might not propagate instantly to TS check here, or valid
                    contactName: data.contactName,
                    contactTitle: data.contactTitle,
                    contactPhone: data.contactPhone,
                    contactEmail: data.contactEmail,
                });
                customerId = customer.id;
            }

            // Then create opportunity
            const opportunity = await api.post("/opportunities", {
                customerId: customerId,
                title: data.title,
                estimatedValue: data.estimatedValue,
                probability: data.probability,
                source: data.source,
                expectedCloseDate: data.expectedCloseDate || undefined,
                salesStage: data.salesStage,
                competitors: data.competitors,
                decisionMakers: data.decisionMakers,
                salesOwner: data.salesOwner,
                dealType: data.dealType,
                deliveryModel: data.deliveryModel,
                estimatedEffort: data.estimatedEffort,
                richDescription: richDescription,
                projectBudget: data.projectBudget,
                businessCost: data.businessCost,
                laborCost: data.laborCost,
                otherCost: data.otherCost,
                grossProfit: data.grossProfit,
                profitMargin: data.profitMargin,
            });

            // Upload pending files
            for (const file of pendingFiles) {
                const formData = new FormData();
                formData.append('file', file);
                await fetch(`http://localhost:3000/opportunities/${opportunity.id}/attachments`, {
                    method: 'POST',
                    body: formData,
                });
            }

            router.push("/opportunities");
        } catch (error) {
            console.error("Failed to create opportunity", error);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-8">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                    <Link href="/opportunities" className="flex items-center text-sm text-slate-600 hover:text-slate-900 mb-4">
                        <ArrowLeft className="h-4 w-4 mr-1" />
                        {t("nav.back")}
                    </Link>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900">{t("form.title")}</h1>
                    <p className="text-slate-500 mt-1">{t("form.desc")}</p>
                </div>

                <form onSubmit={form.handleSubmit(onSubmit)}>
                    <Tabs defaultValue="customer" className="w-full">
                        <TabsList className="grid w-full grid-cols-5 mb-6">
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
                        </TabsList>

                        {/* Customer Information Tab */}
                        <TabsContent value="customer">
                            <Card>
                                <CardHeader>
                                    <CardTitle>{t("form.sections.customer")}</CardTitle>
                                    <CardDescription>{t("form.sectionDesc.customer")}</CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-6">
                                    {/* Customer Selection Mode */}
                                    <div className="flex gap-4 mb-4">
                                        <Button
                                            type="button"
                                            variant={!isNewCustomer ? "default" : "outline"}
                                            onClick={() => setIsNewCustomer(false)}
                                        >
                                            选择现有客户
                                        </Button>
                                        <Button
                                            type="button"
                                            variant={isNewCustomer ? "default" : "outline"}
                                            onClick={() => {
                                                setIsNewCustomer(true);
                                                setSelectedCustomerId("");
                                            }}
                                        >
                                            新建客户
                                        </Button>
                                    </div>

                                    {!isNewCustomer ? (
                                        /* Select Existing Customer */
                                        <div className="space-y-4">
                                            <div>
                                                <Label>{t("form.clientName")} *</Label>
                                                <Select
                                                    value={selectedCustomerId}
                                                    onValueChange={(v) => {
                                                        setSelectedCustomerId(v);
                                                        const customer = customers.find(c => c.id === v);
                                                        if (customer) {
                                                            form.setValue("companyName", customer.companyName);
                                                            form.setValue("industry", customer.industry || "");
                                                            form.setValue("companySize", customer.companySize || "");
                                                            form.setValue("contactName", customer.contactName || "");
                                                            form.setValue("contactTitle", customer.contactTitle || "");
                                                            form.setValue("contactPhone", customer.contactPhone || "");
                                                            form.setValue("contactEmail", customer.contactEmail || "");
                                                        }
                                                    }}
                                                >
                                                    <SelectTrigger className="w-full">
                                                        <SelectValue placeholder="选择客户..." />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        {customers.map((customer) => (
                                                            <SelectItem key={customer.id} value={customer.id}>
                                                                <div className="flex items-center gap-2">
                                                                    <Building2 className="h-4 w-4 text-slate-400" />
                                                                    <span className="font-medium">{customer.companyName}</span>
                                                                    {customer.industry && (
                                                                        <span className="text-xs text-slate-500">({customer.industry})</span>
                                                                    )}
                                                                </div>
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                            </div>

                                            {/* Show selected customer details */}
                                            {selectedCustomerId && (() => {
                                                const customer = customers.find(c => c.id === selectedCustomerId);
                                                if (!customer) return null;
                                                return (
                                                    <div className="p-4 bg-slate-50 rounded-lg border space-y-2">
                                                        <div className="flex items-center gap-2 text-lg font-semibold">
                                                            <Building2 className="h-5 w-5 text-blue-500" />
                                                            {customer.companyName}
                                                        </div>
                                                        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                                                            {customer.industry && (
                                                                <div><span className="text-slate-500">行业:</span> {customer.industry}</div>
                                                            )}
                                                            {customer.companySize && (
                                                                <div><span className="text-slate-500">规模:</span> {customer.companySize}</div>
                                                            )}
                                                            {customer.contactName && (
                                                                <div><span className="text-slate-500">联系人:</span> {customer.contactName}</div>
                                                            )}
                                                            {customer.contactTitle && (
                                                                <div><span className="text-slate-500">职位:</span> {customer.contactTitle}</div>
                                                            )}
                                                            {customer.contactPhone && (
                                                                <div><span className="text-slate-500">电话:</span> {customer.contactPhone}</div>
                                                            )}
                                                            {customer.contactEmail && (
                                                                <div><span className="text-slate-500">邮箱:</span> {customer.contactEmail}</div>
                                                            )}
                                                        </div>
                                                    </div>
                                                );
                                            })()}
                                        </div>
                                    ) : (
                                        /* New Customer Form */
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="col-span-2">
                                                <Label>{t("form.clientName")} *</Label>
                                                <Input {...form.register("companyName")} placeholder={t("form.placeholder.companyName")} />
                                                {form.formState.errors.companyName && (
                                                    <p className="text-sm text-red-500 mt-1">{form.formState.errors.companyName.message}</p>
                                                )}
                                            </div>
                                            <div>
                                                <Label>{t("form.industry")}</Label>
                                                <Select onValueChange={(v) => form.setValue("industry", v)}>
                                                    <SelectTrigger><SelectValue placeholder={t("form.placeholder.selectIndustry")} /></SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="银行">银行</SelectItem>
                                                        <SelectItem value="证券">证券</SelectItem>
                                                        <SelectItem value="保险">保险</SelectItem>
                                                        <SelectItem value="基金">基金</SelectItem>
                                                        <SelectItem value="期货交易所">期货交易所</SelectItem>
                                                        <SelectItem value="证券交易所">证券交易所</SelectItem>
                                                        <SelectItem value="电信">电信</SelectItem>
                                                        <SelectItem value="制造业">制造业</SelectItem>
                                                        <SelectItem value="汽车">汽车</SelectItem>
                                                        <SelectItem value="传媒">传媒</SelectItem>
                                                        <SelectItem value="教育">教育</SelectItem>
                                                        <SelectItem value="交通运输">交通运输</SelectItem>
                                                        <SelectItem value="其他">其他</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div>
                                                <Label>{t("form.companySize")}</Label>
                                                <Select onValueChange={(v) => form.setValue("companySize", v)}>
                                                    <SelectTrigger><SelectValue placeholder={t("form.placeholder.selectSize")} /></SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="中型">中型</SelectItem>
                                                        <SelectItem value="大型">大型</SelectItem>
                                                        <SelectItem value="超大型">超大型</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div>
                                                <Label>城市</Label>
                                                <Input {...form.register("city")} placeholder="例如: 北京" />
                                            </div>
                                            <div>
                                                <Label>{t("form.contactName")}</Label>
                                                <Input {...form.register("contactName")} placeholder={t("form.placeholder.contactName")} />
                                            </div>
                                            <div>
                                                <Label>{t("form.contactTitle")}</Label>
                                                <Input {...form.register("contactTitle")} placeholder={t("form.placeholder.contactTitle")} />
                                            </div>
                                            <div>
                                                <Label>{t("form.contactPhone")}</Label>
                                                <Input {...form.register("contactPhone")} placeholder="+86 138 0000 0000" />
                                            </div>
                                            <div>
                                                <Label>{t("form.contactEmail")}</Label>
                                                <Input {...form.register("contactEmail")} type="email" placeholder="contact@example.com" />
                                            </div>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        </TabsContent>

                        {/* Opportunity Details Tab */}
                        <TabsContent value="opportunity">
                            <Card>
                                <CardHeader>
                                    <CardTitle>{t("form.sections.opportunity")}</CardTitle>
                                    <CardDescription>Project and financial information</CardDescription>
                                </CardHeader>
                                <CardContent className="grid grid-cols-2 gap-4">
                                    <div className="col-span-2">
                                        <Label>{t("form.projectName")} *</Label>
                                        <Input {...form.register("title")} placeholder="e.g. Digital Transformation Project" />
                                        {form.formState.errors.title && (
                                            <p className="text-sm text-red-500 mt-1">{form.formState.errors.title.message}</p>
                                        )}
                                    </div>
                                    <div>
                                        <Label>{t("form.estValue")}</Label>
                                        <Input {...form.register("estimatedValue")} type="number" placeholder="100000" />
                                    </div>
                                    <div>
                                        <Label>{t("form.probability")}</Label>
                                        <Input {...form.register("probability")} type="number" min="0" max="100" placeholder="50" />
                                    </div>
                                    <div>
                                        <Label>{t("form.source")}</Label>
                                        <Select onValueChange={(v) => form.setValue("source", v)}>
                                            <SelectTrigger><SelectValue placeholder="Select source" /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="referral">客户推荐 / Referral</SelectItem>
                                                <SelectItem value="website">官网 / Website</SelectItem>
                                                <SelectItem value="cold_call">主动拓展 / Cold Call</SelectItem>
                                                <SelectItem value="event">展会/活动 / Event</SelectItem>
                                                <SelectItem value="partner">合作伙伴 / Partner</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div>
                                        <Label>{t("form.closeDate")}</Label>
                                        <Input {...form.register("expectedCloseDate")} type="date" />
                                    </div>
                                </CardContent>
                            </Card>
                        </TabsContent>

                        {/* Enterprise Details Tab */}
                        <TabsContent value="enterprise">
                            <Card>
                                <CardHeader>
                                    <CardTitle>{t("form.sections.enterprise")}</CardTitle>
                                    <CardDescription>{t("form.sectionDesc.enterprise")}</CardDescription>
                                </CardHeader>
                                <CardContent className="grid grid-cols-2 gap-4">
                                    <div>
                                        <Label>{t("form.salesStage")}</Label>
                                        <Select onValueChange={(v) => form.setValue("salesStage", v)}>
                                            <SelectTrigger><SelectValue placeholder={t("form.placeholder.selectStage")} /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="initial_contact">{t("options.salesStage.initial_contact")}</SelectItem>
                                                <SelectItem value="requirement">{t("options.salesStage.requirement")}</SelectItem>
                                                <SelectItem value="proposal">{t("options.salesStage.proposal")}</SelectItem>
                                                <SelectItem value="negotiation">{t("options.salesStage.negotiation")}</SelectItem>
                                                <SelectItem value="contract">{t("options.salesStage.contract")}</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div>
                                        <Label>{t("form.dealType")}</Label>
                                        <Select onValueChange={(v) => form.setValue("dealType", v)}>
                                            <SelectTrigger><SelectValue placeholder={t("form.placeholder.selectType")} /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="new">{t("options.dealType.new")}</SelectItem>
                                                <SelectItem value="renewal">{t("options.dealType.renewal")}</SelectItem>
                                                <SelectItem value="upsell">{t("options.dealType.upsell")}</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div>
                                        <Label>{t("form.salesOwner")}</Label>
                                        <Input {...form.register("salesOwner")} placeholder={t("form.placeholder.salesOwner")} />
                                    </div>
                                    <div>
                                        <Label>{t("form.deliveryModel")}</Label>
                                        <Select onValueChange={(v) => form.setValue("deliveryModel", v)}>
                                            <SelectTrigger><SelectValue placeholder={t("form.placeholder.selectModel")} /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="onsite">{t("options.deliveryModel.onsite")}</SelectItem>
                                                <SelectItem value="remote">{t("options.deliveryModel.remote")}</SelectItem>
                                                <SelectItem value="hybrid">{t("options.deliveryModel.hybrid")}</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div>
                                        <Label>{t("form.estimatedEffort")}</Label>
                                        <Input {...form.register("estimatedEffort")} type="number" placeholder="30" />
                                    </div>
                                    <div>
                                        <Label>{t("form.competitors")}</Label>
                                        <Input {...form.register("competitors")} placeholder={t("form.placeholder.competitors")} />
                                    </div>
                                    <div className="col-span-2">
                                        <Label>{t("form.decisionMakers")}</Label>
                                        <Input {...form.register("decisionMakers")} placeholder={t("form.placeholder.decisionMakers")} />
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
                                            <Input
                                                type="number"
                                                {...form.register("projectBudget")}
                                                placeholder="0"
                                                onChange={(e) => {
                                                    form.setValue("projectBudget", Number(e.target.value));
                                                    // Trigger recalculation
                                                    const budget = Number(e.target.value) || 0;
                                                    const businessCost = Number(form.getValues("businessCost")) || 0;
                                                    const laborCost = Number(form.getValues("laborCost")) || 0;
                                                    const otherCost = Number(form.getValues("otherCost")) || 0;
                                                    const totalCost = businessCost + laborCost + otherCost;
                                                    const grossProfit = budget - totalCost;
                                                    const profitMargin = budget > 0 ? Math.round((grossProfit / budget) * 100) : 0;
                                                    form.setValue("grossProfit", grossProfit);
                                                    form.setValue("profitMargin", profitMargin);
                                                }}
                                            />
                                        </div>
                                        <div>
                                            <Label>{t("detail.businessCost")}</Label>
                                            <Input
                                                type="number"
                                                {...form.register("businessCost")}
                                                placeholder="0"
                                                onChange={(e) => {
                                                    form.setValue("businessCost", Number(e.target.value));
                                                    const budget = Number(form.getValues("projectBudget")) || 0;
                                                    const businessCost = Number(e.target.value) || 0;
                                                    const laborCost = Number(form.getValues("laborCost")) || 0;
                                                    const otherCost = Number(form.getValues("otherCost")) || 0;
                                                    const totalCost = businessCost + laborCost + otherCost;
                                                    const grossProfit = budget - totalCost;
                                                    const profitMargin = budget > 0 ? Math.round((grossProfit / budget) * 100) : 0;
                                                    form.setValue("grossProfit", grossProfit);
                                                    form.setValue("profitMargin", profitMargin);
                                                }}
                                            />
                                        </div>
                                        <div>
                                            <Label>{t("detail.laborCost")}</Label>
                                            <Input
                                                type="number"
                                                {...form.register("laborCost")}
                                                placeholder="0"
                                                onChange={(e) => {
                                                    form.setValue("laborCost", Number(e.target.value));
                                                    const budget = Number(form.getValues("projectBudget")) || 0;
                                                    const businessCost = Number(form.getValues("businessCost")) || 0;
                                                    const laborCost = Number(e.target.value) || 0;
                                                    const otherCost = Number(form.getValues("otherCost")) || 0;
                                                    const totalCost = businessCost + laborCost + otherCost;
                                                    const grossProfit = budget - totalCost;
                                                    const profitMargin = budget > 0 ? Math.round((grossProfit / budget) * 100) : 0;
                                                    form.setValue("grossProfit", grossProfit);
                                                    form.setValue("profitMargin", profitMargin);
                                                }}
                                            />
                                        </div>
                                        <div>
                                            <Label>{t("detail.otherCost")}</Label>
                                            <Input
                                                type="number"
                                                {...form.register("otherCost")}
                                                placeholder="0"
                                                onChange={(e) => {
                                                    form.setValue("otherCost", Number(e.target.value));
                                                    const budget = Number(form.getValues("projectBudget")) || 0;
                                                    const businessCost = Number(form.getValues("businessCost")) || 0;
                                                    const laborCost = Number(form.getValues("laborCost")) || 0;
                                                    const otherCost = Number(e.target.value) || 0;
                                                    const totalCost = businessCost + laborCost + otherCost;
                                                    const grossProfit = budget - totalCost;
                                                    const profitMargin = budget > 0 ? Math.round((grossProfit / budget) * 100) : 0;
                                                    form.setValue("grossProfit", grossProfit);
                                                    form.setValue("profitMargin", profitMargin);
                                                }}
                                            />
                                        </div>
                                    </div>

                                    {/* Auto-calculated fields */}
                                    <div className="border-t pt-4">
                                        <div className="text-sm text-slate-500 mb-3">自动计算结果</div>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                                                <Label className="text-green-700">{t("detail.grossProfit")}</Label>
                                                <div className="text-2xl font-bold text-green-600 mt-1">
                                                    ¥{(Number(form.watch("grossProfit")) || 0).toLocaleString()}
                                                </div>
                                                <input type="hidden" {...form.register("grossProfit")} />
                                            </div>
                                            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                                                <Label className="text-blue-700">{t("detail.profitMargin")}</Label>
                                                <div className="text-2xl font-bold text-blue-600 mt-1">
                                                    {(Number(form.watch("profitMargin")) || 0)}%
                                                </div>
                                                <input type="hidden" {...form.register("profitMargin")} />
                                            </div>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </TabsContent>

                        {/* Description Tab with Rich Text Editor */}
                        <TabsContent value="description">
                            <Card>
                                <CardHeader>
                                    <CardTitle>{t("form.sections.description")}</CardTitle>
                                    <CardDescription>Add detailed notes, images, and attachments. You can paste images directly.</CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-6">
                                    <RichTextEditor
                                        content={richDescription}
                                        onChange={setRichDescription}
                                        placeholder="Enter detailed project description... You can paste images directly from clipboard."
                                    />

                                    {/* File Upload Section */}
                                    <div className="border-t pt-4">
                                        <div className="flex items-center justify-between mb-3">
                                            <Label className="text-base font-medium">附件 / Attachments</Label>
                                            <Button
                                                type="button"
                                                variant="outline"
                                                size="sm"
                                                onClick={() => fileInputRef.current?.click()}
                                            >
                                                <Upload className="h-4 w-4 mr-2" />
                                                上传文件
                                            </Button>
                                            <input
                                                type="file"
                                                ref={fileInputRef}
                                                className="hidden"
                                                multiple
                                                onChange={(e) => {
                                                    const files = Array.from(e.target.files || []);
                                                    setPendingFiles(prev => [...prev, ...files]);
                                                    e.target.value = ''; // Reset input
                                                }}
                                            />
                                        </div>

                                        {pendingFiles.length > 0 ? (
                                            <div className="space-y-2">
                                                {pendingFiles.map((file, index) => (
                                                    <div key={index} className="flex items-center gap-3 p-3 rounded-lg border bg-slate-50">
                                                        <FileIcon className="h-5 w-5 text-blue-500" />
                                                        <div className="flex-1">
                                                            <div className="font-medium text-sm">{file.name}</div>
                                                            <div className="text-xs text-slate-500">
                                                                {(file.size / 1024).toFixed(1)} KB
                                                            </div>
                                                        </div>
                                                        <Button
                                                            type="button"
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => {
                                                                setPendingFiles(prev => prev.filter((_, i) => i !== index));
                                                            }}
                                                        >
                                                            <X className="h-4 w-4" />
                                                        </Button>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div
                                                className={`text-center py-8 border-2 border-dashed rounded-lg transition-colors cursor-pointer ${isDragging ? 'border-blue-500 bg-blue-50 text-blue-600' : 'text-slate-500 hover:border-slate-400'}`}
                                                onDragOver={handleDragOver}
                                                onDragLeave={handleDragLeave}
                                                onDrop={handleDrop}
                                                onClick={() => fileInputRef.current?.click()}
                                            >
                                                <Upload className="h-8 w-8 mx-auto mb-2 opacity-50" />
                                                <p>{isDragging ? '松开鼠标上传文件' : '拖拽文件到此处，或点击选择'}</p>
                                                <p className="text-xs mt-1">支持多文件上传</p>
                                            </div>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        </TabsContent>
                    </Tabs>

                    {/* Action Buttons */}
                    <div className="flex justify-end gap-4 mt-6">
                        <Button type="button" variant="outline" onClick={() => router.push("/opportunities")}>
                            {t("form.cancel")}
                        </Button>
                        <Button type="submit" disabled={isSubmitting}>
                            {isSubmitting ? t("form.creating") : t("form.create")}
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    );
}
