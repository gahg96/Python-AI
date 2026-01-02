'use client';

import React, { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from 'next/link';
import { FileText, ArrowRight, Loader2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useI18n } from "@/lib/i18n/I18nContext";

interface Contract {
    id: string;
    contractNumber: string;
    status: string;
    totalContractValue: number;
    createdAt: string;
    opportunity: {
        title: string;
        customer: {
            companyName: string;
        };
    };
}

export default function ContractsListPage() {
    const { t } = useI18n();
    const [contracts, setContracts] = useState<Contract[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        fetchContracts();
    }, []);

    const fetchContracts = async () => {
        try {
            const data = await api.get('/contracts');
            setContracts(data);
        } catch (error) {
            console.error("Failed to fetch contracts", error);
        } finally {
            setLoading(false);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'Draft': return 'bg-slate-500';
            case 'PendingApproval': return 'bg-yellow-500';
            case 'Approved': return 'bg-green-500';
            case 'Rejected': return 'bg-red-500';
            case 'Signed': return 'bg-blue-600';
            default: return 'bg-slate-500';
        }
    };

    if (loading) return <div className="flex justify-center p-8"><Loader2 className="animate-spin" /></div>;

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-slate-800">{t("contract.title")}</h1>
            </div>

            <div className="grid gap-4">
                {contracts.length === 0 ? (
                    <div className="text-center p-8 bg-slate-50 rounded border text-slate-500">
                        {t("contract.noContracts")}
                    </div>
                ) : (
                    contracts.map((contract) => (
                        <Card key={contract.id} className="hover:shadow-md transition-shadow cursor-pointer" onClick={() => router.push(`/contracts/${contract.id}`)}>
                            <CardContent className="p-6 flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className="p-3 bg-blue-50 rounded-lg">
                                        <FileText className="h-6 w-6 text-blue-600" />
                                    </div>
                                    <div>
                                        <div className="font-semibold text-lg">{contract.contractNumber}</div>
                                        <div className="text-sm text-slate-500">
                                            {contract.opportunity.customer.companyName} - {contract.opportunity.title}
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-6">
                                    <div className="text-right">
                                        <div className="font-medium">
                                            {new Intl.NumberFormat('zh-CN', { style: 'currency', currency: 'CNY' }).format(contract.totalContractValue)}
                                        </div>
                                        <div className="text-xs text-slate-400">{t("contract.totalValue")}</div>
                                    </div>
                                    <Badge className={`${getStatusColor(contract.status)} hover:${getStatusColor(contract.status)} text-white`}>
                                        {contract.status}
                                    </Badge>
                                    <ArrowRight className="h-5 w-5 text-slate-400" />
                                </div>
                            </CardContent>
                        </Card>
                    ))
                )}
            </div>
        </div>
    );
}
