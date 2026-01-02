'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { DollarSign, FileText, TrendingUp, AlertCircle } from 'lucide-react';
import { api } from '@/lib/api';
import Link from 'next/link';

export default function FinancePage() {
    const [dashboardData, setDashboardData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchDashboardData();
    }, []);

    const fetchDashboardData = async () => {
        try {
            const data = await api.get('/finance/dashboard');
            setDashboardData(data);
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-lg">加载中...</div>
            </div>
        );
    }

    return (
        <div className="container mx-auto p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold">财务管理</h1>
                    <p className="text-muted-foreground mt-1">发票、收款与财务概览</p>
                </div>
                <Link href="/finance/invoices/new">
                    <Button>
                        <FileText className="mr-2 h-4 w-4" />
                        创建发票
                    </Button>
                </Link>
            </div>

            {/* KPI Cards */}
            <div className="grid gap-4 md:grid-cols-3">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">待开票金额</CardTitle>
                        <AlertCircle className="h-4 w-4 text-amber-600" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-amber-600">
                            ¥{dashboardData?.pendingInvoiceAmount?.toLocaleString() || 0}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                            {dashboardData?.readyToInvoiceMilestones?.length || 0} 个里程碑待开票
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">应收账款</CardTitle>
                        <TrendingUp className="h-4 w-4 text-orange-600" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-orange-600">
                            ¥{dashboardData?.outstandingAmount?.toLocaleString() || 0}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                            已开票未收款
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">已收款金额</CardTitle>
                        <DollarSign className="h-4 w-4 text-green-600" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-green-600">
                            ¥{dashboardData?.paidAmount?.toLocaleString() || 0}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                            本期已收款
                        </p>
                    </CardContent>
                </Card>
            </div>

            {/* Ready to Invoice Milestones */}
            {dashboardData?.readyToInvoiceMilestones && dashboardData.readyToInvoiceMilestones.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>待开票里程碑</CardTitle>
                        <CardDescription>以下里程碑已验收，可以开具发票</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-3">
                            {dashboardData.readyToInvoiceMilestones.map((milestone: any) => (
                                <div
                                    key={milestone.id}
                                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent transition-colors"
                                >
                                    <div className="flex-1">
                                        <div className="font-medium">{milestone.name}</div>
                                        <div className="text-sm text-muted-foreground">
                                            {milestone.contract?.opportunity?.customer?.companyName || '未知客户'}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="text-right">
                                            <div className="font-bold text-lg">¥{Number(milestone.amount).toLocaleString()}</div>
                                            <div className="text-xs text-muted-foreground">{milestone.status}</div>
                                        </div>
                                        <Link href={`/finance/invoices/new?milestoneId=${milestone.id}`}>
                                            <Button size="sm">开票</Button>
                                        </Link>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Recent Invoices */}
            <Card>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle>最近发票</CardTitle>
                            <CardDescription>最近创建的发票记录</CardDescription>
                        </div>
                        <Link href="/finance/invoices">
                            <Button variant="outline" size="sm">查看全部</Button>
                        </Link>
                    </div>
                </CardHeader>
                <CardContent>
                    {dashboardData?.recentInvoices && dashboardData.recentInvoices.length > 0 ? (
                        <div className="space-y-3">
                            {dashboardData.recentInvoices.slice(0, 5).map((invoice: any) => (
                                <div
                                    key={invoice.id}
                                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent transition-colors"
                                >
                                    <div className="flex-1">
                                        <div className="font-medium">{invoice.invoiceNumber}</div>
                                        <div className="text-sm text-muted-foreground">
                                            {invoice.contract?.opportunity?.customer?.companyName || '未知客户'}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="text-right">
                                            <div className="font-bold">¥{invoice.totalAmount?.toLocaleString()}</div>
                                            <div className={`text-xs ${invoice.status === 'Paid' ? 'text-green-600' :
                                                invoice.status === 'Overdue' ? 'text-red-600' :
                                                    invoice.status === 'PartiallyPaid' ? 'text-amber-600' :
                                                        'text-muted-foreground'
                                                }`}>
                                                {invoice.status === 'Draft' && '草稿'}
                                                {invoice.status === 'Issued' && '已开具'}
                                                {invoice.status === 'PartiallyPaid' && '部分收款'}
                                                {invoice.status === 'Paid' && '已收款'}
                                                {invoice.status === 'Overdue' && '已逾期'}
                                                {invoice.status === 'Cancelled' && '已作废'}
                                            </div>
                                        </div>
                                        <Link href={`/finance/invoices/${invoice.id}`}>
                                            <Button variant="ghost" size="sm">查看</Button>
                                        </Link>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8 text-muted-foreground">
                            暂无发票记录
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
