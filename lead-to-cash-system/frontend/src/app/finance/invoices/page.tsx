'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Search, FileText, ArrowLeft } from 'lucide-react';
import { api } from '@/lib/api';
import Link from 'next/link';

export default function InvoicesPage() {
    const [invoices, setInvoices] = useState<any[]>([]);
    const [filteredInvoices, setFilteredInvoices] = useState<any[]>([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchInvoices();
    }, []);

    useEffect(() => {
        if (searchTerm) {
            const filtered = invoices.filter(inv =>
                inv.invoiceNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
                inv.contract?.opportunity?.customer?.companyName?.toLowerCase().includes(searchTerm.toLowerCase())
            );
            setFilteredInvoices(filtered);
        } else {
            setFilteredInvoices(invoices);
        }
    }, [searchTerm, invoices]);

    const fetchInvoices = async () => {
        try {
            const data = await api.get('/finance/invoices');
            setInvoices(data);
            setFilteredInvoices(data);
        } catch (error) {
            console.error('Failed to fetch invoices:', error);
        } finally {
            setLoading(false);
        }
    };

    const getStatusBadge = (status: string) => {
        const variants: any = {
            Draft: 'secondary',
            Issued: 'default',
            PartiallyPaid: 'warning',
            Paid: 'success',
            Overdue: 'destructive',
            Cancelled: 'outline',
        };

        const labels: any = {
            Draft: '草稿',
            Issued: '已开具',
            PartiallyPaid: '部分收款',
            Paid: '已收款',
            Overdue: '已逾期',
            Cancelled: '已作废',
        };

        return (
            <Badge variant={variants[status] || 'default'}>
                {labels[status] || status}
            </Badge>
        );
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
                <div className="flex items-center gap-4">
                    <Link href="/finance">
                        <Button variant="ghost" size="sm">
                            <ArrowLeft className="h-4 w-4 mr-2" />
                            返回
                        </Button>
                    </Link>
                    <div>
                        <h1 className="text-3xl font-bold">发票管理</h1>
                        <p className="text-muted-foreground mt-1">共 {filteredInvoices.length} 张发票</p>
                    </div>
                </div>
                <Link href="/finance/invoices/new">
                    <Button>
                        <FileText className="mr-2 h-4 w-4" />
                        创建发票
                    </Button>
                </Link>
            </div>

            {/* Search */}
            <Card>
                <CardContent className="pt-6">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                        <Input
                            placeholder="搜索发票号或客户名称..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="pl-10"
                        />
                    </div>
                </CardContent>
            </Card>

            {/* Invoice List */}
            <Card>
                <CardHeader>
                    <CardTitle>发票列表</CardTitle>
                </CardHeader>
                <CardContent>
                    {filteredInvoices.length > 0 ? (
                        <div className="space-y-3">
                            {filteredInvoices.map((invoice) => (
                                <Link key={invoice.id} href={`/finance/invoices/${invoice.id}`}>
                                    <div className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent transition-colors cursor-pointer">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-3">
                                                <div className="font-bold text-lg">{invoice.invoiceNumber}</div>
                                                {getStatusBadge(invoice.status)}
                                            </div>
                                            <div className="text-sm text-muted-foreground mt-1">
                                                {invoice.contract?.opportunity?.customer?.companyName || '未知客户'}
                                            </div>
                                            <div className="text-xs text-muted-foreground mt-1">
                                                开票日期: {new Date(invoice.invoiceDate).toLocaleDateString('zh-CN')}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="font-bold text-xl">¥{invoice.totalAmount?.toLocaleString()}</div>
                                            <div className="text-sm text-muted-foreground">
                                                {invoice.type === 'Service' ? '服务 6%' : '产品 13%'}
                                            </div>
                                        </div>
                                    </div>
                                </Link>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-12 text-muted-foreground">
                            {searchTerm ? '未找到匹配的发票' : '暂无发票记录'}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
