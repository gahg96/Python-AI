'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { ArrowLeft } from 'lucide-react';
import { api } from '@/lib/api';
import Link from 'next/link';

export default function NewInvoicePage() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const milestoneId = searchParams.get('milestoneId');

    const [contracts, setContracts] = useState<any[]>([]);
    const [milestone, setMilestone] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        contractId: '',
        milestoneId: milestoneId || '',
        amount: '',
        invoiceDate: new Date().toISOString().split('T')[0],
        dueDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        type: 'Service',
        description: '',
        remarks: '',
    });

    useEffect(() => {
        fetchContracts();
        if (milestoneId) {
            fetchMilestone(milestoneId);
        }
    }, [milestoneId]);

    const fetchContracts = async () => {
        try {
            const data = await api.get('/contracts');
            setContracts(data);
        } catch (error) {
            console.error('Failed to fetch contracts:', error);
        }
    };

    const fetchMilestone = async (id: string) => {
        try {
            const data = await api.get(`/finance/milestones/${id}`);
            setMilestone(data);
            setFormData(prev => ({
                ...prev,
                contractId: data.contractId,
                amount: data.amount.toString(),
                description: `Invoice for milestone: ${data.name}`,
            }));
        } catch (error) {
            console.error('Failed to fetch milestone:', error);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        try {
            const payload = {
                ...formData,
                amount: parseFloat(formData.amount),
            };

            if (milestoneId) {
                await api.post(`/finance/invoices/from-milestone/${milestoneId}`, payload);
            } else {
                await api.post('/finance/invoices', payload);
            }

            router.push('/finance/invoices');
        } catch (error: any) {
            console.error('Failed to create invoice:', error);
            const msg = error.response?.data?.message || error.message || '创建发票失败，请检查输入';
            alert(`创建失败: ${msg}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex items-center gap-4">
                <Link href="/finance/invoices">
                    <Button variant="ghost" size="sm">
                        <ArrowLeft className="h-4 w-4 mr-2" />
                        返回
                    </Button>
                </Link>
                <div>
                    <h1 className="text-3xl font-bold">创建发票</h1>
                    {milestone && (
                        <p className="text-muted-foreground mt-1">
                            里程碑: {milestone.name}
                        </p>
                    )}
                </div>
            </div>

            <form onSubmit={handleSubmit}>
                <Card>
                    <CardHeader>
                        <CardTitle>发票信息</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label htmlFor="contractId">合同 *</Label>
                                <Select
                                    value={formData.contractId}
                                    onValueChange={(value) => setFormData({ ...formData, contractId: value })}
                                    disabled={!!milestoneId}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="选择合同" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {contracts.map((contract) => (
                                            <SelectItem key={contract.id} value={contract.id}>
                                                {contract.opportunity?.customer?.companyName || 'Unknown'} - ¥{contract.totalContractValue?.toLocaleString()}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="type">发票类型 *</Label>
                                <Select
                                    value={formData.type}
                                    onValueChange={(value) => setFormData({ ...formData, type: value })}
                                >
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="Service">服务 (6%)</SelectItem>
                                        <SelectItem value="Product">产品 (13%)</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="amount">金额 *</Label>
                                <Input
                                    id="amount"
                                    type="number"
                                    step="0.01"
                                    value={formData.amount}
                                    onChange={(e) => setFormData({ ...formData, amount: e.target.value })}
                                    required
                                    disabled={!!milestoneId}
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="invoiceDate">开票日期 *</Label>
                                <Input
                                    id="invoiceDate"
                                    type="date"
                                    value={formData.invoiceDate}
                                    onChange={(e) => setFormData({ ...formData, invoiceDate: e.target.value })}
                                    required
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="dueDate">到期日期 *</Label>
                                <Input
                                    id="dueDate"
                                    type="date"
                                    value={formData.dueDate}
                                    onChange={(e) => setFormData({ ...formData, dueDate: e.target.value })}
                                    required
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="description">描述</Label>
                            <Textarea
                                id="description"
                                value={formData.description}
                                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                rows={3}
                            />
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="remarks">备注</Label>
                            <Textarea
                                id="remarks"
                                value={formData.remarks}
                                onChange={(e) => setFormData({ ...formData, remarks: e.target.value })}
                                rows={2}
                            />
                        </div>

                        <div className="flex justify-end gap-3 pt-4">
                            <Link href="/finance/invoices">
                                <Button type="button" variant="outline">
                                    取消
                                </Button>
                            </Link>
                            <Button type="submit" disabled={loading}>
                                {loading ? '创建中...' : '创建发票'}
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </form>
        </div>
    );
}
