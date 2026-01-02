'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { ArrowLeft, Download, DollarSign, Upload, FileText } from 'lucide-react';
import { api } from '@/lib/api';
import Link from 'next/link';

export default function InvoiceDetailPage() {
    const params = useParams();
    const [invoice, setInvoice] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [isEditingRemarks, setIsEditingRemarks] = useState(false);
    const [remarks, setRemarks] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [showPaymentDialog, setShowPaymentDialog] = useState(false);
    const [paymentForm, setPaymentForm] = useState({
        amount: '',
        paymentDate: new Date().toISOString().split('T')[0],
        paymentMethod: 'BankTransfer',
        bankName: '',
        transactionRef: '',
        remarks: '',
    });

    useEffect(() => {
        if (params.id) {
            fetchInvoice();
        }
    }, [params.id]);

    const fetchInvoice = async () => {
        try {
            const data = await api.get(`/finance/invoices/${params.id}`);
            setInvoice(data);
            setRemarks(data.remarks || '');
        } catch (error) {
            console.error('Failed to fetch invoice:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSaveRemarks = async () => {
        try {
            await api.patch(`/finance/invoices/${params.id}`, { remarks });
            setIsEditingRemarks(false);
            fetchInvoice();
        } catch (error) {
            console.error("Failed to save remarks", error);
        }
    };

    const handleUploadReceipt = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;

        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        setIsUploading(true);
        try {
            await api.post(`/finance/invoices/${params.id}/receipt`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            fetchInvoice();
        } catch (error) {
            console.error("Failed to upload receipt", error);
            alert("上传失败");
        } finally {
            setIsUploading(false);
        }
    };

    const handlePaymentSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await api.post('/finance/payments', {
                invoiceId: params.id,
                ...paymentForm,
                amount: parseFloat(paymentForm.amount),
            });
            setShowPaymentDialog(false);
            fetchInvoice(); // Refresh invoice data
            setPaymentForm({
                amount: '',
                paymentDate: new Date().toISOString().split('T')[0],
                paymentMethod: 'BankTransfer',
                bankName: '',
                transactionRef: '',
                remarks: '',
            });
        } catch (error) {
            console.error('Failed to record payment:', error);
            alert('记录收款失败');
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
            <Badge variant={variants[status] || 'default'} className="text-sm">
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

    if (!invoice) {
        return (
            <div className="flex flex-col items-center justify-center h-96">
                <div className="text-lg mb-4">发票未找到</div>
                <Link href="/finance/invoices">
                    <Button>返回发票列表</Button>
                </Link>
            </div>
        );
    }

    const totalPaid = invoice.payments?.reduce((sum: number, p: any) => sum + p.amount, 0) || 0;
    const remaining = invoice.totalAmount - totalPaid;

    return (
        <div className="container mx-auto p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link href="/finance/invoices">
                        <Button variant="ghost" size="sm">
                            <ArrowLeft className="h-4 w-4 mr-2" />
                            返回
                        </Button>
                    </Link>
                    <div>
                        <div className="flex items-center gap-3">
                            <h1 className="text-3xl font-bold">{invoice.invoiceNumber}</h1>
                            {getStatusBadge(invoice.status)}
                        </div>
                        <p className="text-muted-foreground mt-1">
                            {invoice.contract?.opportunity?.customer?.companyName || '未知客户'}
                        </p>
                    </div>
                </div>
                <Button variant="outline">
                    <Download className="mr-2 h-4 w-4" />
                    下载PDF
                </Button>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                {/* Invoice Details */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex justify-between items-center">
                            <span>发票信息</span>
                            {invoice.filePath && (
                                <Badge variant="outline" className="text-green-600 border-green-200 bg-green-50">
                                    已上传回单
                                </Badge>
                            )}
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <div className="text-sm text-muted-foreground">发票号</div>
                                <div className="font-medium">{invoice.invoiceNumber}</div>
                            </div>
                            <div>
                                <div className="text-sm text-muted-foreground">类型</div>
                                <div className="font-medium">
                                    {invoice.type === 'Service' ? '服务 (6%)' : '产品 (13%)'}
                                </div>
                            </div>
                            <div>
                                <div className="text-sm text-muted-foreground">开票日期</div>
                                <div className="font-medium">
                                    {new Date(invoice.invoiceDate).toLocaleDateString('zh-CN')}
                                </div>
                            </div>
                            <div>
                                <div className="text-sm text-muted-foreground">到期日期</div>
                                <div className="font-medium">
                                    {new Date(invoice.dueDate).toLocaleDateString('zh-CN')}
                                </div>
                            </div>
                        </div>

                        {invoice.description && (
                            <div>
                                <div className="text-sm text-muted-foreground">描述</div>
                                <div className="font-medium">{invoice.description}</div>
                            </div>
                        )}

                        {/* Remarks Section */}
                        <div>
                            <div className="flex justify-between items-center mb-1">
                                <div className="text-sm text-muted-foreground">备注</div>
                                {!isEditingRemarks && (
                                    <Button variant="link" size="sm" className="h-auto p-0 text-blue-600" onClick={() => setIsEditingRemarks(true)}>
                                        编辑
                                    </Button>
                                )}
                            </div>

                            {isEditingRemarks ? (
                                <div className="space-y-2">
                                    <Textarea
                                        value={remarks}
                                        onChange={(e) => setRemarks(e.target.value)}
                                        className="min-h-[80px]"
                                    />
                                    <div className="flex gap-2 justify-end">
                                        <Button size="sm" variant="ghost" onClick={() => {
                                            setIsEditingRemarks(false);
                                            setRemarks(invoice.remarks || '');
                                        }}>取消</Button>
                                        <Button size="sm" onClick={handleSaveRemarks}>保存</Button>
                                    </div>
                                </div>
                            ) : (
                                <div className="font-medium whitespace-pre-wrap">{invoice.remarks || '-'}</div>
                            )}
                        </div>

                        {/* Receipt Upload Section */}
                        <div className="border-t pt-4 mt-4">
                            <div className="text-sm font-medium mb-2">电子回单</div>
                            {invoice.filePath ? (
                                <div className="flex items-center justify-between bg-slate-50 p-3 rounded-md border">
                                    <div className="flex items-center gap-2 overflow-hidden">
                                        <FileText className="h-4 w-4 text-blue-500 flex-shrink-0" />
                                        <span className="text-sm truncate" title={invoice.fileName}>{invoice.fileName || '已上传文件'}</span>
                                    </div>
                                    <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={() => window.open(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'}/${invoice.filePath}`, '_blank')}>
                                        <Download className="h-4 w-4" />
                                    </Button>
                                </div>
                            ) : (
                                <div>
                                    <input
                                        type="file"
                                        id="receipt-upload"
                                        className="hidden"
                                        accept=".pdf,.jpg,.jpeg,.png"
                                        onChange={handleUploadReceipt}
                                        disabled={isUploading}
                                    />
                                    <Label
                                        htmlFor="receipt-upload"
                                        className={`flex items-center justify-center gap-2 w-full p-3 border-2 border-dashed rounded-md cursor-pointer hover:bg-slate-50 transition-colors ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
                                    >
                                        <Upload className="h-4 w-4 text-slate-500" />
                                        <span className="text-sm text-slate-500">
                                            {isUploading ? '上传中...' : '点击上传电子回单'}
                                        </span>
                                    </Label>
                                </div>
                            )}
                        </div>
                    </CardContent>
                </Card>

                {/* Amount Details */}
                <Card>
                    <CardHeader>
                        <CardTitle>金额明细</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-muted-foreground">金额</span>
                            <span className="font-medium">¥{invoice.amount?.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-muted-foreground">税率</span>
                            <span className="font-medium">{(invoice.taxRate * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-muted-foreground">税额</span>
                            <span className="font-medium">¥{invoice.taxAmount?.toLocaleString()}</span>
                        </div>
                        <div className="border-t pt-3 flex justify-between">
                            <span className="font-bold">总额</span>
                            <span className="font-bold text-xl">¥{invoice.totalAmount?.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between text-green-600">
                            <span>已收款</span>
                            <span className="font-bold">¥{totalPaid.toLocaleString()}</span>
                        </div>
                        {remaining > 0 && (
                            <div className="flex justify-between text-orange-600">
                                <span>待收款</span>
                                <span className="font-bold">¥{remaining.toLocaleString()}</span>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* Payment Records */}
            <Card>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <CardTitle>收款记录</CardTitle>
                        {invoice.status !== 'Paid' && invoice.status !== 'Cancelled' && (
                            <Dialog open={showPaymentDialog} onOpenChange={setShowPaymentDialog}>
                                <DialogTrigger asChild>
                                    <Button size="sm">
                                        <DollarSign className="mr-2 h-4 w-4" />
                                        记录收款
                                    </Button>
                                </DialogTrigger>
                                <DialogContent>
                                    <DialogHeader>
                                        <DialogTitle>记录收款</DialogTitle>
                                    </DialogHeader>
                                    <form onSubmit={handlePaymentSubmit} className="space-y-4">
                                        <div className="space-y-2">
                                            <Label htmlFor="amount">收款金额 *</Label>
                                            <Input
                                                id="amount"
                                                type="number"
                                                step="0.01"
                                                value={paymentForm.amount}
                                                onChange={(e) => setPaymentForm({ ...paymentForm, amount: e.target.value })}
                                                required
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="paymentDate">收款日期 *</Label>
                                            <Input
                                                id="paymentDate"
                                                type="date"
                                                value={paymentForm.paymentDate}
                                                onChange={(e) => setPaymentForm({ ...paymentForm, paymentDate: e.target.value })}
                                                required
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="paymentMethod">支付方式 *</Label>
                                            <Select
                                                value={paymentForm.paymentMethod}
                                                onValueChange={(value) => setPaymentForm({ ...paymentForm, paymentMethod: value })}
                                            >
                                                <SelectTrigger>
                                                    <SelectValue />
                                                </SelectTrigger>
                                                <SelectContent>
                                                    <SelectItem value="BankTransfer">银行转账</SelectItem>
                                                    <SelectItem value="Check">支票</SelectItem>
                                                    <SelectItem value="Cash">现金</SelectItem>
                                                    <SelectItem value="Other">其他</SelectItem>
                                                </SelectContent>
                                            </Select>
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="bankName">银行名称</Label>
                                            <Input
                                                id="bankName"
                                                value={paymentForm.bankName}
                                                onChange={(e) => setPaymentForm({ ...paymentForm, bankName: e.target.value })}
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="transactionRef">交易参考号</Label>
                                            <Input
                                                id="transactionRef"
                                                value={paymentForm.transactionRef}
                                                onChange={(e) => setPaymentForm({ ...paymentForm, transactionRef: e.target.value })}
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="remarks">备注</Label>
                                            <Input
                                                id="remarks"
                                                value={paymentForm.remarks}
                                                onChange={(e) => setPaymentForm({ ...paymentForm, remarks: e.target.value })}
                                            />
                                        </div>
                                        <div className="flex justify-end gap-3">
                                            <Button type="button" variant="outline" onClick={() => setShowPaymentDialog(false)}>
                                                取消
                                            </Button>
                                            <Button type="submit">
                                                确认收款
                                            </Button>
                                        </div>
                                    </form>
                                </DialogContent>
                            </Dialog>
                        )}
                    </div>
                </CardHeader>
                <CardContent>
                    {invoice.payments && invoice.payments.length > 0 ? (
                        <div className="space-y-3">
                            {invoice.payments.map((payment: any) => (
                                <div
                                    key={payment.id}
                                    className="flex items-center justify-between p-4 border rounded-lg"
                                >
                                    <div>
                                        <div className="font-medium">{payment.paymentNumber}</div>
                                        <div className="text-sm text-muted-foreground">
                                            {new Date(payment.paymentDate).toLocaleDateString('zh-CN')}
                                        </div>
                                        {payment.transactionRef && (
                                            <div className="text-xs text-muted-foreground">
                                                交易号: {payment.transactionRef}
                                            </div>
                                        )}
                                    </div>
                                    <div className="text-right">
                                        <div className="font-bold text-lg text-green-600">
                                            ¥{payment.amount?.toLocaleString()}
                                        </div>
                                        <div className="text-sm text-muted-foreground">
                                            {payment.paymentMethod === 'BankTransfer' && '银行转账'}
                                            {payment.paymentMethod === 'Check' && '支票'}
                                            {payment.paymentMethod === 'Cash' && '现金'}
                                            {payment.paymentMethod === 'Other' && '其他'}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8 text-muted-foreground">
                            暂无收款记录
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
