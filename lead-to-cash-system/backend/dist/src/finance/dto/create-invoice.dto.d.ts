import { InvoiceType } from '@prisma/client';
export declare class CreateInvoiceDto {
    contractId: string;
    projectId?: string;
    milestoneId?: string;
    invoiceDate: string;
    dueDate: string;
    amount: number;
    type: InvoiceType;
    description?: string;
    remarks?: string;
}
