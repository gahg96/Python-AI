import { IsString, IsNumber, IsEnum, IsOptional, IsDateString } from 'class-validator';
import { InvoiceType } from '@prisma/client';

export class CreateInvoiceDto {
    @IsString()
    contractId: string;

    @IsOptional()
    @IsString()
    projectId?: string;

    @IsOptional()
    @IsString()
    milestoneId?: string;

    @IsDateString()
    invoiceDate: string;

    @IsDateString()
    dueDate: string;

    @IsNumber()
    amount: number;

    @IsEnum(InvoiceType)
    type: InvoiceType;

    @IsOptional()
    @IsString()
    description?: string;

    @IsOptional()
    @IsString()
    remarks?: string;
}
