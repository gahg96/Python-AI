import { IsString, IsNumber, IsEnum, IsDateString, IsOptional } from 'class-validator';
import { PaymentMethod } from '@prisma/client';

export class CreatePaymentDto {
    @IsString()
    invoiceId: string;

    @IsDateString()
    paymentDate: string;

    @IsNumber()
    amount: number;

    @IsEnum(PaymentMethod)
    paymentMethod: PaymentMethod;

    @IsString()
    @IsOptional()
    bankName?: string;

    @IsString()
    @IsOptional()
    transactionRef?: string;

    @IsString()
    @IsOptional()
    remarks?: string;
}
