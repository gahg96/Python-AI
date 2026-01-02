import { IsString, IsOptional, IsNumber, IsEnum, IsDateString } from 'class-validator';

export enum ProcurementType {
    DirectQuote = 'DirectQuote',
    Negotiation = 'Negotiation',
    Comparison = 'Comparison',
    Consultation = 'Consultation',
    PublicTender = 'PublicTender',
}

export class CreateProcurementDto {
    @IsString()
    opportunityId: string;

    @IsEnum(ProcurementType)
    type: ProcurementType;

    @IsOptional()
    @IsNumber()
    customerBudget?: number;

    @IsOptional()
    @IsNumber()
    ourQuote?: number;

    @IsOptional()
    @IsDateString()
    submissionDeadline?: string;

    @IsOptional()
    @IsDateString()
    notificationDate?: string;

    @IsOptional()
    @IsString()
    bidLocation?: string;

    @IsOptional()
    @IsNumber()
    depositAmount?: number;

    @IsOptional()
    @IsString()
    notes?: string;
}
