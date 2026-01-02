import { IsString, IsNotEmpty, IsOptional, IsNumber, IsDateString, IsBoolean } from 'class-validator';

export class CreateContractDto {
    @IsNotEmpty()
    @IsString()
    opportunityId: string;

    @IsNotEmpty()
    @IsString()
    contractNumber: string;

    @IsNotEmpty()
    @IsNumber()
    totalContractValue: number;

    @IsOptional()
    @IsString()
    paymentTerms?: string;

    @IsOptional()
    @IsDateString()
    startDate?: string;

    @IsOptional()
    @IsDateString()
    endDate?: string;

    @IsOptional()
    @IsString()
    riskAssessment?: string;

    @IsOptional()
    @IsString()
    scope?: string;

    @IsOptional()
    @IsString()
    sla?: string;

    @IsOptional()
    @IsString()
    liability?: string;

    @IsOptional()
    @IsString()
    paymentTermsDetails?: string;

    // Optional: Drafter ID typically comes from current user, but DTO might allow it
    @IsOptional()
    @IsString()
    drafterId?: string;
}
