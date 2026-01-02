import { PartialType } from '@nestjs/mapped-types';
import { CreateProcurementDto } from './create-procurement.dto';
import { IsOptional, IsEnum, IsString } from 'class-validator';

export enum ProcurementStatus {
    Draft = 'Draft',
    Preparing = 'Preparing',
    Submitted = 'Submitted',
    InProgress = 'InProgress',
    Won = 'Won',
    Lost = 'Lost',
}

export class UpdateProcurementDto extends PartialType(CreateProcurementDto) {
    @IsOptional()
    @IsEnum(ProcurementStatus)
    status?: ProcurementStatus;

    @IsOptional()
    @IsString()
    resultNote?: string;
}
