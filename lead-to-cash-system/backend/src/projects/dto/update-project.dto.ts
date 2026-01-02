import { IsString, IsOptional, IsNumber, IsDateString, IsBoolean, IsEnum } from 'class-validator';

export class UpdateProjectDto {
    @IsEnum(['Initialization', 'Planning', 'Execution', 'Delivery', 'Maintenance', 'Closed'])
    @IsOptional()
    status?: string;

    @IsNumber()
    @IsOptional()
    budget?: number;

    @IsNumber()
    @IsOptional()
    targetProfitMargin?: number;

    @IsNumber()
    @IsOptional()
    laborCost?: number;

    @IsNumber()
    @IsOptional()
    outsourceCost?: number;

    @IsNumber()
    @IsOptional()
    travelCost?: number;

    @IsNumber()
    @IsOptional()
    emergencySupportCost?: number;

    @IsNumber()
    @IsOptional()
    thirdPartyEquipmentCost?: number;

    @IsNumber()
    @IsOptional()
    softwareCost?: number;

    @IsNumber()
    @IsOptional()
    otherWeight?: number;

    @IsEnum(['Low', 'Medium', 'High'])
    @IsOptional()
    complexity?: string;

    @IsString()
    @IsOptional()
    financialRemarks?: string;

    @IsBoolean()
    @IsOptional()
    isDelayed?: boolean;

    @IsDateString()
    @IsOptional()
    startDate?: string;

    @IsDateString()
    @IsOptional()
    endDate?: string;

    @IsString()
    @IsOptional()
    description?: string;
}
