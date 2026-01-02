import { IsString, IsOptional, IsBoolean } from 'class-validator';

export class CreateMilestoneTemplateDto {
    @IsString()
    name: string;

    @IsOptional()
    @IsString()
    description?: string;

    @IsString()
    milestones: string; // JSON string of milestone array

    @IsOptional()
    @IsBoolean()
    isActive?: boolean;
}
