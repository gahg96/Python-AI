import { IsString, IsOptional, IsEnum, IsDateString } from 'class-validator';

export class CreateMeetingDto {
    @IsString()
    title: string;

    @IsEnum(['Kickoff', 'Weekly', 'Monthly', 'Technical', 'Review', 'Adhoc'])
    @IsOptional()
    type?: string;

    @IsDateString()
    planDate: string;

    @IsDateString()
    @IsOptional()
    actualDate?: string;

    @IsString()
    @IsOptional()
    minutes?: string;
}
