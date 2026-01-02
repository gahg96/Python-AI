import { IsString, IsOptional, IsEmail, IsNotEmpty } from 'class-validator';

export class CreateCustomerDto {
    @IsString()
    @IsNotEmpty()
    companyName: string;

    @IsString()
    @IsOptional()
    industry?: string;

    @IsString()
    @IsOptional()
    companySize?: string;

    @IsString()
    @IsOptional()
    city?: string;

    @IsString()
    @IsOptional()
    country?: string;

    @IsString()
    @IsOptional()
    contactName?: string;

    @IsString()
    @IsOptional()
    contactTitle?: string;

    @IsString()
    @IsOptional()
    contactPhone?: string;

    @IsString()
    @IsEmail()
    @IsOptional()
    contactEmail?: string;
}
