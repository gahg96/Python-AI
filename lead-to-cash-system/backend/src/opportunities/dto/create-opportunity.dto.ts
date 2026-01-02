import { IsString, IsNotEmpty, IsOptional, IsNumber, IsIn } from 'class-validator';

export class CreateOpportunityDto {
    @IsString()
    @IsNotEmpty()
    customerId: string;

    @IsString()
    @IsNotEmpty()
    title: string;

    @IsNumber()
    @IsOptional()
    estimatedValue?: number;

    @IsNumber()
    @IsOptional()
    probability?: number;

    @IsString()
    @IsOptional()
    source?: string;

    @IsString()
    @IsOptional()
    status?: 'New' | 'Proposal' | 'Negotiation' | 'Won' | 'Lost';

    @IsString()
    @IsOptional()
    expectedCloseDate?: string; // ISO Date String

    // Phase 5: Enterprise Fields
    @IsString()
    @IsOptional()
    salesStage?: string;

    @IsString()
    @IsOptional()
    competitors?: string;

    @IsString()
    @IsOptional()
    decisionMakers?: string;

    @IsString()
    @IsOptional()
    salesOwner?: string;

    @IsString()
    @IsOptional()
    dealType?: string;

    @IsString()
    @IsOptional()
    deliveryModel?: string;

    @IsNumber()
    @IsOptional()
    estimatedEffort?: number;

    @IsString()
    @IsOptional()
    richDescription?: string;

    // Phase 6: Financial Fields
    @IsNumber()
    @IsOptional()
    projectBudget?: number;   // 客户预算

    @IsNumber()
    @IsOptional()
    businessCost?: number;    // 商务费用

    @IsNumber()
    @IsOptional()
    laborCost?: number;       // 人力成本

    @IsNumber()
    @IsOptional()
    otherCost?: number;       // 其他成本

    @IsNumber()
    @IsOptional()
    grossProfit?: number;     // 毛利润

    @IsNumber()
    @IsOptional()
    profitMargin?: number;    // 毛利率
}
