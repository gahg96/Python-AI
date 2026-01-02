export declare class CreateOpportunityDto {
    customerId: string;
    title: string;
    estimatedValue?: number;
    probability?: number;
    source?: string;
    status?: 'New' | 'Proposal' | 'Negotiation' | 'Won' | 'Lost';
    expectedCloseDate?: string;
    salesStage?: string;
    competitors?: string;
    decisionMakers?: string;
    salesOwner?: string;
    dealType?: string;
    deliveryModel?: string;
    estimatedEffort?: number;
    richDescription?: string;
    projectBudget?: number;
    businessCost?: number;
    laborCost?: number;
    otherCost?: number;
    grossProfit?: number;
    profitMargin?: number;
}
