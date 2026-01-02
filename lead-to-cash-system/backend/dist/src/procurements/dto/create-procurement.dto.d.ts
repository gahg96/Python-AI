export declare enum ProcurementType {
    DirectQuote = "DirectQuote",
    Negotiation = "Negotiation",
    Comparison = "Comparison",
    Consultation = "Consultation",
    PublicTender = "PublicTender"
}
export declare class CreateProcurementDto {
    opportunityId: string;
    type: ProcurementType;
    customerBudget?: number;
    ourQuote?: number;
    submissionDeadline?: string;
    notificationDate?: string;
    bidLocation?: string;
    depositAmount?: number;
    notes?: string;
}
