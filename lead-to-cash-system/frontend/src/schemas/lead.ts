import { z } from "zod";

export const leadSchema = z.object({
    // Customer fields
    companyName: z.string().min(1, "Company Name is required"),
    industry: z.string().optional(),
    companySize: z.string().optional(),
    city: z.string().optional(), // Added city
    contactName: z.string().optional(),
    contactTitle: z.string().optional(),
    contactPhone: z.string().optional(),
    contactEmail: z.string().email().optional().or(z.literal('')),

    // Opportunity fields
    title: z.string().min(1, "Project Name is required"),
    estimatedValue: z.coerce.number().min(0, "Value must be positive").default(0),
    probability: z.coerce.number().min(0).max(100).default(50),
    source: z.string().optional(),
    expectedCloseDate: z.string().optional(),

    // Enterprise fields
    salesStage: z.string().optional(),
    competitors: z.string().optional(),
    decisionMakers: z.string().optional(),
    salesOwner: z.string().optional(),
    dealType: z.string().optional(),
    deliveryModel: z.string().optional(),
    estimatedEffort: z.coerce.number().min(0).optional(),
    richDescription: z.string().optional(),

    // Financial fields
    projectBudget: z.coerce.number().min(0).optional(),
    businessCost: z.coerce.number().min(0).optional(),
    laborCost: z.coerce.number().min(0).optional(),
    otherCost: z.coerce.number().min(0).optional(),
    grossProfit: z.coerce.number().min(0).optional(),
    profitMargin: z.coerce.number().min(0).max(100).optional(),
});

export type LeadFormValues = z.infer<typeof leadSchema>;
