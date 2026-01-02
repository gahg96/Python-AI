import { DashboardService } from './dashboard.service';
export declare class DashboardController {
    private readonly dashboardService;
    constructor(dashboardService: DashboardService);
    getStats(): Promise<{
        totalCashIn: number;
        pendingInvoices: number;
        projectedRevenue: number | import("@prisma/client/runtime/library").Decimal;
        activeContractCount: number;
        activeDealsCount: number;
        winRate: number;
    }>;
    getFunnel(): Promise<{
        stage: string;
        count: number;
    }[]>;
    getTrend(): Promise<{
        month: string;
        fullDate: string;
        revenue: number;
        pipeline: number;
        opportunityIds: any[];
    }[]>;
    fixDates(): Promise<{
        message: string;
    }>;
    getGeo(): Promise<{
        name: string;
        city: {
            en: string;
            zh: string;
        };
        coords: [number, number];
        value: number;
    }[]>;
}
