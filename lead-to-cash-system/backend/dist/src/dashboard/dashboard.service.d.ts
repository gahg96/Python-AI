import { PrismaService } from '../prisma/prisma.service';
export declare class DashboardService {
    private prisma;
    constructor(prisma: PrismaService);
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
    getGeoDistribution(): Promise<{
        name: string;
        city: {
            en: string;
            zh: string;
        };
        coords: [number, number];
        value: number;
    }[]>;
}
