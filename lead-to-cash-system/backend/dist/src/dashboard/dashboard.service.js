"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DashboardService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
let DashboardService = class DashboardService {
    prisma;
    constructor(prisma) {
        this.prisma = prisma;
    }
    async getStats() {
        const totalPayments = await this.prisma.payment.aggregate({
            _sum: { amount: true },
        });
        const totalCashIn = totalPayments._sum.amount || 0;
        const pendingInvoices = await this.prisma.invoice.aggregate({
            where: {
                status: { notIn: ['Paid', 'Draft', 'Cancelled'] }
            },
            _sum: { totalAmount: true },
        });
        const pendingInvoicesAmount = pendingInvoices._sum.totalAmount || 0;
        const pipeline = await this.prisma.opportunity.aggregate({
            where: { status: { in: ['New', 'Proposal', 'Negotiation', 'Won'] } },
            _sum: { estimatedValue: true },
            _count: true,
        });
        const projectedRevenue = pipeline._sum.estimatedValue || 0;
        const activeDealsCount = pipeline._count;
        const wonDeals = await this.prisma.opportunity.count({ where: { status: 'Won' } });
        const lostDeals = await this.prisma.opportunity.count({ where: { status: 'Lost' } });
        const totalClosed = wonDeals + lostDeals;
        const winRate = totalClosed > 0 ? Math.round((wonDeals / totalClosed) * 100) : 0;
        return {
            totalCashIn,
            pendingInvoices: pendingInvoicesAmount,
            projectedRevenue,
            activeContractCount: 0,
            activeDealsCount,
            winRate,
        };
    }
    async getFunnel() {
        const stages = ['New', 'Proposal', 'Negotiation', 'Won'];
        const funnelData = [];
        for (const stage of stages) {
            const count = await this.prisma.opportunity.count({
                where: { status: stage },
            });
            funnelData.push({ stage, count });
        }
        return funnelData;
    }
    async getTrend() {
        const months = [];
        const now = new Date();
        for (let i = 5; i >= 0; i--) {
            const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
            months.push(d.toISOString().slice(0, 7));
        }
        const payments = await this.prisma.payment.findMany({
            select: { amount: true, paymentDate: true }
        });
        const opportunities = await this.prisma.opportunity.findMany({
            where: { status: { not: 'Lost' } },
            select: { id: true, estimatedValue: true, expectedCloseDate: true }
        });
        const trendData = months.map(month => {
            const revenue = payments
                .filter(p => p.paymentDate && p.paymentDate.toISOString().startsWith(month))
                .reduce((sum, p) => sum + Number(p.amount), 0);
            const monthOpps = opportunities
                .filter(o => o.expectedCloseDate && o.expectedCloseDate.toISOString().startsWith(month));
            const pipeline = monthOpps.reduce((sum, o) => sum + Number(o.estimatedValue), 0);
            const opportunityIds = monthOpps.map(o => o.id);
            const dateObj = new Date(month + "-01");
            const monthName = dateObj.toLocaleString('en-US', { month: 'short' });
            return {
                month: monthName,
                fullDate: month,
                revenue,
                pipeline,
                opportunityIds
            };
        });
        return trendData;
    }
    async fixDates() {
        const now = new Date();
        const opps = await this.prisma.opportunity.findMany();
        const dates = [
            new Date(now.getFullYear(), now.getMonth() - 4, 15),
            new Date(now.getFullYear(), now.getMonth() - 1, 15),
            new Date(now.getFullYear(), now.getMonth() + 2, 15),
        ];
        for (let i = 0; i < opps.length; i++) {
            await this.prisma.opportunity.update({
                where: { id: opps[i].id },
                data: { expectedCloseDate: dates[i % dates.length] }
            });
        }
        const payments = await this.prisma.payment.findMany();
        const payDates = [
            new Date(now.getFullYear(), now.getMonth() - 5, 10),
            new Date(now.getFullYear(), now.getMonth() - 3, 20),
            new Date(now.getFullYear(), now.getMonth() - 1, 5),
        ];
        for (let i = 0; i < payments.length; i++) {
            await this.prisma.payment.update({
                where: { id: payments[i].id },
                data: { paymentDate: payDates[i % payDates.length] }
            });
        }
        const customers = await this.prisma.customer.findMany();
        const cities = [
            { name: "London", country: "UK" },
            { name: "Beijing", country: "China" },
            { name: "Seattle", country: "USA" }
        ];
        for (let i = 0; i < customers.length; i++) {
            const loc = cities[i % cities.length];
            await this.prisma.customer.update({
                where: { id: customers[i].id },
                data: { city: loc.name, country: loc.country }
            });
        }
        return { message: "Dates shifted and Cities populated" };
    }
    async getGeoDistribution() {
        const customers = await this.prisma.customer.findMany({
            include: {
                opportunities: true,
            }
        });
        const cityStats = new Map();
        const CITY_DATA = [
            { keys: ["london", "伦敦"], en: "London", zh: "伦敦", coords: [-0.1276, 51.5074] },
            { keys: ["beijing", "北京"], en: "Beijing", zh: "北京", coords: [116.4074, 39.9042] },
            { keys: ["shanghai", "上海"], en: "Shanghai", zh: "上海", coords: [121.4737, 31.2304] },
            { keys: ["seattle", "西雅图"], en: "Seattle", zh: "西雅图", coords: [-122.3321, 47.6062] },
            { keys: ["new york", "纽约"], en: "New York", zh: "纽约", coords: [-74.0060, 40.7128] },
            { keys: ["singapore", "新加坡"], en: "Singapore", zh: "新加坡", coords: [103.8198, 1.3521] },
            { keys: ["tokyo", "东京"], en: "Tokyo", zh: "东京", coords: [139.6503, 35.6762] },
            { keys: ["paris", "巴黎"], en: "Paris", zh: "巴黎", coords: [2.3522, 48.8566] },
            { keys: ["berlin", "柏林"], en: "Berlin", zh: "柏林", coords: [13.4050, 52.5200] },
            { keys: ["sydney", "悉尼"], en: "Sydney", zh: "悉尼", coords: [151.2093, -33.8688] },
            { keys: ["dubai", "迪拜"], en: "Dubai", zh: "迪拜", coords: [55.2708, 25.2048] },
            { keys: ["hong kong", "香港"], en: "Hong Kong", zh: "香港", coords: [114.1694, 22.3193] },
            { keys: ["shenzhen", "深圳"], en: "Shenzhen", zh: "深圳", coords: [114.0579, 22.5431] },
            { keys: ["guangzhou", "广州"], en: "Guangzhou", zh: "广州", coords: [113.2644, 23.1291] },
            { keys: ["chengdu", "成都"], en: "Chengdu", zh: "成都", coords: [104.0668, 30.5728] },
            { keys: ["hangzhou", "杭州"], en: "Hangzhou", zh: "杭州", coords: [120.1551, 30.2741] },
        ];
        const aggregated = new Map();
        for (const customer of customers) {
            if (!customer.city)
                continue;
            const totalValue = customer.opportunities.reduce((sum, opp) => sum + Number(opp.estimatedValue || 0), 0);
            const lowerCity = customer.city.toLowerCase();
            const matchedCity = CITY_DATA.find(c => c.keys.some(k => lowerCity.includes(k)));
            if (matchedCity) {
                const current = aggregated.get(matchedCity.en) || 0;
                aggregated.set(matchedCity.en, current + totalValue);
            }
        }
        const results = [];
        for (const [enName, value] of aggregated.entries()) {
            const cityInfo = CITY_DATA.find(c => c.en === enName);
            if (cityInfo) {
                results.push({
                    name: `${cityInfo.en} Cluster`,
                    city: { en: cityInfo.en, zh: cityInfo.zh },
                    coords: cityInfo.coords,
                    value: value
                });
            }
        }
        return results;
    }
};
exports.DashboardService = DashboardService;
exports.DashboardService = DashboardService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], DashboardService);
//# sourceMappingURL=dashboard.service.js.map