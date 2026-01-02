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
exports.OpportunitiesService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
let OpportunitiesService = class OpportunitiesService {
    prisma;
    constructor(prisma) {
        this.prisma = prisma;
    }
    async create(createOpportunityDto) {
        const year = new Date().getFullYear();
        const prefix = `OPP-${year}-`;
        const lastOpp = await this.prisma.opportunity.findFirst({
            where: {
                opportunityNumber: { startsWith: prefix }
            },
            orderBy: { opportunityNumber: 'desc' }
        });
        let nextNum = 1;
        if (lastOpp && lastOpp.opportunityNumber) {
            const lastNum = parseInt(lastOpp.opportunityNumber.split('-')[2], 10);
            nextNum = lastNum + 1;
        }
        const opportunityNumber = `${prefix}${nextNum.toString().padStart(4, '0')}`;
        return this.prisma.opportunity.create({
            data: {
                ...createOpportunityDto,
                opportunityNumber,
                expectedCloseDate: createOpportunityDto.expectedCloseDate ? new Date(createOpportunityDto.expectedCloseDate) : undefined,
            },
        });
    }
    async update(id, updateOpportunityDto) {
        const data = { ...updateOpportunityDto };
        if (data.expectedCloseDate && typeof data.expectedCloseDate === 'string') {
            data.expectedCloseDate = new Date(data.expectedCloseDate);
        }
        return this.prisma.opportunity.update({
            where: { id },
            data,
            include: {
                customer: true,
            },
        });
    }
    async findAll() {
        return this.prisma.opportunity.findMany({
            include: {
                customer: true,
            },
            orderBy: { createdAt: 'desc' },
        });
    }
    async findOne(id) {
        return this.prisma.opportunity.findUnique({
            where: { id },
            include: {
                customer: true,
                contracts: true,
                followUps: {
                    orderBy: { createdAt: 'desc' },
                },
                attachments: {
                    orderBy: { createdAt: 'desc' },
                },
            },
        });
    }
    async createFollowUp(opportunityId, dto, user) {
        return this.prisma.followUp.create({
            data: {
                content: dto.content,
                opportunityId,
                createdById: user?.userId,
            },
        });
    }
    async getFollowUps(opportunityId) {
        return this.prisma.followUp.findMany({
            where: { opportunityId },
            include: { user: true },
            orderBy: { createdAt: 'desc' },
        });
    }
    async createAttachment(opportunityId, fileData, user) {
        return this.prisma.attachment.create({
            data: {
                ...fileData,
                opportunityId,
            },
        });
    }
    async getAttachments(opportunityId) {
        return this.prisma.attachment.findMany({
            where: { opportunityId },
            orderBy: { createdAt: 'desc' },
        });
    }
};
exports.OpportunitiesService = OpportunitiesService;
exports.OpportunitiesService = OpportunitiesService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], OpportunitiesService);
//# sourceMappingURL=opportunities.service.js.map