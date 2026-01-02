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
exports.ProjectsService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
let ProjectsService = class ProjectsService {
    prisma;
    constructor(prisma) {
        this.prisma = prisma;
    }
    async create(contractId, data) {
        return this.prisma.project.create({
            data: {
                contractId,
                ...data,
                budget: data.budget ? parseFloat(data.budget) : 0,
                targetProfitMargin: data.targetProfitMargin ? parseFloat(data.targetProfitMargin) : 0,
            },
            include: {
                contract: true,
                resources: true,
            }
        });
    }
    async findAll() {
        return this.prisma.project.findMany({
            include: {
                contract: {
                    include: {
                        opportunity: {
                            include: { customer: true }
                        }
                    }
                },
                resources: {
                    include: { user: true }
                }
            },
            orderBy: { createdAt: 'desc' }
        });
    }
    async findOne(id) {
        return this.prisma.project.findUnique({
            where: { id },
            include: {
                contract: {
                    include: {
                        opportunity: {
                            include: { customer: true }
                        },
                        milestones: true
                    }
                },
                resources: {
                    include: { user: true }
                },
                meetings: {
                    orderBy: { planDate: 'desc' }
                },
                risks: {
                    orderBy: { createdAt: 'desc' }
                },
                invoices: {
                    orderBy: { createdAt: 'desc' }
                }
            }
        });
    }
    async update(id, data) {
        try {
            const updateData = { ...data };
            if (updateData.budget !== undefined)
                updateData.budget = parseFloat(updateData.budget);
            if (updateData.targetProfitMargin !== undefined)
                updateData.targetProfitMargin = parseFloat(updateData.targetProfitMargin);
            if (updateData.laborCost !== undefined)
                updateData.laborCost = parseFloat(updateData.laborCost);
            if (updateData.outsourceCost !== undefined)
                updateData.outsourceCost = parseFloat(updateData.outsourceCost);
            if (updateData.travelCost !== undefined)
                updateData.travelCost = parseFloat(updateData.travelCost);
            if (updateData.emergencySupportCost !== undefined)
                updateData.emergencySupportCost = parseFloat(updateData.emergencySupportCost);
            if (updateData.thirdPartyEquipmentCost !== undefined)
                updateData.thirdPartyEquipmentCost = parseFloat(updateData.thirdPartyEquipmentCost);
            if (updateData.softwareCost !== undefined)
                updateData.softwareCost = parseFloat(updateData.softwareCost);
            if (updateData.otherWeight !== undefined)
                updateData.otherWeight = parseFloat(updateData.otherWeight);
            if (updateData.complexity !== undefined)
                updateData.complexity = updateData.complexity;
            console.log('Updating project:', id, 'with data:', updateData);
            return await this.prisma.project.update({
                where: { id },
                data: updateData
            });
        }
        catch (error) {
            console.error('Error updating project:', error);
            throw error;
        }
    }
    async addResource(projectId, data) {
        return this.prisma.projectResource.create({
            data: {
                projectId,
                ...data,
                allocationPct: parseInt(data.allocationPct) || 100
            }
        });
    }
    async removeResource(id) {
        return this.prisma.projectResource.delete({
            where: { id }
        });
    }
    async createMeeting(projectId, data) {
        return this.prisma.projectMeeting.create({
            data: {
                projectId,
                ...data,
                planDate: new Date(data.planDate),
                actualDate: data.actualDate ? new Date(data.actualDate) : undefined
            }
        });
    }
    async updateMeeting(id, data) {
        const updateData = { ...data };
        if (updateData.planDate)
            updateData.planDate = new Date(updateData.planDate);
        if (updateData.actualDate)
            updateData.actualDate = new Date(updateData.actualDate);
        return this.prisma.projectMeeting.update({
            where: { id },
            data: updateData
        });
    }
    async addRisk(projectId, data) {
        return this.prisma.projectRisk.create({
            data: {
                projectId,
                ...data
            }
        });
    }
    async updateRisk(id, data) {
        return this.prisma.projectRisk.update({
            where: { id },
            data
        });
    }
    async addMeetingAttachment(meetingId, file) {
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');
        return this.prisma.projectMeeting.update({
            where: { id: meetingId },
            data: {
                filepath: file.path,
                filename: decodedFilename,
                mimetype: file.mimetype
            }
        });
    }
    async deleteMeeting(id) {
        return this.prisma.projectMeeting.delete({
            where: { id }
        });
    }
};
exports.ProjectsService = ProjectsService;
exports.ProjectsService = ProjectsService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], ProjectsService);
//# sourceMappingURL=projects.service.js.map