import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class ProjectsService {
    constructor(private prisma: PrismaService) { }

    async create(contractId: string, data: any) {
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

    async findOne(id: string) {
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

    async update(id: string, data: any) {
        try {
            const updateData = { ...data };
            if (updateData.budget !== undefined) updateData.budget = parseFloat(updateData.budget);
            if (updateData.targetProfitMargin !== undefined) updateData.targetProfitMargin = parseFloat(updateData.targetProfitMargin);
            if (updateData.laborCost !== undefined) updateData.laborCost = parseFloat(updateData.laborCost);
            if (updateData.outsourceCost !== undefined) updateData.outsourceCost = parseFloat(updateData.outsourceCost);
            if (updateData.travelCost !== undefined) updateData.travelCost = parseFloat(updateData.travelCost);
            if (updateData.emergencySupportCost !== undefined) updateData.emergencySupportCost = parseFloat(updateData.emergencySupportCost);
            if (updateData.thirdPartyEquipmentCost !== undefined) updateData.thirdPartyEquipmentCost = parseFloat(updateData.thirdPartyEquipmentCost);
            if (updateData.softwareCost !== undefined) updateData.softwareCost = parseFloat(updateData.softwareCost);
            if (updateData.otherWeight !== undefined) updateData.otherWeight = parseFloat(updateData.otherWeight);
            if (updateData.complexity !== undefined) updateData.complexity = updateData.complexity;

            console.log('Updating project:', id, 'with data:', updateData);

            return await this.prisma.project.update({
                where: { id },
                data: updateData
            });
        } catch (error) {
            console.error('Error updating project:', error);
            throw error;
        }
    }

    // Resource Management
    async addResource(projectId: string, data: any) {
        return this.prisma.projectResource.create({
            data: {
                projectId,
                ...data,
                allocationPct: parseInt(data.allocationPct) || 100
            }
        });
    }

    async removeResource(id: string) {
        return this.prisma.projectResource.delete({
            where: { id }
        });
    }

    // Meeting Management
    async createMeeting(projectId: string, data: any) {
        return this.prisma.projectMeeting.create({
            data: {
                projectId,
                ...data,
                planDate: new Date(data.planDate),
                actualDate: data.actualDate ? new Date(data.actualDate) : undefined
            }
        });
    }

    async updateMeeting(id: string, data: any) {
        const updateData = { ...data };
        if (updateData.planDate) updateData.planDate = new Date(updateData.planDate);
        if (updateData.actualDate) updateData.actualDate = new Date(updateData.actualDate);

        return this.prisma.projectMeeting.update({
            where: { id },
            data: updateData
        });
    }

    // Risk Management
    async addRisk(projectId: string, data: any) {
        return this.prisma.projectRisk.create({
            data: {
                projectId,
                ...data
            }
        });
    }

    async updateRisk(id: string, data: any) {
        return this.prisma.projectRisk.update({
            where: { id },
            data
        });
    }

    async addMeetingAttachment(meetingId: string, file: Express.Multer.File) {
        // Decode Chinese filename properly
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

    async deleteMeeting(id: string) {
        return this.prisma.projectMeeting.delete({
            where: { id }
        });
    }
}
