import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateOpportunityDto } from './dto/create-opportunity.dto';
import { CreateFollowUpDto } from './dto/create-follow-up.dto';
import { UpdateOpportunityDto } from './dto/update-opportunity.dto';

@Injectable()
export class OpportunitiesService {
    constructor(private prisma: PrismaService) { }

    async create(createOpportunityDto: CreateOpportunityDto) {
        // Generate opportunity number: OPP-YYYY-NNNN
        const year = new Date().getFullYear();
        const prefix = `OPP-${year}-`;

        // Find the highest existing number for this year
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

    async update(id: string, updateOpportunityDto: UpdateOpportunityDto) {
        // Prepare data with proper date conversion
        const data: any = { ...updateOpportunityDto };

        // Convert expectedCloseDate string to Date if provided
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

    async findOne(id: string) {
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

    // FollowUp methods
    async createFollowUp(opportunityId: string, dto: CreateFollowUpDto, user?: any) {
        return this.prisma.followUp.create({
            data: {
                content: dto.content,
                opportunityId,
                createdById: user?.userId,
            },
        });
    }

    async getFollowUps(opportunityId: string) {
        return this.prisma.followUp.findMany({
            where: { opportunityId },
            include: { user: true }, // Include user details
            orderBy: { createdAt: 'desc' },
        });
    }

    // Attachment methods
    async createAttachment(opportunityId: string, fileData: any, user?: any) {
        // fileData contains filename, filepath, mimetype, size
        return this.prisma.attachment.create({ // NOTE: Attachment model didn't have uploadedBy in previous schema read, need to check if I updated it? 
            // Wait, I only updated ProcurementDocument in schema. Is Attachment on Opportunity needing this too?
            // User request: "上传技术方案和商务方案" (Procurement related) primarily. 
            // So focus on ProcurementsService later.
            data: {
                ...fileData,
                opportunityId,
                // uploadedById: user?.userId 
            },
        });
    }

    async getAttachments(opportunityId: string) {
        return this.prisma.attachment.findMany({
            where: { opportunityId },
            orderBy: { createdAt: 'desc' },
        });
    }
}
