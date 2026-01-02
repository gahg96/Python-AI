import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateProcurementDto } from './dto/create-procurement.dto';
import { UpdateProcurementDto } from './dto/update-procurement.dto';

// Default tasks by procurement type
const DEFAULT_TASKS: Record<string, { name: string; category: string }[]> = {
    DirectQuote: [
        { name: '报价单', category: 'commercial' },
    ],
    Negotiation: [
        { name: '技术说明书', category: 'technical' },
        { name: '报价文件', category: 'commercial' },
    ],
    Comparison: [
        { name: '技术说明书', category: 'technical' },
        { name: '报价文件', category: 'commercial' },
        { name: '资质证明', category: 'qualification' },
    ],
    Consultation: [
        { name: '技术方案', category: 'technical' },
        { name: '商务报价', category: 'commercial' },
        { name: '资质证明', category: 'qualification' },
        { name: '投标保证金', category: 'commercial' },
    ],
    PublicTender: [
        { name: '技术方案', category: 'technical' },
        { name: '商务报价', category: 'commercial' },
        { name: '资质证明', category: 'qualification' },
        { name: '投标保证金', category: 'commercial' },
        { name: '授权书', category: 'qualification' },
    ],
};

@Injectable()
export class ProcurementsService {
    constructor(private prisma: PrismaService) { }

    async create(createProcurementDto: CreateProcurementDto) {
        // Generate procurement number: BID-YYYY-NNNN
        const year = new Date().getFullYear();
        const prefix = `BID-${year}-`;

        const lastProc = await this.prisma.procurement.findFirst({
            where: {
                procurementNumber: { startsWith: prefix }
            },
            orderBy: { procurementNumber: 'desc' }
        });

        let nextNum = 1;
        if (lastProc && lastProc.procurementNumber) {
            const lastNum = parseInt(lastProc.procurementNumber.split('-')[2], 10);
            nextNum = lastNum + 1;
        }

        const procurementNumber = `${prefix}${nextNum.toString().padStart(4, '0')}`;

        // Prepare data with proper date conversion
        const data: any = {
            ...createProcurementDto,
            procurementNumber,
        };

        if (data.submissionDeadline && typeof data.submissionDeadline === 'string') {
            data.submissionDeadline = new Date(data.submissionDeadline);
        }
        if (data.notificationDate && typeof data.notificationDate === 'string') {
            data.notificationDate = new Date(data.notificationDate);
        }

        // Create procurement
        const procurement = await this.prisma.procurement.create({
            data,
        });

        // Auto-generate default tasks based on type
        const defaultTasks = DEFAULT_TASKS[createProcurementDto.type] || [];
        for (let i = 0; i < defaultTasks.length; i++) {
            await this.prisma.biddingTask.create({
                data: {
                    procurementId: procurement.id,
                    name: defaultTasks[i].name,
                    category: defaultTasks[i].category,
                    sortOrder: i,
                },
            });
        }

        // Return complete procurement with relations
        return this.prisma.procurement.findUnique({
            where: { id: procurement.id },
            include: {
                opportunity: {
                    include: { customer: true }
                },
                documents: true,
                tasks: { orderBy: { sortOrder: 'asc' } },
            },
        });
    }

    async findAll() {
        return this.prisma.procurement.findMany({
            include: {
                opportunity: {
                    include: { customer: true }
                },
                documents: true,
                tasks: { orderBy: { sortOrder: 'asc' } },
            },
            orderBy: { createdAt: 'desc' },
        });
    }

    async findByOpportunity(opportunityId: string) {
        return this.prisma.procurement.findMany({
            where: { opportunityId },
            include: {
                opportunity: {
                    include: { customer: true }
                },
                documents: true,
                tasks: { orderBy: { sortOrder: 'asc' } },
            },
            orderBy: { createdAt: 'desc' },
        });
    }

    async findOne(id: string) {
        return this.prisma.procurement.findUnique({
            where: { id },
            include: {
                opportunity: {
                    include: { customer: true }
                },
                documents: true,
                tasks: { orderBy: { sortOrder: 'asc' } },
            },
        });
    }

    async update(id: string, updateProcurementDto: UpdateProcurementDto) {
        const data: any = { ...updateProcurementDto };

        if (data.submissionDeadline && typeof data.submissionDeadline === 'string') {
            data.submissionDeadline = new Date(data.submissionDeadline);
        }
        if (data.notificationDate && typeof data.notificationDate === 'string') {
            data.notificationDate = new Date(data.notificationDate);
        }

        return this.prisma.procurement.update({
            where: { id },
            data,
            include: {
                opportunity: {
                    include: { customer: true }
                },
                documents: true,
                tasks: { orderBy: { sortOrder: 'asc' } },
            },
        });
    }

    async remove(id: string) {
        return this.prisma.procurement.delete({
            where: { id },
        });
    }

    // Get active procurements for dashboard
    async findActive() {
        return this.prisma.procurement.findMany({
            where: {
                status: {
                    in: ['Draft', 'Preparing', 'Submitted', 'InProgress']
                }
            },
            include: {
                opportunity: {
                    include: { customer: true }
                },
                documents: true,
                tasks: { orderBy: { sortOrder: 'asc' } },
            },
            orderBy: { submissionDeadline: 'asc' },
        });
    }

    // Update task completion status
    async updateTask(taskId: string, isCompleted: boolean, assignee?: string) {
        return this.prisma.biddingTask.update({
            where: { id: taskId },
            data: {
                isCompleted,
                ...(assignee !== undefined && { assignee }),
            },
        });
    }

    // Add document to procurement
    async addDocument(procurementId: string, doc: {
        docType: string;
        filename: string;
        filepath: string;
        mimetype: string;
        size: number;
    }, user?: any) {
        return this.prisma.procurementDocument.create({
            data: {
                procurementId,
                docType: doc.docType,
                filename: doc.filename,
                filepath: doc.filepath,
                mimetype: doc.mimetype,
                size: doc.size,
                uploadedById: user?.userId
            },
        });
    }

    // Get documents for a procurement
    async getDocuments(procurementId: string) {
        return this.prisma.procurementDocument.findMany({
            where: { procurementId },
            orderBy: { createdAt: 'desc' },
        });
    }
}
