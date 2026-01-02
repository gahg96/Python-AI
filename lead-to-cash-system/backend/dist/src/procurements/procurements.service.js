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
exports.ProcurementsService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
const DEFAULT_TASKS = {
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
let ProcurementsService = class ProcurementsService {
    prisma;
    constructor(prisma) {
        this.prisma = prisma;
    }
    async create(createProcurementDto) {
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
        const data = {
            ...createProcurementDto,
            procurementNumber,
        };
        if (data.submissionDeadline && typeof data.submissionDeadline === 'string') {
            data.submissionDeadline = new Date(data.submissionDeadline);
        }
        if (data.notificationDate && typeof data.notificationDate === 'string') {
            data.notificationDate = new Date(data.notificationDate);
        }
        const procurement = await this.prisma.procurement.create({
            data,
        });
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
    async findByOpportunity(opportunityId) {
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
    async findOne(id) {
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
    async update(id, updateProcurementDto) {
        const data = { ...updateProcurementDto };
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
    async remove(id) {
        return this.prisma.procurement.delete({
            where: { id },
        });
    }
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
    async updateTask(taskId, isCompleted, assignee) {
        return this.prisma.biddingTask.update({
            where: { id: taskId },
            data: {
                isCompleted,
                ...(assignee !== undefined && { assignee }),
            },
        });
    }
    async addDocument(procurementId, doc, user) {
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
    async getDocuments(procurementId) {
        return this.prisma.procurementDocument.findMany({
            where: { procurementId },
            orderBy: { createdAt: 'desc' },
        });
    }
};
exports.ProcurementsService = ProcurementsService;
exports.ProcurementsService = ProcurementsService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], ProcurementsService);
//# sourceMappingURL=procurements.service.js.map