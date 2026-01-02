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
exports.ContractsService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
const client_1 = require("@prisma/client");
let ContractsService = class ContractsService {
    prisma;
    constructor(prisma) {
        this.prisma = prisma;
    }
    async create(createContractDto, userId) {
        return this.prisma.contract.create({
            data: {
                ...createContractDto,
                status: client_1.ContractStatus.Draft,
                drafterId: userId,
            },
            include: {
                drafter: true,
                opportunity: true,
            },
        });
    }
    findAll() {
        return this.prisma.contract.findMany({
            include: {
                opportunity: {
                    include: { customer: true }
                },
                drafter: true,
                approver: true,
            },
            orderBy: { createdAt: 'desc' }
        });
    }
    findOne(id) {
        return this.prisma.contract.findUnique({
            where: { id },
            include: {
                opportunity: { include: { customer: true } },
                milestones: { orderBy: { createdAt: 'asc' } },
                documents: { include: { uploadedBy: true }, orderBy: { createdAt: 'desc' } },
                drafter: true,
                approver: true,
                project: true,
            },
        });
    }
    update(id, updateContractDto) {
        return this.prisma.contract.update({
            where: { id },
            data: updateContractDto,
        });
    }
    async submitForApproval(id) {
        return this.prisma.contract.update({
            where: { id },
            data: { status: client_1.ContractStatus.PendingApproval },
        });
    }
    async approve(id, approverId) {
        return this.prisma.contract.update({
            where: { id },
            data: {
                status: client_1.ContractStatus.Approved,
                approverId,
            },
        });
    }
    async reject(id, approverId) {
        return this.prisma.contract.update({
            where: { id },
            data: {
                status: client_1.ContractStatus.Rejected,
                approverId,
            },
        });
    }
    async sign(id) {
        return this.prisma.contract.update({
            where: { id },
            data: { status: client_1.ContractStatus.Signed },
        });
    }
    async addDocument(contractId, file, userId) {
        return this.prisma.contractDocument.create({
            data: {
                contractId,
                filename: file.filename,
                filepath: file.path,
                mimetype: file.mimetype,
                size: file.size,
                uploadedById: userId,
            },
        });
    }
    async addMilestone(contractId, data) {
        const milestoneData = { ...data };
        if (milestoneData.dueDate && typeof milestoneData.dueDate === 'string') {
            milestoneData.dueDate = new Date(milestoneData.dueDate);
        }
        return this.prisma.milestone.create({
            data: {
                contractId,
                ...milestoneData,
                amount: milestoneData.amount ? parseFloat(milestoneData.amount) : 0,
            },
        });
    }
    async updateMilestone(id, data) {
        const updateData = { ...data };
        if (updateData.amount) {
            updateData.amount = parseFloat(updateData.amount);
        }
        if (updateData.dueDate && typeof updateData.dueDate === 'string') {
            updateData.dueDate = new Date(updateData.dueDate);
        }
        return this.prisma.milestone.update({
            where: { id },
            data: updateData,
        });
    }
    async deleteMilestone(id) {
        return this.prisma.milestone.delete({
            where: { id },
        });
    }
    remove(id) {
        return this.prisma.contract.delete({
            where: { id },
        });
    }
};
exports.ContractsService = ContractsService;
exports.ContractsService = ContractsService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], ContractsService);
//# sourceMappingURL=contracts.service.js.map