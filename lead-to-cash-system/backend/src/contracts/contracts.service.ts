import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateContractDto } from './dto/create-contract.dto';
import { UpdateContractDto } from './dto/update-contract.dto';
import { ContractStatus } from '@prisma/client';

@Injectable()
export class ContractsService {
    constructor(private prisma: PrismaService) { }

    async create(createContractDto: CreateContractDto, userId: string) {
        return this.prisma.contract.create({
            data: {
                ...createContractDto,
                status: ContractStatus.Draft,
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

    findOne(id: string) {
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

    update(id: string, updateContractDto: UpdateContractDto) {
        return this.prisma.contract.update({
            where: { id },
            data: updateContractDto,
        });
    }

    // Approval Workflow Actions
    async submitForApproval(id: string) {
        return this.prisma.contract.update({
            where: { id },
            data: { status: ContractStatus.PendingApproval },
        });
    }

    async approve(id: string, approverId: string) {
        return this.prisma.contract.update({
            where: { id },
            data: {
                status: ContractStatus.Approved,
                approverId,
            },
        });
    }

    async reject(id: string, approverId: string) {
        return this.prisma.contract.update({
            where: { id },
            data: {
                status: ContractStatus.Rejected,
                approverId, // Optional: keep track of who rejected
            },
        });
    }

    async sign(id: string) {
        return this.prisma.contract.update({
            where: { id },
            data: { status: ContractStatus.Signed },
        });
    }

    // Document Management
    async addDocument(contractId: string, file: any, userId: string) {
        return this.prisma.contractDocument.create({
            data: {
                contractId,
                filename: file.filename, // Using the decoded/processed filename
                filepath: file.path, // Multer uses .path
                mimetype: file.mimetype,
                size: file.size,
                uploadedById: userId,
            },
        });
    }


    // Milestone Management
    async addMilestone(contractId: string, data: any) {
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

    async updateMilestone(id: string, data: any) {
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

    async deleteMilestone(id: string) {
        return this.prisma.milestone.delete({
            where: { id },
        });
    }

    remove(id: string) {
        return this.prisma.contract.delete({
            where: { id },
        });
    }
}

