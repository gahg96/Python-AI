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
exports.FinanceService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
const client_1 = require("@prisma/client");
let FinanceService = class FinanceService {
    prisma;
    constructor(prisma) {
        this.prisma = prisma;
    }
    async generateInvoiceNumber() {
        const year = new Date().getFullYear();
        const prefix = `RM-${year}-`;
        const lastInvoice = await this.prisma.invoice.findFirst({
            where: {
                invoiceNumber: {
                    startsWith: prefix,
                },
            },
            orderBy: {
                invoiceNumber: 'desc',
            },
        });
        let sequence = 1;
        if (lastInvoice) {
            const lastSequence = parseInt(lastInvoice.invoiceNumber.split('-')[2]);
            sequence = lastSequence + 1;
        }
        return `${prefix}${sequence.toString().padStart(5, '0')}`;
    }
    calculateTax(amount, type) {
        const taxRate = type === client_1.InvoiceType.Service ? 0.06 : 0.13;
        return Math.round(amount * taxRate * 100) / 100;
    }
    async createInvoice(dto) {
        const contract = await this.prisma.contract.findUnique({
            where: { id: dto.contractId },
        });
        if (!contract) {
            throw new common_1.NotFoundException('Contract not found');
        }
        if (dto.milestoneId) {
            const milestone = await this.prisma.milestone.findUnique({
                where: { id: dto.milestoneId },
            });
            if (!milestone) {
                throw new common_1.NotFoundException('Milestone not found');
            }
            if (milestone.invoiceDate) {
                throw new common_1.BadRequestException('Milestone already has an invoice');
            }
        }
        const invoiceNumber = await this.generateInvoiceNumber();
        const taxRate = dto.type === client_1.InvoiceType.Service ? 0.06 : 0.13;
        const taxAmount = this.calculateTax(dto.amount, dto.type);
        const totalAmount = dto.amount + taxAmount;
        const invoice = await this.prisma.invoice.create({
            data: {
                invoiceNumber,
                contractId: dto.contractId,
                projectId: dto.projectId,
                milestoneId: dto.milestoneId,
                invoiceDate: new Date(dto.invoiceDate),
                dueDate: new Date(dto.dueDate),
                amount: dto.amount,
                taxRate,
                taxAmount,
                totalAmount,
                type: dto.type,
                status: client_1.InvoiceStatus.Draft,
                description: dto.description,
                remarks: dto.remarks,
            },
            include: {
                contract: {
                    include: {
                        opportunity: {
                            include: {
                                customer: true,
                            },
                        },
                    },
                },
                project: true,
                milestone: true,
            },
        });
        if (dto.milestoneId) {
            await this.prisma.milestone.update({
                where: { id: dto.milestoneId },
                data: {
                    invoiceDate: new Date(dto.invoiceDate),
                    status: client_1.MilestoneStatus.Invoiced,
                },
            });
        }
        return invoice;
    }
    async findOneMilestone(id) {
        return this.prisma.milestone.findUnique({
            where: { id },
            include: {
                contract: {
                    include: {
                        project: true
                    }
                },
                invoice: true,
            },
        });
    }
    async createInvoiceFromMilestone(milestoneId, dto) {
        const milestone = await this.prisma.milestone.findUnique({
            where: { id: milestoneId },
            include: {
                contract: {
                    include: { project: true }
                },
            },
        });
        if (!milestone) {
            throw new common_1.NotFoundException('Milestone not found');
        }
        if (milestone.invoiceDate) {
            throw new common_1.BadRequestException('Milestone already has an invoice');
        }
        const invoiceDto = {
            contractId: milestone.contractId,
            projectId: milestone.contract?.project?.id,
            milestoneId: milestone.id,
            amount: Number(milestone.amount),
            invoiceDate: dto.invoiceDate || new Date().toISOString(),
            dueDate: dto.dueDate || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
            type: dto.type || client_1.InvoiceType.Service,
            description: dto.description || `Invoice for milestone: ${milestone.name}`,
            remarks: dto.remarks,
        };
        return this.createInvoice(invoiceDto);
    }
    async findAll() {
        return this.prisma.invoice.findMany({
            include: {
                contract: {
                    include: {
                        opportunity: {
                            include: {
                                customer: true,
                            },
                        },
                    },
                },
                project: true,
                milestone: true,
                payments: true,
            },
            orderBy: {
                createdAt: 'desc',
            },
        });
    }
    async findOne(id) {
        const invoice = await this.prisma.invoice.findUnique({
            where: { id },
            include: {
                contract: {
                    include: {
                        opportunity: {
                            include: {
                                customer: true,
                            },
                        },
                    },
                },
                project: true,
                milestone: true,
                payments: true,
            },
        });
        if (!invoice) {
            throw new common_1.NotFoundException('Invoice not found');
        }
        return invoice;
    }
    async updateStatus(id, status) {
        const invoice = await this.findOne(id);
        return this.prisma.invoice.update({
            where: { id },
            data: { status },
        });
    }
    async createPayment(dto) {
        const invoice = await this.findOne(dto.invoiceId);
        const year = new Date().getFullYear();
        const prefix = `PAY-${year}-`;
        const lastPayment = await this.prisma.payment.findFirst({
            where: { paymentNumber: { startsWith: prefix } },
            orderBy: { paymentNumber: 'desc' },
        });
        let sequence = 1;
        if (lastPayment) {
            const lastSeq = parseInt(lastPayment.paymentNumber.split('-')[2]);
            sequence = lastSeq + 1;
        }
        const paymentNumber = `${prefix}${sequence.toString().padStart(5, '0')}`;
        const payment = await this.prisma.payment.create({
            data: {
                paymentNumber,
                invoiceId: dto.invoiceId,
                paymentDate: new Date(dto.paymentDate),
                amount: dto.amount,
                paymentMethod: dto.paymentMethod,
                bankName: dto.bankName,
                transactionRef: dto.transactionRef,
                remarks: dto.remarks,
            },
        });
        const totalPayments = await this.prisma.payment.aggregate({
            where: { invoiceId: dto.invoiceId },
            _sum: { amount: true },
        });
        const totalPaid = totalPayments._sum.amount || 0;
        let newStatus = invoice.status;
        if (totalPaid >= invoice.totalAmount) {
            newStatus = client_1.InvoiceStatus.Paid;
            if (invoice.milestoneId) {
                await this.prisma.milestone.update({
                    where: { id: invoice.milestoneId },
                    data: {
                        paymentDate: new Date(dto.paymentDate),
                        status: client_1.MilestoneStatus.Paid,
                    },
                });
            }
        }
        else if (totalPaid > 0) {
            newStatus = client_1.InvoiceStatus.PartiallyPaid;
        }
        if (newStatus !== invoice.status) {
            await this.updateStatus(dto.invoiceId, newStatus);
        }
        return payment;
    }
    async getDashboardData() {
        const invoices = await this.prisma.invoice.findMany({
            include: {
                payments: true,
                milestone: true,
            },
        });
        let pendingInvoiceAmount = 0;
        let outstandingAmount = 0;
        let paidAmount = 0;
        for (const invoice of invoices) {
            const totalPaid = invoice.payments.reduce((sum, p) => sum + p.amount, 0);
            if (invoice.status === client_1.InvoiceStatus.Draft) {
            }
            else if (invoice.status === client_1.InvoiceStatus.Paid) {
                paidAmount += invoice.totalAmount;
            }
            else {
                outstandingAmount += (invoice.totalAmount - totalPaid);
            }
        }
        const readyToInvoiceMilestones = await this.prisma.milestone.findMany({
            where: {
                status: {
                    in: [client_1.MilestoneStatus.Verified, client_1.MilestoneStatus.Ready_to_Invoice],
                },
            },
            include: {
                contract: {
                    include: {
                        opportunity: {
                            include: {
                                customer: true,
                            },
                        },
                    },
                },
            },
        });
        const pendingInvoiceAmountFromMilestones = readyToInvoiceMilestones.reduce((sum, m) => sum + Number(m.amount), 0);
        return {
            pendingInvoiceAmount: pendingInvoiceAmountFromMilestones,
            outstandingAmount,
            paidAmount,
            readyToInvoiceMilestones,
            recentInvoices: invoices.slice(0, 10),
        };
    }
    async createMilestoneTemplate(data) {
        return this.prisma.milestoneTemplate.create({
            data: {
                name: data.name,
                description: data.description,
                milestones: data.milestones,
                isActive: data.isActive ?? true,
            },
        });
    }
    async findAllTemplates() {
        return this.prisma.milestoneTemplate.findMany({
            where: { isActive: true },
            orderBy: { createdAt: 'desc' },
        });
    }
    async findOneTemplate(id) {
        const template = await this.prisma.milestoneTemplate.findUnique({
            where: { id },
        });
        if (!template) {
            throw new common_1.NotFoundException('Template not found');
        }
        return template;
    }
    async updateTemplate(id, data) {
        await this.findOneTemplate(id);
        return this.prisma.milestoneTemplate.update({
            where: { id },
            data,
        });
    }
    async deleteTemplate(id) {
        await this.findOneTemplate(id);
        return this.prisma.milestoneTemplate.update({
            where: { id },
            data: { isActive: false },
        });
    }
    async uploadReceipt(id, file) {
        const invoice = await this.prisma.invoice.findUnique({ where: { id } });
        if (!invoice)
            throw new common_1.NotFoundException('Invoice not found');
        return this.prisma.invoice.update({
            where: { id },
            data: {
                filePath: file.path,
                fileName: file.filename,
            },
        });
    }
};
exports.FinanceService = FinanceService;
exports.FinanceService = FinanceService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], FinanceService);
//# sourceMappingURL=finance.service.js.map