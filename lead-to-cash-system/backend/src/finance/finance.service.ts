import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateInvoiceDto } from './dto/create-invoice.dto';
import { CreatePaymentDto } from './dto/create-payment.dto';
import { InvoiceType, InvoiceStatus, MilestoneStatus } from '@prisma/client';

@Injectable()
export class FinanceService {
    constructor(private prisma: PrismaService) { }

    /**
     * Generate invoice number in format: RM-YYYY-XXXXX
     */
    async generateInvoiceNumber(): Promise<string> {
        const year = new Date().getFullYear();
        const prefix = `RM-${year}-`;

        // Find the last invoice for this year
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

    /**
     * Calculate tax amount based on invoice type
     * Service: 6%, Product: 13%
     */
    calculateTax(amount: number, type: InvoiceType): number {
        const taxRate = type === InvoiceType.Service ? 0.06 : 0.13;
        return Math.round(amount * taxRate * 100) / 100; // Round to 2 decimal places
    }

    /**
     * Create invoice
     */
    async createInvoice(dto: CreateInvoiceDto) {
        // Validate contract exists
        const contract = await this.prisma.contract.findUnique({
            where: { id: dto.contractId },
        });

        if (!contract) {
            throw new NotFoundException('Contract not found');
        }

        // If milestone is specified, validate it
        if (dto.milestoneId) {
            const milestone = await this.prisma.milestone.findUnique({
                where: { id: dto.milestoneId },
            });

            if (!milestone) {
                throw new NotFoundException('Milestone not found');
            }

            // Check if milestone already has an invoice
            if (milestone.invoiceDate) {
                throw new BadRequestException('Milestone already has an invoice');
            }
        }

        // Generate invoice number
        const invoiceNumber = await this.generateInvoiceNumber();

        // Calculate tax
        const taxRate = dto.type === InvoiceType.Service ? 0.06 : 0.13;
        const taxAmount = this.calculateTax(dto.amount, dto.type);
        const totalAmount = dto.amount + taxAmount;

        // Create invoice
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
                status: InvoiceStatus.Draft,
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

        // If linked to milestone, update milestone invoice date
        if (dto.milestoneId) {
            await this.prisma.milestone.update({
                where: { id: dto.milestoneId },
                data: {
                    invoiceDate: new Date(dto.invoiceDate),
                    status: MilestoneStatus.Invoiced,
                },
            });
        }

        return invoice;
    }

    async findOneMilestone(id: string) {
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

    /**
     * Create invoice from milestone
     */
    async createInvoiceFromMilestone(milestoneId: string, dto: Partial<CreateInvoiceDto>) {
        const milestone = await this.prisma.milestone.findUnique({
            where: { id: milestoneId },
            include: {
                contract: {
                    include: { project: true } // Include project to get ID
                },
            },
        });

        if (!milestone) {
            throw new NotFoundException('Milestone not found');
        }

        // Allow invoicing for Pending/WIP (e.g. Advance Payment) - removed strict check
        // if (milestone.status !== MilestoneStatus.Verified && milestone.status !== MilestoneStatus.Ready_to_Invoice) {
        //     throw new BadRequestException('Milestone must be verified before creating invoice');
        // }

        if (milestone.invoiceDate) {
            throw new BadRequestException('Milestone already has an invoice');
        }

        const invoiceDto: CreateInvoiceDto = {
            contractId: milestone.contractId,
            projectId: milestone.contract?.project?.id, // Link to project
            milestoneId: milestone.id,
            amount: Number(milestone.amount),
            invoiceDate: dto.invoiceDate || new Date().toISOString(),
            dueDate: dto.dueDate || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days from now
            type: dto.type || InvoiceType.Service,
            description: dto.description || `Invoice for milestone: ${milestone.name}`,
            remarks: dto.remarks,
        };

        return this.createInvoice(invoiceDto);
    }

    /**
     * Get all invoices
     */
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

    /**
     * Get invoice by ID
     */
    async findOne(id: string) {
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
            throw new NotFoundException('Invoice not found');
        }

        return invoice;
    }

    /**
     * Update invoice status
     */
    async updateStatus(id: string, status: InvoiceStatus) {
        const invoice = await this.findOne(id);

        return this.prisma.invoice.update({
            where: { id },
            data: { status },
        });
    }

    /**
     * Create payment record
     */
    async createPayment(dto: CreatePaymentDto) {
        const invoice = await this.findOne(dto.invoiceId);

        // Generate payment number
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

        // Create payment
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

        // Calculate total payments for this invoice
        const totalPayments = await this.prisma.payment.aggregate({
            where: { invoiceId: dto.invoiceId },
            _sum: { amount: true },
        });

        const totalPaid = totalPayments._sum.amount || 0;

        // Update invoice status
        let newStatus = invoice.status;
        if (totalPaid >= invoice.totalAmount) {
            newStatus = InvoiceStatus.Paid;

            // Update milestone status if linked
            if (invoice.milestoneId) {
                await this.prisma.milestone.update({
                    where: { id: invoice.milestoneId },
                    data: {
                        paymentDate: new Date(dto.paymentDate),
                        status: MilestoneStatus.Paid,
                    },
                });
            }
        } else if (totalPaid > 0) {
            newStatus = InvoiceStatus.PartiallyPaid;
        }

        if (newStatus !== invoice.status) {
            await this.updateStatus(dto.invoiceId, newStatus);
        }

        return payment;
    }

    /**
     * Get dashboard data
     */
    async getDashboardData() {
        // Get all invoices
        const invoices = await this.prisma.invoice.findMany({
            include: {
                payments: true,
                milestone: true,
            },
        });

        // Calculate KPIs
        let pendingInvoiceAmount = 0;
        let outstandingAmount = 0;
        let paidAmount = 0;

        for (const invoice of invoices) {
            const totalPaid = invoice.payments.reduce((sum, p) => sum + p.amount, 0);

            if (invoice.status === InvoiceStatus.Draft) {
                // Not counted
            } else if (invoice.status === InvoiceStatus.Paid) {
                paidAmount += invoice.totalAmount;
            } else {
                outstandingAmount += (invoice.totalAmount - totalPaid);
            }
        }

        // Get milestones ready to invoice
        const readyToInvoiceMilestones = await this.prisma.milestone.findMany({
            where: {
                status: {
                    in: [MilestoneStatus.Verified, MilestoneStatus.Ready_to_Invoice],
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

        const pendingInvoiceAmountFromMilestones = readyToInvoiceMilestones.reduce(
            (sum, m) => sum + Number(m.amount),
            0
        );

        return {
            pendingInvoiceAmount: pendingInvoiceAmountFromMilestones,
            outstandingAmount,
            paidAmount,
            readyToInvoiceMilestones,
            recentInvoices: invoices.slice(0, 10),
        };
    }

    /**
     * Milestone Template Management
     */
    async createMilestoneTemplate(data: { name: string; description?: string; milestones: string; isActive?: boolean }) {
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

    async findOneTemplate(id: string) {
        const template = await this.prisma.milestoneTemplate.findUnique({
            where: { id },
        });

        if (!template) {
            throw new NotFoundException('Template not found');
        }

        return template;
    }

    async updateTemplate(id: string, data: Partial<{ name: string; description?: string; milestones: string; isActive: boolean }>) {
        await this.findOneTemplate(id);

        return this.prisma.milestoneTemplate.update({
            where: { id },
            data,
        });
    }

    async deleteTemplate(id: string) {
        await this.findOneTemplate(id);

        return this.prisma.milestoneTemplate.update({
            where: { id },
            data: { isActive: false },
        });
    }

    async uploadReceipt(id: string, file: any) {
        const invoice = await this.prisma.invoice.findUnique({ where: { id } });
        if (!invoice) throw new NotFoundException('Invoice not found');

        return this.prisma.invoice.update({
            where: { id },
            data: {
                filePath: file.path,
                fileName: file.filename,
            },
        });
    }
}
