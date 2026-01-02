import {
    Controller,
    Get,
    Post,
    Body,
    Param,
    Patch,
    UseGuards,
    UseInterceptors,
    UploadedFile,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { extname } from 'path';
import { FinanceService } from './finance.service';
import { CreateInvoiceDto } from './dto/create-invoice.dto';
import { CreatePaymentDto } from './dto/create-payment.dto';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { InvoiceStatus } from '@prisma/client';

@Controller('finance')
@UseGuards(JwtAuthGuard)
export class FinanceController {
    constructor(private readonly financeService: FinanceService) {
        // Ensure upload directory exists
        const fs = require('fs');
        if (!fs.existsSync('./uploads/invoices')) {
            fs.mkdirSync('./uploads/invoices', { recursive: true });
        }
    }

    /**
     * Get dashboard data
     * GET /finance/dashboard
     */
    @Get('dashboard')
    getDashboard() {
        return this.financeService.getDashboardData();
    }

    /**
     * Get all invoices
     * GET /finance/invoices
     */
    @Get('invoices')
    findAllInvoices() {
        return this.financeService.findAll();
    }

    /**
     * Create invoice
     * POST /finance/invoices
     */
    @Post('invoices')
    createInvoice(@Body() createInvoiceDto: CreateInvoiceDto) {
        return this.financeService.createInvoice(createInvoiceDto);
    }

    /**
     * Create invoice from milestone
     * POST /finance/invoices/from-milestone/:milestoneId
     */
    @Post('invoices/from-milestone/:milestoneId')
    createInvoiceFromMilestone(
        @Param('milestoneId') milestoneId: string,
        @Body() dto: Partial<CreateInvoiceDto>,
    ) {
        return this.financeService.createInvoiceFromMilestone(milestoneId, dto);
    }

    /**
     * Get invoice by ID
     * GET /finance/invoices/:id
     */
    @Get('invoices/:id')
    findOneInvoice(@Param('id') id: string) {
        return this.financeService.findOne(id);
    }

    /**
     * Update invoice status
     * PATCH /finance/invoices/:id/status
     */
    @Patch('invoices/:id/status')
    updateInvoiceStatus(
        @Param('id') id: string,
        @Body('status') status: InvoiceStatus,
    ) {
        return this.financeService.updateStatus(id, status);
    }

    /**
     * Create payment
     * POST /finance/payments
     */
    @Post('payments')
    createPayment(@Body() createPaymentDto: CreatePaymentDto) {
        return this.financeService.createPayment(createPaymentDto);
    }

    /**
     * Get all milestone templates
     * GET /finance/milestone-templates
     */
    @Get('milestone-templates')
    findAllTemplates() {
        return this.financeService.findAllTemplates();
    }

    /**
     * Create milestone template
     * POST /finance/milestone-templates
     */
    @Post('milestone-templates')
    createTemplate(@Body() dto: any) {
        return this.financeService.createMilestoneTemplate(dto);
    }

    /**
     * Get template by ID
     * GET /finance/milestone-templates/:id
     */
    @Get('milestone-templates/:id')
    findOneTemplate(@Param('id') id: string) {
        return this.financeService.findOneTemplate(id);
    }

    /**
     * Get milestone by ID
     * GET /finance/milestones/:id
     */
    @Get('milestones/:id')
    findOneMilestone(@Param('id') id: string) {
        return this.financeService.findOneMilestone(id);
    }

    /**
     * Update template
     * PATCH /finance/milestone-templates/:id
     */
    @Patch('milestone-templates/:id')
    updateTemplate(@Param('id') id: string, @Body() dto: any) {
        return this.financeService.updateTemplate(id, dto);
    }

    /**
     * Upload invoice receipt
     * POST /finance/invoices/:id/receipt
     */
    @Post('invoices/:id/receipt')
    @UseInterceptors(FileInterceptor('file', {
        storage: diskStorage({
            destination: './uploads/invoices',
            filename: (req, file, cb) => {
                const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
                cb(null, uniqueSuffix + extname(file.originalname));
            },
        }),
    }))
    uploadReceipt(
        @Param('id') id: string,
        @UploadedFile() file: Express.Multer.File,
    ) {
        if (!file) throw new Error("File upload failed");

        // Decode Chinese filename
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');

        return this.financeService.uploadReceipt(id, {
            ...file,
            filename: decodedFilename,
        });
    }
}
