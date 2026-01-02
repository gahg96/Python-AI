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
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.FinanceController = void 0;
const common_1 = require("@nestjs/common");
const platform_express_1 = require("@nestjs/platform-express");
const multer_1 = require("multer");
const path_1 = require("path");
const finance_service_1 = require("./finance.service");
const create_invoice_dto_1 = require("./dto/create-invoice.dto");
const create_payment_dto_1 = require("./dto/create-payment.dto");
const jwt_auth_guard_1 = require("../auth/jwt-auth.guard");
const client_1 = require("@prisma/client");
let FinanceController = class FinanceController {
    financeService;
    constructor(financeService) {
        this.financeService = financeService;
        const fs = require('fs');
        if (!fs.existsSync('./uploads/invoices')) {
            fs.mkdirSync('./uploads/invoices', { recursive: true });
        }
    }
    getDashboard() {
        return this.financeService.getDashboardData();
    }
    findAllInvoices() {
        return this.financeService.findAll();
    }
    createInvoice(createInvoiceDto) {
        return this.financeService.createInvoice(createInvoiceDto);
    }
    createInvoiceFromMilestone(milestoneId, dto) {
        return this.financeService.createInvoiceFromMilestone(milestoneId, dto);
    }
    findOneInvoice(id) {
        return this.financeService.findOne(id);
    }
    updateInvoiceStatus(id, status) {
        return this.financeService.updateStatus(id, status);
    }
    createPayment(createPaymentDto) {
        return this.financeService.createPayment(createPaymentDto);
    }
    findAllTemplates() {
        return this.financeService.findAllTemplates();
    }
    createTemplate(dto) {
        return this.financeService.createMilestoneTemplate(dto);
    }
    findOneTemplate(id) {
        return this.financeService.findOneTemplate(id);
    }
    findOneMilestone(id) {
        return this.financeService.findOneMilestone(id);
    }
    updateTemplate(id, dto) {
        return this.financeService.updateTemplate(id, dto);
    }
    uploadReceipt(id, file) {
        if (!file)
            throw new Error("File upload failed");
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');
        return this.financeService.uploadReceipt(id, {
            ...file,
            filename: decodedFilename,
        });
    }
};
exports.FinanceController = FinanceController;
__decorate([
    (0, common_1.Get)('dashboard'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "getDashboard", null);
__decorate([
    (0, common_1.Get)('invoices'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "findAllInvoices", null);
__decorate([
    (0, common_1.Post)('invoices'),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [create_invoice_dto_1.CreateInvoiceDto]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "createInvoice", null);
__decorate([
    (0, common_1.Post)('invoices/from-milestone/:milestoneId'),
    __param(0, (0, common_1.Param)('milestoneId')),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "createInvoiceFromMilestone", null);
__decorate([
    (0, common_1.Get)('invoices/:id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "findOneInvoice", null);
__decorate([
    (0, common_1.Patch)('invoices/:id/status'),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.Body)('status')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, String]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "updateInvoiceStatus", null);
__decorate([
    (0, common_1.Post)('payments'),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [create_payment_dto_1.CreatePaymentDto]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "createPayment", null);
__decorate([
    (0, common_1.Get)('milestone-templates'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "findAllTemplates", null);
__decorate([
    (0, common_1.Post)('milestone-templates'),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "createTemplate", null);
__decorate([
    (0, common_1.Get)('milestone-templates/:id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "findOneTemplate", null);
__decorate([
    (0, common_1.Get)('milestones/:id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "findOneMilestone", null);
__decorate([
    (0, common_1.Patch)('milestone-templates/:id'),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "updateTemplate", null);
__decorate([
    (0, common_1.Post)('invoices/:id/receipt'),
    (0, common_1.UseInterceptors)((0, platform_express_1.FileInterceptor)('file', {
        storage: (0, multer_1.diskStorage)({
            destination: './uploads/invoices',
            filename: (req, file, cb) => {
                const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
                cb(null, uniqueSuffix + (0, path_1.extname)(file.originalname));
            },
        }),
    })),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.UploadedFile)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object]),
    __metadata("design:returntype", void 0)
], FinanceController.prototype, "uploadReceipt", null);
exports.FinanceController = FinanceController = __decorate([
    (0, common_1.Controller)('finance'),
    (0, common_1.UseGuards)(jwt_auth_guard_1.JwtAuthGuard),
    __metadata("design:paramtypes", [finance_service_1.FinanceService])
], FinanceController);
//# sourceMappingURL=finance.controller.js.map