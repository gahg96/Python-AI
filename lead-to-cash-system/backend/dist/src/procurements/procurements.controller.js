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
exports.ProcurementsController = void 0;
const common_1 = require("@nestjs/common");
const platform_express_1 = require("@nestjs/platform-express");
const multer_1 = require("multer");
const path_1 = require("path");
const procurements_service_1 = require("./procurements.service");
const create_procurement_dto_1 = require("./dto/create-procurement.dto");
const update_procurement_dto_1 = require("./dto/update-procurement.dto");
const jwt_auth_guard_1 = require("../auth/jwt-auth.guard");
const fs_1 = require("fs");
const uploadsDir = './uploads/procurements';
if (!(0, fs_1.existsSync)(uploadsDir)) {
    (0, fs_1.mkdirSync)(uploadsDir, { recursive: true });
}
let ProcurementsController = class ProcurementsController {
    procurementsService;
    constructor(procurementsService) {
        this.procurementsService = procurementsService;
    }
    create(createProcurementDto) {
        return this.procurementsService.create(createProcurementDto);
    }
    findAll(opportunityId) {
        if (opportunityId) {
            return this.procurementsService.findByOpportunity(opportunityId);
        }
        return this.procurementsService.findAll();
    }
    findActive() {
        return this.procurementsService.findActive();
    }
    findOne(id) {
        return this.procurementsService.findOne(id);
    }
    update(id, updateProcurementDto) {
        return this.procurementsService.update(id, updateProcurementDto);
    }
    remove(id) {
        return this.procurementsService.remove(id);
    }
    updateTask(taskId, body) {
        return this.procurementsService.updateTask(taskId, body.isCompleted ?? false, body.assignee);
    }
    async uploadDocument(procurementId, file, docType, req) {
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');
        return this.procurementsService.addDocument(procurementId, {
            docType: docType || 'other',
            filename: decodedFilename,
            filepath: file.path,
            mimetype: file.mimetype,
            size: file.size,
        }, req.user);
    }
    getDocuments(id) {
        return this.procurementsService.getDocuments(id);
    }
};
exports.ProcurementsController = ProcurementsController;
__decorate([
    (0, common_1.Post)(),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [create_procurement_dto_1.CreateProcurementDto]),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "create", null);
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, common_1.Query)('opportunityId')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "findAll", null);
__decorate([
    (0, common_1.Get)('active'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "findActive", null);
__decorate([
    (0, common_1.Get)(':id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "findOne", null);
__decorate([
    (0, common_1.Patch)(':id'),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, update_procurement_dto_1.UpdateProcurementDto]),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "update", null);
__decorate([
    (0, common_1.Delete)(':id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "remove", null);
__decorate([
    (0, common_1.Patch)('tasks/:taskId'),
    __param(0, (0, common_1.Param)('taskId')),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object]),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "updateTask", null);
__decorate([
    (0, common_1.Post)(':id/documents'),
    (0, common_1.UseGuards)(jwt_auth_guard_1.JwtAuthGuard),
    (0, common_1.UseInterceptors)((0, platform_express_1.FileInterceptor)('file', {
        storage: (0, multer_1.diskStorage)({
            destination: './uploads/procurements',
            filename: (req, file, cb) => {
                const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
                cb(null, uniqueSuffix + (0, path_1.extname)(file.originalname));
            },
        }),
    })),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.UploadedFile)()),
    __param(2, (0, common_1.Body)('docType')),
    __param(3, (0, common_1.Request)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object, String, Object]),
    __metadata("design:returntype", Promise)
], ProcurementsController.prototype, "uploadDocument", null);
__decorate([
    (0, common_1.Get)(':id/documents'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], ProcurementsController.prototype, "getDocuments", null);
exports.ProcurementsController = ProcurementsController = __decorate([
    (0, common_1.Controller)('procurements'),
    __metadata("design:paramtypes", [procurements_service_1.ProcurementsService])
], ProcurementsController);
//# sourceMappingURL=procurements.controller.js.map