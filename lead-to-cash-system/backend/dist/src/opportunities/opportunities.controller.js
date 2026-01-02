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
exports.OpportunitiesController = void 0;
const common_1 = require("@nestjs/common");
const platform_express_1 = require("@nestjs/platform-express");
const multer_1 = require("multer");
const path_1 = require("path");
const opportunities_service_1 = require("./opportunities.service");
const create_opportunity_dto_1 = require("./dto/create-opportunity.dto");
const create_follow_up_dto_1 = require("./dto/create-follow-up.dto");
const update_opportunity_dto_1 = require("./dto/update-opportunity.dto");
const fs_1 = require("fs");
const jwt_auth_guard_1 = require("../auth/jwt-auth.guard");
const uploadsDir = './uploads/opportunities';
if (!(0, fs_1.existsSync)(uploadsDir)) {
    (0, fs_1.mkdirSync)(uploadsDir, { recursive: true });
}
let OpportunitiesController = class OpportunitiesController {
    opportunitiesService;
    constructor(opportunitiesService) {
        this.opportunitiesService = opportunitiesService;
    }
    create(createOpportunityDto, req) {
        return this.opportunitiesService.create(createOpportunityDto);
    }
    findAll() {
        return this.opportunitiesService.findAll();
    }
    findOne(id) {
        return this.opportunitiesService.findOne(id);
    }
    update(id, updateOpportunityDto) {
        return this.opportunitiesService.update(id, updateOpportunityDto);
    }
    createFollowUp(id, dto, req) {
        return this.opportunitiesService.createFollowUp(id, dto, req.user);
    }
    getFollowUps(id) {
        return this.opportunitiesService.getFollowUps(id);
    }
    async uploadAttachment(id, file) {
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');
        return this.opportunitiesService.createAttachment(id, {
            filename: decodedFilename,
            filepath: file.filename,
            mimetype: file.mimetype,
            size: file.size,
        });
    }
    getAttachments(id) {
        return this.opportunitiesService.getAttachments(id);
    }
    downloadFile(filename, res) {
        const filePath = (0, path_1.join)(uploadsDir, filename);
        return res.sendFile(filePath);
    }
};
exports.OpportunitiesController = OpportunitiesController;
__decorate([
    (0, common_1.Post)(),
    (0, common_1.UseGuards)(jwt_auth_guard_1.JwtAuthGuard),
    __param(0, (0, common_1.Body)()),
    __param(1, (0, common_1.Request)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [create_opportunity_dto_1.CreateOpportunityDto, Object]),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "create", null);
__decorate([
    (0, common_1.Get)(),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "findAll", null);
__decorate([
    (0, common_1.Get)(':id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "findOne", null);
__decorate([
    (0, common_1.Patch)(':id'),
    (0, common_1.UseGuards)(jwt_auth_guard_1.JwtAuthGuard),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, update_opportunity_dto_1.UpdateOpportunityDto]),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "update", null);
__decorate([
    (0, common_1.Post)(':id/follow-ups'),
    (0, common_1.UseGuards)(jwt_auth_guard_1.JwtAuthGuard),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.Body)()),
    __param(2, (0, common_1.Request)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, create_follow_up_dto_1.CreateFollowUpDto, Object]),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "createFollowUp", null);
__decorate([
    (0, common_1.Get)(':id/follow-ups'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "getFollowUps", null);
__decorate([
    (0, common_1.Post)(':id/attachments'),
    (0, common_1.UseInterceptors)((0, platform_express_1.FileInterceptor)('file', {
        storage: (0, multer_1.diskStorage)({
            destination: uploadsDir,
            filename: (req, file, cb) => {
                const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
                cb(null, uniqueSuffix + (0, path_1.extname)(file.originalname));
            },
        }),
    })),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.UploadedFile)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object]),
    __metadata("design:returntype", Promise)
], OpportunitiesController.prototype, "uploadAttachment", null);
__decorate([
    (0, common_1.Get)(':id/attachments'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "getAttachments", null);
__decorate([
    (0, common_1.Get)('attachments/download/:filename'),
    __param(0, (0, common_1.Param)('filename')),
    __param(1, (0, common_1.Res)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object]),
    __metadata("design:returntype", void 0)
], OpportunitiesController.prototype, "downloadFile", null);
exports.OpportunitiesController = OpportunitiesController = __decorate([
    (0, common_1.Controller)('opportunities'),
    __metadata("design:paramtypes", [opportunities_service_1.OpportunitiesService])
], OpportunitiesController);
//# sourceMappingURL=opportunities.controller.js.map