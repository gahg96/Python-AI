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
exports.CreateProcurementDto = exports.ProcurementType = void 0;
const class_validator_1 = require("class-validator");
var ProcurementType;
(function (ProcurementType) {
    ProcurementType["DirectQuote"] = "DirectQuote";
    ProcurementType["Negotiation"] = "Negotiation";
    ProcurementType["Comparison"] = "Comparison";
    ProcurementType["Consultation"] = "Consultation";
    ProcurementType["PublicTender"] = "PublicTender";
})(ProcurementType || (exports.ProcurementType = ProcurementType = {}));
class CreateProcurementDto {
    opportunityId;
    type;
    customerBudget;
    ourQuote;
    submissionDeadline;
    notificationDate;
    bidLocation;
    depositAmount;
    notes;
}
exports.CreateProcurementDto = CreateProcurementDto;
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], CreateProcurementDto.prototype, "opportunityId", void 0);
__decorate([
    (0, class_validator_1.IsEnum)(ProcurementType),
    __metadata("design:type", String)
], CreateProcurementDto.prototype, "type", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsNumber)(),
    __metadata("design:type", Number)
], CreateProcurementDto.prototype, "customerBudget", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsNumber)(),
    __metadata("design:type", Number)
], CreateProcurementDto.prototype, "ourQuote", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsDateString)(),
    __metadata("design:type", String)
], CreateProcurementDto.prototype, "submissionDeadline", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsDateString)(),
    __metadata("design:type", String)
], CreateProcurementDto.prototype, "notificationDate", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], CreateProcurementDto.prototype, "bidLocation", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsNumber)(),
    __metadata("design:type", Number)
], CreateProcurementDto.prototype, "depositAmount", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], CreateProcurementDto.prototype, "notes", void 0);
//# sourceMappingURL=create-procurement.dto.js.map