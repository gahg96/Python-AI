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
exports.UpdateContractDto = void 0;
const mapped_types_1 = require("@nestjs/mapped-types");
const create_contract_dto_1 = require("./create-contract.dto");
const class_validator_1 = require("class-validator");
const client_1 = require("@prisma/client");
class UpdateContractDto extends (0, mapped_types_1.PartialType)(create_contract_dto_1.CreateContractDto) {
    status;
    approverId;
    isActive;
}
exports.UpdateContractDto = UpdateContractDto;
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsEnum)(client_1.ContractStatus),
    __metadata("design:type", String)
], UpdateContractDto.prototype, "status", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], UpdateContractDto.prototype, "approverId", void 0);
__decorate([
    (0, class_validator_1.IsOptional)(),
    (0, class_validator_1.IsBoolean)(),
    __metadata("design:type", Boolean)
], UpdateContractDto.prototype, "isActive", void 0);
//# sourceMappingURL=update-contract.dto.js.map