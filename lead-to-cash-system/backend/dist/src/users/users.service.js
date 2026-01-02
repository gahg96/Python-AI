"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UsersService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
const bcrypt = __importStar(require("bcrypt"));
let UsersService = class UsersService {
    prisma;
    constructor(prisma) {
        this.prisma = prisma;
    }
    async onModuleInit() {
        try {
            const admin = await this.prisma.user.findFirst({ where: { role: 'ADMIN' } });
            if (!admin) {
                console.log('Seeding default admin user...');
                await this.create({
                    username: 'admin',
                    passwordHash: 'admin123',
                    displayName: 'System Admin',
                    role: 'ADMIN',
                });
            }
            const staffToSeed = [
                { username: 'ceo', display: '张总 (CEO)', role: 'MANAGER' },
                { username: 'sales', display: '销售王 (Sales)', role: 'SALES' },
                { username: 'commerce', display: '商务李 (商务)', role: 'COMMERCIAL' },
                { username: 'approver', display: '审批陈 (合同审批人)', role: 'MANAGER' },
                { username: 'drafter', display: '拟稿赵 (合同拟稿)', role: 'USER' },
                { username: 'pm_zhou', display: '项目经理周 (PM)', role: 'MANAGER' },
                { username: 'dev_han', display: '开发韩 (软件开发工程师)', role: 'DEVELOPER' },
                { username: 'qa_wu', display: '测试吴 (软件测试工程师)', role: 'TECHNICAL' },
                { username: 'architect_feng', display: '架构冯 (架构师)', role: 'TECHNICAL' },
                { username: 'ai_shen', display: 'AI沈 (AI工程师)', role: 'TECHNICAL' },
            ];
            for (const staff of staffToSeed) {
                try {
                    const existing = await this.prisma.user.findUnique({ where: { username: staff.username } });
                    if (!existing) {
                        console.log(`Seeding user: ${staff.username}`);
                        await this.create({
                            username: staff.username,
                            passwordHash: 'user123',
                            displayName: staff.display,
                            role: staff.role,
                        });
                    }
                }
                catch (err) {
                    console.error(`Failed to seed user ${staff.username}:`, err);
                }
            }
        }
        catch (globalError) {
            console.error('UsersService.onModuleInit failed:', globalError);
        }
    }
    async findOne(username) {
        return this.prisma.user.findUnique({ where: { username } });
    }
    async findById(id) {
        return this.prisma.user.findUnique({ where: { id } });
    }
    async create(data) {
        const salt = await bcrypt.genSalt();
        const hash = await bcrypt.hash(data.passwordHash, salt);
        return this.prisma.user.create({
            data: {
                ...data,
                passwordHash: hash,
            }
        });
    }
    async findAll() {
        try {
            const users = await this.prisma.user.findMany();
            return JSON.parse(JSON.stringify(users));
        }
        catch (error) {
            console.error('UsersService.findAll error:', error);
            throw error;
        }
    }
};
exports.UsersService = UsersService;
exports.UsersService = UsersService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], UsersService);
//# sourceMappingURL=users.service.js.map