import { Injectable, OnModuleInit } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { User, Prisma } from '@prisma/client';
import * as bcrypt from 'bcrypt';

@Injectable()
export class UsersService implements OnModuleInit {
    constructor(private prisma: PrismaService) { }

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
                            role: staff.role as any,
                        });
                    }
                } catch (err) {
                    console.error(`Failed to seed user ${staff.username}:`, err);
                }
            }
        } catch (globalError) {
            console.error('UsersService.onModuleInit failed:', globalError);
        }
    }

    async findOne(username: string): Promise<User | null> {
        return this.prisma.user.findUnique({ where: { username } });
    }

    async findById(id: string): Promise<User | null> {
        return this.prisma.user.findUnique({ where: { id } });
    }

    async create(data: Prisma.UserCreateInput): Promise<User> {
        // Assuming data.passwordHash contains the RAW password here for simplicity in initial call
        // In a real DTO we would separate them
        const salt = await bcrypt.genSalt();
        const hash = await bcrypt.hash(data.passwordHash, salt);
        return this.prisma.user.create({
            data: {
                ...data,
                passwordHash: hash,
            }
        });
    }

    async findAll(): Promise<User[]> {
        try {
            const users = await this.prisma.user.findMany();
            // Ensure plain objects for serialization stability
            return JSON.parse(JSON.stringify(users));
        } catch (error) {
            console.error('UsersService.findAll error:', error);
            throw error;
        }
    }
}
