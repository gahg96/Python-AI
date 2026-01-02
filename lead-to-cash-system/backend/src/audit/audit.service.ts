import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class AuditService {
    constructor(private prisma: PrismaService) { }

    async log(userId: string, action: string, tableName: string, recordId: string, details?: any) {
        try {
            await this.prisma.auditLog.create({
                data: {
                    userId,
                    action,
                    tableName,
                    recordId,
                    details: details ? JSON.stringify(details) : undefined,
                }
            });
        } catch (e) {
            console.error('Failed to create audit log', e);
        }
    }
}
