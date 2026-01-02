import { PrismaService } from '../prisma/prisma.service';
export declare class AuditService {
    private prisma;
    constructor(prisma: PrismaService);
    log(userId: string, action: string, tableName: string, recordId: string, details?: any): Promise<void>;
}
