import { Module } from '@nestjs/common';
import { ProcurementsController } from './procurements.controller';
import { ProcurementsService } from './procurements.service';
import { PrismaModule } from '../prisma/prisma.module';

@Module({
    imports: [PrismaModule],
    controllers: [ProcurementsController],
    providers: [ProcurementsService],
    exports: [ProcurementsService],
})
export class ProcurementsModule { }
