import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { CustomersModule } from './customers/customers.module';
import { OpportunitiesModule } from './opportunities/opportunities.module';
import { ContractsModule } from './contracts/contracts.module';
import { MilestonesModule } from './milestones/milestones.module';
import { ProcurementsModule } from './procurements/procurements.module';

import { PrismaModule } from './prisma/prisma.module';

import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { AuditModule } from './audit/audit.module';
import { DashboardModule } from './dashboard/dashboard.module';
import { ProjectsModule } from './projects/projects.module';
import { FinanceModule } from './finance/finance.module';

@Module({
  imports: [PrismaModule, CustomersModule, OpportunitiesModule, ContractsModule, MilestonesModule, ProcurementsModule, AuthModule, UsersModule, AuditModule, DashboardModule, ProjectsModule, FinanceModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule { }
