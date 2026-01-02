import { Controller, Get, UseGuards } from '@nestjs/common';
import { DashboardService } from './dashboard.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';

@Controller('dashboard')
@UseGuards(JwtAuthGuard)
export class DashboardController {
  constructor(private readonly dashboardService: DashboardService) { }

  @Get('stats')
  getStats() {
    return this.dashboardService.getStats();
  }

  @Get('funnel')
  getFunnel() {
    return this.dashboardService.getFunnel();
  }

  @Get('trend')
  getTrend() {
    return this.dashboardService.getTrend();
  }

  @Get('fix-dates')
  fixDates() {
    return this.dashboardService.fixDates();
  }

  @Get('geo')
  getGeo() {
    return this.dashboardService.getGeoDistribution();
  }
}

