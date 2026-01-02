import { Controller, Get, Post, Body, Patch, Param, Delete, Query, UseInterceptors, UploadedFile, UseGuards, Request } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { extname } from 'path';
import { ProcurementsService } from './procurements.service';
import { CreateProcurementDto } from './dto/create-procurement.dto';
import { UpdateProcurementDto } from './dto/update-procurement.dto';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';

import { existsSync, mkdirSync } from 'fs';

const uploadsDir = './uploads/procurements';
if (!existsSync(uploadsDir)) {
    mkdirSync(uploadsDir, { recursive: true });
}

@Controller('procurements')
export class ProcurementsController {
    constructor(private readonly procurementsService: ProcurementsService) { }

    @Post()
    create(@Body() createProcurementDto: CreateProcurementDto) {
        return this.procurementsService.create(createProcurementDto);
    }

    @Get()
    findAll(@Query('opportunityId') opportunityId?: string) {
        if (opportunityId) {
            return this.procurementsService.findByOpportunity(opportunityId);
        }
        return this.procurementsService.findAll();
    }

    @Get('active')
    findActive() {
        return this.procurementsService.findActive();
    }

    @Get(':id')
    findOne(@Param('id') id: string) {
        return this.procurementsService.findOne(id);
    }

    @Patch(':id')
    update(@Param('id') id: string, @Body() updateProcurementDto: UpdateProcurementDto) {
        return this.procurementsService.update(id, updateProcurementDto);
    }

    @Delete(':id')
    remove(@Param('id') id: string) {
        return this.procurementsService.remove(id);
    }

    @Patch('tasks/:taskId')
    updateTask(
        @Param('taskId') taskId: string,
        @Body() body: { isCompleted?: boolean; assignee?: string }
    ) {
        return this.procurementsService.updateTask(taskId, body.isCompleted ?? false, body.assignee);
    }

    @Post(':id/documents')
    @UseGuards(JwtAuthGuard)
    @UseInterceptors(FileInterceptor('file', {
        storage: diskStorage({
            destination: './uploads/procurements',
            filename: (req, file, cb) => {
                const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
                cb(null, uniqueSuffix + extname(file.originalname));
            },
        }),
    }))
    async uploadDocument(
        @Param('id') procurementId: string,
        @UploadedFile() file: Express.Multer.File,
        @Body('docType') docType: string,
        @Request() req,
    ) {
        // Decode Chinese filename properly
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');

        return this.procurementsService.addDocument(procurementId, {
            docType: docType || 'other',
            filename: decodedFilename,
            filepath: file.path,
            mimetype: file.mimetype,
            size: file.size,
        }, req.user);
    }

    @Get(':id/documents')
    getDocuments(@Param('id') id: string) {
        return this.procurementsService.getDocuments(id);
    }
}

