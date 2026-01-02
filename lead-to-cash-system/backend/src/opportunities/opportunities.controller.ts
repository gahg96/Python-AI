import { Controller, Get, Post, Patch, Body, Param, UseInterceptors, UploadedFile, Res, UseGuards, Request } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { extname, join } from 'path';
import type { Response } from 'express';
import { OpportunitiesService } from './opportunities.service';
import { CreateOpportunityDto } from './dto/create-opportunity.dto';
import { CreateFollowUpDto } from './dto/create-follow-up.dto';
import { UpdateOpportunityDto } from './dto/update-opportunity.dto';
import { existsSync, mkdirSync } from 'fs';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';

// Ensure uploads directory exists - use project root/uploads/opportunities
const uploadsDir = './uploads/opportunities';
if (!existsSync(uploadsDir)) {
    mkdirSync(uploadsDir, { recursive: true });
}

@Controller('opportunities')
export class OpportunitiesController {
    constructor(private readonly opportunitiesService: OpportunitiesService) { }

    @Post()
    @UseGuards(JwtAuthGuard)
    create(@Body() createOpportunityDto: CreateOpportunityDto, @Request() req) { // TODO: Add user attribution to Opp creation too?
        return this.opportunitiesService.create(createOpportunityDto);
    }

    @Get()
    findAll() {
        return this.opportunitiesService.findAll();
    }

    @Get(':id')
    findOne(@Param('id') id: string) {
        return this.opportunitiesService.findOne(id);
    }

    @Patch(':id')
    @UseGuards(JwtAuthGuard)
    update(@Param('id') id: string, @Body() updateOpportunityDto: UpdateOpportunityDto) {
        return this.opportunitiesService.update(id, updateOpportunityDto);
    }

    // FollowUp Endpoints
    @Post(':id/follow-ups')
    @UseGuards(JwtAuthGuard)
    createFollowUp(@Param('id') id: string, @Body() dto: CreateFollowUpDto, @Request() req) {
        return this.opportunitiesService.createFollowUp(id, dto, req.user);
    }

    @Get(':id/follow-ups')
    getFollowUps(@Param('id') id: string) {
        return this.opportunitiesService.getFollowUps(id);
    }

    // Attachment Endpoints
    @Post(':id/attachments')
    @UseInterceptors(FileInterceptor('file', {
        storage: diskStorage({
            destination: uploadsDir,
            filename: (req, file, cb) => {
                const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
                cb(null, uniqueSuffix + extname(file.originalname));
            },
        }),
    }))
    async uploadAttachment(@Param('id') id: string, @UploadedFile() file: Express.Multer.File) {
        // Fix Chinese filename encoding: Multer encodes as latin1, need to convert to UTF-8
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');
        return this.opportunitiesService.createAttachment(id, {
            filename: decodedFilename,
            filepath: file.filename, // Just the filename, not the full path
            mimetype: file.mimetype,
            size: file.size,
        });
    }

    @Get(':id/attachments')
    getAttachments(@Param('id') id: string) {
        return this.opportunitiesService.getAttachments(id);
    }

    // Serve uploaded files
    @Get('attachments/download/:filename')
    downloadFile(@Param('filename') filename: string, @Res() res: Response) {
        const filePath = join(uploadsDir, filename);
        return res.sendFile(filePath);
    }
}
