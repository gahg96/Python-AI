import { Controller, Get, Post, Body, Patch, Param, Delete, UseGuards, UseInterceptors, UploadedFile, BadRequestException } from '@nestjs/common';
import { ProjectsService } from './projects.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { UpdateProjectDto } from './dto/update-project.dto';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { extname } from 'path';
import * as fs from 'fs';

@Controller('projects')
@UseGuards(JwtAuthGuard)
export class ProjectsController {
    constructor(private readonly projectsService: ProjectsService) { }

    @Post()
    create(@Body() data: { contractId: string;[key: string]: any }) {
        const { contractId, ...rest } = data;
        return this.projectsService.create(contractId, rest);
    }

    @Get()
    findAll() {
        return this.projectsService.findAll();
    }

    @Get(':id')
    findOne(@Param('id') id: string) {
        return this.projectsService.findOne(id);
    }

    @Patch(':id')
    update(@Param('id') id: string, @Body() data: UpdateProjectDto) {
        return this.projectsService.update(id, data);
    }

    // Resources
    @Post(':id/resources')
    addResource(@Param('id') id: string, @Body() data: any) {
        return this.projectsService.addResource(id, data);
    }

    @Delete('resources/:rid')
    removeResource(@Param('rid') rid: string) {
        return this.projectsService.removeResource(rid);
    }

    // Meetings
    @Post(':id/meetings')
    createMeeting(@Param('id') id: string, @Body() data: any) {
        return this.projectsService.createMeeting(id, data);
    }

    @Patch('meetings/:mid')
    updateMeeting(@Param('mid') mid: string, @Body() data: any) {
        return this.projectsService.updateMeeting(mid, data);
    }

    @Post('meetings/:mid/upload')
    @UseInterceptors(FileInterceptor('file', {
        storage: diskStorage({
            destination: (req, file, cb) => {
                const uploadPath = './uploads/projects/meetings';
                if (!fs.existsSync(uploadPath)) {
                    fs.mkdirSync(uploadPath, { recursive: true });
                }
                cb(null, uploadPath);
            },
            filename: (req, file, cb) => {
                const randomName = Array(32).fill(null).map(() => (Math.round(Math.random() * 16)).toString(16)).join('');
                cb(null, `${randomName}${extname(file.originalname)}`);
            }
        })
    }))
    uploadMeetingMinutes(@Param('mid') mid: string, @UploadedFile() file: Express.Multer.File) {
        if (!file) {
            throw new BadRequestException('File is required');
        }
        return this.projectsService.addMeetingAttachment(mid, file);
    }

    @Delete('meetings/:mid')
    deleteMeeting(@Param('mid') mid: string) {
        return this.projectsService.deleteMeeting(mid);
    }

    // Risks
    @Post(':id/risks')
    addRisk(@Param('id') id: string, @Body() data: any) {
        return this.projectsService.addRisk(id, data);
    }

    @Patch('risks/:rid')
    updateRisk(@Param('rid') rid: string, @Body() data: any) {
        return this.projectsService.updateRisk(rid, data);
    }
}
