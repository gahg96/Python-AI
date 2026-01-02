import { Controller, Get, Post, Body, Patch, Param, Delete, UseGuards, Request, UseInterceptors, UploadedFile } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { extname } from 'path';
import { ContractsService } from './contracts.service';
import { CreateContractDto } from './dto/create-contract.dto';
import { UpdateContractDto } from './dto/update-contract.dto';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';

@Controller('contracts')
@UseGuards(JwtAuthGuard)
export class ContractsController {
    constructor(private readonly contractsService: ContractsService) {
        // Ensure upload directory exists
        const fs = require('fs');
        if (!fs.existsSync('./uploads/contracts')) {
            fs.mkdirSync('./uploads/contracts', { recursive: true });
        }
    }

    @Post()
    create(@Body() createContractDto: CreateContractDto, @Request() req) {
        return this.contractsService.create(createContractDto, req.user.userId);
    }

    @Get()
    findAll() {
        return this.contractsService.findAll();
    }

    @Get(':id')
    findOne(@Param('id') id: string) {
        return this.contractsService.findOne(id);
    }

    @Patch(':id')
    update(@Param('id') id: string, @Body() updateContractDto: UpdateContractDto) {
        return this.contractsService.update(id, updateContractDto);
    }

    // Approval Workflow Endpoints
    @Post(':id/submit')
    submit(@Param('id') id: string) {
        return this.contractsService.submitForApproval(id);
    }

    @Post(':id/approve')
    approve(@Param('id') id: string, @Request() req) {
        // TODO: Verify user role is MANAGER (handled by Guards later, simpler for now)
        return this.contractsService.approve(id, req.user.userId);
    }

    @Post(':id/reject')
    reject(@Param('id') id: string, @Request() req) {
        return this.contractsService.reject(id, req.user.userId);
    }

    @Post(':id/sign')
    sign(@Param('id') id: string) {
        return this.contractsService.sign(id);
    }

    @Delete(':id')
    remove(@Param('id') id: string) {
        return this.contractsService.remove(id);
    }

    // Document Upload
    @Post(':id/documents')
    @UseInterceptors(FileInterceptor('file', {
        storage: diskStorage({
            destination: './uploads/contracts',
            filename: (req, file, cb) => {
                // Generates a random filename to avoid conflicts and encoding issues
                const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
                cb(null, uniqueSuffix + extname(file.originalname));
            },
        }),
    }))
    async uploadDocument(
        @Param('id') contractId: string,
        @UploadedFile() file: Express.Multer.File,
        @Request() req,
    ) {
        // Handle potentially missing file
        if (!file) {
            throw new Error("File upload failed");
        }

        // Decode Chinese filename properly for display
        const decodedFilename = Buffer.from(file.originalname, 'latin1').toString('utf8');

        return this.contractsService.addDocument(contractId, {
            ...file,
            filename: decodedFilename,
        }, req.user.userId);
    }

    // Milestones
    @Post(':id/milestones')
    addMilestone(@Param('id') id: string, @Body() data: any) {
        return this.contractsService.addMilestone(id, data);
    }

    @Patch('milestones/:mid')
    updateMilestone(@Param('mid') mid: string, @Body() data: any) {
        return this.contractsService.updateMilestone(mid, data);
    }

    @Delete('milestones/:mid')
    deleteMilestone(@Param('mid') mid: string) {
        return this.contractsService.deleteMilestone(mid);
    }
}

