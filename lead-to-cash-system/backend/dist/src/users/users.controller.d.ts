import { UsersService } from './users.service';
export declare class UsersController {
    private readonly usersService;
    constructor(usersService: UsersService);
    findAll(): Promise<{
        id: string;
        createdAt: Date;
        username: string;
        passwordHash: string;
        displayName: string;
        role: import("@prisma/client").$Enums.UserRole;
        email: string | null;
    }[]>;
}
