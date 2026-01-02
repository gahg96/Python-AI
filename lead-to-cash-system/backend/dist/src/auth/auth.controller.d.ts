import { AuthService } from './auth.service';
export declare class AuthController {
    private authService;
    constructor(authService: AuthService);
    login(req: any): Promise<{
        access_token: string;
        user: {
            username: any;
            sub: any;
            role: any;
            name: any;
        };
    }>;
    getProfile(req: any): any;
}
