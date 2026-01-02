import { CustomersService } from './customers.service';
import { CreateCustomerDto } from './dto/create-customer.dto';
import { UpdateCustomerDto } from './dto/update-customer.dto';
export declare class CustomersController {
    private readonly customersService;
    constructor(customersService: CustomersService);
    create(createCustomerDto: CreateCustomerDto): Promise<{
        id: string;
        createdAt: Date;
        companyName: string;
        industry: string | null;
        companySize: string | null;
        city: string | null;
        country: string | null;
        contactName: string | null;
        contactTitle: string | null;
        contactPhone: string | null;
        contactEmail: string | null;
    }>;
    findAll(): Promise<{
        id: string;
        createdAt: Date;
        companyName: string;
        industry: string | null;
        companySize: string | null;
        city: string | null;
        country: string | null;
        contactName: string | null;
        contactTitle: string | null;
        contactPhone: string | null;
        contactEmail: string | null;
    }[]>;
    findOne(id: string): Promise<{
        id: string;
        createdAt: Date;
        companyName: string;
        industry: string | null;
        companySize: string | null;
        city: string | null;
        country: string | null;
        contactName: string | null;
        contactTitle: string | null;
        contactPhone: string | null;
        contactEmail: string | null;
    } | null>;
    update(id: string, updateCustomerDto: UpdateCustomerDto): Promise<{
        id: string;
        createdAt: Date;
        companyName: string;
        industry: string | null;
        companySize: string | null;
        city: string | null;
        country: string | null;
        contactName: string | null;
        contactTitle: string | null;
        contactPhone: string | null;
        contactEmail: string | null;
    }>;
}
