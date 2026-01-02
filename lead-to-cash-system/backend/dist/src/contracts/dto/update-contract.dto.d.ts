import { CreateContractDto } from './create-contract.dto';
import { ContractStatus } from '@prisma/client';
declare const UpdateContractDto_base: import("@nestjs/mapped-types").MappedType<Partial<CreateContractDto>>;
export declare class UpdateContractDto extends UpdateContractDto_base {
    status?: ContractStatus;
    approverId?: string;
    isActive?: boolean;
}
export {};
