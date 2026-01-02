"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const client_1 = require("@prisma/client");
const prisma = new client_1.PrismaClient();
async function main() {
    try {
        console.log('Testing connection...');
        const count = await prisma.customer.count();
        console.log('Customer count:', count);
        console.log('Creating customer...');
        const customer = await prisma.customer.create({
            data: {
                companyName: 'Debug Customer',
                city: 'Debug City'
            }
        });
        console.log('Created:', customer);
    }
    catch (e) {
        console.error('Error:', e);
    }
    finally {
        await prisma.$disconnect();
    }
}
main();
//# sourceMappingURL=debug_prisma.js.map