"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const client_1 = require("@prisma/client");
const prisma = new client_1.PrismaClient();
async function main() {
    console.log('Checking database state...');
    const userCount = await prisma.user.count();
    console.log(`Users: ${userCount}`);
    const customerCount = await prisma.customer.count();
    console.log(`Customers: ${customerCount}`);
    const oppCount = await prisma.opportunity.count();
    console.log(`Opportunities: ${oppCount}`);
    if (oppCount === 0) {
        console.log('No opportunities found. Seeding one sample...');
        let user = await prisma.user.findFirst();
        if (!user) {
        }
        let customer = await prisma.customer.findFirst();
        if (!customer) {
            customer = await prisma.customer.create({
                data: {
                    companyName: 'Acme Corp',
                    industry: 'Tech',
                    companySize: '100-500',
                    contactName: 'John Doe',
                    contactEmail: 'john@acme.com',
                    contactPhone: '1234567890'
                }
            });
            console.log('Created sample customer.');
        }
        const opp = await prisma.opportunity.create({
            data: {
                title: 'Cloud Migration Project',
                opportunityNumber: 'OPP-2025-0001',
                customerId: customer.id,
                status: 'New',
                estimatedValue: 500000,
                probability: 20,
                salesStage: 'initial_contact',
                source: 'website',
                expectedCloseDate: new Date('2025-12-31'),
            }
        });
        console.log(`Created sample opportunity: ${opp.title} (${opp.id})`);
    }
    else {
        console.log('Opportunities exist. Fetching first 3...');
        const opps = await prisma.opportunity.findMany({ take: 3 });
        console.log(JSON.stringify(opps, null, 2));
    }
}
main()
    .catch((e) => {
    console.error(e);
    process.exit(1);
})
    .finally(async () => {
    await prisma.$disconnect();
});
//# sourceMappingURL=debug_seed.js.map