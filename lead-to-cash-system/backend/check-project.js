const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
    const project = await prisma.project.findFirst({
        include: {
            contract: true
        }
    });
    console.log('Project details:', JSON.stringify(project, null, 2));
}

main()
    .catch((e) => console.error(e))
    .finally(async () => {
        await prisma.$disconnect();
    });
