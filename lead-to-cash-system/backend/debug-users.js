const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
    try {
        const users = await prisma.user.findMany();
        console.log('Users found:', users.length);
        console.log('Sample user:', JSON.stringify(users[0], null, 2));
    } catch (e) {
        console.error('Error in findMany:', e);
    }
}

main()
    .catch((e) => console.error(e))
    .finally(async () => {
        await prisma.$disconnect();
    });
