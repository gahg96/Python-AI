const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcrypt');
const prisma = new PrismaClient();

async function main() {
    console.log('🚀 开始填充完整测试数据...\n');

    // 默认密码: adc1234
    const defaultPassword = await bcrypt.hash('adc1234', 10);

    // 1. 创建用户
    console.log('👥 创建用户...');
    const users = await Promise.all([
        prisma.user.upsert({ where: { username: 'admin' }, update: {}, create: { username: 'admin', passwordHash: defaultPassword, displayName: '系统管理员', role: 'ADMIN' } }),
        prisma.user.upsert({ where: { username: 'sales_wang' }, update: {}, create: { username: 'sales_wang', passwordHash: defaultPassword, displayName: '王销售', role: 'SALES' } }),
        prisma.user.upsert({ where: { username: 'pm_zhang' }, update: {}, create: { username: 'pm_zhang', passwordHash: defaultPassword, displayName: '张项目', role: 'MANAGER' } }),
        prisma.user.upsert({ where: { username: 'dev_li' }, update: {}, create: { username: 'dev_li', passwordHash: defaultPassword, displayName: '李开发', role: 'DEVELOPER' } }),
        prisma.user.upsert({ where: { username: 'finance_chen' }, update: {}, create: { username: 'finance_chen', passwordHash: defaultPassword, displayName: '陈财务', role: 'USER' } }),
    ]);
    console.log(`✓ ${users.length} 个用户\n`);

    // 2. 创建客户
    console.log('🏢 创建客户...');
    const customers = await Promise.all([
        prisma.customer.create({ data: { companyName: 'TechGiant Corp', industry: 'Technology', companySize: 'Large', contactName: '张总', contactTitle: 'CTO', contactPhone: '13800138000', contactEmail: 'zhang@techgiant.com' } }),
        prisma.customer.create({ data: { companyName: 'Rocket Startup', industry: 'Internet', companySize: 'Medium', contactName: '李总', contactTitle: 'CEO', contactPhone: '13900139000', contactEmail: 'li@rocket.com' } }),
        prisma.customer.create({ data: { companyName: 'Global Finance Ltd', industry: 'Finance', companySize: 'Large', contactName: '赵总', contactTitle: 'VP', contactPhone: '13700137000', contactEmail: 'zhao@globalfinance.com' } }),
    ]);
    console.log(`✓ ${customers.length} 个客户\n`);

    // 3. 创建商机
    console.log('💼 创建商机...');
    const opportunities = await Promise.all([
        prisma.opportunity.create({ data: { opportunityNumber: 'OPP-2024-00001', customerId: customers[0].id, title: 'Enterprise Software Development', estimatedValue: 500000, probability: 90, status: 'Won', expectedCloseDate: new Date('2024-03-31'), salesStage: 'Closed Won', salesOwner: '王销售' } }),
        prisma.opportunity.create({ data: { opportunityNumber: 'OPP-2024-00002', customerId: customers[1].id, title: 'Cloud Platform Migration', estimatedValue: 300000, probability: 70, status: 'Negotiation', expectedCloseDate: new Date('2024-06-30'), salesStage: 'Negotiation', salesOwner: '王销售' } }),
        prisma.opportunity.create({ data: { opportunityNumber: 'OPP-2024-00003', customerId: customers[2].id, title: 'Financial System Upgrade', estimatedValue: 800000, probability: 60, status: 'Proposal', expectedCloseDate: new Date('2024-09-30'), salesStage: 'Proposal', salesOwner: '王销售' } }),
    ]);
    console.log(`✓ ${opportunities.length} 个商机\n`);

    // 4. 创建招投标
    console.log('📋 创建招投标...');
    const procurements = await Promise.all([
        prisma.procurement.create({
            data: {
                procurementNumber: 'BID-2024-00001',
                type: 'PublicTender',
                opportunityId: opportunities[0].id,
                customerBudget: 500000,
                status: 'Won',
                submissionDeadline: new Date('2024-02-28'),
                tasks: {
                    create: [
                        { name: '技术方案编写', assignee: '张项目', isCompleted: true },
                        { name: '商务报价', assignee: '王销售', isCompleted: true }
                    ]
                }
            }
        }),
        prisma.procurement.create({
            data: {
                procurementNumber: 'BID-2024-00002',
                type: 'Negotiation',
                opportunityId: opportunities[1].id,
                customerBudget: 300000,
                status: 'Preparing',
                submissionDeadline: new Date('2024-05-31'),
                tasks: {
                    create: [
                        { name: '需求分析', assignee: '张项目', isCompleted: false }
                    ]
                }
            }
        }),
    ]);
    console.log(`✓ ${procurements.length} 个招投标\n`);

    // 5. 创建合同
    console.log('📝 创建合同...');
    const contracts = await Promise.all([
        prisma.contract.create({ data: { contractNumber: 'CON-2024-00001', opportunityId: opportunities[0].id, totalContractValue: 500000, status: 'Signed', startDate: new Date('2024-04-01'), endDate: new Date('2024-12-31'), isActive: true, milestones: { create: [{ name: 'Kickoff (50%)', amount: 250000, status: 'Invoiced', dueDate: new Date('2024-05-31') }, { name: 'Phase 1 Delivery', amount: 150000, status: 'Verified', dueDate: new Date('2024-08-31') }, { name: 'Final Delivery', amount: 100000, status: 'WIP', dueDate: new Date('2024-12-31') }] } } }),
        prisma.contract.create({ data: { contractNumber: 'CON-2024-00002', opportunityId: opportunities[1].id, totalContractValue: 300000, status: 'Signed', startDate: new Date('2024-07-01'), endDate: new Date('2025-03-31'), isActive: true, milestones: { create: [{ name: 'Initial Deposit', amount: 150000, status: 'Ready_to_Invoice', dueDate: new Date('2024-07-15') }, { name: 'Final Payment', amount: 150000, status: 'Pending', dueDate: new Date('2025-03-31') }] } } }),
    ]);
    console.log(`✓ ${contracts.length} 个合同\n`);

    // 6. 创建项目
    console.log('🚧 创建项目...');
    const projects = await Promise.all([
        prisma.project.create({
            data: {
                contractId: contracts[0].id,
                status: 'Execution',
                budget: 500000,
                targetProfitMargin: 25,
                laborCost: 200000,
                outsourceCost: 50000,
                travelCost: 10000,
                emergencySupportCost: 20000,
                thirdPartyEquipmentCost: 30000,
                softwareCost: 15000,
                otherWeight: 5000,
                complexity: 'Medium',
                startDate: new Date('2024-04-01'),
                endDate: new Date('2024-12-31'),
                description: 'Enterprise software development project',
                resources: {
                    create: [
                        { userId: users[2].id, role: 'PM', allocationPct: 100 },
                        { userId: users[3].id, role: 'Developer', allocationPct: 80 }
                    ]
                }
            }
        }),
    ]);
    console.log(`✓ ${projects.length} 个项目\n`);

    // 7. 创建发票和收款
    console.log('💰 创建发票和收款...');
    const milestone1 = await prisma.milestone.findFirst({ where: { contractId: contracts[0].id, name: 'Kickoff (50%)' } });
    const invoice1 = await prisma.invoice.create({ data: { invoiceNumber: 'RM-2024-00001', contractId: contracts[0].id, projectId: projects[0].id, milestoneId: milestone1.id, invoiceDate: new Date('2024-05-01'), dueDate: new Date('2024-06-01'), amount: 250000, taxRate: 0.06, taxAmount: 15000, totalAmount: 265000, type: 'Service', status: 'Paid', description: 'Invoice for Kickoff milestone' } });
    await prisma.payment.create({ data: { paymentNumber: 'PAY-2024-00001', invoiceId: invoice1.id, paymentDate: new Date('2024-05-15'), amount: 265000, paymentMethod: 'BankTransfer', bankName: '中国银行', transactionRef: 'TXN20240515001' } });

    const milestone2 = await prisma.milestone.findFirst({ where: { contractId: contracts[0].id, name: 'Phase 1 Delivery' } });
    const invoice2 = await prisma.invoice.create({ data: { invoiceNumber: 'RM-2024-00002', contractId: contracts[0].id, projectId: projects[0].id, milestoneId: milestone2.id, invoiceDate: new Date('2024-09-01'), dueDate: new Date('2024-10-01'), amount: 150000, taxRate: 0.06, taxAmount: 9000, totalAmount: 159000, type: 'Service', status: 'PartiallyPaid', description: 'Invoice for Phase 1 Delivery' } });
    await prisma.payment.create({ data: { paymentNumber: 'PAY-2024-00002', invoiceId: invoice2.id, paymentDate: new Date('2024-09-10'), amount: 80000, paymentMethod: 'BankTransfer', bankName: '工商银行', transactionRef: 'TXN20240910001' } });
    console.log('✓ 2 张发票和 2 笔收款\n');

    console.log('📊 数据统计：');
    console.log('═══════════════════════════════════════');
    console.log(`👥 用户: ${users.length} | 🏢 客户: ${customers.length}`);
    console.log(`💼 商机: ${opportunities.length} | 📋 招投标: ${procurements.length}`);
    console.log(`📝 合同: ${contracts.length} | 📍 里程碑: 5`);
    console.log(`🚧 项目: ${projects.length} | 💰 发票: 2 | 💵 收款: 2`);
    console.log('═══════════════════════════════════════\n');
    console.log('✅ 完整测试数据填充成功！\n');
    console.log('📝 登录信息：');
    console.log('  用户名: admin / sales_wang / pm_zhang / dev_li / finance_chen');
    console.log('  密码: adc1234');
}

main().catch((e) => { console.error('❌ 错误:', e); process.exit(1); }).finally(async () => { await prisma.$disconnect(); });
