import { PrismaClient } from '@prisma/client';
import * as bcrypt from 'bcrypt';

const prisma = new PrismaClient();

async function main() {
  console.log('Seeding database...');

  // --- 1. Cleanup ---
  await prisma.milestone.deleteMany();
  await prisma.contract.deleteMany();
  await prisma.opportunity.deleteMany();
  await prisma.customer.deleteMany();

  // --- 2. Create Customers (参考客户logo墙) ---

  // 银行类
  const icbc = await prisma.customer.create({
    data: {
      companyName: '中国工商银行',
      industry: '银行',
      companySize: '超大型',
      contactName: '张经理',
      contactTitle: '科技部项目经理',
      contactEmail: 'zhang@icbc.com.cn',
      contactPhone: '021-88888001',
    },
  });

  const bocom = await prisma.customer.create({
    data: {
      companyName: '交通银行',
      industry: '银行',
      companySize: '超大型',
      contactName: '李总监',
      contactTitle: '信息技术部总监',
      contactEmail: 'li@bankcomm.com',
      contactPhone: '021-88888002',
    },
  });

  const cib = await prisma.customer.create({
    data: {
      companyName: '兴业银行',
      industry: '银行',
      companySize: '大型',
      contactName: '王主管',
      contactTitle: '数字金融部主管',
      contactEmail: 'wang@cib.com.cn',
      contactPhone: '021-88888003',
    },
  });

  const spdb = await prisma.customer.create({
    data: {
      companyName: '浦发银行',
      industry: '银行',
      companySize: '大型',
      contactName: '陈经理',
      contactTitle: '信息科技部经理',
      contactEmail: 'chen@spdb.com.cn',
      contactPhone: '021-88888004',
    },
  });

  const nbbank = await prisma.customer.create({
    data: {
      companyName: '宁波银行',
      industry: '银行',
      companySize: '中型',
      contactName: '赵总',
      contactTitle: '科技创新部总经理',
      contactEmail: 'zhao@nbcb.com.cn',
      contactPhone: '0574-8888005',
    },
  });

  // 证券类
  const sse = await prisma.customer.create({
    data: {
      companyName: '上海证券交易所',
      industry: '证券交易所',
      companySize: '大型',
      contactName: '刘处长',
      contactTitle: '技术开发部处长',
      contactEmail: 'liu@sse.com.cn',
      contactPhone: '021-88888006',
    },
  });

  const shfe = await prisma.customer.create({
    data: {
      companyName: '上海期货交易所',
      industry: '期货交易所',
      companySize: '大型',
      contactName: '孙主任',
      contactTitle: '信息技术部主任',
      contactEmail: 'sun@shfe.com.cn',
      contactPhone: '021-88888007',
    },
  });

  const gtja = await prisma.customer.create({
    data: {
      companyName: '国泰君安证券',
      industry: '证券',
      companySize: '大型',
      contactName: '周经理',
      contactTitle: '金融科技部经理',
      contactEmail: 'zhou@gtja.com',
      contactPhone: '021-88888008',
    },
  });

  const everbright = await prisma.customer.create({
    data: {
      companyName: '光大证券',
      industry: '证券',
      companySize: '大型',
      contactName: '吴总监',
      contactTitle: '信息技术总监',
      contactEmail: 'wu@ebscn.com',
      contactPhone: '021-88888009',
    },
  });

  // 保险/基金类
  const chinaTaiping = await prisma.customer.create({
    data: {
      companyName: '中国太平',
      industry: '保险',
      companySize: '超大型',
      contactName: '郑副总',
      contactTitle: '科技运营部副总经理',
      contactEmail: 'zheng@cntaiping.com',
      contactPhone: '021-88888010',
    },
  });

  const citicPru = await prisma.customer.create({
    data: {
      companyName: '中信保诚',
      industry: '保险',
      companySize: '大型',
      contactName: '黄经理',
      contactTitle: '数字化转型部经理',
      contactEmail: 'huang@citicpru.com.cn',
      contactPhone: '021-88888011',
    },
  });

  const htfFund = await prisma.customer.create({
    data: {
      companyName: '汇添富基金',
      industry: '基金',
      companySize: '大型',
      contactName: '林总',
      contactTitle: 'IT部总经理',
      contactEmail: 'lin@htffund.com',
      contactPhone: '021-88888012',
    },
  });

  // 央企/国企类
  const chinaMobile = await prisma.customer.create({
    data: {
      companyName: '中国移动',
      industry: '电信',
      companySize: '超大型',
      contactName: '杨处长',
      contactTitle: '信息技术中心处长',
      contactEmail: 'yang@chinamobile.com',
      contactPhone: '021-88888013',
    },
  });

  const shanghaiElectric = await prisma.customer.create({
    data: {
      companyName: '上海电气',
      industry: '制造业',
      companySize: '超大型',
      contactName: '徐主任',
      contactTitle: '数字化部主任',
      contactEmail: 'xu@shanghai-electric.com',
      contactPhone: '021-88888014',
    },
  });

  const saicMotor = await prisma.customer.create({
    data: {
      companyName: '上汽集团',
      industry: '汽车',
      companySize: '超大型',
      contactName: '何经理',
      contactTitle: '智能化研发中心经理',
      contactEmail: 'he@saicmotor.com',
      contactPhone: '021-88888015',
    },
  });

  const orientalPearl = await prisma.customer.create({
    data: {
      companyName: '东方明珠',
      industry: '传媒',
      companySize: '大型',
      contactName: '马总监',
      contactTitle: '技术研发总监',
      contactEmail: 'ma@opg.cn',
      contactPhone: '021-88888016',
    },
  });

  const xiamenUniv = await prisma.customer.create({
    data: {
      companyName: '厦门大学',
      industry: '教育',
      companySize: '大型',
      contactName: '谢主任',
      contactTitle: '信息与网络中心主任',
      contactEmail: 'xie@xmu.edu.cn',
      contactPhone: '0592-8888017',
    },
  });

  const shanghaiAirport = await prisma.customer.create({
    data: {
      companyName: '上海机场',
      industry: '交通运输',
      companySize: '大型',
      contactName: '胡经理',
      contactTitle: '信息管理部经理',
      contactEmail: 'hu@shairport.com',
      contactPhone: '021-88888018',
    },
  });

  // --- 3. Scenario A: 工商银行 - 核心系统升级项目 (已签约) ---
  const dealA = await prisma.opportunity.create({
    data: {
      customerId: icbc.id,
      opportunityNumber: 'OPP-2023-0001',
      title: '核心银行系统智能化升级',
      status: 'Won',
      estimatedValue: 2800000,
      probability: 100,
      salesStage: 'contract',
      dealType: 'new',
      deliveryModel: 'onsite',
      salesOwner: '张伟',
      expectedCloseDate: new Date('2023-12-01'),
    },
  });

  const contractA = await prisma.contract.create({
    data: {
      contractNumber: 'CTR-2024-001',
      opportunityId: dealA.id,
      totalContractValue: 2800000,
      startDate: new Date('2024-01-01'),
      endDate: new Date('2024-06-30'),
      isActive: true,
      status: 'Signed',
    },
  });

  await prisma.milestone.createMany({
    data: [
      {
        contractId: contractA.id,
        name: '首付款 (30%)',
        amount: 840000,
        status: 'Paid',
        dueDate: new Date('2024-01-05'),
      },
      {
        contractId: contractA.id,
        name: '一期交付验收',
        amount: 1120000,
        status: 'WIP',
        dueDate: new Date('2024-03-31'),
      },
      {
        contractId: contractA.id,
        name: '终验尾款',
        amount: 840000,
        status: 'Pending',
        dueDate: new Date('2024-06-30'),
      },
    ],
  });

  // --- 4. Scenario B: 上汽集团 - 智能驾驶平台 (刚签约) ---
  const dealB = await prisma.opportunity.create({
    data: {
      customerId: saicMotor.id,
      opportunityNumber: 'OPP-2024-0001',
      title: '智能驾驶数据分析平台',
      status: 'Won',
      estimatedValue: 1500000,
      probability: 100,
      salesStage: 'contract',
      dealType: 'new',
      deliveryModel: 'hybrid',
      salesOwner: '李明',
      expectedCloseDate: new Date('2024-02-15'),
    },
  });

  const contractB = await prisma.contract.create({
    data: {
      contractNumber: 'CTR-2024-002',
      opportunityId: dealB.id,
      totalContractValue: 1500000,
      startDate: new Date('2024-03-01'),
      endDate: new Date('2024-08-31'),
      isActive: true,
      status: 'Draft',
    },
  });

  await prisma.milestone.createMany({
    data: [
      {
        contractId: contractB.id,
        name: '预付款 (50%)',
        amount: 750000,
        status: 'Ready_to_Invoice',
        dueDate: new Date('2024-03-01'),
      },
      {
        contractId: contractB.id,
        name: '上线验收款',
        amount: 750000,
        status: 'Pending',
        dueDate: new Date('2024-08-31'),
      },
    ],
  });

  // --- 5. Scenario C: 国泰君安 - 量化交易系统 (提案中) ---
  await prisma.opportunity.create({
    data: {
      customerId: gtja.id,
      opportunityNumber: 'OPP-2024-0002',
      title: '量化交易策略分析平台',
      status: 'Proposal',
      estimatedValue: 980000,
      probability: 60,
      salesStage: 'proposal',
      dealType: 'new',
      deliveryModel: 'remote',
      salesOwner: '王芳',
      expectedCloseDate: new Date('2024-04-01'),
    },
  });

  // --- 6. Scenario D: 上海证券交易所 - 监控系统 (谈判中) ---
  await prisma.opportunity.create({
    data: {
      customerId: sse.id,
      opportunityNumber: 'OPP-2024-0003',
      title: '交易监控智能预警系统',
      status: 'Negotiation',
      estimatedValue: 1200000,
      probability: 75,
      salesStage: 'negotiation',
      dealType: 'new',
      deliveryModel: 'onsite',
      salesOwner: '张伟',
      expectedCloseDate: new Date('2024-05-15'),
    },
  });

  // --- 7. Scenario E: 中国移动 - 运维平台 (新线索) ---
  await prisma.opportunity.create({
    data: {
      customerId: chinaMobile.id,
      opportunityNumber: 'OPP-2026-0001',
      title: 'AI智能运维管理平台',
      status: 'New',
      estimatedValue: 650000,
      probability: 20,
      salesStage: 'initial_contact',
      dealType: 'new',
      salesOwner: '李明',
      expectedCloseDate: new Date('2026-06-30'),
    },
  });

  // --- 8. Scenario F: 汇添富基金 - 投资分析系统 (新线索) ---
  await prisma.opportunity.create({
    data: {
      customerId: htfFund.id,
      opportunityNumber: 'OPP-2026-0002',
      title: '智能投资决策分析系统',
      status: 'New',
      estimatedValue: 420000,
      probability: 30,
      salesStage: 'requirement',
      dealType: 'new',
      deliveryModel: 'remote',
      salesOwner: '王芳',
      expectedCloseDate: new Date('2026-05-01'),
    },
  });

  // --- 9. Create Specific Users (Per Requirement) ---

  // zhangying (Contract Drafter)
  const zhangying = await prisma.user.upsert({
    where: { username: 'zhangying' },
    update: {},
    create: {
      username: 'zhangying',
      passwordHash: '$2b$10$wT.f.q/o2I.w.R.l.l.l.u.e.r.s.P.a.s.s.w.o.r.d', // password (placeholder)
      displayName: 'Zhang Ying',
      role: 'SALES',
      email: 'zhangying@example.com',
    },
  });

  // hugang (Contract Approver)
  const hugang = await prisma.user.upsert({
    where: { username: 'hugang' },
    update: {},
    create: {
      username: 'hugang',
      passwordHash: '$2b$10$wT.f.q/o2I.w.R.l.l.l.u.e.r.s.P.a.s.s.w.o.r.d', // password (placeholder)
      displayName: 'Hu Gang',
      role: 'MANAGER',
      email: 'hugang@example.com',
    },
  });

  // admin (System Administrator)
  const salt = await bcrypt.genSalt();
  const adminPassword = await bcrypt.hash('admin123', salt);
  const admin = await prisma.user.upsert({
    where: { username: 'admin' },
    update: {
      passwordHash: adminPassword,
    },
    create: {
      username: 'admin',
      passwordHash: adminPassword,
      displayName: 'System Admin',
      role: 'ADMIN',
      email: 'admin@example.com',
    },
  });

  console.log('Seeding finished. New users created: zhangying, hugang, admin');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
