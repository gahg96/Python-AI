-- Extension for UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Customers: 客户表
CREATE TABLE customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_name VARCHAR(255) NOT NULL, -- 公司名称
    contact_name VARCHAR(255),          -- 联系人姓名
    email VARCHAR(255),
    phone VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Opportunities: 商机/潜在交易 (The "Lead" part)
CREATE TABLE opportunities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID REFERENCES customers(id),
    title VARCHAR(255) NOT NULL,
    -- 状态：新建 -> 方案 -> 谈判 -> 赢单/输单
    status VARCHAR(50) CHECK (status IN ('New', 'Proposal', 'Negotiation', 'Won', 'Lost')) DEFAULT 'New',
    estimated_value DECIMAL(12, 2), -- 预计金额
    expected_close_date DATE,       -- 预计成交日期
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Contracts: 合同 (连接销售与执行的桥梁)
-- 在这个 Lite 模型中，合同直接链接到商机
CREATE TABLE contracts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    opportunity_id UUID REFERENCES opportunities(id),
    total_contract_value DECIMAL(12, 2) NOT NULL, -- 合同总额
    start_date DATE,
    end_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Milestones: 里程碑 (Lite 系统的核心)
-- 结合了项目执行(Phase)和收款(Billing)
CREATE TABLE milestones (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES contracts(id),
    name VARCHAR(255) NOT NULL, -- 例如: "30% 预付款", "MVP 交付"
    amount DECIMAL(12, 2) NOT NULL,
    
    -- 状态追踪工作进度和资金回笼
    -- Pending: 未开始
    -- WIP (Work In Progress): 进行中
    -- Ready_to_Invoice: 已完工，待开票
    -- Invoiced: 已开票
    -- Paid: 已收款
    status VARCHAR(50) CHECK (status IN ('Pending', 'WIP', 'Ready_to_Invoice', 'Invoiced', 'Paid')) DEFAULT 'Pending',
    
    due_date DATE, -- 预计完成/收款日期
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for quick dashboard lookups (为仪表盘快速查询建立索引)
CREATE INDEX idx_milestones_status ON milestones(status);
CREATE INDEX idx_opportunities_status ON opportunities(status);
