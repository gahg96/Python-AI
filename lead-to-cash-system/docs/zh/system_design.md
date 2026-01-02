# Lead-to-Cash 系统设计文档

**系统名称**: Lead-to-Cash (L2C) Enterprise System  
**版本**: 1.0  
**架构师**: Antigravity  
**日期**: 2026-01-02

---

## 1. 简介 (Introduction)

本系统旨在为中大型服务型企业提供从线索（Lead）到回款（Cash）的全链路数字化管理能力。系统集成了CRM（客户关系管理）、CLM（合同生命周期管理）、PM（项目管理）和 Finance（财务管理）四大核心领域。

### 1.1 设计目标
-   **数据一致性**: 确保销售端承诺（合同）与交付端执行（项目）以及财务端结算（发票）的数据源单一且一致。
-   **高扩展性**: 采用前后端分离架构，便于未来扩展移动端或第三方集成。
-   **安全性**: 基于RBAC（基于角色的访问控制）的权限体系，确保数据安全。

---

## 2. 技术架构 (Architecture)

系统采用现代化的 **Monorepo** 全栈架构。

### 2.1 技术栈 (Tech Stack)

#### 前端 (Frontend)
-   **框架**: Next.js 14 (App Router) - 提供服务端渲染(SSR)和静态生成(SSG)能力，SEO友好且首屏加载快。
-   **语言**: TypeScript - 强类型约束，减少运行时错误。
-   **UI组件库**: Shadcn UI (基于 Radix UI) + Tailwind CSS - 现代化、可定制的原子化CSS方案。
-   **状态管理**: React Hooks + URL Search Params - 保持轻量级状态管理。
-   **HTTP客户端**: Axios - 封装了统一的拦截器用于处理JWT Token和错误响应。
-   **国际化**: 自研 i18n 方案 (Support ZH/EN)。

#### 后端 (Backend)
-   **框架**: NestJS - 渐进式Node.js框架，模块化设计(Module/Controller/Service)，易于维护。
-   **语言**: TypeScript。
-   **ORM**: Prisma - 下一代ORM，提供类型安全的数据库查询。
-   **数据库**:
    -   开发环境: SQLite (轻量，开箱即用)。
    -   生产环境: PostgreSQL (推荐) 或 MySQL。
-   **认证**: Passport + JWT (JSON Web Tokens)。
-   **文件上传**: Multer (本地磁盘存储 或 可扩展至对象存储 S3/OSS)。

### 2.2 系统拓扑图

```mermaid
graph TD
    User[用户 (Browser)] -->|HTTPS| LB[负载均衡 / Nginx]
    LB -->|Next.js| FE[前端应用 Server]
    FE -->|API Calls (JSON)| BE[后端 API Server (NestJS)]
    BE -->|Query| DB[(PostgreSQL 数据库)]
    BE -->|R/W| FS[文件存储 (Uploads)]
```

---

## 3. 目录结构说明 (Directory Structure)

项目采用 Monorepo 结构，根目录下包含 frontend 和 backend。

### 3.1 Backend (`/backend`)
```
backend/
├── src/
│   ├── app.module.ts       # 根模块，聚合所有子模块
│   ├── main.ts             # 入口文件，配置Swagger, Cors, Pipes
│   ├── prisma/             # Prisma 服务与 Schema
│   │   ├── schema.prisma   # 数据库模型定义 (核心)
│   ├── auth/               # 认证模块 (Login, JWT Strategy)
│   ├── users/              # 用户管理 (CRUD, Profiles)
│   ├── opportunities/      # 商机模块
│   ├── contracts/          # 合同模块 (含 Milestones, Approvals)
│   ├── projects/           # 项目模块 (含 Resources, Risks)
│   ├── finance/            # 财务模块 (Invoices, Payments)
│   ├── procurement/        # 采购模块
│   └── uploads/            # 文件上传目录 (按模块分文件夹)
├── test/                   # E2E 测试
└── package.json
```

### 3.2 Frontend (`/frontend`)
```
frontend/
├── src/
│   ├── app/                # Next.js App Router 页面路由
│   │   ├── login/          # 登录页
│   │   ├── dashboard/      # 仪表盘
│   │   ├── opportunities/  # 商机列表与详情页 ([id])
│   │   ├── contracts/      # 合同页面
│   │   ├── projects/       # 项目页面
│   │   └── finance/        # 财务页面
│   ├── components/         # 公共组件
│   │   ├── ui/             # Shadcn 基础组件 (Button, Input, Card...)
│   │   └── layout/         # 布局组件 (Sidebar, Navbar)
│   ├── lib/                # 工具库 (api.ts, utils.ts, i18n)
│   └── contexts/           # 全局上下文 (AuthContext)
├── public/                 # 静态资源
└── components.json         # Shadcn 配置
```

---

## 4. 核心业务流程设计 (Core Workflows)

### 4.1 线索转项目流程 (L2C Main Flow)
1.  **商机 (Opportunity)**: 销售创建，状态流转至 `Won`。
2.  **合同 (Contract)**: 基于 `Won` 商机创建，继承数据。
    -   *校验*: 必须定义里程碑金额等于合同总额。
    -   *审批*: 提交 -> 经理同意 -> 状态 `Approved`。
3.  **项目 (Project)**: 合同签署 (`Signed`) 后触发 Event，自动创建 Project 记录。
    -   *数据流*: Contract ID -> Project.contractId。
4.  **交付 (Delivery)**: PM 分配资源，执行项目。
5.  **验收 (Verification)**: PM 确认里程碑完成 (`Verified`)。

### 4.2 财务结算流程 (Billing Flow)
1.  **触发**: 监听到 Milestone 状态变为 `Verified`。
2.  **开票**: 财务生成 Invoice。
    -   *关联*: Invoice.milestoneId = Milestone.id。
    -   *状态更新*: Milestone.status -> `Invoiced`。
    -   *附件*: 上传电子发票文件。
3.  **收款**: 财务录入 Payment 记录。
    -   *核销*: 计算 Total Paid vs Invoice Amount。
    -   *完成*: 若结清，Invoice.status -> `Paid`, Milestone.status -> `Paid`。

---

## 5. 数据库设计概要 (Database Overview)

详细设计请参考 `database_design.md`。核心实体关系如下：

-   **User**: 系统的操作主体，关联所有创建记录。
-   **Customer**: 客户主体，关联 Opportunities。
-   **Opportunity**: 销售机会，是一切的起点。
-   **Contract**: 法律实体，定义了金额和付款条件（Milestones）。
-   **Project**: 执行实体，关联 Contract，管理 Resources 和 Risks。
-   **Invoice**: 财务实体，关联 Contract 或 Specific Milestone。
-   **Payment**: 资金流水，弱关联 Invoice。

---

## 6. 接口设计规范 (API Design)

基于 RESTful 风格设计的 API。

-   **Base URL**: `/api` (通过 Nginx 或 NestJS Global Prefix 配置)
-   **Authentication**: Bearer Token (Header: `Authorization: Bearer <token>`)
-   **Response Format**:
    ```json
    // 成功 (200/201)
    {
      "id": "uuid...",
      "data": "..."
    }
    
    // 错误 (4xx/5xx)
    {
      "statusCode": 400,
      "message": "Error description",
      "error": "Bad Request"
    }
    ```

---

## 7. 安全设计 (Security)

1.  **认证**: 使用 JWT 鉴权，Token 有效期设置（如 7天），部分敏感接口需二次验证（规划中）。
2.  **密码**: 使用 `bcrypt` 或 `argon2` 进行加盐哈希存储，绝不明文存储。
3.  **CORS**: 严格限制跨域请求来源（Origin），防止 CSRF 攻击。
4.  **输入校验**: 后端使用 `class-validator` (DTO) 严格校验所有输入字段，防止 SQL 注入和 NoSQL 注入。
5.  **权限控制**: 接口级 Guard，验证 `User.role` (ADMIN, MANAGER, SALES, EMPLOYEE)。

---

## 8. 部署与运维 (Deployment)

推荐使用 Docker 容器化部署。

### Dockerfile 示例 (Backend)
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npx prisma generate
RUN npm run build
CMD ["npm", "run", "start:prod"]
```

### 环境变量 (.env)
-   `DATABASE_URL`: 数据库连接串。
-   `JWT_SECRET`: 签名密钥。
-   `PORT`: 监听端口。

---

*设计文档结束*
