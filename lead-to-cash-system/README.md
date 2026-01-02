# Lead-to-Cash System

A comprehensive Enterprise Resource Planning (ERP) system for managing the entire business lifecycle from sales opportunities to financial collection.

## 📚 Documentation

The documentation is available in multiple languages:

### [🇨🇳 中文文档 (Chinese Docs)](./docs/zh/user_manual.md)
- **[用户手册 (User Manual)](./docs/zh/user_manual.md)**: 详细的功能操作指南。
- **[系统设计 (System Design)](./docs/zh/system_design.md)**: 架构、技术栈与目录结构。
- **[数据库设计 (DB Design)](./docs/zh/database_design.md)**: ER图与数据字典。
- **[API 参考 (API Reference)](./docs/zh/api_reference.md)**: 接口文档。
- **[数据流图 (Data Flow)](./docs/data_flow.md)**: (通用图表，参考旧版或英文版)

### [🇺🇸 English Documentation](./docs/en/user_manual.md)
- **[User Manual](./docs/en/user_manual.md)**: Comprehensive user guide.
- **[System Design](./docs/en/system_design.md)**: Architecture and tech stack.
- **[Database Design](./docs/en/database_design.md)**: Schema and ERD.
- **[API Reference](./docs/en/api_reference.md)**: Endpoints reference.

---

## 🚀 Quick Start

### Prerequisites
- Node.js (v18+)
- npm / yarn

### Backend Setup
```bash
cd backend
npm install
npx prisma generate
npx prisma db push
npm run start:dev
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Tech Stack
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Shadcn UI.
- **Backend**: NestJS, TypeScript, Prisma, SQLite/PostgreSQL.
