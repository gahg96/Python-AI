# Lead-to-Cash System

A comprehensive Enterprise Resource Planning (ERP) system for managing the entire business lifecycle from sales opportunities to financial collection.

## 📚 Documentation
- [User Manual](./docs/user_manual.md): Detailed usage instructions.
- [System Design](./docs/system_design.md): Architecture and technology stack.
- [Database Design](./docs/database_design.md): ER Diagram and Schema.
- [API Reference](./docs/api_reference.md): API Endpoint list.
- [Data Flow](./docs/data_flow.md): System data flow diagrams.

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

## Features
- **CRM**: Opportunity tracking.
- **Contract Management**: Drafting, Approval, Signing.
- **Project Management**: Resource allocation, Profit analysis.
- **Finance**: Invoicing, Payments, Electronic Receipts.
- **Bidding**: Tender management.

## Tech Stack
- **Frontend**: Next.js, Taiwan CSS, Shadcn UI.
- **Backend**: NestJS, Prisma, SQLite/PostgreSQL.
