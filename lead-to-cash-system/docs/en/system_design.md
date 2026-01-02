# Lead-to-Cash System Design Document

**System Name**: Lead-to-Cash (L2C) Enterprise System  
**Version**: 1.0  
**Architect**: Antigravity  
**Date**: 2026-01-02

---

## 1. Introduction

This system is designed to provide comprehensive digital management capabilities for medium-to-large service-oriented enterprises, covering the entire chain from Lead to Cash. It integrates four core domains: CRM (Customer Relationship Management), CLM (Contract Lifecycle Management), PM (Project Management), and Finance.

### 1.1 Design Goals
-   **Data Consistency**: Ensure a single source of truth for promises made by Sales (Contracts), executed by Delivery (Projects), and settled by Finance (Invoices).
-   **Scalability**: Decoupled frontend and backend architecture to facilitate future mobile extensions or third-party integrations.
-   **Security**: Role-Based Access Control (RBAC) to ensure data security and privacy.

---

## 2. Architecture

The system adopts a modern **Monorepo** full-stack architecture.

### 2.1 Tech Stack

#### Frontend
-   **Framework**: Next.js 14 (App Router) - Provides Server-Side Rendering (SSR) and Static Site Generation (SSG) for SEO and performance.
-   **Language**: TypeScript - Strong typing to reduce runtime errors.
-   **UI Library**: Shadcn UI (based on Radix UI) + Tailwind CSS - A modern, customizable atomic CSS solution.
-   **State Management**: React Hooks + URL Search Params - Lightweight state management closer to the platform.
-   **HTTP Client**: Axios - Unified interceptors for JWT Token handling and global error responses.
-   **I18n**: Custom Internationalization solution (ZH/EN).

#### Backend
-   **Framework**: NestJS - A progressive Node.js framework with modular design (Module/Controller/Service), enabling easy maintenance.
-   **Language**: TypeScript.
-   **ORM**: Prisma - Next-generation ORM providing type-safe database queries.
-   **Database**:
    -   Development: SQLite (Lightweight).
    -   Production: PostgreSQL (Recommended) or MySQL.
-   **Auth**: Passport + JWT.
-   **File Storage**: Multer (Local Disk Storage, extensible to S3/OSS).

### 2.2 System Topology

```mermaid
graph TD
    User[User (Browser)] -->|HTTPS| LB[Load Balancer / Nginx]
    LB -->|Next.js| FE[Frontend App Server]
    FE -->|API Calls (JSON)| BE[Backend API Server (NestJS)]
    BE -->|Query| DB[(PostgreSQL Database)]
    BE -->|R/W| FS[File Storage (Uploads)]
```

---

## 3. Directory Structure

The project uses a Monorepo structure, containing `frontend` and `backend` in the root.

### 3.1 Backend (`/backend`)
```
backend/
├── src/
│   ├── app.module.ts       # Root module, aggregates all sub-modules
│   ├── main.ts             # Entry point, Swagger, Cors, Pipes config
│   ├── prisma/             # Prisma Service & Schema
│   │   ├── schema.prisma   # Database Model Schema (Core)
│   ├── auth/               # Auth Module (Login, JWT Strategy)
│   ├── users/              # User Management (CRUD, Profiles)
│   ├── opportunities/      # Sales/Opportunity Module
│   ├── contracts/          # Contract Module (Milestones, Approvals)
│   ├── projects/           # Project Module (Resources, Risks)
│   ├── finance/            # Finance Module (Invoices, Payments)
│   ├── procurement/        # Procurement Module
│   └── uploads/            # Uploads directory (Organized by module)
├── test/                   # E2E Tests
└── package.json
```

### 3.2 Frontend (`/frontend`)
```
frontend/
├── src/
│   ├── app/                # Next.js App Router Pages
│   │   ├── login/          # Login Page
│   │   ├── dashboard/      # Dashboard
│   │   ├── opportunities/  # Opportunity List & Detail ([id])
│   │   ├── contracts/      # Contract Pages
│   │   ├── projects/       # Project Pages
│   │   └── finance/        # Finance Pages
│   ├── components/         # Shared Components
│   │   ├── ui/             # Shadcn Base Components (Button, Input...)
│   │   └── layout/         # Layout Components (Sidebar, Navbar)
│   ├── lib/                # Utilities (api.ts, utils.ts, i18n)
│   └── contexts/           # Global Contexts (AuthContext)
├── public/                 # Static Assets
└── components.json         # Shadcn Config
```

---

## 4. Core Workflows

### 4.1 Lead-to-Project Flow
1.  **Opportunity**: Created by sales, status moves to `Won`.
2.  **Contract**: Created based on `Won` opportunity.
    -   *Validation*: Milestone total must match Contract Value.
    -   *Approval*: Submit -> Manager Approves -> Status `Approved`.
3.  **Project**: Automatically created when Contract status becomes `Signed`.
    -   *Link*: Contract ID -> Project.contractId.
4.  **Delivery**: PM assigns resources and executes the project.
5.  **Verification**: PM validates Milestone completion (`Verified`).

### 4.2 Billing Flow
1.  **Trigger**: Milestone status becomes `Verified`.
2.  **invoice**: Finance generates Invoice.
    -   *Link*: Invoice.milestoneId = Milestone.id.
    -   *Status Update*: Milestone.status -> `Invoiced`.
    -   *Attachment*: Upload electronic invoice receipt.
3.  **Payment**: Finance records Payment.
    -   *Reconciliation*: Calculate Total Paid vs Invoice Amount.
    -   *Completion*: If fully paid, Invoice.status -> `Paid`, Milestone.status -> `Paid`.

---

## 5. Database Overview

See `database_design.md` for details. Core Entities:

-   **User**: Operator of the system.
-   **Customer**: Client entity.
-   **Opportunity**: Sales lead.
-   **Contract**: Legal entity defining value and terms.
-   **Project**: Execution entity managing resources and execution risks.
-   **Invoice**: Financial entity linked to Contracts/Milestones.
-   **Payment**: Financial transaction record.

---

## 6. API Design Standards

RESTful API Design.

-   **Base URL**: `/api`
-   **Authentication**: Bearer Token
-   **Response Format**: Standard JSON envelope for Success/Error.

---

## 7. Security Design

1.  **Auth**: JWT (JSON Web Tokens).
2.  **Password**: Hashed using `bcrypt` or strong hashing algorithms.
3.  **CORS**: Restricted Origin Policy.
4.  **Validation**: Strict DTO validation using `class-validator` to prevent injection.
5.  **RBAC**: Role-based Guards (ADMIN, MANAGER, SALES, EMPLOYEE).

---

## 8. Deployment

Docker containerization is recommended.

### Environment Variables
-   `DATABASE_URL`: Connection string.
-   `JWT_SECRET`: Secret key for token signing.
-   `PORT`: Application port.

---

*End of System Design*
