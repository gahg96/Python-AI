# System Design Document

## 1. Overview
The **Lead-to-Cash (L2C) System** is a comprehensive enterprise resource planning solution designed to manage the entire customer lifecycle from initial lead generation to final payment collection.

## 2. Architecture
The system follows a standard **Monorepo** structure with a clear separation of concerns between Frontend and Backend.

```mermaid
graph TD
    Client[Browser / Client] <--> Frontend[Next.js Frontend]
    Frontend <--> Backend[NestJS Backend API]
    Backend <--> DB[(SQLite/Postgres Database)]
    Backend <--> FS[File Storage (Uploads)]
```

### 2.1 Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **UI Library**: Shadcn UI + Tailwind CSS
- **State Management**: React Hooks + URL State
- **Data Fetching**: Axios (Custom API Wrapper)

### 2.2 Backend
- **Framework**: NestJS
- **Language**: TypeScript
- **Database ORM**: Prisma
- **Database**: SQLite (Dev) / PostgreSQL (Prod)
- **Authentication**: JWT (JSON Web Tokens)
- **File Handling**: Multer (Local Disk Storage)

## 3. Key Modules

### 3.1 Opportunity Management (CRM)
- Tracks sales leads.
- Manages customer relationships.
- Document storage for proposals.

### 3.2 Contract Management
- Lifecycle management: Draft -> Review -> Approval -> Sign.
- Milestone definition for billing.
- Risk assessment.

### 3.3 Project Delivery
- Initializes upon Contract signing.
- Resource management and allocation.
- Cost tracking (Labor, Travel, Software, etc.).
- Margin analysis.

### 3.4 Finance
- Invoice generation (integrated with Contract Milestones).
- Payment tracking.
- Financial Dashboard (KPIs: Outstanding, Paid, Pending).

## 4. Configuration
- **Environment Variables**: `.env` file manages DB connection, JWT secrets, etc.
- **Database**: configured in `prisma/schema.prisma`.
