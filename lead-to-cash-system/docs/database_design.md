# Database Design

## 1. Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    User ||--o{ Opportunity : "sales owner"
    User ||--o{ Contract : "drafter/approver"
    Customer ||--o{ Opportunity : "has"
    Opportunity ||--o{ Contract : "generates"
    Opportunity ||--o{ FollowUp : "has"
    Opportunity ||--o{ Procurement : "has"
    Contract ||--o{ Milestone : "has"
    Contract ||--o| Project : "executes"
    Contract ||--o{ Invoice : "billed via"
    Project ||--o{ ProjectResource : "staffed by"
    Project ||--o{ Invoice : "billing"
    Invoice ||--o{ Payment : "paid by"
    Milestone ||--o| Invoice : "generates"

    User {
        string id PK
        string username
        string role "SALES, ADMIN, etc."
    }

    Customer {
        string id PK
        string companyName
        string industry
    }
    
    Opportunity {
        string id PK
        string title
        decimal estimatedValue
        string status "New, Proposal, Won, Lost"
    }

    Contract {
        string id PK
        string contractNumber
        decimal totalContractValue
        string status "Draft, Approved, Signed"
    }

    Milestone {
        string id PK
        string name
        decimal amount
        string status "Pending, Verified, Invoiced, Paid"
    }

    Project {
        string id PK
        string status "Initialization, Execution, Delivery"
        decimal budget
        decimal laborCost
    }

    Invoice {
        string id PK
        string invoiceNumber
        decimal amount
        string status "Draft, Issued, Paid"
    }

    Payment {
        string id PK
        decimal amount
        date paymentDate
    }
```

## 2. Table Configurations

### Core Business Entities
- **Customer**: Stores client information.
- **Opportunity**: Sales opportunities with stages, value, and probability.
- **Contract**: Legal agreements linked to opportunities. Includes status workflow (Draft -> Approved -> Signed).
- **Project**: Execution phase after contract signing. Tracks budget, costs, and resources.

### Financial Entities
- **Milestone**: Payment milestones defined in the contract. Tracks statuses: `Pending` -> `Verified` -> `Ready_to_Invoice` -> `Invoiced` -> `Paid`.
- **Invoice**: Created from milestones or manually. Tracks tax, attachments (receipts), and payment status.
- **Payment**: Records actual incoming funds against invoices.

### Bidding & Procurement
- **Procurement**: Tracks bidding projects (Tenders).
- **BiddingTask**: Tasks assigned to team members for preparing bid documents.

### System
- **User**: System users with Roles (RBAC).
- **AuditLog**: Tracks critical actions (Create/Update/Delete).
