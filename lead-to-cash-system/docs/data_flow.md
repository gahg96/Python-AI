# Data Flow Diagrams (DFD)

## Level 0: Context Diagram

```mermaid
graph LR
    User[User] -- 1. Enters Opportunity info --> System
    User -- 2. Drafts Contract --> System
    Manager[Manager] -- 3. Approves Contract --> System
    System -- 4. Initializes Project --> User
    User -- 5. Verifies Milestone --> System
    System -- 6. Generates Invoice --> User
    User -- 7. Records Payment --> System
```

## Level 1: Lead-to-Cash Flow

```mermaid
sequenceDiagram
    participant Sales as Sales Rep
    participant Manager as Manager
    participant PM as Project Manager
    participant Finance as Finance
    participant Sys as System

    %% Opportunity
    Sales->>Sys: Create Opportunity
    Sales->>Sys: Update Status to "Won"

    %% Contract
    Sales->>Sys: Draft Contract (linked to Opp)
    Sales->>Sys: Define Milestones
    Sales->>Sys: Submit for Approval
    Manager->>Sys: Approve Contract
    Sales->>Sys: Sign Contract

    %% Project
    Sys->>PM: Auto-Initialize Project
    PM->>Sys: Assign Resources & Plan
    PM->>Sys: Execute & Verify Milestones

    %% Finance
    PM->>Sys: Request Invoice (Milestone Verified)
    Finance->>Sys: Generate Invoice
    Finance->>Sys: Send Invoice to Customer (Offline)
    Finance->>Sys: Record Payment
    Sys->>Sys: Update Milestone to "Paid"
```

## Data States

### Contract Status Flow
`Draft` -> `PendingApproval` -> `Approved` -> `Signed`

### Milestone Status Flow
`Pending` -> `Verified` -> `Ready_to_Invoice` -> `Invoiced` -> `Paid`

### Invoice Status Flow
`Draft` -> `Issued` -> `PartiallyPaid` -> `Paid`
