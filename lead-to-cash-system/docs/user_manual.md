# Lead-to-Cash System User Manual

## 1. Introduction
Welcome to the Lead-to-Cash (L2C) System. This guide covers all functionalities from managing sales opportunities to collecting payments.

## 2. Getting Started
- **Login**: Access the system via the login page. Enter your credentials.
- **Dashboard**: Upon login, you will see the main dashboard with key metrics (Opportunities, Contracts, Revenue).

## 3. Opportunity Management (CRM)
**Goal**: Manage sales leads and convert them to contracts.

1. **Create Opportunity**:
   - Navigate to "Opportunities" -> "New".
   - Fill in details: Title, Customer, Estimated Value.
   - Click "Create".
2. **Manage Opportunity**:
   - Click on an opportunity to view details.
   - Use the "Follow Up" tab to add notes.
   - Upload attachments in the "Files" tab.
3. **Close Opportunity**:
   - Change status to "Won" when a deal is agreed. This enables Contract creation.

## 4. Contract Management
**Goal**: Draft, approve, and sign contracts.

1. **Draft Contract**:
   - From a "Won" Opportunity, click "Create Contract".
   - Enter Contract Number (auto-generated `CON-XXXX`), Value, and Dates.
   - **Milestones**: Define payment milestones (Name, Amount, Date).
2. **Risk Assessment**:
   - In the "Risk" tab, enter risk details.
3. **Approval Workflow**:
   - Click "Submit for Approval".
   - Manager logs in and clicks "Approve" (or "Reject").
4. **Sign Contract**:
   - Once Approved, click "Sign Contract". This locks the contract and **Initializes the Project**.

## 5. Project Delivery
**Goal**: Execute the project and verify milestones for billing.

1. **Project Overview**:
   - Navigate to "Projects". Click on the project (auto-linked to Contract).
   - View Budget, Margin, and Timeline.
2. **Resource Management**:
   - In "Team" tab, assign users (Developers, PMs) to the project.
3. **Billing & Milestones (IMPORTANT)**:
   - Go to "Billing" tab.
   - You will see milestones defined in the Contract.
   - **Verify**: When a milestone is completed, click **"Verify"**.
   - **Invoice**: Once verified, a "Generate Invoice" button appears.

## 6. Finance & Invoicing
**Goal**: Issue invoices and collect payments.

1. **Create Invoice**:
   - **Method A (Recommended)**: From Project "Billing" tab, click "Generate Invoice" on a verified milestone.
   - **Method B**: Go to "Finance" -> "Invoices" -> "New Invoice". Select Customer and Contract manually.
2. **Manage Invoice**:
   - **Remarks**: Edit remarks if needed.
   - **Receipts**: Upload electronic receipt files (PDF/Image) in the invoice detail page.
3. **Record Payment**:
   - When payment is received, open the Invoice.
   - Click "Record Payment".
   - Enter Amount, Date, and Method.
   - Status updates to "Paid".

## 7. Bidding (Procurement)
**Goal**: Manage incoming tenders/bids.

1. **Create Bid**: Navigate to "Tenders" -> "New".
2. **Manage Tasks**: Assign tasks (Technical, Commercial) to team members.
3. **Track Outcome**: Mark as Won/Lost.

## 8. Reports & Analytics
- **Finance Dashboard**: View outstanding payments, cash flow, and invoiced amounts.
- **Profit Analysis**: In Project details, view real-time profit margin based on costs vs. contract value.
