# Lead-to-Cash Enterprise System User Manual

**Version**: 1.0  
**Date**: 2026-01-02  
**Target Audience**: Sales Managers, Project Managers, Finance Specialists, System Admins

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Quick Start](#2-quick-start)
3. [Opportunity Management (CRM)](#3-opportunity-management-crm)
4. [Contract Lifecycle Management (CLM)](#4-contract-lifecycle-management-clm)
5. [Project Delivery & Execution](#5-project-delivery--execution)
6. [Finance & Invoicing](#6-finance--invoicing)
7. [Procurement Management](#7-procurement-management)
8. [Customer & Master Data](#8-customer--master-data)
9. [FAQ](#9-faq)

---

## 1. System Overview

The **Lead-to-Cash (L2C)** system is a comprehensive Enterprise Resource Planning (ERP) solution designed for modern service-oriented businesses. It bridges the gap between "Lead Discovery" and "Cash Collection," eliminating information silos between Sales, Delivery, and Finance departments.

### Core Value Proposition
- **Process Synergy**: Automatically triggers project initiation upon contract signing, eliminating manual handovers.
- **Financial Compliance**: Invoicing is strictly tied to contract milestone verification, ensuring "Revenue Recognition" compliance.
- **Risk Control**: Built-in risk assessment during contract approval and real-time margin monitoring during project execution.

### Functional Modules
1.  **Opportunities**: Sales funnel, follow-ups, quotation management.
2.  **Contracts**: Drafting, online approval, E-signature, milestone definition.
3.  **Projects**: Task assignment, resource planning, cost tracking, progress monitoring.
4.  **Finance**: Invoicing, receipt management, payment reconciliation, financial reporting.
5.  **Procurement**: Tender management and task allocation.

---

## 2. Quick Start

### 2.1 Login
1.  Open your browser and navigate to the system URL (e.g., `http://localhost:3000`).
2.  Enter your username and password on the login page.
3.  Click the "Login" button.
    *   *Note*: Contact your system administrator if you need a password reset.

### 2.2 Dashboard
Upon successful login, you will see the main Dashboard, aggregating key performance indicators (KPIs):
-   **Sales Overview**: New opportunities this quarter, estimated pipeline value.
-   **To-Do List**: Contracts awaiting your approval, milestones needing verification.
-   **Finance Brief**: Outstanding payments for this month, overdue invoice alerts.

### 2.3 Navigation
The sidebar provides access to all major modules:
-   **Dashboard**: Return to home.
-   **Opportunities**: CRM and sales leads.
-   **Contracts**: Contract management.
-   **Projects**: Delivery management.
-   **Finance**: Invoicing and payments.
-   **Customers**: Client database.
-   **Procurement**: Tenders and bidding.

---

## 3. Opportunity Management (CRM)

Primary Users: **Sales Representatives** and **Sales Directors**.

### 3.1 Creating a New Opportunity
1.  Navigate to **Opportunities**.
2.  Click the **"New Opportunity"** button at the top right.
3.  Fill in the details form:
    *   **Title**: A concise description (e.g., "XYZ Group 2026 Digital Transformation").
    *   **Customer**: Select from existing customers. Create new customers in the "Customers" module if needed.
    *   **Estimated Value**: The projected contract value.
    *   **Win Probability**: Estimated chance of closing (0-100%).
    *   **Close Date**: Expected signing date.
4.  Click **"Create"** to save.

### 3.2 Follow-up & Collaboration
In the Opportunity Detail page, utilize the tabs:
-   **Overview**: Update basic info and sales stage (New -> Proposal -> Negotiation -> Won).
-   **Follow Ups**: Log meeting minutes, calls, and visits. Effective for tracking history and manager review.
-   **Attachments**: Upload requirements docs, RFPs, or technical proposals.

### 3.3 Closing (Won/Lost)
When a deal is finalized:
-   **Won**: Change status to "Won". **Note**: Only "Won" opportunities can be converted to Contracts.
-   **Lost**: If the deal is lost to a competitor, select "Lost" and provide a reason in the remarks (Price, Tech, Relationship) for future analysis.

---

## 4. Contract Lifecycle Management (CLM)

The bridge between Sales and Delivery. Accurate contract data is crucial for billing.

### 4.1 Drafting a Contract
1.  **Entry Point**: From a "Won" Opportunity, click the **"Create Contract"** button.
2.  The system pre-fills the Title, Customer, and Estimated Value.
3.  **Refine Details**:
    *   **Contract Number**: Auto-generated unique ID starting with `CON-` (e.g., `CON-2026-0001`).
    *   **Total Value**: The final signed amount.
    *   **Dates**: Effective Date and End Date.

### 4.2 Defining Milestones - **Critical Step**
Payment terms are defined here. You must define phases clearly as they dictate future invoicing.
1.  In the Contract Detail page, locate the **"Milestones"** section.
2.  Click **"Add Milestone"**.
3.  Enter info:
    *   **Name**: e.g., "Down Payment", "Blueprint Sign-off", "UAT Acceptance", "Retention".
    *   **Amount**: The receivable amount for this phase.
    *   **Description**: Acceptance criteria.
4.  Repeat until the sum equals the Contract Total Value.

### 4.3 Risk Assessment
Before submission, switch to the **"Risk Assessment"** tab:
-   Technical Risks (e.g., Unproven technology).
-   Commercial Risks (e.g., Long payment terms).
-   Delivery Risks (e.g., Tight schedule).

### 4.4 Approval & Signing
1.  **Submit**: Click **"Submit for Approval"**. Status becomes `Pending Approval`.
2.  **Approve**: A Manager logs in, reviews details, and clicks **"Approve"** (or "Reject").
3.  **Sign**: Once Approved, complete offline stamping. Then click **"Sign Contract"** in the system. You can upload the scanned copy.
    *   *System Action*: Signing automatically locks the contract and **Initializes a Project** (Status: `Initialization`).

---

## 5. Project Delivery & Execution

Primary Users: **Project Managers (PM)** and **Delivery Team**.

### 5.1 Project Initialization
Projects are auto-created upon Contract signing.
1.  Navigate to **Projects**.
2.  Find the new project (marked as `Initialization`).

### 5.2 Team Building
1.  Go to the **"Team"** tab in Project Details.
2.  Click **"Add Resource"**.
3.  Select a specific User, assign a Role (Developer, Consultant, PM), and set dates.
4.  System calculates projected Labor Cost based on the user's Rate.

### 5.3 Progress & Risk Management
-   **Dashboard**: Monitor Budget vs Actual Cost, and Real-time Margin.
-   **Risks**: Track risks continuously throughout execution.

### 5.4 Billing & Verification (The Revenue Link)
The most critical step for cash flow.
1.  Switch to the **"Milestones & Billing"** tab.
2.  This lists all milestones inherited from the Contract.
3.  **Verify**: When a phase is completed (e.g., Customer signed UAT), the PM must click **"Verify"**.
    *   Status: `Pending` -> `Verified`.
4.  **Notify Finance**: Once `Verified`, a **"Generate Invoice"** button appears, signaling Finance to proceed.

---

## 6. Finance & Invoicing

Primary Users: **Finance Specialists**.

### 6.1 Invoicing
Two modes available:

#### Mode A: Milestone-Based (Recommended)
1.  Check **Finance** -> **Dashboard** for "Ready to Invoice" items.
2.  Or go to **Project** details -> **Billing** tab, identifying verified milestones.
3.  Click **"Generate Invoice"**.
4.  Draft is auto-created with correct amounts and customer data. Review tax rates (6% or 13%) and Save.

#### Mode B: Manual Invoicing
1.  Go to **Finance** -> **Invoices** -> **New Invoice**.
2.  Manually select Customer, Contract, and enter Amount. Used for non-standard billing.

### 6.2 Invoice Management & Receipts
1.  **View**: Click invoice number to see details.
2.  **Remarks**: Edit remarks for special instructions (e.g., "Please wire to new account").
3.  **Receipt Upload**:
    *   After sending the invoice to the customer, upload the customized PDF or tax system export.
    *   Click the **"Upload Receipt"** area at the bottom of the detail page.
    *   Files (PDF/Images) can be downloaded later for audit.

### 6.3 Recording Payments
1.  When funds arrive in the bank, open the specific Invoice.
2.  Click **"Record Payment"**.
3.  Enter:
    *   **Amount**: Actual received (supports partial payments).
    *   **Date**: Bank receipt date.
    *   **Method**: Transfer, Check, Cash.
4.  If Amount >= Invoice Total, status updates to `Paid`. The linked Milestone also becomes `Paid`.

---

## 7. Procurement Management

Manages pre-sales tendering processes.

1.  **New Tender**: Record incoming RFPs in **Procurement**.
2.  **Task Breakdown**: Assign "Tech Proposal Writing" or "Commercial Pricing" tasks to team members.
3.  **Tracking**: Mark tenders as Won/Lost to build a historical database.

---

## 8. Customer & Master Data

Foundation of all business data. Keep clean and unique.

-   **Fields**: Company Name, Tax ID, Address, Contact Person.
-   **Uniqueness**: System checks Tax ID to prevent duplicate customer entries.

---

## 9. FAQ

**Q1: Why can't I create a contract?**
A: Check the Opportunity status. Only `Won` opportunities permit contract creation.

**Q2: Why is the "Verify" button disabled for milestones?**
A: The Project must be in `Execution` stage, and you must have PM permissions.

**Q3: How to fix an incorrect invoice amount?**
A: If status is `Draft`, edit directly. If a Payment is recorded, delete the payment first, then Cancel the invoice and recreate it.

**Q4: Can we change the Contract Number format?**
A: The default is `CON-YYYY-XXXX`. Custom formats require backend configuration by IT support.

**Q5: How to export reports?**
A: Look for the "Export" button at the top-right of list views (Invoices, Contracts) to download Excel/CSV files.

---

*End of Document*
