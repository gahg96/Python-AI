# API Reference Documentation

**Version**: 1.0  
**Base URL**: `/api`  
**Authentication**: Headers: `Authorization: Bearer <Token>` required for all non-public endpoints.

---

## 1. Authentication

### 1.1 Login
**POST** `/auth/login`

Obtain access token (JWT).

- **Request Body**:
  ```json
  {
    "username": "admin",
    "password": "password123"
  }
  ```
- **Response (201)**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsIn..."
  }
  ```

### 1.2 Get Profile
**GET** `/auth/profile`

Get current user info.

---

## 2. Opportunities

### 2.1 Create Opportunity
**POST** `/opportunities`

- **Request Body**:
  ```json
  {
    "title": "2026 Cloud Migration Project",
    "customerId": "uuid-of-customer",
    "estimatedValue": 500000,
    "winProbability": 60,
    "closeDate": "2026-06-30T00:00:00Z"
  }
  ```

### 2.2 List Opportunities
**GET** `/opportunities`

Supports pagination.

- **Query Params**:
  - `page`: (default 1)
  - `limit`: (default 10)
  - `status`: Filter by status.

---

## 3. Contracts

### 3.1 Create Contract
**POST** `/contracts`

- **Request Body**:
  ```json
  {
    "opportunityId": "uuid-of-opportunity",
    "totalValue": 500000,
    "startDate": "2026-07-01T00:00:00Z",
    "endDate": "2027-06-30T00:00:00Z"
  }
  ```

### 3.2 Submit for Approval
**POST** `/contracts/:id/submit`

Changes status to `Pending Approval`.

### 3.3 Adjudicate Contract (Manager Only)
**POST** `/contracts/:id/approve` or `/contracts/:id/reject`

- **Request Body** (Reject only):
  ```json
  {
    "reason": "Margin too low."
  }
  ```

### 3.4 Sign Contract
**POST** `/contracts/:id/sign`

Locks the contract and triggers Project creation.

### 3.5 Add Milestone
**POST** `/contracts/:id/milestones`

- **Body**:
  ```json
  {
    "name": "Down Payment",
    "amount": 150000,
    "description": "Upon signing"
  }
  ```

---

## 4. Finance

### 4.1 Create Invoice
**POST** `/finance/invoices`

Generic creation. Use `from-milestone` preferred.

### 4.2 Create Invoice from Milestone
**POST** `/finance/invoices/from-milestone/:milestoneId`

- **Response**: Returns Invoice object linked to milestone. Status set to `Draft`.

### 4.3 Upload Receipt
**POST** `/finance/invoices/:id/receipt`

- **Body**: `multipart/form-data`
  - `file`: (Binary)

### 4.4 Record Payment
**POST** `/finance/payments`

- **Body**:
  ```json
  {
    "invoiceId": "uuid...",
    "amount": 10000,
    "paymentDate": "2026-08-05",
    "paymentMethod": "BankTransfer",
    "transactionRef": "TXN12345678"
  }
  ```

---

## 5. Projects

### 5.1 Get Project Details
**GET** `/projects/:id`

Includes Team, Milestones, and Financials.

### 5.2 Verify Milestone (PM Only)
**PATCH** `/contracts/milestones/:id`

PM use this to confirm milestone completion.

- **Body**:
  ```json
  {
    "status": "Verified"
  }
  ```

---

## Error Response

- **400**: Bad Request (Validation Error)
- **401**: Unauthorized
- **403**: Forbidden (Role mismatch)
- **404**: Not Found
- **500**: Internal Server Error

---

*End of API Reference*
