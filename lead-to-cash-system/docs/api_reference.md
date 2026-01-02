# API Reference

## Authentication (`/auth`)
- `POST /auth/login`: User login (returns JWT).
- `GET /auth/profile`: Get current user profile.

## Opportunities (`/opportunities`)
- `POST /opportunities`: Create new opportunity.
- `GET /opportunities`: List all opportunities (supports pagination/filtering).
- `GET /opportunities/:id`: Get opportunity details.
- `PATCH /opportunities/:id`: Update opportunity.
- `DELETE /opportunities/:id`: Delete opportunity.
- `POST /opportunities/:id/follow-ups`: Add follow-up record.
- `POST /opportunities/:id/attachments`: Upload attachment.

## Contracts (`/contracts`)
- `POST /contracts`: Create contract (from Opportunity).
- `GET /contracts`: List contracts.
- `GET /contracts/:id`: Get contract details.
- `PATCH /contracts/:id`: Update contract details.
- `POST /contracts/:id/submit`: Submit for approval.
- `POST /contracts/:id/approve`: Approve contract.
- `POST /contracts/:id/reject`: Reject contract.
- `POST /contracts/:id/sign`: Mark as Signed.
- `POST /contracts/:id/documents`: Upload contract document.
- `POST /contracts/:id/milestones`: Add milestone.
- `PATCH /contracts/milestones/:id`: Update milestone (e.g., Verify).

## Projects (`/projects`)
- `POST /projects`: Initialize project (usually auto-created from Contract).
- `GET /projects`: List projects.
- `GET /projects/:id`: Get project details.
- `PATCH /projects/:id`: Update project (status, budget, costs).
- `POST /projects/:id/resources`: Add team member.
- `DELETE /projects/resources/:id`: Remove team member.
- `POST /projects/:id/risks`: Add risk.
- `PATCH /projects/risks/:id`: Update risk status.

## Finance (`/finance`)
- `GET /finance/dashboard`: Get financial KPI data.
- `GET /finance/invoices`: List invoices.
- `POST /finance/invoices`: Create direct invoice.
- `POST /finance/invoices/from-milestone/:id`: Create invoice from milestone.
- `GET /finance/invoices/:id`: Get invoice details.
- `PATCH /finance/invoices/:id`: Update invoice (e.g. Remarks).
- `POST /finance/invoices/:id/receipt`: Upload electronic receipt.
- `POST /finance/payments`: Record payment.

## Procurements (`/procurements`)
- `POST /procurements`: Create tender/bid.
- `GET /procurements`: List bids.
- `GET /procurements/:id`: Get bid details.
- `PATCH /procurements/:id`: Update bid status/result.
- `POST /procurements/:id/tasks`: Add bidding task.

## Customers (`/customers`)
- `POST /customers`: Create customer.
- `GET /customers`: List customers.
