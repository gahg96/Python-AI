# API 接口参考文档 (API Reference)

**版本**: 1.0  
**Base URL**: `/api`  
**认证**: 所有接口（除登录外）均需要 Header `Authorization: Bearer <Token>`

---

## 1. 认证 (Auth)

### 1.1 用户登录
**POST** `/auth/login`

用于获取访问令牌 (JWT)。

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

### 1.2 获取个人信息
**GET** `/auth/profile`

获取当前登录用户的基本信息。

- **Response (200)**:
  ```json
  {
    "userId": "uuid...",
    "username": "admin",
    "role": "ADMIN"
  }
  ```

---

## 2. 商机 (Opportunities)

### 2.1 创建商机
**POST** `/opportunities`

- **Request Body**:
  ```json
  {
    "title": "2026年度云服务项目",
    "customerId": "uuid-of-customer",
    "estimatedValue": 500000,
    "winProbability": 60,
    "closeDate": "2026-06-30T00:00:00Z"
  }
  ```

### 2.2 查询商机列表
**GET** `/opportunities`

支持分页查询。

- **Query Params**:
  - `page`: 页码 (default 1)
  - `limit`: 每页数量 (default 10)
  - `status`: 按状态筛选 (New, Won, etc.)

---

## 3. 合同 (Contracts)

### 3.1 创建合同
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

### 3.2 提交审批
**POST** `/contracts/:id/submit`

将合同状态变更为 `Pending Approval`。

### 3.3 审核合同 (Manager Only)
**POST** `/contracts/:id/approve` 或 `/contracts/:id/reject`

- **Request Body** (Reject时):
  ```json
  {
    "reason": "利润率过低，请重新评估"
  }
  ```

### 3.4 签署合同
**POST** `/contracts/:id/sign`

合同生效，并自动触发项目创建。

### 3.5 添加里程碑
**POST** `/contracts/:id/milestones`

- **Request Body**:
  ```json
  {
    "name": "首付款",
    "amount": 150000,
    "description": "合同签订后5个工作日内支付"
  }
  ```

---

## 4. 财务 (Finance)

### 4.1 创建发票
**POST** `/finance/invoices`

通用发票创建接口。建议使用 `.../from-milestone` 接口。

- **Request Body**:
  ```json
  {
    "contractId": "uuid...",
    "amount": 10000,
    "type": "Service",
    "invoiceDate": "2026-08-01"
  }
  ```

### 4.2 基于里程碑创建发票
**POST** `/finance/invoices/from-milestone/:milestoneId`

- **Response**: 返回创建成功的发票对象，并自动关联里程碑。

### 4.3 上传回单
**POST** `/finance/invoices/:id/receipt`

- **Body**: `multipart/form-data`
  - `file`: (Binary File)

### 4.4 记录收款
**POST** `/finance/payments`

- **Request Body**:
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

## 5. 项目 (Projects)

### 5.1 获取项目详情
**GET** `/projects/:id`

包含关联的 Contract, Team Members, Milestones 信息。

### 5.2 验证里程碑 (PM Only)
**PATCH** `/contracts/milestones/:id`

用于 PM 确认里程碑完成。

- **Request Body**:
  ```json
  {
    "status": "Verified"
  }
  ```

---

## 错误码 (Error Codes)

- **400 Bad Request**: 参数校验失败。
- **401 Unauthorized**: Token 无效或过期。
- **403 Forbidden**: 权限不足 (如 Sales 尝试审批合同)。
- **404 Not Found**: 资源不存在。
- **500 Internal Server Error**: 服务器内部错误。

---

*API 文档结束*
