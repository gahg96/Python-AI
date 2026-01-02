/*
  Warnings:

  - You are about to drop the column `email` on the `customers` table. All the data in the column will be lost.
  - You are about to drop the column `phone` on the `customers` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "opportunities" ADD COLUMN "competitors" TEXT;
ALTER TABLE "opportunities" ADD COLUMN "deal_type" TEXT;
ALTER TABLE "opportunities" ADD COLUMN "decision_makers" TEXT;
ALTER TABLE "opportunities" ADD COLUMN "delivery_model" TEXT;
ALTER TABLE "opportunities" ADD COLUMN "estimated_effort" INTEGER;
ALTER TABLE "opportunities" ADD COLUMN "rich_description" TEXT;
ALTER TABLE "opportunities" ADD COLUMN "sales_owner" TEXT;
ALTER TABLE "opportunities" ADD COLUMN "sales_stage" TEXT;

-- CreateTable
CREATE TABLE "follow_ups" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "opportunity_id" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "created_by" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "follow_ups_opportunity_id_fkey" FOREIGN KEY ("opportunity_id") REFERENCES "opportunities" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "attachments" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "opportunity_id" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "filepath" TEXT NOT NULL,
    "mimetype" TEXT NOT NULL,
    "size" INTEGER NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "attachments_opportunity_id_fkey" FOREIGN KEY ("opportunity_id") REFERENCES "opportunities" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_customers" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "company_name" TEXT NOT NULL,
    "industry" TEXT,
    "company_size" TEXT,
    "contact_name" TEXT,
    "contact_title" TEXT,
    "contact_phone" TEXT,
    "contact_email" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "new_customers" ("company_name", "contact_name", "created_at", "id") SELECT "company_name", "contact_name", "created_at", "id" FROM "customers";
DROP TABLE "customers";
ALTER TABLE "new_customers" RENAME TO "customers";
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
