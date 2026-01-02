-- AlterTable
ALTER TABLE "opportunities" ADD COLUMN "probability" INTEGER DEFAULT 0;
ALTER TABLE "opportunities" ADD COLUMN "source" TEXT;
