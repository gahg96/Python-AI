"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useState, useEffect } from "react";
import { ArrowLeft, ArrowRight, CheckCircle2, DollarSign, AlertCircle, Clock, FileText, Briefcase } from "lucide-react";
import { useI18n } from "@/lib/i18n/I18nContext";
import { LanguageToggle } from "@/components/LanguageToggle";
import Link from "next/link";
// import { NewLeadDialog } from "@/components/NewLeadDialog";
import { api } from "@/lib/api";

interface Procurement {
  id: string;
  procurementNumber: string;
  type: string;
  status: string;
  submissionDeadline: string | null;
  notificationDate: string | null;
  opportunity: {
    id: string;
    title: string;
    opportunityNumber: string;
    customer: { companyName: string };
  };
}

export default function Home() {
  const { t } = useI18n();

  const [kpiData, setKpiData] = useState({
    totalCashIn: 0,
    pendingInvoices: 0,
    projectedRevenue: 0,
    activeDealsCount: 0,
  });

  const [projects, setProjects] = useState<any[]>([]);
  const [procurements, setProcurements] = useState<Procurement[]>([]);

  useEffect(() => {
    // Fetch KPI Data
    api.get("/dashboard/stats").then((data) => {
      setKpiData({
        totalCashIn: Number(data.totalCashIn) || 0,
        pendingInvoices: Number(data.pendingInvoices) || 0,
        projectedRevenue: Number(data.projectedRevenue) || 0,
        activeDealsCount: Number(data.activeDealsCount) || 0,
      });
    }).catch(console.error);

    // Fetch opportunities from backend
    api.get("/opportunities").then((data) => {
      const mappedProjects = data.map((opp: any) => ({
        id: opp.id,
        client: opp.customer?.companyName || "Unknown",
        project: opp.title,
        value: Number(opp.estimatedValue) || 0,
        collected: 0, // This could be fetched if opportunities are linked to payments
        status: opp.status,
        nextMilestone: "TBD",
        health: "Good",
      }));
      setProjects(mappedProjects);
    });

    // Fetch active procurements
    api.get("/procurements/active").then((data) => {
      setProcurements(data);
    }).catch(console.error);
  }, []);

  // ... (rest of the file)

  // Later in the JSX, I need to match the line ... wait.
  // I should use separate replacement chunks for state init vs JSX usage if lines are far apart?
  // Previous view_file showed lines 34-60 roughly for state.
  // And line 154 for JSX.
  // They are far apart. I should use MULTI_REPLACE.

  // NOTE: replace_file_content is for SINGLE CONTIGUOUS block.
  // I must use multi_replace_file_content.


  const milestones = [
    {
      id: 1,
      client: "Rocket Startup",
      name: "Kickoff (50%)",
      amount: 25000,
      status: "Ready_to_Invoice",
      date: "2024-03-01",
    },
    {
      id: 2,
      client: "TechGiant Corp",
      name: "Phase 1 Delivery",
      amount: 40000,
      status: "WIP",
      date: "2024-03-31",
    },
    {
      id: 3,
      client: "TechGiant Corp",
      name: "Initial Deposit",
      amount: 30000,
      status: "Paid",
      date: "2024-01-05",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-900">{t("dashboard.title")}</h1>
            <p className="text-slate-500">{t("dashboard.subtitle")}</p>
          </div>
          <div className="flex items-center gap-4">
            <LanguageToggle />

            <Link href="/opportunities/new">
              <Button>
                <CheckCircle2 className="mr-2 h-4 w-4" />
                {t("dashboard.newLead")}
              </Button>
            </Link>
          </div>
        </div>

        {/* KPI Cards */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{t("dashboard.cashInBank")}</CardTitle>
              <DollarSign className="h-4 w-4 text-emerald-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-emerald-600">${kpiData.totalCashIn.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground">{t("dashboard.kpi.cashGrowth")}</p>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-amber-500">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{t("dashboard.readyToInvoice")}</CardTitle>
              <AlertCircle className="h-4 w-4 text-amber-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-amber-600">${kpiData.pendingInvoices.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground">{t("dashboard.kpi.actionInvoice")}</p>
            </CardContent>
          </Card>

          <Link href="/dashboard/analytics">
            <Card className="cursor-pointer hover:shadow-md transition-all">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{t("dashboard.pipelineValue")}</CardTitle>
                <Clock className="h-4 w-4 text-slate-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">${kpiData.projectedRevenue.toLocaleString()}</div>
                <p className="text-xs text-muted-foreground">{t("dashboard.kpi.activeCount", { count: kpiData.activeDealsCount })}</p>
              </CardContent>
            </Card>
          </Link>
        </div>

        {/* Main Content Area */}
        {/* Main Content Area */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">

          {/* Project List */}
          <Card className="col-span-4">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{t("dashboard.activeProjects")}</CardTitle>
                <Link href="/opportunities" className="text-sm text-blue-600 hover:text-blue-800 flex items-center">
                  {t("dashboard.viewAll")} <ArrowRight className="h-4 w-4 ml-1" />
                </Link>
              </div>
              <CardDescription>{t("dashboard.projDesc")}</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("dashboard.table.client")}</TableHead>
                    <TableHead>{t("dashboard.table.project")}</TableHead>
                    <TableHead className="w-[40%]">{t("dashboard.table.progress")}</TableHead>
                    <TableHead className="text-right">{t("dashboard.table.action")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {projects.map((project, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="font-medium">{project.client}</TableCell>
                      <TableCell>
                        <Link
                          href={`/opportunities/${project.id}`}
                          className="text-blue-600 hover:text-blue-800 hover:underline"
                        >
                          {project.project}
                        </Link>
                      </TableCell>
                      <TableCell>
                        <div className="font-medium">
                          ${project.value.toLocaleString()}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="outline">{project.status}</Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                  {projects.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center text-muted-foreground h-24">
                        No active projects
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Milestones Card */}
          <Card className="col-span-3">
            <CardHeader>
              <CardTitle>{t("dashboard.milestoneWatch")}</CardTitle>
              <CardDescription>{t("dashboard.milestoneDesc")}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-8">
                {milestones.map((milestone) => (
                  <div key={milestone.id} className="flex items-center">
                    <div className="space-y-1">
                      <p className="text-sm font-medium leading-none">{milestone.name}</p>
                      <p className="text-sm text-muted-foreground">{milestone.client}</p>
                    </div>
                    <div className="ml-auto font-medium">
                      {new Intl.NumberFormat('zh-CN', { style: 'currency', currency: 'CNY' }).format(milestone.amount)}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Active Procurements Section */}
        {procurements.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Briefcase className="h-5 w-5 text-purple-600" />
                    投标进行中
                  </CardTitle>
                  <CardDescription>正在进行的投标/应标项目</CardDescription>
                </div>
                <Link href="/procurements" className="text-sm text-blue-600 hover:text-blue-800 flex items-center">
                  查看全部 <ArrowRight className="h-4 w-4 ml-1" />
                </Link>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {procurements.slice(0, 6).map((proc) => (
                  <Link key={proc.id} href={`/opportunities/${proc.opportunity?.id}`} className="block">
                    <div className="p-4 rounded-lg border hover:border-purple-300 hover:bg-purple-50/50 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-mono text-sm text-purple-600">{proc.procurementNumber}</span>
                        <Badge variant={
                          proc.status === 'Won' ? 'default' :
                            proc.status === 'Lost' ? 'destructive' :
                              proc.status === 'Submitted' || proc.status === 'InProgress' ? 'secondary' : 'outline'
                        }>
                          {proc.status === 'Draft' ? '草稿' :
                            proc.status === 'Preparing' ? '准备中' :
                              proc.status === 'Submitted' ? '已提交' :
                                proc.status === 'InProgress' ? '进行中' :
                                  proc.status === 'Won' ? '中标' : '未中标'}
                        </Badge>
                      </div>
                      <div className="font-medium text-slate-900 mb-1 line-clamp-1">{proc.opportunity?.title}</div>
                      <div className="text-sm text-slate-500 mb-2">{proc.opportunity?.customer?.companyName}</div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-500">
                          类型: {proc.type === 'DirectQuote' ? '单一来源' :
                            proc.type === 'Negotiation' ? '商务谈判' :
                              proc.type === 'Comparison' ? '比选' :
                                proc.type === 'Consultation' ? '磋商' : '公开招标'}
                        </span>
                        {proc.submissionDeadline && (
                          <span className="text-orange-600">
                            截止: {new Date(proc.submissionDeadline).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
