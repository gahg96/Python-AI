"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Plus, Search, Filter, ArrowLeft } from "lucide-react";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import { useI18n } from "@/lib/i18n/I18nContext";
import { useRouter } from "next/navigation";

export default function OpportunitiesPage() {
    const { t } = useI18n();
    const router = useRouter();
    const [opportunities, setOpportunities] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        fetchOpportunities();
    }, []);

    const fetchOpportunities = async () => {
        try {
            const data = await api.get("/opportunities");
            setOpportunities(data);
        } catch (error) {
            console.error("Failed to fetch opportunities", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-8">
            <div className="max-w-7xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Button variant="ghost" size="icon" onClick={() => router.push("/")}>
                            <ArrowLeft className="h-4 w-4" />
                        </Button>
                        <div>
                            <h1 className="text-3xl font-bold tracking-tight text-slate-900">{t("nav.opportunities")}</h1>
                            <p className="text-slate-500 mt-1">Manage your sales pipeline and track deals.</p>
                        </div>
                    </div>
                    <Link href="/opportunities/new">
                        <Button className="bg-indigo-600 hover:bg-indigo-700">
                            <Plus className="mr-2 h-4 w-4" />
                            {t("action.newLead")}
                        </Button>
                    </Link>
                </div>

                {/* Filters (Placeholder) */}
                <div className="flex items-center gap-4 bg-white p-4 rounded-lg border shadow-sm">
                    <div className="relative flex-1 max-w-sm">
                        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-slate-500" />
                        <Input placeholder="Search opportunities..." className="pl-9" />
                    </div>
                    <Button variant="outline" className="gap-2">
                        <Filter className="h-4 w-4" />
                        Filter
                    </Button>
                </div>

                {/* Data Table */}
                <Card className="shadow-sm border-slate-200">
                    <CardContent className="p-0">
                        <Table>
                            <TableHeader>
                                <TableRow className="bg-slate-50 hover:bg-slate-50">
                                    <TableHead className="w-[100px]">ID</TableHead>
                                    <TableHead>{t("table.client")}</TableHead>
                                    <TableHead>{t("table.project")}</TableHead>
                                    <TableHead>{t("table.value")}</TableHead>
                                    <TableHead>{t("form.probability")}</TableHead>
                                    <TableHead>{t("table.status")}</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={7} className="h-24 text-center">Loading...</TableCell>
                                    </TableRow>
                                ) : opportunities.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={7} className="h-24 text-center text-slate-500">No opportunities found.</TableCell>
                                    </TableRow>
                                ) : (
                                    opportunities.map((opp) => (
                                        <TableRow key={opp.id} className="hover:bg-slate-50/50 cursor-pointer" onClick={() => router.push(`/opportunities/${opp.id}`)}>
                                            <TableCell className="font-mono text-xs text-slate-600">{opp.opportunityNumber || `#${opp.id.slice(0, 8)}`}</TableCell>
                                            <TableCell className="font-medium text-slate-900">{opp.customer?.companyName || "Unknown"}</TableCell>
                                            <TableCell>{opp.title}</TableCell>
                                            <TableCell className="font-semibold text-slate-700">
                                                {new Intl.NumberFormat('zh-CN', { style: 'currency', currency: 'CNY' }).format(opp.estimatedValue)}
                                            </TableCell>
                                            <TableCell>{opp.probability}%</TableCell>
                                            <TableCell>
                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                                            ${opp.status === 'New' ? 'bg-blue-100 text-blue-800' :
                                                        opp.status === 'Won' ? 'bg-emerald-100 text-emerald-800' :
                                                            opp.status === 'Lost' ? 'bg-slate-100 text-slate-800' :
                                                                'bg-yellow-100 text-yellow-800'}`}>
                                                    {opp.status}
                                                </span>
                                            </TableCell>
                                            <TableCell className="text-right">
                                                <Button variant="ghost" size="sm">View</Button>
                                            </TableCell>
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
