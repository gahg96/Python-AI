'use client';

import React, { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n/I18nContext";
import { Loader2, Plus, Calendar, Users, TrendingUp } from 'lucide-react';
import Link from 'next/link';

export default function ProjectDeliveryPage() {
    const { t } = useI18n();
    const [projects, setProjects] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchProjects();
    }, []);

    const fetchProjects = async () => {
        try {
            const data = await api.get('/projects');
            setProjects(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'Initialization': return 'bg-blue-100 text-blue-800';
            case 'Planning': return 'bg-yellow-100 text-yellow-800';
            case 'Execution': return 'bg-green-100 text-green-800';
            case 'Delivery': return 'bg-purple-100 text-purple-800';
            default: return 'bg-slate-100 text-slate-800';
        }
    };

    if (loading) return <div className="flex justify-center p-8"><Loader2 className="animate-spin" /></div>;

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-slate-900">{t("project.title")}</h1>
                    <p className="text-slate-500">{t("project.subtitle")}</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between pb-2">
                        <CardTitle className="text-sm font-medium">{t("dashboard.activeProjects")}</CardTitle>
                        <Calendar className="h-4 w-4 text-slate-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{projects.length}</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between pb-2">
                        <CardTitle className="text-sm font-medium">{t("project.fields.avgMargin")}</CardTitle>
                        <TrendingUp className="h-4 w-4 text-slate-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">32.4%</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between pb-2">
                        <CardTitle className="text-sm font-medium">{t("project.fields.teamLoad")}</CardTitle>
                        <Users className="h-4 w-4 text-slate-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">85%</div>
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>{t("project.fields.activeEngagements")}</CardTitle>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>{t("project.fields.projectContract")}</TableHead>
                                <TableHead>{t("project.fields.customer")}</TableHead>
                                <TableHead>{t("project.fields.status")}</TableHead>
                                <TableHead>{t("project.fields.timeline")}</TableHead>
                                <TableHead>{t("project.fields.resources")}</TableHead>
                                <TableHead className="text-right">{t("project.fields.action")}</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {projects.map((proj) => (
                                <TableRow key={proj.id}>
                                    <TableCell>
                                        <div className="font-medium">{proj.contract.opportunity.title}</div>
                                        <div className="text-xs text-slate-500">{proj.contract.contractNumber}</div>
                                    </TableCell>
                                    <TableCell>{proj.contract.opportunity.customer.companyName}</TableCell>
                                    <TableCell>
                                        <Badge className={getStatusColor(proj.status)} variant="outline">
                                            {t(`project.status.${proj.status.toLowerCase()}`)}
                                        </Badge>
                                    </TableCell>
                                    <TableCell>
                                        <div className="text-sm">
                                            {proj.startDate ? new Date(proj.startDate).toLocaleDateString() : t("project.placeholders.tbd")} -
                                            {proj.endDate ? new Date(proj.endDate).toLocaleDateString() : t("project.placeholders.tbd")}
                                        </div>
                                    </TableCell>
                                    <TableCell>
                                        <div className="flex -space-x-2">
                                            {proj.resources.map((r: any) => (
                                                <div key={r.id} className="h-8 w-8 rounded-full bg-slate-200 border-2 border-white flex items-center justify-center text-xs font-bold" title={r.user.displayName}>
                                                    {r.user.displayName.charAt(0)}
                                                </div>
                                            ))}
                                            {proj.resources.length === 0 && <span className="text-xs text-slate-400">{t("project.placeholders.unassigned")}</span>}
                                        </div>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <Link href={`/delivery/${proj.id}`}>
                                            <Button variant="ghost" size="sm">{t("project.actions.viewDetails")}</Button>
                                        </Link>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {projects.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={6} className="text-center py-8 text-slate-500">
                                        {t("project.placeholders.noProjects")}
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>
        </div>
    );
}
