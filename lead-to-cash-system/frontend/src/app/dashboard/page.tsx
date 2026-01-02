'use client';

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from '@/lib/api';
import {
    FunnelChart,
    Funnel,
    LabelList,
    Tooltip,
    ResponsiveContainer,
    Cell
} from 'recharts';
import { Loader2, DollarSign, Briefcase, TrendingUp } from 'lucide-react';

interface DashboardStats {
    totalContractValue: number;
    activeContractCount: number;
    activeDealsCount: number;
    winRate: number;
}

interface FunnelData {
    stage: string;
    count: number;
    fill?: string;
}

export default function DashboardPage() {
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [funnelData, setFunnelData] = useState<FunnelData[]>([]);
    const [loading, setLoading] = useState(true);

    const COLORS = ['#8884d8', '#83a6ed', '#8dd1e1', '#82ca9d', '#a4de6c'];

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [statsRes, funnelRes] = await Promise.all([
                    api.get('/dashboard/stats'),
                    api.get('/dashboard/funnel')
                ]);
                setStats(statsRes);
                // Assign colors
                const funnelWithColors = (funnelRes as FunnelData[]).map((item, index) => ({
                    ...item,
                    fill: COLORS[index % COLORS.length]
                }));
                setFunnelData(funnelWithColors);
            } catch (error) {
                console.error("Failed to fetch dashboard data", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    if (loading) {
        return <div className="flex justify-center items-center h-screen"><Loader2 className="animate-spin h-8 w-8 text-blue-600" /></div>;
    }

    return (
        <div className="container mx-auto p-6 space-y-8">
            <h1 className="text-3xl font-bold text-slate-800">Dashboard</h1>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium text-slate-500">
                            Total Contract Value
                        </CardTitle>
                        <DollarSign className="h-4 w-4 text-slate-400" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">
                            ¥{stats?.totalContractValue.toLocaleString()}
                        </div>
                        <p className="text-xs text-slate-500">
                            {stats?.activeContractCount} Active/Signed Contracts
                        </p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium text-slate-500">
                            Active Opportunities
                        </CardTitle>
                        <Briefcase className="h-4 w-4 text-slate-400" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{stats?.activeDealsCount}</div>
                        <p className="text-xs text-slate-500">
                            In Pipeline (New - Negotiation)
                        </p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium text-slate-500">
                            Win Rate
                        </CardTitle>
                        <TrendingUp className="h-4 w-4 text-slate-400" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{stats?.winRate}%</div>
                        <p className="text-xs text-slate-500">
                            Based on Won/Lost deals
                        </p>
                    </CardContent>
                </Card>
            </div>

            {/* Sales Funnel */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="col-span-1 min-h-[400px]">
                    <CardHeader>
                        <CardTitle>Sales Funnel</CardTitle>
                    </CardHeader>
                    <CardContent className="h-[350px]">
                        {funnelData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <FunnelChart>
                                    <Tooltip />
                                    <Funnel
                                        dataKey="count"
                                        data={funnelData}
                                        isAnimationActive
                                    >
                                        <LabelList position="right" fill="#000" stroke="none" dataKey="stage" />
                                        <LabelList position="center" fill="#fff" stroke="none" dataKey="count" />
                                    </Funnel>
                                </FunnelChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex items-center justify-center h-full text-slate-400">
                                No data available
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Placeholder for Recent Activity or other chart */}
                <Card className="col-span-1 min-h-[400px] flex items-center justify-center bg-slate-50 border-dashed">
                    <div className="text-center text-slate-400">
                        <p>More Insights Coming Soon</p>
                        <p className="text-sm">(Drill-down views, Recent Activities)</p>
                    </div>
                </Card>
            </div>
        </div>
    );
}
