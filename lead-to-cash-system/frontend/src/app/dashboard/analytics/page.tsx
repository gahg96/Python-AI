"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useI18n } from "@/lib/i18n/I18nContext";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    FunnelChart,
    Funnel,
    LabelList
} from "recharts";
import { ComposableMap, Geographies, Geography, Marker } from "react-simple-maps";

export default function AnalyticsPage() {
    const { t, language } = useI18n();
    const [funnelData, setFunnelData] = useState<any[]>([]);

    useEffect(() => {
        // Fetch Funnel Data
        api.get("/dashboard/funnel").then((data) => {
            // Map stages to colors
            const colors = {
                "New": "#3b82f6", // blue-500
                "Proposal": "#eab308", // yellow-500
                "Negotiation": "#f97316", // orange-500
                "Won": "#22c55e", // green-500
            };

            const formatted = data.map((item: any) => ({
                ...item,
                name: t(`dashboard.milestones.${item.stage.toLowerCase()}`) || item.stage,
                fill: colors[item.stage as keyof typeof colors] || "#8884d8",
            })).sort((a: any, b: any) => b.count - a.count); // Sort descending to ensure funnel shape
            setFunnelData(formatted);

        }).catch(console.error);

        // Fetch Trend Data
        api.get("/dashboard/trend").then((data) => {
            setTrendData(data);
        }).catch(console.error);

        // Fetch Geo Data
        api.get("/dashboard/geo").then((data) => {
            setGeoData(data);
        }).catch(console.error);
    }, []); // eslint-disable-line

    const [trendData, setTrendData] = useState<any[]>([]);
    const [geoData, setGeoData] = useState<any[]>([]);

    const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

    return (
        <div className="min-h-screen bg-slate-50 p-8 space-y-8">
            {/* Header */}
            <div className="flex items-center gap-4">
                <Link href="/">
                    <Button variant="ghost" size="icon">
                        <ArrowLeft className="h-4 w-4" />
                    </Button>
                </Link>
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900">{t("dashboard.analytics.title")}</h1>
                    <p className="text-slate-500">{t("dashboard.analytics.subtitle")}</p>
                </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                {/* Sales Funnel */}
                <Card>
                    <CardHeader>
                        <CardTitle>{t("dashboard.analytics.funnelTitle")}</CardTitle>
                        <CardDescription>{t("dashboard.analytics.funnelDesc")}</CardDescription>
                    </CardHeader>
                    <CardContent className="h-[400px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <FunnelChart>
                                <Tooltip />
                                <Funnel
                                    dataKey="count"
                                    data={funnelData}
                                    isAnimationActive
                                >
                                    <LabelList
                                        position="right"
                                        fill="#666"
                                        stroke="none"
                                        dataKey="displayName"
                                        style={{ fontSize: 12 }}
                                    />
                                </Funnel>
                            </FunnelChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Revenue Trend */}
                <Card>
                    <CardHeader>
                        <CardTitle>{t("dashboard.analytics.trendTitle")}</CardTitle>
                        <CardDescription>{t("dashboard.analytics.trendDesc")}</CardDescription>
                    </CardHeader>
                    <CardContent className="h-[400px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={trendData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="month" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="revenue" fill="#22c55e" name={t("dashboard.analytics.revenue")} />
                                <Bar
                                    dataKey="pipeline"
                                    fill="#3b82f6"
                                    name={t("dashboard.analytics.pipeline")}
                                    cursor="pointer"
                                    onClick={(data) => {
                                        if (data && data.opportunityIds && data.opportunityIds.length > 0) {
                                            if (data.opportunityIds.length === 1) {
                                                window.location.href = `/opportunities/${data.opportunityIds[0]}`;
                                            } else {
                                                // Navigate to list view (could add filter in future)
                                                console.log("Navigating to opportunities list for IDs:", data.opportunityIds);
                                                window.location.href = `/opportunities`;
                                            }
                                        }
                                    }}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Geo Map */}
                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>{t("dashboard.analytics.geoTitle")}</CardTitle>
                        <CardDescription>{t("dashboard.analytics.geoDesc")}</CardDescription>
                    </CardHeader>
                    <CardContent className="h-[300px]">
                        <ComposableMap projection="geoMercator" projectionConfig={{ scale: 100 }}>
                            <Geographies geography={geoUrl}>
                                {({ geographies }) =>
                                    geographies.map((geo) => (
                                        <Geography
                                            key={geo.rsmKey}
                                            geography={geo}
                                            fill="#EAEAEC"
                                            stroke="#D6D6DA"
                                            onClick={() => window.location.href = `/opportunities`}
                                            style={{
                                                default: { outline: "none" },
                                                hover: { fill: "#DDD", outline: "none", cursor: "pointer" },
                                                pressed: { outline: "none" },
                                            }}
                                        />
                                    ))
                                }
                            </Geographies>
                            {geoData.map(({ name, coords, city }) => (
                                <Marker key={name} coordinates={coords as [number, number]}
                                    onClick={() => window.location.href = `/opportunities?city=${(city as any).en}`}
                                    style={{
                                        default: { outline: "none", cursor: "pointer" },
                                        hover: { outline: "none", cursor: "pointer" },
                                        pressed: { outline: "none" },
                                    }}>
                                    <circle r={6} fill="#F00" stroke="#fff" strokeWidth={2} />
                                    <text
                                        textAnchor="middle"
                                        y={-12}
                                        style={{ fontFamily: "system-ui", fill: "#5D5A6D", fontSize: "10px", fontWeight: "bold" }}
                                    >
                                        {"🚩 " + (language === 'zh' ? (city as any).zh : (city as any).en)}
                                    </text>
                                </Marker>
                            ))}
                        </ComposableMap>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
