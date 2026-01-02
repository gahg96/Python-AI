'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useI18n } from '@/lib/i18n/I18nContext';
import {
    LayoutDashboard,
    FileText,
    Truck,
    CreditCard,
    LogOut,
    Briefcase
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';

export default function Sidebar() {
    const pathname = usePathname();
    const { t } = useI18n();
    const { logout, user } = useAuth();

    const navItems = [
        { name: t("nav.dashboard") || "Dashboard", href: '/', icon: LayoutDashboard },
        { name: t("nav.opportunities") || "Opportunities", href: '/opportunities', icon: Briefcase }, // Use Briefcase or Lightbulb
        { name: t("nav.contracts") || "Contracts", href: '/contracts', icon: FileText },
        { name: t("nav.delivery") || "Project Delivery", href: '/delivery', icon: Truck },
        { name: t("nav.finance") || "Finance & Payment", href: '/finance', icon: CreditCard },
    ];

    return (
        <div className="flex flex-col h-screen w-64 bg-slate-900 text-white fixed left-0 top-0">
            <div className="p-6">
                <h1 className="text-xl font-bold flex items-center gap-2">
                    <Briefcase className="text-blue-400" />
                    Lead2Cash
                </h1>
            </div>

            <nav className="flex-1 px-4 space-y-2">
                {navItems.map((item) => {
                    const isActive = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href));
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive
                                ? 'bg-blue-600 text-white'
                                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                                }`}
                        >
                            <item.icon className="h-5 w-5" />
                            <span>{item.name}</span>
                        </Link>
                    )
                })}
            </nav>

            <div className="p-4 border-t border-slate-800">
                <div className="mb-4 px-4">
                    <div className="text-sm font-medium text-white">{user?.username || 'User'}</div>
                    <div className="text-xs text-slate-500 capitalize">{user?.role?.toLowerCase() || 'Role'}</div>
                </div>
                <Button
                    variant="ghost"
                    className="w-full justify-start text-red-400 hover:text-red-300 hover:bg-slate-800"
                    onClick={logout}
                >
                    <LogOut className="h-4 w-4 mr-2" />
                    {t("nav.logout") || "Logout"}
                </Button>
            </div>
        </div>
    );
}
