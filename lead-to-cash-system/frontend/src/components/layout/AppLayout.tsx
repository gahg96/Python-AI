'use client';

import { useAuth } from '@/contexts/AuthContext';
import { usePathname } from 'next/navigation';
import Sidebar from './Sidebar';
import { Loader2 } from 'lucide-react';

export default function AppLayout({ children }: { children: React.ReactNode }) {
    const { user, loading } = useAuth();
    const pathname = usePathname();

    // Pages that don't need the sidebar (e.g. login)
    const isPublicPage = pathname === '/login' || pathname === '/register';

    if (loading) {
        return <div className="h-screen flex items-center justify-center"><Loader2 className="animate-spin h-8 w-8 text-blue-500" /></div>;
    }

    if (!user || isPublicPage) {
        return <main>{children}</main>;
    }

    return (
        <div className="flex h-screen bg-slate-50">
            <Sidebar />
            <main className="flex-1 ml-64 overflow-y-auto">
                {children}
            </main>
        </div>
    );
}
