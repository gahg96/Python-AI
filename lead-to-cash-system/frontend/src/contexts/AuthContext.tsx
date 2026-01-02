"use client";
import React, { createContext, useContext, useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { api } from '@/lib/api';

interface User {
    userId: string;
    username: string;
    displayName: string;
    role: string;
}

interface AuthContextType {
    user: User | null;
    login: (username: string, pass: string) => Promise<void>;
    logout: () => void;
    isAuthenticated: boolean;
    loading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const router = useRouter();
    const pathname = usePathname();

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            const storedUser = localStorage.getItem('user');
            if (storedUser) {
                try {
                    setUser(JSON.parse(storedUser));
                } catch (e) {
                    console.error("Failed to parse user from storage", e);
                }
            }
        }
        setLoading(false);
    }, []);

    useEffect(() => {
        // Redirect to login if not authenticated and not already on login page
        // Also allow access to public paths if any (e.g., /register)
        if (!loading && !user && pathname !== '/login') {
            const token = localStorage.getItem('token'); // Double check storage event
            if (!token) {
                router.push('/login');
            }
        }
    }, [user, loading, pathname, router]);

    const login = async (username: string, pass: string) => {
        const res = await api.post('/auth/login', { username, password: pass });
        localStorage.setItem('token', res.access_token);
        localStorage.setItem('user', JSON.stringify(res.user));
        setUser(res.user);
    };

    const logout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setUser(null);
        window.location.href = '/login';
    };

    if (loading) {
        return <div className="min-h-screen flex items-center justify-center text-slate-500">Loading system...</div>;
    }

    return (
        <AuthContext.Provider value={{ user, login, logout, isAuthenticated: !!user, loading }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};
