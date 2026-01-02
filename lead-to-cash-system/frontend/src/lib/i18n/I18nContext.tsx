"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";
import en from "./en.json";
import zh from "./zh.json";

type Language = "en" | "zh";
type Translations = typeof en;

interface I18nContextType {
    language: Language;
    setLanguage: (lang: Language) => void;
    t: (key: string, params?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextType | undefined>(undefined);

const translations: Record<Language, any> = {
    en,
    zh,
};

export function I18nProvider({ children }: { children: ReactNode }) {
    const [language, setLanguage] = useState<Language>("zh"); // Default to Chinese as per user request

    const t = (key: string, params?: Record<string, string | number>) => {
        const keys = key.split(".");
        let value = translations[language];
        for (const k of keys) {
            if (value && value[k]) {
                value = value[k];
            } else {
                return key;
            }
        }
        let str = value as string;
        if (params) {
            Object.entries(params).forEach(([k, v]) => {
                str = str.replace(new RegExp(`{{${k}}}`, 'g'), String(v));
            });
        }
        return str;
    };

    return (
        <I18nContext.Provider value={{ language, setLanguage, t }}>
            {children}
        </I18nContext.Provider>
    );
}

export function useI18n() {
    const context = useContext(I18nContext);
    if (!context) {
        throw new Error("useI18n must be used within an I18nProvider");
    }
    return context;
}
