"use client";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n/I18nContext";

export function LanguageToggle() {
    const { language, setLanguage } = useI18n();

    return (
        <Button
            variant="outline"
            size="sm"
            onClick={() => setLanguage(language === "en" ? "zh" : "en")}
        >
            {language === "en" ? "中文" : "English"}
        </Button>
    );
}
