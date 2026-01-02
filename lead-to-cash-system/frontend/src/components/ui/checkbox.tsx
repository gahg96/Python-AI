"use client"

import * as React from "react"
import { Check } from "lucide-react"
import { cn } from "@/lib/utils"

function Checkbox({
    className,
    checked,
    onCheckedChange,
    disabled,
    ...props
}: {
    className?: string
    checked?: boolean
    onCheckedChange?: (checked: boolean) => void
    disabled?: boolean
    id?: string
    [key: string]: any
}) {
    return (
        <div
            data-slot="checkbox"
            className={cn(
                "peer relative flex h-5 w-5 shrink-0 items-center justify-center rounded-sm border border-primary ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
                checked ? "bg-primary text-primary-foreground" : "bg-transparent",
                className
            )}
            onClick={() => !disabled && onCheckedChange?.(!checked)}
            {...props}
        >
            {checked && <Check className="h-3.5 w-3.5" />}
            <input
                type="checkbox"
                className="sr-only"
                checked={checked}
                disabled={disabled}
                readOnly
            />
        </div>
    )
}

export { Checkbox }
