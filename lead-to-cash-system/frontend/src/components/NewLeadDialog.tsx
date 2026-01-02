"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { Loader2, Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/lib/i18n/I18nContext";
import { api } from "@/lib/api";
import { leadSchema, LeadFormValues } from "@/schemas/lead";

export function NewLeadDialog() {
    const [open, setOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const { t } = useI18n();
    const router = useRouter();

    const form = useForm<LeadFormValues>({
        resolver: zodResolver(leadSchema),
        defaultValues: {
            companyName: "",
            title: "",
            estimatedValue: 0,
        },
    });

    async function onSubmit(data: LeadFormValues) {
        setIsLoading(true);
        try {
            // 1. Create Customer
            const customer = await api.post("/customers", {
                companyName: data.companyName,
            });

            // 2. Create Opportunity
            await api.post("/opportunities", {
                customerId: customer.id,
                title: data.title,
                estimatedValue: Number(data.estimatedValue),
                status: "New",
            });

            setOpen(false);
            form.reset();
            router.refresh();
        } catch (error) {
            console.error("Failed to create lead", error);
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
                <Button>
                    <Plus className="mr-2 h-4 w-4" />
                    {t("dashboard.newLead")}
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>{t("form.create")}</DialogTitle>
                    <DialogDescription>{t("form.desc")}</DialogDescription>
                </DialogHeader>
                <Form {...form}>
                    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                        <FormField
                            control={form.control}
                            name="companyName"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>{t("form.clientName")}</FormLabel>
                                    <FormControl>
                                        <Input placeholder="Acme Corp" {...field} />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="title"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>{t("form.projectName")}</FormLabel>
                                    <FormControl>
                                        <Input placeholder="Website Redesign" {...field} />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="estimatedValue"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>{t("form.estValue")}</FormLabel>
                                    <FormControl>
                                        <Input
                                            type="number"
                                            placeholder="10000"
                                            {...field}
                                            onChange={(e) => field.onChange(e.target.valueAsNumber)}
                                        />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <DialogFooter>
                            <Button type="submit" disabled={isLoading}>
                                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                                {isLoading ? t("form.creating") : t("form.create")}
                            </Button>
                        </DialogFooter>
                    </form>
                </Form>
            </DialogContent>
        </Dialog>
    );
}
