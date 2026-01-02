"use client";

import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Image from '@tiptap/extension-image';
import Link from '@tiptap/extension-link';
import Placeholder from '@tiptap/extension-placeholder';
import { Button } from '@/components/ui/button';
import { Bold, Italic, List, ListOrdered, ImageIcon, Link as LinkIcon, Heading2 } from 'lucide-react';
import { useCallback } from 'react';

interface RichTextEditorProps {
    content?: string;
    onChange?: (html: string) => void;
    placeholder?: string;
    editable?: boolean;
}

export function RichTextEditor({ content = '', onChange, placeholder = 'Write something...', editable = true }: RichTextEditorProps) {
    const editor = useEditor({
        immediatelyRender: false, // Fix SSR hydration mismatch
        extensions: [
            StarterKit,
            Image.configure({
                inline: true,
                allowBase64: true, // Allow pasted images as base64
            }),
            Link.configure({
                openOnClick: false,
            }),
            Placeholder.configure({
                placeholder,
            }),
        ],
        content,
        editable,
        onUpdate: ({ editor }) => {
            onChange?.(editor.getHTML());
        },
        editorProps: {
            attributes: {
                class: 'prose prose-sm max-w-none focus:outline-none min-h-[150px] p-3',
            },
            handlePaste: (view, event) => {
                const items = event.clipboardData?.items;
                if (!items) return false;

                for (const item of items) {
                    if (item.type.startsWith('image/')) {
                        event.preventDefault();
                        const file = item.getAsFile();
                        if (!file) return false;

                        const reader = new FileReader();
                        reader.onload = (e) => {
                            const base64 = e.target?.result as string;
                            view.dispatch(
                                view.state.tr.replaceSelectionWith(
                                    view.state.schema.nodes.image.create({ src: base64 })
                                )
                            );
                        };
                        reader.readAsDataURL(file);
                        return true;
                    }
                }
                return false;
            },
        },
    });

    const addImage = useCallback(() => {
        const url = window.prompt('Enter image URL');
        if (url && editor) {
            editor.chain().focus().setImage({ src: url }).run();
        }
    }, [editor]);

    const setLink = useCallback(() => {
        const url = window.prompt('Enter URL');
        if (url && editor) {
            editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run();
        }
    }, [editor]);

    if (!editor) {
        return null;
    }

    return (
        <div className="border rounded-md overflow-hidden">
            {editable && (
                <div className="flex items-center gap-1 border-b bg-slate-50 p-2">
                    <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => editor.chain().focus().toggleBold().run()}
                        className={editor.isActive('bold') ? 'bg-slate-200' : ''}
                    >
                        <Bold className="h-4 w-4" />
                    </Button>
                    <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => editor.chain().focus().toggleItalic().run()}
                        className={editor.isActive('italic') ? 'bg-slate-200' : ''}
                    >
                        <Italic className="h-4 w-4" />
                    </Button>
                    <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
                        className={editor.isActive('heading', { level: 2 }) ? 'bg-slate-200' : ''}
                    >
                        <Heading2 className="h-4 w-4" />
                    </Button>
                    <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => editor.chain().focus().toggleBulletList().run()}
                        className={editor.isActive('bulletList') ? 'bg-slate-200' : ''}
                    >
                        <List className="h-4 w-4" />
                    </Button>
                    <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => editor.chain().focus().toggleOrderedList().run()}
                        className={editor.isActive('orderedList') ? 'bg-slate-200' : ''}
                    >
                        <ListOrdered className="h-4 w-4" />
                    </Button>
                    <Button type="button" variant="ghost" size="sm" onClick={addImage}>
                        <ImageIcon className="h-4 w-4" />
                    </Button>
                    <Button type="button" variant="ghost" size="sm" onClick={setLink}>
                        <LinkIcon className="h-4 w-4" />
                    </Button>
                </div>
            )}
            <EditorContent editor={editor} />
        </div>
    );
}
