import { useState, useEffect, useCallback, useRef } from "react";
import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { Table } from "@tiptap/extension-table";
import { TableRow } from "@tiptap/extension-table-row";
import { TableCell } from "@tiptap/extension-table-cell";
import { TableHeader } from "@tiptap/extension-table-header";
import { Markdown } from "tiptap-markdown";
import * as api from "../api";
import VoiceMicButton from "./VoiceMicButton";

interface Props {
  filePath: string;
  onClose: () => void;
}

function ToolbarButton({
  onClick,
  active,
  title,
  children,
}: {
  onClick: () => void;
  active?: boolean;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      className={`toolbar-btn${active ? " toolbar-btn-active" : ""}`}
      onClick={onClick}
      title={title}
      type="button"
    >
      {children}
    </button>
  );
}

function ToolbarDivider() {
  return <span className="toolbar-divider" />;
}

export default function ContextEditor({ filePath, onClose }: Props) {
  const [original, setOriginal] = useState("");
  const [saving, setSaving] = useState(false);
  const [formatting, setFormatting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const originalRef = useRef("");

  const editor = useEditor({
    extensions: [
      StarterKit,
      Table.configure({ resizable: false }),
      TableRow,
      TableCell,
      TableHeader,
      Markdown.configure({
        html: false,
        transformCopiedText: true,
        transformPastedText: true,
      }),
    ],
    content: "",
    onUpdate: ({ editor }) => {
      if (!loaded) return;
      const md = (editor.storage as any).markdown.getMarkdown();
      setDirty(md !== originalRef.current);
    },
  });

  useEffect(() => {
    api
      .readFile(filePath)
      .then((data) => {
        setOriginal(data.content);
        originalRef.current = data.content;
        if (editor) {
          editor.commands.setContent(data.content);
          setLoaded(true);
        }
      })
      .catch((err) => setError(err.message));
  }, [filePath, editor]);

  const handleSave = useCallback(async () => {
    if (!editor) return;
    setSaving(true);
    setError(null);
    try {
      const md = (editor.storage as any).markdown.getMarkdown();
      await api.writeFile(filePath, md);
      setOriginal(md);
      originalRef.current = md;
      setDirty(false);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }, [editor, filePath]);

  const handleFormat = useCallback(async () => {
    if (!editor) return;
    setFormatting(true);
    setError(null);
    try {
      const md = (editor.storage as any).markdown.getMarkdown();
      const result = await api.formatFile(md);
      editor.commands.setContent(result.content);
      setDirty(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setFormatting(false);
    }
  }, [editor]);

  // Cmd+S / Ctrl+S keyboard shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "s" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSave();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [handleSave]);

  const filename = filePath.split("/").pop() || filePath;

  if (!editor) return null;

  return (
    <div className="context-editor">
      <div className="editor-header">
        <span className="editor-title">
          {filename}
          {dirty && <span className="dirty-marker"> (unsaved)</span>}
        </span>
        <div className="editor-actions">
          <VoiceMicButton
            onTranscript={(text) => editor.commands.insertContent(text)}
          />
          <button
            className="btn btn-sm"
            onClick={handleFormat}
            disabled={formatting}
            title="Format with Claude"
          >
            {formatting ? "Formatting..." : "Format"}
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleSave}
            disabled={saving || !dirty}
          >
            {saving ? "Saving..." : "Save"}
          </button>
          <button className="btn btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      {error && <div className="editor-error">{error}</div>}

      <div className="editor-toolbar">
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleBold().run()}
          active={editor.isActive("bold")}
          title="Bold (Cmd+B)"
        >
          <strong>B</strong>
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleItalic().run()}
          active={editor.isActive("italic")}
          title="Italic (Cmd+I)"
        >
          <em>I</em>
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleStrike().run()}
          active={editor.isActive("strike")}
          title="Strikethrough"
        >
          <s>S</s>
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleCode().run()}
          active={editor.isActive("code")}
          title="Inline Code"
        >
          {"<>"}
        </ToolbarButton>

        <ToolbarDivider />

        <ToolbarButton
          onClick={() =>
            editor.chain().focus().toggleHeading({ level: 1 }).run()
          }
          active={editor.isActive("heading", { level: 1 })}
          title="Heading 1"
        >
          H1
        </ToolbarButton>
        <ToolbarButton
          onClick={() =>
            editor.chain().focus().toggleHeading({ level: 2 }).run()
          }
          active={editor.isActive("heading", { level: 2 })}
          title="Heading 2"
        >
          H2
        </ToolbarButton>
        <ToolbarButton
          onClick={() =>
            editor.chain().focus().toggleHeading({ level: 3 }).run()
          }
          active={editor.isActive("heading", { level: 3 })}
          title="Heading 3"
        >
          H3
        </ToolbarButton>

        <ToolbarDivider />

        <ToolbarButton
          onClick={() => editor.chain().focus().toggleBulletList().run()}
          active={editor.isActive("bulletList")}
          title="Bullet List"
        >
          &bull; List
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleOrderedList().run()}
          active={editor.isActive("orderedList")}
          title="Ordered List"
        >
          1. List
        </ToolbarButton>

        <ToolbarDivider />

        <ToolbarButton
          onClick={() => editor.chain().focus().toggleCodeBlock().run()}
          active={editor.isActive("codeBlock")}
          title="Code Block"
        >
          {"{ }"}
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleBlockquote().run()}
          active={editor.isActive("blockquote")}
          title="Blockquote"
        >
          &ldquo; &rdquo;
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().setHorizontalRule().run()}
          title="Horizontal Rule"
        >
          &mdash;
        </ToolbarButton>

        <ToolbarDivider />

        <ToolbarButton
          onClick={() =>
            editor
              .chain()
              .focus()
              .insertTable({ rows: 3, cols: 3, withHeaderRow: true })
              .run()
          }
          active={editor.isActive("table")}
          title="Insert Table"
        >
          Table
        </ToolbarButton>
        {editor.isActive("table") && (
          <>
            <ToolbarButton
              onClick={() => editor.chain().focus().addColumnAfter().run()}
              title="Add Column"
            >
              +Col
            </ToolbarButton>
            <ToolbarButton
              onClick={() => editor.chain().focus().addRowAfter().run()}
              title="Add Row"
            >
              +Row
            </ToolbarButton>
            <ToolbarButton
              onClick={() => editor.chain().focus().deleteColumn().run()}
              title="Delete Column"
            >
              -Col
            </ToolbarButton>
            <ToolbarButton
              onClick={() => editor.chain().focus().deleteRow().run()}
              title="Delete Row"
            >
              -Row
            </ToolbarButton>
            <ToolbarButton
              onClick={() => editor.chain().focus().deleteTable().run()}
              title="Delete Table"
            >
              Del Table
            </ToolbarButton>
          </>
        )}
      </div>

      <div className="tiptap-editor">
        <EditorContent editor={editor} />
      </div>
    </div>
  );
}
