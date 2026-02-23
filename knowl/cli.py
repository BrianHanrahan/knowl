"""Knowl CLI — manage context from the command line."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from knowl.context import store


def cmd_init(args: argparse.Namespace) -> None:
    store.init_store()
    print(f"Knowl store initialized at {store.KNOWL_DIR}")


def cmd_project_create(args: argparse.Namespace) -> None:
    path = store.create_project(args.name)
    print(f"Created project: {args.name} ({path})")


def cmd_project_list(args: argparse.Namespace) -> None:
    projects = store.list_projects()
    if not projects:
        print("No projects yet. Create one with: knowl project create <name>")
        return
    config = store.load_config()
    active = config.get("active_project")
    for p in projects:
        marker = " *" if p == active else ""
        print(f"  {p}{marker}")


def cmd_project_switch(args: argparse.Namespace) -> None:
    projects = store.list_projects()
    if args.name not in projects:
        print(f"Project '{args.name}' not found. Available: {', '.join(projects) or 'none'}")
        sys.exit(1)
    config = store.load_config()
    config["active_project"] = args.name
    store.save_config(config)
    print(f"Switched to project: {args.name}")


def cmd_context_list(args: argparse.Namespace) -> None:
    if args.scope == "global" or (not args.scope and not args.project):
        files = store.list_global_files()
        if files:
            print("Global context files:")
            for f in files:
                tokens = store.estimate_file_tokens(f)
                print(f"  {f.name}  (~{tokens} tokens)")
        else:
            print("No global context files.")

    project = args.project
    if not project and args.scope != "global":
        config = store.load_config()
        project = config.get("active_project")

    if project:
        files = store.list_project_files(project)
        active = [f.name for f in store.list_active_files(project)]
        if files:
            print(f"\nProject '{project}' context files:")
            for f in files:
                tokens = store.estimate_file_tokens(f)
                marker = " [active]" if f.name in active else ""
                print(f"  {f.name}  (~{tokens} tokens){marker}")
        else:
            print(f"\nNo context files in project '{project}'.")


def cmd_context_add(args: argparse.Namespace) -> None:
    if args.scope == "global":
        target_dir = store.GLOBAL_DIR
        scope_label = "global"
    else:
        project = args.project
        if not project:
            config = store.load_config()
            project = config.get("active_project")
        if not project:
            print("No project specified and no active project. Use --project or --global.")
            sys.exit(1)
        target_dir = store.PROJECTS_DIR / project
        scope_label = f"project/{project}"

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / args.file

    if not sys.stdin.isatty():
        # Accept content from stdin
        content = sys.stdin.read()
        store.write_context_file(target, content)
        print(f"Created {scope_label}/{args.file}")
        return

    # Try to open in $EDITOR
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", ""))
    if editor:
        if not target.exists():
            target.write_text(f"# {args.file}\n\n", encoding="utf-8")
        subprocess.run([editor, str(target)])
        print(f"Saved {scope_label}/{args.file}")
    else:
        # Fallback: prompt for content
        if not target.exists():
            target.write_text(f"# {args.file}\n\n", encoding="utf-8")
        print(f"Created {scope_label}/{args.file} — edit it at: {target}")


def cmd_context_show(args: argparse.Namespace) -> None:
    # Try global first, then active project
    path = store.GLOBAL_DIR / args.file
    if not path.exists():
        config = store.load_config()
        project = config.get("active_project")
        if project:
            path = store.PROJECTS_DIR / project / args.file
    if not path.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)
    content = store.read_context_file(path)
    if content is not None:
        print(content)


def cmd_context_delete(args: argparse.Namespace) -> None:
    path = store.GLOBAL_DIR / args.file
    if not path.exists():
        config = store.load_config()
        project = config.get("active_project")
        if project:
            path = store.PROJECTS_DIR / project / args.file
    if not path.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)
    confirm = input(f"Delete {path}? [y/N] ").strip().lower()
    if confirm == "y":
        store.delete_context_file(path)
        print(f"Deleted {args.file}")
    else:
        print("Cancelled.")


def cmd_context_promote(args: argparse.Namespace) -> None:
    project = args.project
    if not project:
        config = store.load_config()
        project = config.get("active_project")
    if not project:
        print("No project specified. Use --from <project>.")
        sys.exit(1)
    result = store.promote_to_global(project, args.file)
    if result:
        print(f"Promoted {args.file} from {project} to global context.")
    else:
        print(f"File not found: {project}/{args.file}")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    config = store.load_config()
    active_project = config.get("active_project")
    global_files = store.list_global_files()
    model = config.get("llm", {}).get("model", "claude-sonnet-4-6")

    print(f"Knowl store: {store.KNOWL_DIR}")
    print(f"Model: {model}")
    print(f"Global files: {len(global_files)}")
    print(f"Projects: {len(store.list_projects())}")
    print(f"Active project: {active_project or '(none)'}")

    if active_project:
        tokens = store.estimate_active_context_tokens(active_project)
        total = tokens.pop("total", 0)
        print(f"\nActive context ({total} tokens):")
        for name, count in tokens.items():
            print(f"  {name}: ~{count} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(prog="knowl", description="Knowl — Personal Context Manager for AI")
    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Initialize the Knowl store")

    # project
    project_parser = sub.add_parser("project", help="Manage projects")
    project_sub = project_parser.add_subparsers(dest="project_command")

    create_p = project_sub.add_parser("create", help="Create a new project")
    create_p.add_argument("name", help="Project name")

    project_sub.add_parser("list", help="List all projects")

    switch_p = project_sub.add_parser("switch", help="Switch active project")
    switch_p.add_argument("name", help="Project name")

    # context
    context_parser = sub.add_parser("context", help="Manage context files")
    context_sub = context_parser.add_subparsers(dest="context_command")

    list_p = context_sub.add_parser("list", help="List context files")
    list_p.add_argument("--project", default=None, help="Project name")
    list_g = list_p.add_mutually_exclusive_group()
    list_g.add_argument("--global", dest="scope", action="store_const", const="global", help="Show global files only")

    add_p = context_sub.add_parser("add", help="Add a context file")
    add_p.add_argument("file", help="Filename to create")
    add_p.add_argument("--project", default=None, help="Project name")
    add_g = add_p.add_mutually_exclusive_group()
    add_g.add_argument("--global", dest="scope", action="store_const", const="global", help="Add to global scope")

    show_p = context_sub.add_parser("show", help="Show a context file")
    show_p.add_argument("file", help="Filename to show")

    del_p = context_sub.add_parser("delete", help="Delete a context file")
    del_p.add_argument("file", help="Filename to delete")

    prom_p = context_sub.add_parser("promote", help="Promote a project file to global")
    prom_p.add_argument("file", help="Filename to promote")
    prom_p.add_argument("--from", dest="project", default=None, help="Source project")

    # status
    sub.add_parser("status", help="Show Knowl status")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "project":
        if args.project_command == "create":
            cmd_project_create(args)
        elif args.project_command == "list":
            cmd_project_list(args)
        elif args.project_command == "switch":
            cmd_project_switch(args)
        else:
            project_parser.print_help()
    elif args.command == "context":
        if args.context_command == "list":
            cmd_context_list(args)
        elif args.context_command == "add":
            cmd_context_add(args)
        elif args.context_command == "show":
            cmd_context_show(args)
        elif args.context_command == "delete":
            cmd_context_delete(args)
        elif args.context_command == "promote":
            cmd_context_promote(args)
        else:
            context_parser.print_help()
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
