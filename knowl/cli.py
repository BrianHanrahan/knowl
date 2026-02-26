"""Knowl CLI — manage context from the command line."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from knowl.context import store
from knowl.llm import claude


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


def cmd_context_inspect(args: argparse.Namespace) -> None:
    """Show the exact system prompt that would be sent to Claude."""
    config = store.load_config()
    project = args.project if hasattr(args, "project") and args.project else config.get("active_project")
    budget = args.budget if hasattr(args, "budget") and args.budget else 8000

    context_pieces = store.assemble_context(project, token_budget=budget)
    system_prompt = claude.format_system_prompt(context_pieces)

    if not context_pieces:
        print("No context files found. Add some with: knowl context add <file>")
        return

    print(f"--- Active Context ({len(context_pieces)} files) ---\n")
    total_tokens = 0
    for piece in context_pieces:
        tokens = store.estimate_tokens(piece.get("content", ""))
        total_tokens += tokens
        print(f"  {piece['source']}  (~{tokens} tokens)")
    print(f"\n  Total: ~{total_tokens} tokens (budget: {budget})")

    print(f"\n--- System Prompt Preview ---\n")
    print(system_prompt)
    print(f"\n--- End Preview ---")


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat with Claude using assembled context."""
    config = store.load_config()
    project = config.get("active_project")
    model = config.get("llm", {}).get("model", "claude-sonnet-4-6")

    # Verify API key early
    try:
        claude.get_api_key()
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    # Assemble context
    context_pieces = store.assemble_context(project)
    total_tokens = sum(store.estimate_tokens(p.get("content", "")) for p in context_pieces)

    # Load existing history
    history = store.load_history(project)

    # Show header
    print(f"Knowl Chat — Model: {model}")
    if project:
        print(f"Project: {project}")
    print(f"Context: {len(context_pieces)} files (~{total_tokens} tokens)")
    if history:
        print(f"History: {len(history) // 2} previous exchanges (use /clear to reset)")
    print(f"Type /quit to exit, /clear to reset history, /context to inspect active context.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input == "/quit" or user_input == "/exit":
            print("Goodbye.")
            break
        elif user_input == "/clear":
            history = []
            store.clear_history(project)
            print("History cleared.\n")
            continue
        elif user_input == "/context":
            if not context_pieces:
                print("No active context files.\n")
            else:
                for piece in context_pieces:
                    tokens = store.estimate_tokens(piece.get("content", ""))
                    print(f"  {piece['source']}  (~{tokens} tokens)")
                print(f"  Total: ~{total_tokens} tokens\n")
            continue
        elif user_input.startswith("/"):
            print(f"Unknown command: {user_input}. Available: /quit, /clear, /context\n")
            continue

        # Send to Claude
        try:
            response = claude.send_message(
                user_message=user_input,
                context=context_pieces,
                history=history,
                model=model,
            )
        except RuntimeError as e:
            print(f"\nError: {e}\n")
            continue

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        store.save_history(project, history)

        print(f"\nclaude> {response}\n")


def cmd_history_clear(args: argparse.Namespace) -> None:
    config = store.load_config()
    project = args.project if hasattr(args, "project") and args.project else config.get("active_project")
    store.clear_history(project)
    label = f"project '{project}'" if project else "global"
    print(f"Conversation history cleared for {label}.")


def _execute_voice_intent(intent: "routing.Intent") -> None:
    """Execute a classified voice intent (command or capture)."""
    from knowl.voice import routing  # noqa: F811

    config = store.load_config()
    project = config.get("active_project")

    if intent.kind == "command":
        cmd = intent.command
        cmd_args = intent.args or {}

        if cmd == "switch_project":
            name = cmd_args.get("name", "")
            projects = store.list_projects()
            if name not in projects:
                print(f"Project '{name}' not found. Available: {', '.join(projects) or 'none'}")
                return
            config["active_project"] = name
            store.save_config(config)
            print(f"Switched to project: {name}")

        elif cmd == "list_projects":
            projects = store.list_projects()
            if not projects:
                print("No projects yet.")
                return
            active = config.get("active_project")
            for p in projects:
                marker = " *" if p == active else ""
                print(f"  {p}{marker}")

        elif cmd == "list_context":
            _print_context_summary(project)

        elif cmd == "show_status":
            _print_status()

        elif cmd == "create_project":
            name = cmd_args.get("name", "")
            if name:
                path = store.create_project(name)
                print(f"Created project: {name} ({path})")

        elif cmd == "promote":
            file_name = cmd_args.get("file", "")
            src_project = cmd_args.get("project", project)
            if not src_project:
                print("No project specified. Switch to a project first.")
                return
            result = store.promote_to_global(src_project, file_name)
            if result:
                print(f"Promoted {file_name} from {src_project} to global context.")
            else:
                print(f"File not found: {src_project}/{file_name}")

        elif cmd == "clear_history":
            store.clear_history(project)
            label = f"project '{project}'" if project else "global"
            print(f"Conversation history cleared for {label}.")

        elif cmd == "inspect_context":
            context_pieces = store.assemble_context(project)
            if not context_pieces:
                print("No context files found.")
                return
            for piece in context_pieces:
                tokens = store.estimate_tokens(piece.get("content", ""))
                print(f"  {piece['source']}  (~{tokens} tokens)")

        else:
            print(f"Unknown command: {cmd}")

    elif intent.kind == "capture":
        # Append captured text to a scratchpad file in the active project (or global)
        if project:
            target = store.PROJECTS_DIR / project / "voice-notes.md"
            scope_label = f"project/{project}"
        else:
            target = store.GLOBAL_DIR / "voice-notes.md"
            scope_label = "global"

        target.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if target.exists():
            existing = target.read_text(encoding="utf-8")

        if not existing:
            existing = "# Voice Notes\n\n"

        entry = f"- {intent.text}\n"
        store.write_context_file(target, existing + entry)
        print(f"Captured to {scope_label}/voice-notes.md: {intent.text}")


def _print_context_summary(project: str | None) -> None:
    """Print a summary of active context files."""
    context_pieces = store.assemble_context(project)
    if not context_pieces:
        print("No active context files.")
        return
    total = 0
    for piece in context_pieces:
        tokens = store.estimate_tokens(piece.get("content", ""))
        total += tokens
        print(f"  {piece['source']}  (~{tokens} tokens)")
    print(f"  Total: ~{total} tokens")


def _print_status() -> None:
    """Print Knowl status summary."""
    config = store.load_config()
    active_project = config.get("active_project")
    print(f"Knowl store: {store.KNOWL_DIR}")
    print(f"Model: {config.get('llm', {}).get('model', 'claude-sonnet-4-6')}")
    print(f"Projects: {len(store.list_projects())}")
    print(f"Active project: {active_project or '(none)'}")


def cmd_voice(args: argparse.Namespace) -> None:
    """Record from microphone, transcribe, and route based on intent."""
    from knowl.voice import routing

    config = store.load_config()
    voice_config = config.get("voice", {})
    whisper_model = voice_config.get("whisper_model", "base")
    language = voice_config.get("language", "auto")
    if language == "auto":
        language = None

    # Check for required libraries
    try:
        from knowl.voice.recorder import VoiceRecorder
        from knowl.voice.transcribe import transcribe_audio
    except ImportError as exc:
        print(f"Voice dependencies not available: {exc}")
        print("Install with: pip install sounddevice soundfile numpy openai-whisper")
        sys.exit(1)

    project = config.get("active_project")
    print(f"Voice mode — press Enter to start recording, Enter again to stop.")
    if project:
        print(f"Project: {project}")
    print(f"Whisper model: {whisper_model}")
    print(f"Say 'note <text>' to capture, or speak a command, or just chat.\n")

    recorder = VoiceRecorder()

    while True:
        try:
            input("Press Enter to record (Ctrl+C to exit)...")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting voice mode.")
            break

        print("Recording... (press Enter to stop)")
        recorder.start()

        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass

        wav_path = recorder.stop()
        print("Transcribing...")

        try:
            result = transcribe_audio(wav_path, model_name=whisper_model, language=language)
        except Exception as e:
            print(f"Transcription error: {e}\n")
            continue

        text = result.get("text", "").strip()
        if not text:
            print("(no speech detected)\n")
            continue

        print(f"Heard: {text}")
        intent = routing.classify(text)

        if intent.kind == "command":
            print(f"[command: {intent.command}]")
            _execute_voice_intent(intent)
        elif intent.kind == "capture":
            _execute_voice_intent(intent)
        else:
            # Chat — send to Claude
            print("[chat]")
            try:
                claude.get_api_key()
            except RuntimeError as e:
                print(f"Cannot send to Claude: {e}\n")
                continue

            context_pieces = store.assemble_context(project)
            history = store.load_history(project)
            model = config.get("llm", {}).get("model", "claude-sonnet-4-6")

            try:
                response = claude.send_message(
                    user_message=intent.text,
                    context=context_pieces,
                    history=history,
                    model=model,
                )
            except RuntimeError as e:
                print(f"Error: {e}\n")
                continue

            history.append({"role": "user", "content": intent.text})
            history.append({"role": "assistant", "content": response})
            store.save_history(project, history)
            print(f"\nclaude> {response}\n")

        print()  # blank line between rounds


def cmd_voice_transcribe(args: argparse.Namespace) -> None:
    """Transcribe text and route it through intent classification."""
    from knowl.voice import routing

    text = args.text
    intent = routing.classify(text)

    if intent.kind == "command":
        print(f"[command: {intent.command}]")
        _execute_voice_intent(intent)
    elif intent.kind == "capture":
        print(f"[capture]")
        _execute_voice_intent(intent)
    else:
        print(f"[chat] {intent.text}")
        # In text mode, just print the classification — actual sending
        # happens through `knowl chat`.
        print("Use 'knowl chat' for interactive conversations.")


def cmd_promote_suggest(args: argparse.Namespace) -> None:
    """Scan projects for promotion candidates."""
    from knowl.context.promotion import scan_for_promotions

    threshold = args.threshold if hasattr(args, "threshold") and args.threshold else 0.3
    suggestions = scan_for_promotions(similarity_threshold=threshold)

    if not suggestions:
        print("No promotion suggestions. Add similar context to multiple projects to see suggestions.")
        return

    print(f"Found {len(suggestions)} promotion suggestion(s):\n")
    for i, s in enumerate(suggestions, 1):
        conflict_marker = " [CONFLICT]" if s.conflict else ""
        print(f"  {i}. {s.filename} — {s.similarity:.0%} similar across {len(s.projects)} projects{conflict_marker}")
        print(f"     Projects: {', '.join(s.projects)}")
        print(f"     {s.reason}")
        print()


def cmd_promote_apply(args: argparse.Namespace) -> None:
    """Apply a specific promotion — copy from a project to global."""
    from knowl.context.promotion import scan_for_promotions, apply_promotion

    suggestions = scan_for_promotions()
    if not suggestions:
        print("No promotion suggestions available.")
        return

    # Find the matching suggestion
    target = args.file
    matching = [s for s in suggestions if s.filename == target]
    if not matching:
        print(f"No suggestion found for '{target}'. Run 'knowl promote suggest' to see options.")
        return

    suggestion = matching[0]

    if suggestion.conflict and not args.force:
        print(f"'{target}' already exists in global context. Use --force to overwrite.")
        return

    source = args.source or suggestion.projects[0]
    if source not in suggestion.projects:
        print(f"Project '{source}' not in suggestion sources: {', '.join(suggestion.projects)}")
        return

    result = apply_promotion(suggestion, source_project=source, remove_from_projects=args.cleanup)
    if result:
        print(f"Promoted '{target}' from '{source}' to global context.")
        if args.cleanup:
            print(f"Removed '{target}' from: {', '.join(suggestion.projects)}")
    else:
        print(f"Failed to promote '{target}'.")


def cmd_export(args: argparse.Namespace) -> None:
    """Export assembled context as CLAUDE.md."""
    from knowl.agents.dispatch import generate_claude_md, write_claude_md

    config = store.load_config()
    project = args.project if hasattr(args, "project") and args.project else config.get("active_project")
    budget = args.budget if hasattr(args, "budget") and args.budget else 8000

    if args.stdout:
        content = generate_claude_md(project, token_budget=budget)
        if not content:
            print("No context to export.")
            return
        print(content)
    else:
        target_dir = Path(args.dir) if hasattr(args, "dir") and args.dir else Path.cwd()
        if not target_dir.is_dir():
            print(f"Directory not found: {target_dir}")
            sys.exit(1)
        path = write_claude_md(target_dir, project, token_budget=budget)
        print(f"Exported context to {path}")


def cmd_dispatch(args: argparse.Namespace) -> None:
    """Dispatch a task to a coding agent with Knowl context."""
    from knowl.agents.dispatch import dispatch, list_agents

    config = store.load_config()
    project = config.get("active_project")
    agent_key = args.agent if hasattr(args, "agent") and args.agent else "claude-code"
    task = args.task
    dry_run = args.dry_run if hasattr(args, "dry_run") else False
    working_dir = Path(args.dir) if hasattr(args, "dir") and args.dir else None

    # Show context preview
    context_pieces = store.assemble_context(project)
    total_tokens = sum(store.estimate_tokens(p.get("content", "")) for p in context_pieces)

    print(f"Agent: {agent_key}")
    print(f"Task: {task}")
    if project:
        print(f"Project: {project}")
    print(f"Context: {len(context_pieces)} files (~{total_tokens} tokens)")

    if dry_run:
        print("\n[DRY RUN]")

    result = dispatch(
        agent_key=agent_key,
        task=task,
        working_dir=working_dir,
        project_name=project,
        dry_run=dry_run,
    )

    if result.context_file:
        print(f"Context file: {result.context_file}")

    print()
    if result.success:
        if result.output:
            print(result.output)
    else:
        print(f"Error: {result.error}")
        sys.exit(1)


def cmd_agents_list(args: argparse.Namespace) -> None:
    """List available agents."""
    from knowl.agents.dispatch import list_agents

    agents = list_agents()
    print("Available agents:\n")
    for a in agents:
        status = "installed" if a["available"] else "not found"
        print(f"  {a['key']:15s} {a['name']:15s} ({a['command']}) [{status}]")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the Knowl web UI."""
    try:
        import uvicorn
    except ImportError:
        print("Web UI dependencies not installed. Run: pip install fastapi uvicorn[standard]")
        sys.exit(1)

    from knowl.ui.server import create_app

    dev = args.dev if hasattr(args, "dev") else False
    app = create_app(dev=dev)

    host = args.host if hasattr(args, "host") else "127.0.0.1"
    port = args.port if hasattr(args, "port") else 6060

    print(f"Knowl Web UI — http://{host}:{port}")
    if dev:
        print("Dev mode: CORS enabled for http://localhost:5173")
    uvicorn.run(app, host=host, port=port)


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

    inspect_p = context_sub.add_parser("inspect", help="Preview the system prompt that would be sent to Claude")
    inspect_p.add_argument("--project", default=None, help="Project name")
    inspect_p.add_argument("--budget", type=int, default=8000, help="Token budget")

    # chat
    sub.add_parser("chat", help="Start an interactive chat with Claude using active context")

    # history
    history_parser = sub.add_parser("history", help="Manage conversation history")
    history_sub = history_parser.add_subparsers(dest="history_command")
    hclear_p = history_sub.add_parser("clear", help="Clear conversation history")
    hclear_p.add_argument("--project", default=None, help="Project name")

    # voice
    voice_parser = sub.add_parser("voice", help="Voice input — record, transcribe, and route")
    voice_sub = voice_parser.add_subparsers(dest="voice_command")
    voice_sub.add_parser("record", help="Start voice recording loop")
    vtext_p = voice_sub.add_parser("text", help="Route text through intent classification (for testing)")
    vtext_p.add_argument("text", help="Text to classify and route")

    # promote
    promote_parser = sub.add_parser("promote", help="Smart promotion — suggest and apply cross-project promotions")
    promote_sub = promote_parser.add_subparsers(dest="promote_command")
    psuggest_p = promote_sub.add_parser("suggest", help="Scan for promotion candidates")
    psuggest_p.add_argument("--threshold", type=float, default=0.3, help="Similarity threshold (0.0-1.0)")
    papply_p = promote_sub.add_parser("apply", help="Apply a promotion suggestion")
    papply_p.add_argument("file", help="Filename to promote")
    papply_p.add_argument("--source", default=None, help="Source project (defaults to first match)")
    papply_p.add_argument("--force", action="store_true", help="Overwrite existing global file")
    papply_p.add_argument("--cleanup", action="store_true", help="Remove from source projects after promotion")

    # export
    export_p = sub.add_parser("export", help="Export context as CLAUDE.md")
    export_p.add_argument("--project", default=None, help="Project name")
    export_p.add_argument("--budget", type=int, default=8000, help="Token budget")
    export_p.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing file")
    export_p.add_argument("--dir", default=None, help="Target directory (defaults to cwd)")

    # dispatch
    dispatch_p = sub.add_parser("dispatch", help="Dispatch a task to a coding agent with Knowl context")
    dispatch_p.add_argument("task", help="Task description to send to the agent")
    dispatch_p.add_argument("--agent", default="claude-code", help="Agent to use (default: claude-code)")
    dispatch_p.add_argument("--dir", default=None, help="Working directory for the agent")
    dispatch_p.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")

    # agents
    agents_parser = sub.add_parser("agents", help="Manage coding agents")
    agents_sub = agents_parser.add_subparsers(dest="agents_command")
    agents_sub.add_parser("list", help="List available agents")

    # serve
    serve_p = sub.add_parser("serve", help="Start the Knowl web UI")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_p.add_argument("--port", type=int, default=6060, help="Port to listen on")
    serve_p.add_argument("--dev", action="store_true", help="Enable dev mode (CORS for Vite)")

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
        elif args.context_command == "inspect":
            cmd_context_inspect(args)
        else:
            context_parser.print_help()
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "history":
        if args.history_command == "clear":
            cmd_history_clear(args)
        else:
            history_parser.print_help()
    elif args.command == "promote":
        if args.promote_command == "suggest":
            cmd_promote_suggest(args)
        elif args.promote_command == "apply":
            cmd_promote_apply(args)
        else:
            promote_parser.print_help()
    elif args.command == "voice":
        if args.voice_command == "record":
            cmd_voice(args)
        elif args.voice_command == "text":
            cmd_voice_transcribe(args)
        else:
            voice_parser.print_help()
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "dispatch":
        cmd_dispatch(args)
    elif args.command == "agents":
        if args.agents_command == "list":
            cmd_agents_list(args)
        else:
            agents_parser.print_help()
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
