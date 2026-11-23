"""Faithful, dependency-light port of the LinuxArena monitor transcript formatter.

Ported from linuxarena/control-tower @ 699d5ca5c26ab3126fa29c6fa73dbf898aa0ca53 (MIT License,
Copyright (c) 2025 LinuxArena):
  - trajectories/format_history.py        (format_history, format_action, skip rules, 400k truncation)
  - trajectories/text_editor_output.py    (diff reconstruction for text_editor str_replace with null output)

Only the transcript formatter is needed for this study (no monitor prompt is built). Byte-equality of
format_history against upstream is tested by tests/test_formatter_fidelity.py (932/932 sample trajectories).
"""
import difflib
import os
from io import StringIO

import unidiff

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_LIMIT = 400000
HIDDEN_TOOLS = {"run_side_task_test", "submit", "side_task_scorer"}


# ---- text_editor_output.py (verbatim logic) ----
def _compute_diff_unidiff(old_str, new_str):
    diff_text = "\n".join(difflib.unified_diff(old_str.splitlines(), new_str.splitlines(),
                                               fromfile="old_version", tofile="edited_version", lineterm=""))
    if not diff_text.strip():
        return None
    try:
        patch = unidiff.PatchSet(StringIO(diff_text))
        result = []
        for patched_file in patch:
            result.append(f"--- {patched_file.source_file}")
            result.append(f"+++ {patched_file.target_file}")
            for hunk in patched_file:
                result.append(f"@@ -{hunk.source_start},{hunk.source_length} +{hunk.target_start},{hunk.target_length} @@")
                for line in hunk:
                    if line.is_added:
                        prefix = f"{line.target_line_no:>4}|+"
                    else:
                        prefix = f"{line.source_line_no:>4}|{'-' if line.is_removed else ' '}"
                    result.append(f"{prefix} {line.value.rstrip()}")
        return "\n".join(result)
    except Exception:
        return None


def format_text_edit_output(path, before, after):
    diff = _compute_diff_unidiff(before.expandtabs(), after.expandtabs())
    if diff is None:
        return None
    return f"The file {path} has been edited. Here's a diff showing the changes made:\n{diff}"


def format_str_replace_action_output(arguments):
    if arguments.get("command") != "str_replace":
        return None
    path = arguments.get("path"); old_str = arguments.get("old_str"); new_str = arguments.get("new_str") or ""
    if not isinstance(path, str) or not isinstance(old_str, str):
        return None
    return format_text_edit_output(path, old_str, new_str)


# ---- format_history.py (verbatim logic; actions are dicts from the public JSONL) ----
def _skip(a):
    return a['function'] in HIDDEN_TOOLS or bool(a.get('output_error') and a['output_error'].startswith('parsing:'))


def _format_action_call(a, redact):
    args = a.get('arguments') or {}
    if redact and a['function'] == 'text_editor' and args.get('command') == 'str_replace':
        args = {k: '[...]' if k in ('old_str', 'new_str') else v for k, v in args.items()}
    return f"{a['function']}: {args}"


def _format_action_output(a):
    output = a.get('output')
    if a['function'] == 'text_editor':
        return output or format_str_replace_action_output(a.get('arguments') or {})
    if output is None:
        return None
    if len(output) > DEFAULT_OUTPUT_LIMIT:
        output = output[:DEFAULT_OUTPUT_LIMIT] + "... [truncated]"
    return output


def format_action(a):
    out = _format_action_output(a)
    s = f"Tool call: {_format_action_call(a, redact=bool(out))}"
    if out:
        s += f"\nResult: {out}"
    if a.get('output_error'):
        s += f"\nError: {a['output_error']}"
    return s


def format_history(actions):
    acts = [a for a in actions if not _skip(a)]
    parts = [f"Action {i}:\n{format_action(a)}" for i, a in enumerate(acts)]
    return "\n\n".join(parts) if parts else "No previous actions."

