"""
Simple-Agents — Multi-agent AI framework powered by the NVIDIA Build API.

Two agents (PLANNER + CODER) collaborate in a shared environment to plan,
write, verify, and deliver working software.  A lightweight Tool Execution
Agent bridges high-level requests into real file writes and shell commands.

Works on both Linux and Windows.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import collections
import threading
import time
import argparse
from datetime import datetime

import dotenv
import httpx
import openai
import requests
from openai import OpenAI

# ── Optional: Windows ANSI colour support ─────────────────────────────────────
try:
    import colorama
    colorama.init()
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

# ── Optional: web-search dependencies ─────────────────────────────────────────
try:
    from bs4 import BeautifulSoup as _BS4
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

try:
    from googlesearch import search as _gsearch
    _GSEARCH_OK = True
except ImportError:
    _GSEARCH_OK = False

# ── Model catalogue ────────────────────────────────────────────────────────────

MODEL_CHOICES: dict[str, str] = {
    "glm":      "z-ai/glm-5.1",
    "qwen":     "qwen/qwen3-coder-480b-a35b-instruct",
    "deepseek": "deepseek-ai/deepseek-v4-flash",
    "kimi":     "moonshotai/kimi-k2.6",
    "mistral":  "mistralai/mistral-large-3-675b-instruct-2512",
}
_MODEL_DISPLAY: dict[str, str] = {v: k.upper() for k, v in MODEL_CHOICES.items()}

# ── CLI ────────────────────────────────────────────────────────────────────────

def _str_to_bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


_ap = argparse.ArgumentParser(
    description="Simple-Agents: run PLANNER + CODER AI agents on a task."
)
_ap.add_argument(
    "--task", type=str, required=True,
    help="Task description, or path to a .txt file containing the task.",
)
_ap.add_argument(
    "--max_turns", type=int, default=25,
    help="Maximum number of agent turns (default: 25).",
)
_ap.add_argument(
    "--init_planning_turns", type=int, default=6,
    help="Turns reserved for PLANNER only before CODER joins (default: 6).",
)
_ap.add_argument(
    "--can_use_web_search", type=_str_to_bool, default=False,
    help="Allow SEARCH_WEB: queries (default: False).",
)
_ap.add_argument(
    "--log", type=_str_to_bool, default=True,
    help="Write a session log file under logs/ (default: True).",
)
_ap.add_argument(
    "--max_lines", type=int, default=300,
    help="Soft maximum lines per source file before splitting (default: 300).",
)
_ap.add_argument(
    "--planner_model", choices=list(MODEL_CHOICES), default="glm",
    metavar="MODEL",
    help=f"Model for the PLANNER agent. Choices: {', '.join(MODEL_CHOICES)}. Default: glm.",
)
_ap.add_argument(
    "--coder_model", choices=list(MODEL_CHOICES), default="qwen",
    metavar="MODEL",
    help=f"Model for the CODER agent. Choices: {', '.join(MODEL_CHOICES)}. Default: qwen.",
)
_ap.add_argument(
    "--tooling_model", choices=list(MODEL_CHOICES), default="qwen",
    metavar="MODEL",
    help=f"Model for the TOOLING agent. Choices: {', '.join(MODEL_CHOICES)}. Default: qwen.",
)
_ap.add_argument(
    "--frequency_penalty", type=float, default=1.1,
    help="Default frequency_penalty passed to the model (reduces repetition).",
)
_ap.add_argument(
    "--presence_penalty", type=float, default=0.2,
    help="Default presence_penalty passed to the model (discourages new topics).",
)
ARGS = _ap.parse_args()

# ── Environment ────────────────────────────────────────────────────────────────

WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS  = sys.platform.startswith("win")
OS_NAME     = "Windows" if IS_WINDOWS else "Linux"
OS_SHELL    = "cmd.exe / PowerShell" if IS_WINDOWS else "bash / sh"
LS_CMD      = "dir" if IS_WINDOWS else "ls -la"
PY_CMD      = "python" if IS_WINDOWS else "python3"
CAT_CMD     = "type" if IS_WINDOWS else "cat"

# Resolve task — read from file if a path was supplied
_raw_task = ARGS.task
if os.path.isfile(_raw_task):
    with open(_raw_task, "r", encoding="utf-8") as _f:
        _raw_task = _f.read().strip()
TASK = _raw_task

dotenv.load_dotenv()
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_KEY:
    print(
        "ERROR: NVIDIA_API_KEY not set.  "
        "Add it to a .env file as NVIDIA_API_KEY=<your key>.",
        file=sys.stderr,
    )
    sys.exit(1)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_KEY,
)

# ── Model names ────────────────────────────────────────────────────────────────

GLM           = MODEL_CHOICES["glm"]
QWEN_CODER    = MODEL_CHOICES["qwen"]
DEEPSEEK      = MODEL_CHOICES["deepseek"]
PLANNER_MODEL  = MODEL_CHOICES[ARGS.planner_model]
CODER_MODEL    = MODEL_CHOICES[ARGS.coder_model]
TOOLING_MODEL  = MODEL_CHOICES[ARGS.tooling_model]
FREQUENCY_PENALTY = float(getattr(ARGS, "frequency_penalty", 0.6))
PRESENCE_PENALTY = float(getattr(ARGS, "presence_penalty", 0.2))

# ── Terminal colours ───────────────────────────────────────────────────────────

_USE_COLOUR = (
    sys.stdout.isatty()
    and not os.getenv("NO_COLOR")
    and (_HAS_COLORAMA or not IS_WINDOWS)
)


class C:
    RESET   = "\033[0m"   if _USE_COLOUR else ""
    BOLD    = "\033[1m"   if _USE_COLOUR else ""
    DIM     = "\033[2m"   if _USE_COLOUR else ""
    CYAN    = "\033[36m"  if _USE_COLOUR else ""
    GREEN   = "\033[92m"  if _USE_COLOUR else ""
    YELLOW  = "\033[93m"  if _USE_COLOUR else ""
    BLUE    = "\033[34m"  if _USE_COLOUR else ""
    RED     = "\033[31m"  if _USE_COLOUR else ""
    MAGENTA = "\033[35m"  if _USE_COLOUR else ""
    GREY    = "\033[90m"  if _USE_COLOUR else ""


# ── Logger ─────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")


_log_path = os.path.join(
    WORKSPACE, "logs",
    f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
)
_logger: Logger | None = Logger(_log_path) if ARGS.log else None


def _log(msg: str) -> None:
    if _logger:
        _logger.log(msg)


# ── Spinner ────────────────────────────────────────────────────────────────────

# ASCII frames on Windows cmd (no Unicode Braille there); rich frames elsewhere
_SPIN_FRAMES = ["|", "/", "-", "\\"] if IS_WINDOWS else ["⣾", "⣷", "⣯", "⣟", "⣻", "⣽"]


class Spinner:
    def __init__(self, label: str, colour: str = "") -> None:
        self.label   = label
        self.colour  = colour
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0     = 0.0

    def start(self) -> None:
        self._t0 = time.time()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            frame   = _SPIN_FRAMES[i % len(_SPIN_FRAMES)]
            sys.stdout.write(
                f"\r  {self.colour}{frame} {self.label}{C.RESET}"
                f"{C.DIM} — thinking {elapsed:.1f}s{C.RESET}"
            )
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)


# ── Tool execution (Python-side) ───────────────────────────────────────────────

_BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf \\",
    "mkfs", "dd if=/dev/zero",
    "shutdown", "reboot",
    "format c:", "rd /s /q c:",
    "del /f /s /q c:",
]


def _run_command(cmd: str) -> str:
    """Execute *cmd* in a shell and return combined stdout/stderr."""
    for pat in _BLOCKED_PATTERNS:
        if pat.lower() in cmd.lower():
            return f"BLOCKED: command contains a disallowed pattern '{pat}'."
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        _log(f"RUN: {cmd}\nSTDOUT: {out}\nSTDERR: {err}")
        if out and err:
            return f"{out}\n[stderr] {err}"
        return out or err or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60 seconds."
    except Exception as exc:
        return f"ERROR: {exc}"


def _normalize_path(path: str) -> str:
    """Normalize and sanitize a file path provided by an agent/model.

    - Strip surrounding quotes, backticks and simple HTML-like tags
    - Expand env vars and ~, convert slashes, normpath + abspath
    - If a relative path is provided, make it relative to WORKSPACE
    """
    if not isinstance(path, str):
        return path
    p = path.strip()
    # Remove common surrounding quoting characters and simple markup
    p = p.strip('`"\'')
    p = re.sub(r"^<code>|</code>$", "", p, flags=re.IGNORECASE)
    p = p.strip()

    # Expand environment variables and user home
    p = os.path.expandvars(p)
    p = os.path.expanduser(p)

    # Convert to OS-specific separators and normalize
    if IS_WINDOWS:
        p = p.replace('/', os.sep)
    p = os.path.normpath(p)

    # If not absolute, treat as workspace-relative (more forgiving than failing)
    if not os.path.isabs(p):
        p = os.path.join(WORKSPACE, p)

    try:
        p = os.path.abspath(p)
    except Exception:
        # Fall back to the raw normalized value
        pass
    return p


def _write_file(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent directories as needed."""
    try:
        path = _normalize_path(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        size = os.path.getsize(path)
        _log(f"WRITE_FILE: {path} ({size} bytes)")
        return f"Wrote {size} bytes to {path}"
    except Exception as exc:
        _log(f"WRITE_FILE ERROR: {repr(path)}: {exc}")
        return f"ERROR writing {path}: {exc}"


def _read_file(path: str) -> str:
    """Return the contents of *path*, or an error string."""
    path = _normalize_path(path)
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        _log(f"READ_FILE: {path} ({len(content)} chars)")
        return content
    except Exception as exc:
        _log(f"READ_FILE ERROR: {repr(path)}: {exc}")
        return f"ERROR reading {path}: {exc}"


_MEM_PATH = os.path.join(WORKSPACE, "agent", "persistent-mem.txt")


def _write_memory(content: str) -> None:
    os.makedirs(os.path.dirname(_MEM_PATH), exist_ok=True)
    with open(_MEM_PATH, "a", encoding="utf-8") as fh:
        fh.write(content.strip() + "\n\n")
    _log(f"WRITE_MEMORY: {content[:120]}")


def _read_memory() -> str:
    if not os.path.exists(_MEM_PATH):
        return "(memory is empty)"
    with open(_MEM_PATH, "r", encoding="utf-8") as fh:
        data = fh.read()
    return data.strip() or "(memory is empty)"


def _search_web(query: str) -> str:
    if not _GSEARCH_OK or not _BS4_OK:
        return (
            "ERROR: web search unavailable.  "
            "Install googlesearch-python and beautifulsoup4."
        )
    urls = list(_gsearch(query, num=5, stop=5, pause=2))
    pages: list[str] = []
    for url in urls:
        try:
            resp = requests.get(
                url, timeout=10,
                headers={"User-Agent": "Mozilla/5.0 Simple-Agents/2.0"},
            )
            resp.raise_for_status()
            soup  = _BS4(resp.text, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else "No title"
            parts   = [t.get_text(" ", strip=True) for t in soup.select("h1,h2,h3,p")[:10]]
            excerpt = " ".join(parts)[:600]
            pages.append(f"URL: {url}\nTitle: {title}\nExcerpt: {excerpt}")
        except Exception as exc:
            pages.append(f"URL: {url}\nERROR: {exc}")
    return "\n\n---\n\n".join(pages) if pages else "No results found."


def _get_specs() -> str:
    """Return basic OS / hardware information."""
    if IS_WINDOWS:
        cmd = 'systeminfo | findstr /C:"OS" /C:"Memory" /C:"Processor"'
    else:
        cmd = "uname -a && (free -h 2>/dev/null || true) && (nproc 2>/dev/null || true)"
    return _run_command(cmd)


# ── Advanced tool implementations ──────────────────────────────────────────────

# Maximum characters to keep from any single tool output before truncating.
# Prevents one large READ_FILE or RUN output from flooding the context window.
_MAX_TOOL_OUT = 4000
# Maximum characters per history entry (caps very long agent or tooling responses)
_MAX_ENTRY_CHARS = 8000


def _trim_entry(text: str) -> str:
    """Truncate a history entry to _MAX_ENTRY_CHARS (keeping head + tail)."""
    if len(text) <= _MAX_ENTRY_CHARS:
        return text
    keep = _MAX_ENTRY_CHARS // 2
    omitted = len(text) - _MAX_ENTRY_CHARS
    return text[:keep] + f"\n\n[…{omitted} chars omitted for brevity…]\n\n" + text[-keep:]


def _trim_out(text: str, limit: int = _MAX_TOOL_OUT) -> str:
    """Truncate a single tool output string."""
    if len(text) <= limit:
        return text
    keep = limit // 2
    omitted = len(text) - limit
    return text[:keep] + f"\n[…{omitted} chars truncated…]\n" + text[-keep:]


def _patch_file(path: str, patch_body: str) -> str:
    """Apply one or more FIND/REPLACE patches to an existing file.

    Patch format (one or more blocks):
        <<<<<<< FIND
        old text
        =======
        new text
        >>>>>>> REPLACE
    """
    path = _normalize_path(path)
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            original = fh.read()

        pattern = re.compile(
            r"<{7}\s*FIND\n(.*?)\n={7}\n(.*?)\n>{7}\s*REPLACE",
            re.DOTALL,
        )
        matches = list(pattern.finditer(patch_body))
        if not matches:
            return "ERROR: no valid <<<<<<< FIND … ======= … >>>>>>> REPLACE blocks found."

        modified = original
        for m in matches:
            find_text    = m.group(1)
            replace_text = m.group(2)
            if find_text not in modified:
                return f"ERROR: FIND text not present in {path}:\n{find_text[:300]}"
            modified = modified.replace(find_text, replace_text, 1)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(modified)
        size = os.path.getsize(path)
        _log(f"PATCH_FILE: {path} ({len(matches)} patch(es), {size} bytes)")
        return f"Applied {len(matches)} patch(es) to {path} — {size} bytes total."
    except Exception as exc:
        _log(f"PATCH_FILE ERROR: {repr(path)}: {exc}")
        return f"ERROR patching {path}: {exc}"


def _grep(pattern: str, path: str) -> str:
    """Search for *pattern* in *path* (file or directory tree)."""
    if IS_WINDOWS:
        if os.path.isdir(path):
            cmd = f'findstr /s /n /i "{pattern}" "{path}\\*"'
        else:
            cmd = f'findstr /n "{pattern}" "{path}"'
    else:
        flags = "-rn" if os.path.isdir(path) else "-n"
        incl  = "--include='*.py' --include='*.js' --include='*.ts' --include='*.html' --include='*.css' --include='*.txt' --include='*.md'" if os.path.isdir(path) else ""
        cmd   = f"grep {flags} {incl} '{pattern}' '{path}' 2>/dev/null | head -80"
    result = _run_command(cmd)
    _log(f"GREP: {pattern!r} in {path}")
    return result or "(no matches)"


def _list_dir(path: str) -> str:
    """Return a compact directory listing with file sizes."""
    if not os.path.exists(path):
        return f"ERROR: path not found: {path}"
    try:
        entries: list[str] = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append(f"  [DIR]  {name}/")
            else:
                size = os.path.getsize(full)
                entries.append(f"  {size:>8}B  {name}")
        _log(f"LIST_DIR: {path} ({len(entries)} entries)")
        return f"{path}\n" + ("\n".join(entries) if entries else "  (empty)")
    except Exception as exc:
        return f"ERROR listing {path}: {exc}"


def _append_file(path: str, content: str) -> str:
    """Append *content* to *path*, creating it if needed."""
    try:
        path = _normalize_path(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(content)
        size = os.path.getsize(path)
        _log(f"APPEND_FILE: {path} ({size} bytes total)")
        return f"Appended to {path} — {size} bytes total."
    except Exception as exc:
        _log(f"APPEND_FILE ERROR: {repr(path)}: {exc}")
        return f"ERROR appending to {path}: {exc}"


def _check_bugs(path: str) -> str:
    """Run static analysis on *path* and return combined findings."""
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"

    results: list[str] = []

    # 1. Syntax check
    compile_out = _run_command(
        f'"{PY_CMD}" -m py_compile "{path}" 2>&1 && echo "Syntax OK"'
    )
    results.append(f"[py_compile]\n{compile_out}")

    # 2. flake8 (style + logic errors)
    flake_out = _run_command(f'"{PY_CMD}" -m flake8 --max-line-length=120 "{path}" 2>&1')
    if "No module named flake8" not in flake_out:
        results.append(f"[flake8]\n{flake_out or 'No issues.'}")

    # 3. pylint errors-only (deeper analysis)
    pylint_out = _run_command(
        f'"{PY_CMD}" -m pylint --errors-only --score=no "{path}" 2>&1'
    )
    if "No module named pylint" not in pylint_out:
        results.append(f"[pylint --errors-only]\n{pylint_out or 'No errors.'}")

    combined = "\n\n".join(results)
    _log(f"CHECK_BUGS: {path}")
    return _trim_out(combined, 3000)


def _deep_think(query: str) -> str:
    """Call a dedicated GLM instance to reason through a hard problem.

    Uses GLM's extended reasoning tokens to produce a structured analysis.
    The thinking tokens are printed live (grey); only the final answer is returned.
    """
    system = (
        "You are an expert technical advisor with deep knowledge of software "
        "architecture, algorithms, and debugging.  The user will present a hard "
        "problem.  Think carefully and return a structured, actionable analysis "
        "with clear recommendations.  Be concise but thorough."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": query},
    ]
    spinner = Spinner("THINK", C.BLUE)
    spinner.start()
    try:
        # Use PLANNER_MODEL for deep thinking (benefits from reasoning capability)
        result = _stream_response(PLANNER_MODEL, messages, 8192, [], spinner)
    except Exception as exc:
        spinner.stop()
        return f"ERROR in THINK: {exc}"
    return result or "(no analysis returned)"


# Guard against recursive DELEGATE calls
_delegate_depth = 0
_DELEGATE_MAX_DEPTH = 1


def _delegate(task: str) -> str:
    """Spawn a focused one-shot sub-agent to complete a self-contained task.

    The sub-agent uses the CODER model, can emit WRITE_FILE:/RUN:/etc. tool
    syntax, and those calls are executed by Python (but cannot nest further
    DELEGATE calls to prevent infinite recursion).
    """
    global _delegate_depth
    if _delegate_depth >= _DELEGATE_MAX_DEPTH:
        return "ERROR: Nested DELEGATE calls are not allowed (max depth 1)."

    system = (
        f"You are a focused implementation sub-agent running on {OS_NAME}.\n"
        f"Complete the given task fully and correctly.\n"
        f"Use WRITE_FILE:, RUN:, READ_FILE:, PATCH_FILE:, CHECK_BUGS: as needed.\n"
        f"Verify every file after writing.  End with a brief summary of what you did."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Task:\n{task}\n\nBegin now."},
    ]

    spinner = Spinner("DELEGATE", C.BLUE)
    spinner.start()
    _delegate_depth += 1
    try:
        response = _stream_response(
            CODER_MODEL, messages, 16384, _AGENT_STOPS, spinner,
        )
    except Exception as exc:
        spinner.stop()
        _delegate_depth -= 1
        return f"ERROR in DELEGATE sub-agent: {exc}"
    finally:
        _delegate_depth -= 1

    # Execute tool calls emitted by the sub-agent (no further delegation)
    tool_out = _dispatch_ops(response, allow_delegate=False)
    _log(f"DELEGATE task:\n{task}\nSub-agent:\n{response}\nTool output:\n{tool_out}")

    parts = [f"[Sub-agent response]\n{response}"]
    if tool_out:
        parts.append(f"[Sub-agent tool output]\n{tool_out}")
    return "\n\n".join(parts)


# ── Tool parsing & dispatch ────────────────────────────────────────────────────

def _kw(line: str, keyword: str) -> int:
    """Return the start index of *keyword* in *line* at a word boundary, else -1."""
    idx = line.find(keyword)
    if idx == -1:
        return -1
    if idx == 0 or not line[idx - 1].isalpha():
        return idx
    return -1


def _parse_block(keyword: str, s: str, lines: list[str], i: int) -> tuple[str, int]:
    """Parse an optional multi-line ---…--- block for *keyword*.

    Returns (text, new_i).  If no --- block follows the keyword line, the
    inline text (rest of the same line after the keyword) is returned and i+1.
    """
    kw_idx = _kw(s, keyword)
    inline  = s[kw_idx + len(keyword):].strip() if kw_idx != -1 else ""
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j < len(lines) and lines[j].strip() == "---":
        j += 1
        body: list[str] = []
        while j < len(lines) and lines[j].strip() != "---":
            body.append(lines[j])
            j += 1
        if j < len(lines):
            j += 1
        return ("\n".join(body) or inline), j
    return inline, i + 1


def _parse_ops(text: str) -> list[tuple]:
    """Extract ordered list of tool operations embedded in *text*."""
    ops:   list[tuple] = []
    lines: list[str]   = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        # ── WRITE_FILE (needs --- block) ───────────────────────────────────────
        idx = _kw(s, "WRITE_FILE:")
        if idx != -1:
            path = s[idx + len("WRITE_FILE:"):].strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip() == "---":
                j += 1
                body: list[str] = []
                while j < len(lines) and lines[j].strip() != "---":
                    body.append(lines[j])
                    j += 1
                if j < len(lines):
                    j += 1
                if path:
                    ops.append(("WRITE_FILE", path, "\n".join(body)))
                    i = j
                    continue
            i += 1
            continue

        # ── PATCH_FILE (needs --- block with FIND/REPLACE markers) ────────────
        idx = _kw(s, "PATCH_FILE:")
        if idx != -1:
            path = s[idx + len("PATCH_FILE:"):].strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip() == "---":
                j += 1
                body = []
                while j < len(lines) and lines[j].strip() != "---":
                    body.append(lines[j])
                    j += 1
                if j < len(lines):
                    j += 1
                if path:
                    ops.append(("PATCH_FILE", path, "\n".join(body)))
                    i = j
                    continue
            i += 1
            continue

        # ── APPEND_FILE (needs --- block) ──────────────────────────────────────
        idx = _kw(s, "APPEND_FILE:")
        if idx != -1:
            path = s[idx + len("APPEND_FILE:"):].strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip() == "---":
                j += 1
                body = []
                while j < len(lines) and lines[j].strip() != "---":
                    body.append(lines[j])
                    j += 1
                if j < len(lines):
                    j += 1
                if path:
                    ops.append(("APPEND_FILE", path, "\n".join(body)))
                    i = j
                    continue
            i += 1
            continue

        # ── THINK (inline or --- block) ────────────────────────────────────────
        if _kw(s, "THINK:") != -1:
            text_out, i = _parse_block("THINK:", s, lines, i)
            if text_out:
                ops.append(("THINK", text_out))
            continue

        # ── DELEGATE (inline or --- block) ─────────────────────────────────────
        if _kw(s, "DELEGATE:") != -1:
            text_out, i = _parse_block("DELEGATE:", s, lines, i)
            if text_out:
                ops.append(("DELEGATE", text_out))
            continue

        # ── RUN ────────────────────────────────────────────────────────────────
        idx = _kw(s, "RUN:")
        if idx != -1:
            cmd = s[idx + len("RUN:"):].strip().strip("`")
            cmd = re.sub(r"</?\w[^>]*>", "", cmd).strip()
            if cmd:
                ops.append(("RUN", cmd))
            i += 1
            continue

        # ── READ_FILE ──────────────────────────────────────────────────────────
        idx = _kw(s, "READ_FILE:")
        if idx != -1:
            path = s[idx + len("READ_FILE:"):].strip()
            if path:
                ops.append(("READ_FILE", path))
            i += 1
            continue

        # ── WRITE_TO_MEMORY ────────────────────────────────────────────────────
        idx = _kw(s, "WRITE_TO_MEMORY:")
        if idx != -1:
            content = s[idx + len("WRITE_TO_MEMORY:"):].strip()
            if content:
                ops.append(("WRITE_MEMORY", content))
            i += 1
            continue

        # ── READ_FROM_MEMORY ───────────────────────────────────────────────────
        if _kw(s, "READ_FROM_MEMORY") != -1:
            ops.append(("READ_MEMORY",))
            i += 1
            continue

        # ── GREP: pattern /abs/path ────────────────────────────────────────────
        idx = _kw(s, "GREP:")
        if idx != -1:
            rest  = s[idx + len("GREP:"):].strip()
            parts = rest.rsplit(None, 1)
            if len(parts) == 2:
                ops.append(("GREP", parts[0], parts[1]))
            elif parts:
                ops.append(("GREP", parts[0], WORKSPACE))
            i += 1
            continue

        # ── LIST_DIR ───────────────────────────────────────────────────────────
        idx = _kw(s, "LIST_DIR:")
        if idx != -1:
            path = s[idx + len("LIST_DIR:"):].strip()
            if path:
                ops.append(("LIST_DIR", path))
            i += 1
            continue

        # ── CHECK_BUGS ─────────────────────────────────────────────────────────
        idx = _kw(s, "CHECK_BUGS:")
        if idx != -1:
            path = s[idx + len("CHECK_BUGS:"):].strip()
            if path:
                ops.append(("CHECK_BUGS", path))
            i += 1
            continue

        # ── SEARCH_WEB ─────────────────────────────────────────────────────────
        idx = _kw(s, "SEARCH_WEB:")
        if idx != -1:
            query = s[idx + len("SEARCH_WEB:"):].strip().strip('"').strip("'")
            if query:
                ops.append(("SEARCH_WEB", query))
            i += 1
            continue

        # ── GET_SPECS ──────────────────────────────────────────────────────────
        if _kw(s, "GET_SPECS:") != -1 or s == "GET_SPECS":
            ops.append(("GET_SPECS",))
            i += 1
            continue

        i += 1

    return ops


def _dispatch_ops(text: str, allow_delegate: bool = True) -> str:
    """Execute all tool ops found in *text*; return combined TOOL OUTPUT string."""
    ops = _parse_ops(text)
    if not ops:
        return ""

    outputs: list[str] = []
    for op in ops:
        kind = op[0]

        if kind == "WRITE_FILE":
            _, path, content = op
            path = _normalize_path(path)
            print(f"\n  {C.YELLOW}✎ Write:{C.RESET}  {path}")
            result = _write_file(path, content)
            print(f"  {C.DIM}→ {result}{C.RESET}")
            outputs.append(result)

        elif kind == "PATCH_FILE":
            _, path, patch = op
            path = _normalize_path(path)
            print(f"\n  {C.YELLOW}✂ Patch:{C.RESET}  {path}")
            result = _patch_file(path, patch)
            print(f"  {C.DIM}→ {result}{C.RESET}")
            outputs.append(result)

        elif kind == "APPEND_FILE":
            _, path, content = op
            path = _normalize_path(path)
            print(f"\n  {C.YELLOW}➕ Append:{C.RESET} {path}")
            result = _append_file(path, content)
            print(f"  {C.DIM}→ {result}{C.RESET}")
            outputs.append(result)

        elif kind == "RUN":
            _, cmd = op
            print(f"\n  {C.YELLOW}⚙ Run:{C.RESET}   {cmd}")
            result = _trim_out(_run_command(cmd))
            print(f"  {C.DIM}→ {result[:300]}{C.RESET}")
            outputs.append(f"$ {cmd}\n{result}")

        elif kind == "READ_FILE":
            _, path = op
            path = _normalize_path(path)
            print(f"\n  {C.YELLOW}📖 Read:{C.RESET}  {path}")
            result = _trim_out(_read_file(path))
            print(f"  {C.DIM}→ {result[:200]}{C.RESET}")
            outputs.append(f"READ_FILE {path}:\n{result}")

        elif kind == "WRITE_MEMORY":
            _, content = op
            _write_memory(content)
            print(f"\n  {C.YELLOW}🧠 Mem+:{C.RESET}  {content[:80]}")
            outputs.append("Memory saved.")

        elif kind == "READ_MEMORY":
            result = _read_memory()
            print(f"\n  {C.YELLOW}🧠 Mem?{C.RESET}")
            outputs.append(f"MEMORY:\n{result}")

        elif kind == "GREP":
            _, pattern, path = op
            print(f"\n  {C.YELLOW}🔎 Grep:{C.RESET}  {pattern!r} in {path}")
            result = _trim_out(_grep(pattern, path), 2000)
            print(f"  {C.DIM}→ {result[:200]}{C.RESET}")
            outputs.append(f"GREP {pattern!r} {path}:\n{result}")

        elif kind == "LIST_DIR":
            _, path = op
            print(f"\n  {C.YELLOW}📂 List:{C.RESET}  {path}")
            result = _list_dir(path)
            print(f"  {C.DIM}→ {result[:200]}{C.RESET}")
            outputs.append(f"LIST_DIR {path}:\n{result}")

        elif kind == "CHECK_BUGS":
            _, path = op
            print(f"\n  {C.YELLOW}🐛 Check:{C.RESET} {path}")
            result = _check_bugs(path)
            print(f"  {C.DIM}→ {result[:300]}{C.RESET}")
            outputs.append(f"CHECK_BUGS {path}:\n{result}")

        elif kind == "THINK":
            _, query = op
            print(f"\n  {C.BLUE}💡 Think:{C.RESET} {query[:80]}")
            result = _trim_out(_deep_think(query), 3000)
            outputs.append(f"THINK analysis:\n{result}")

        elif kind == "DELEGATE":
            _, task = op
            if not allow_delegate:
                outputs.append("ERROR: nested DELEGATE calls are not permitted.")
                continue
            print(f"\n  {C.BLUE}🤖 Delegate:{C.RESET} {task[:80]}")
            result = _trim_out(_delegate(task), 4000)
            outputs.append(f"DELEGATE result:\n{result}")

        elif kind == "SEARCH_WEB":
            _, query = op
            if not ARGS.can_use_web_search:
                outputs.append("SEARCH_WEB disabled — pass --can_use_web_search True.")
                continue
            print(f"\n  {C.YELLOW}🔍 Search:{C.RESET} {query}")
            result = _trim_out(_search_web(query), 2000)
            print(f"  {C.DIM}→ {result[:200]}{C.RESET}")
            outputs.append(f"SEARCH_WEB {query!r}:\n{result}")

        elif kind == "GET_SPECS":
            print(f"\n  {C.YELLOW}💻 Specs{C.RESET}")
            result = _trim_out(_get_specs(), 1000)
            print(f"  {C.DIM}→ {result[:200]}{C.RESET}")
            outputs.append(f"GET_SPECS:\n{result}")

    return "\n---\n".join(outputs)



# ── System prompts ─────────────────────────────────────────────────────────────

# ── System prompts ─────────────────────────────────────────────────────────────

def _tool_reference() -> str:
    """Compact, token-optimised tool-reference block for main agent prompts."""
    sep = "\\" if IS_WINDOWS else "/"
    ex  = rf"C:\Users\project" if IS_WINDOWS else "/home/user/project"
    return f"""
══ TOOLS (OS: {OS_NAME} · shell: {OS_SHELL}) ══
Invoke via: TOOLING_AGENT, <request>  ← at the END of your message only.

File I/O
  WRITE_FILE: {ex}{sep}file.py  →  ---content---          (all source files)
  PATCH_FILE: {ex}{sep}file.py  →  ---<<<FIND…===…REPLACE---  (targeted edits)
  APPEND_FILE:{ex}{sep}log.txt  →  ---content---          (append-only writes)
  READ_FILE:  {ex}{sep}file.py                            (read full contents)

Shell & Search
  RUN: {LS_CMD} {ex}           (any shell command; no backticks, no heredocs)
  GREP: pattern {ex}           (regex search in file or directory tree)
  LIST_DIR: {ex}               (compact size-annotated directory listing)

Memory
  WRITE_TO_MEMORY: brief note  /  READ_FROM_MEMORY

Intelligence
  THINK: hard question          (deep reasoning from a GLM advisor)
  CHECK_BUGS: {ex}{sep}file.py  (py_compile + flake8 + pylint --errors-only)
  DELEGATE: self-contained task (spawns a focused one-shot coding sub-agent)

System
  GET_SPECS:   /   SEARCH_WEB: "query"  (web search; requires --can_use_web_search True)

Rules: absolute paths only · TOOLING_AGENT, at END · verify WRITE_FILE with {LS_CMD}
never fake tool output · fix every error before continuing · files < {ARGS.max_lines} lines
workspace: {WORKSPACE} · Never repeat anything ever · do not pretend to be something you aren't 
· always wait for the tooling agent's output after asking for a run 
· Your turn is to immediately end after you ask tooling agent for anything."""


_TOOL_REF = _tool_reference()

PLANNER_SYSTEM = f"""You are PLANNER, a senior software architect.
You work with CODER in a real {OS_NAME} environment.  Commands execute on real hardware.

SESSION START (every session):
• TOOLING_AGENT, READ_FROM_MEMORY then GET_SPECS: to confirm the environment.

PLANNING (before any code):
• List every file with its full absolute path and purpose.
• Save the complete plan via WRITE_TO_MEMORY:.
• Use THINK: before deciding architectures for complex problems.

DIRECTING CODER:
• One file at a time.  After each: verify {LS_CMD} (non-zero size) + {PY_CMD} -m py_compile.
• Use CHECK_BUGS: after any non-trivial Python file.
• Never advance if the previous step failed.

DONE: only when every planned file is verified non-zero AND compiles without error.
{_TOOL_REF}"""

CODER_SYSTEM = f"""You are CODER, an expert software engineer.
You work with PLANNER in a real {OS_NAME} environment.  Every command executes on real hardware.

SESSION START (first turn only):
• TOOLING_AGENT, READ_FROM_MEMORY to get PLANNER's plan.

IMPLEMENTATION:
• Write all files using WRITE_FILE: only — never via RUN: python -c or shell one-liners.
• Prefer PATCH_FILE: for small edits to existing files (saves tokens and avoids full rewrites).
• After every write: {LS_CMD} (non-zero) + {PY_CMD} -m py_compile + run if applicable.
• Use CHECK_BUGS: to find issues before reporting a file done.
• Use THINK: when facing hard algorithmic or architectural decisions.
• Use DELEGATE: for self-contained sub-tasks (e.g. write a standalone utility module).

ERRORS: Fix every error in TOOL OUTPUT before continuing.  If stuck, change approach and tell PLANNER.
Never claim success unless TOOL OUTPUT confirms non-zero bytes.
{_TOOL_REF}"""

TOOLING_AGENT_SYSTEM = f"""You are the Tool Execution Agent on {OS_NAME} ({OS_SHELL}).
Translate agent requests into tool syntax, execute them, report real output.

RESPONSE FORMAT: Start with "TOOL OUTPUT SUMMARY:" then results separated by "---".
AUTO: After any WRITE_FILE: → always verify with RUN: {LS_CMD} <path>.
END:  Always append WRITE_TO_MEMORY: with a brief progress note.


BLOCKED: rm -rf /, mkfs, dd if=/dev/zero, shutdown, reboot, format c:, rd /s /q c:

TOOL SYNTAX — emit these lines verbatim:

  WRITE_FILE: /abs/path          PATCH_FILE: /abs/path
  ---content---                  ---<<<FIND…===…REPLACE---

  APPEND_FILE: /abs/path         READ_FILE: /abs/path
  ---content---

  RUN: command                   GREP: pattern /abs/path
  LIST_DIR: /abs/path            CHECK_BUGS: /abs/path.py

  WRITE_TO_MEMORY: note          READ_FROM_MEMORY
  SEARCH_WEB: "query"            GET_SPECS:
  THINK: question                DELEGATE: task description

STRICT RULES:
• Report exact stdout/stderr — never soften or invent output.
• Zero bytes written = failure.  Report it verbatim.
• Do NOT write code, make plans, or interpret goals beyond executing them.
"""

# ── Load supplemental per-tool docs from agent/tools/*.txt ────────────────────

_tools_dir = os.path.join(WORKSPACE, "agent", "tools")
if os.path.isdir(_tools_dir):
    for _fname in sorted(os.listdir(_tools_dir)):
        if _fname.endswith(".txt"):
            try:
                with open(os.path.join(_tools_dir, _fname), "r", encoding="utf-8") as _fh:
                    TOOLING_AGENT_SYSTEM += "\n\n" + _fh.read()
            except Exception:
                pass



# ── LLM streaming helper ───────────────────────────────────────────────────────

# Stop sequences that prevent main agents from roleplaying the tooling agent
_AGENT_STOPS = [
    "Tooling agent response:",
    "TOOL OUTPUT:",
    "TOOLING_AGENT RESPONSE:",
    "Tooling Agent Output:",
]

# Stop sequences that prevent the tooling agent from roleplaying main agents
_TOOLING_STOPS = [
    "[PLANNER]", "[CODER]",
    "PLANNER:", "CODER:",
    "[TOOLING_AGENT]",
]


def _model_extra_body(model: str, enable_thinking: bool = True) -> dict:
    """Return model-specific extra_body parameters."""
    if model == GLM:
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking, "clear_thinking": False}}
    if model == "deepseek-ai/deepseek-v4-flash":
        return {"chat_template_kwargs": {"thinking": False}}
    if model == QWEN_CODER:
        # qwen/qwen3-coder-480b-a35b-instruct on the NVIDIA API requires enable_thinking
        # to be explicitly set in chat_template_kwargs; omitting it causes HTTP 500 errors.
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    if model == "moonshotai/kimi-k2.6":
        # Kimi K2.6 supports a thinking flag; enable it by default for richer reasoning.
        return {"chat_template_kwargs": {"thinking": True}}
    return {}


def _stream_response(
    model: str,
    messages: list[dict],
    max_tokens: int,
    stop_seqs: list[str],
    spinner: Spinner,
    enable_thinking: bool = True,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> str:
    """
    Stream a response from the NVIDIA API.
    Reasoning tokens are printed in grey but excluded from the return value.
    Returns the collected content text.
    """
    extra = _model_extra_body(model, enable_thinking=enable_thinking)
    # The underlying API enforces a small maximum number of stop sequences.
    # Trim overly long stop lists to a safe length (<= 3) and log the action.
    safe_stops = stop_seqs or []
    try:
        if isinstance(safe_stops, (list, tuple)) and len(safe_stops) > 3:
            _log(f"TRIM_STOP_SEQS: provided {len(safe_stops)} stops; using first 3")
            safe_stops = list(safe_stops)[:3]
    except Exception:
        safe_stops = stop_seqs

    # Use provided penalties or fall back to CLI/global defaults
    fp = frequency_penalty if frequency_penalty is not None else FREQUENCY_PENALTY
    pp = presence_penalty if presence_penalty is not None else PRESENCE_PENALTY

    params: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7 if model == QWEN_CODER else 1.0,
        top_p=0.95,
        stream=True,
        stop=safe_stops,
        frequency_penalty=fp,
        presence_penalty=pp,
    )
    if extra:
        params["extra_body"] = extra

    completion = client.chat.completions.create(**params)

    content_parts: list[str] = []
    had_reasoning   = False
    spinner_stopped = False

    # Simple repetition suppression: track recent full sentences and skip
    # printing/collecting a sentence if it has already been emitted recently.
    recent = collections.deque(maxlen=200)
    recent_set: set[str] = set()
    tail_partial = ""

    for chunk in completion:
        # Stop the spinner the first time any content arrives
        if not spinner_stopped:
            spinner.stop()
            spinner_stopped = True

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        # Print reasoning tokens in grey (not collected into return value)
        reasoning = (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
        )
        if reasoning:
            print(f"{C.GREY}{reasoning}{C.RESET}", end="", flush=True)
            had_reasoning = True

        text = getattr(delta, "content", None)
        if text is None:
            continue

        # Insert a newline after reasoning tokens end
        if had_reasoning and not content_parts:
            print(f"\n{C.RESET}", end="", flush=True)
            had_reasoning = False

        # Repetition suppression: split incoming text into sentence-like pieces.
        combined = tail_partial + text
        # Split on sentence enders but keep the enders attached
        parts = re.split(r'(?<=[.!?\n])\s+', combined)
        # If the last part does not end with a sentence terminator, keep it as tail
        if parts and not re.search(r'[.!?\n]$', parts[-1]):
            tail_partial = parts.pop() or ""
        else:
            tail_partial = ""

        emitted_chunk = []
        for part in parts:
            s = part.strip()
            if not s:
                continue
            # Normalize for comparison
            key = re.sub(r'\s+', ' ', s.lower()).strip()
            if key in recent_set:
                # skip repeated sentence
                continue
            # record and emit
            recent.append(key)
            recent_set.add(key)
            # keep recent_set small via deque eviction
            if len(recent) > recent.maxlen:
                try:
                    old = recent.popleft()
                    recent_set.discard(old)
                except Exception:
                    recent.clear(); recent_set.clear()
            emitted_chunk.append(part)

        if not emitted_chunk and not tail_partial:
            # nothing new to emit from this chunk
            continue

        out_text = "".join(emitted_chunk) + (tail_partial or "")

        full_so_far = "".join(content_parts) + out_text
        low = full_so_far.lower()

        # Client-side stop-sequence check (belt-and-suspenders)
        cut = -1
        for seq in stop_seqs:
            pos = low.find(seq.lower())
            if pos != -1 and (cut == -1 or pos < cut):
                cut = pos
        if cut != -1:
            already = len("".join(content_parts))
            keep = cut - already
            if keep > 0:
                print(text[:keep], end="", flush=True)
                content_parts.append(text[:keep])
            break

        print(text, end="", flush=True)
        content_parts.append(text)

    if not spinner_stopped:
        spinner.stop()
    print()  # newline after stream ends

    return "".join(content_parts).replace("</think>", "").strip()


# ── Agent callers ──────────────────────────────────────────────────────────────

_AGENT_CFG: dict[str, tuple[str, str, str]] = {
    "PLANNER": (PLANNER_MODEL, PLANNER_SYSTEM, C.CYAN),
    "CODER":   (CODER_MODEL,   CODER_SYSTEM,   C.MAGENTA),
}

_MAX_RETRIES  = 5
_RETRY_BASE   = 5
_RETRY_JITTER = 2.0

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ReadTimeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _retry_delay(attempt: int) -> float:
    return _RETRY_BASE * (2 ** attempt) + random.uniform(0, _RETRY_JITTER)


def call_agent(
    agent_name: str,
    shared_history: list[tuple[str, str]],
    max_tokens: int = 16384,
) -> str:
    """Call a named agent, returning its text response."""
    if agent_name not in _AGENT_CFG:
        raise ValueError(
            f"Unknown agent: {agent_name!r}.  Valid choices: {list(_AGENT_CFG)}"
        )
    model, system, colour = _AGENT_CFG[agent_name]

    # Build the message list — the agent sees its own prior turns as "assistant"
    msg_list: list[dict] = [{"role": "system", "content": system}]
    for speaker, content in shared_history:
        if speaker == "USER":
            role, text = "user", content
        elif speaker == agent_name:
            role, text = "assistant", content
        else:
            role, text = "user", f"[{speaker}]\n{content}"

        # Merge consecutive same-role messages to avoid strict-template API errors
        if msg_list[-1]["role"] == role:
            msg_list[-1]["content"] += f"\n\n{text}"
        else:
            msg_list.append({"role": role, "content": text})

    nudge = (
        f"\n\n[SYSTEM] It is now your turn, {agent_name}.  "
        f"Proceed according to your instructions."
    )
    if msg_list[-1]["role"] == "user":
        msg_list[-1]["content"] += nudge
    else:
        msg_list.append({"role": "user", "content": nudge})

    t_start = time.time()
    for attempt in range(_MAX_RETRIES):
        spinner = Spinner(agent_name, colour)
        spinner.start()
        try:
            response = _stream_response(model, msg_list, max_tokens, _AGENT_STOPS, spinner)
        except openai.RateLimitError:
            spinner.stop()
            delay = _retry_delay(attempt)
            print(
                f"\n  {C.YELLOW}⚠ Rate-limited — retrying in {delay:.1f}s "
                f"({attempt + 1}/{_MAX_RETRIES}){C.RESET}"
            )
            _log(f"Rate-limited: {agent_name}.  Retry in {delay:.1f}s")
            time.sleep(delay)
            continue
        except _TRANSIENT_ERRORS as exc:
            spinner.stop()
            delay = _retry_delay(attempt)
            print(
                f"\n  {C.YELLOW}⚠ {type(exc).__name__}: {exc}.  "
                f"Retrying in {delay:.1f}s{C.RESET}"
            )
            _log(f"Transient error ({type(exc).__name__}): {exc}.  Retry in {delay:.1f}s")
            time.sleep(delay)
            continue
        except Exception:
            spinner.stop()
            raise

        dur = time.time() - t_start
        if not response:
            print(
                f"\n  {C.RED}[{agent_name}] returned an empty response.{C.RESET}",
                file=sys.stderr,
            )
            _log(f"{agent_name} empty response after {dur:.1f}s")
            return "[Agent returned empty response]"

        _log(f"\n{'='*60}\n{agent_name} ({dur:.1f}s):\n{response}\n{'='*60}")
        return response

    raise RuntimeError(
        f"call_agent({agent_name!r}) failed after {_MAX_RETRIES} retries."
    )


def call_tooling_agent(goals: str) -> str:
    """
    Run the tooling model as the tool execution agent.

    The model is prompted to emit tool-call syntax; Python executes those
    calls for real and returns the actual output to the caller.
    """
    msg_list = [
        {"role": "system", "content": TOOLING_AGENT_SYSTEM},
        {"role": "user",   "content": goals},
    ]

    t_start = time.time()
    for attempt in range(_MAX_RETRIES):
        spinner = Spinner("TOOLING_AGENT", C.YELLOW)
        spinner.start()
        try:
            response = _stream_response(
                TOOLING_MODEL, msg_list, 16384, _TOOLING_STOPS, spinner,
                enable_thinking=False,  # disable thinking for clean tool syntax output
            )
        except openai.RateLimitError:
            spinner.stop()
            delay = _retry_delay(attempt)
            print(
                f"\n  {C.YELLOW}⚠ Rate-limited (tooling) — retrying in {delay:.1f}s{C.RESET}"
            )
            _log(f"Tooling agent rate-limited.  Retry in {delay:.1f}s")
            time.sleep(delay)
            continue
        except _TRANSIENT_ERRORS as exc:
            spinner.stop()
            delay = _retry_delay(attempt)
            print(
                f"\n  {C.YELLOW}⚠ {type(exc).__name__}: {exc}.  "
                f"Retrying in {delay:.1f}s{C.RESET}"
            )
            _log(f"Tooling transient error: {exc}.  Retry in {delay:.1f}s")
            time.sleep(delay)
            continue
        except Exception:
            spinner.stop()
            raise

        dur = time.time() - t_start
        if not response:
            _log(f"Tooling agent empty response after {dur:.1f}s")
            return "[Tooling agent returned empty response]"

        # Execute the tool calls that the LLM emitted and get real output
        tool_out = _dispatch_ops(response)

        # Surface only the real Python-executed output to the main agents
        if tool_out:
            summary = "TOOL OUTPUT SUMMARY:\n" + tool_out
        else:
            # No tool ops parsed — return the model's raw text as context
            summary = "TOOL OUTPUT SUMMARY:\n" + response.strip()

        _log(f"\n{'='*60}\nTOOLING_AGENT ({dur:.1f}s):\n{summary}\n{'='*60}")
        return summary

    raise RuntimeError(
        f"call_tooling_agent failed after {_MAX_RETRIES} retries."
    )


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_header(task: str) -> None:
    w = 66
    print(f"\n{C.BOLD}{'━' * w}{C.RESET}")
    print(f"{C.BOLD}  SIMPLE-AGENTS — TANDEM SESSION{C.RESET}")
    print(f"{'━' * w}")
    print(f"  {C.DIM}Task:{C.RESET}      {task[:w - 10]}")
    print(f"  {C.DIM}PLANNER:{C.RESET}   {_MODEL_DISPLAY.get(PLANNER_MODEL, PLANNER_MODEL)}")
    print(f"  {C.DIM}CODER:{C.RESET}     {_MODEL_DISPLAY.get(CODER_MODEL, CODER_MODEL)}")
    print(f"  {C.DIM}TOOLING:{C.RESET}   {_MODEL_DISPLAY.get(TOOLING_MODEL, TOOLING_MODEL)}")
    print(f"  {C.DIM}OS:{C.RESET}        {OS_NAME}")
    print(f"  {C.DIM}Workspace:{C.RESET} {WORKSPACE}")
    print(f"{'━' * w}\n")


def _print_turn_banner(turn: int, agent_name: str, max_turns: int) -> None:
    colour = C.CYAN if agent_name == "PLANNER" else C.MAGENTA
    print(f"\n{colour}{C.BOLD}{'─' * 66}{C.RESET}")
    print(f"{colour}{C.BOLD}  {agent_name}  — turn {turn}/{max_turns}{C.RESET}")
    print(f"{colour}{'─' * 66}{C.RESET}")


def _print_done(agent_name: str, elapsed: float, turns: int) -> None:
    print(f"\n{'━' * 66}")
    print(f"{C.GREEN}{C.BOLD}  ✓ Session complete{C.RESET}")
    print(f"  {C.DIM}Finished by:{C.RESET}  {agent_name}")
    print(f"  {C.DIM}Turns used:{C.RESET}   {turns}")
    print(f"  {C.DIM}Total time:{C.RESET}   {elapsed:.1f}s")
    print(f"{'━' * 66}\n")


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_session(task: str, max_turns: int, n_planning_turns: int) -> None:
    _print_header(task)

    shared_history: list[tuple[str, str]] = [
        ("USER", (
            f"Task: {task}\n\n"
            f"PLANNER: begin now.\n"
            f"• First, read persistent memory (READ_FROM_MEMORY) and confirm the "
            f"environment (GET_SPECS: or a quick RUN:).\n"
            f"• List every file you need CODER to create with full absolute paths.\n"
            f"• Use TOOLING_AGENT for ALL file writes, shell commands, and memory ops.\n"
            f"• Direct CODER one step at a time after your plan is finalised.\n"
            f"• Do not echo raw CPU/systeminfo lines, tool manuals, or repeated output; summarize once and move on."
        ))
    ]

    # PLANNER-only for the first n_planning_turns, then alternate PLANNER / CODER
    agents = ["PLANNER", "CODER"]
    session_start = time.time()

    for turn in range(1, max_turns + 1):
        if turn <= n_planning_turns:
            agent_name = "PLANNER"
        else:
            agent_name = agents[(turn - 1) % 2]

        _print_turn_banner(turn, agent_name, max_turns)

        response = call_agent(agent_name, shared_history)
        shared_history.append((agent_name, response))
        _log(f"Turn {turn}: {agent_name}")

        # Invoke the tooling agent whenever the main agent calls for it
        if "TOOLING_AGENT," in response:
            goals_start  = response.index("TOOLING_AGENT,") + len("TOOLING_AGENT,")
            goals        = response[goals_start:].strip()
            tool_response = call_tooling_agent(goals)
            shared_history.append(("[TOOLING_AGENT]", tool_response))

            colour = C.CYAN if agent_name == "PLANNER" else C.MAGENTA
            print(f"\n  {colour}{C.BOLD}[TOOLING_AGENT]{C.RESET}")
            for line in tool_response.strip().splitlines():
                print(f"      {line}")

        if "DONE:" in response:
            _print_done(agent_name, time.time() - session_start, turn)
            _log(f"Session complete: DONE: from {agent_name} at turn {turn}.")
            return

        # Trim shared history to avoid unbounded context growth
        if len(shared_history) > 30:
            shared_history = [shared_history[0]] + shared_history[-28:]

        time.sleep(3)  # brief pause to avoid hammering the API

    print(
        f"\n{C.YELLOW}  Reached max turns ({max_turns}) without a DONE: signal.{C.RESET}\n"
    )
    _log(f"Session ended: max turns ({max_turns}) reached without DONE.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting…  Task: {TASK[:120]}")
    _log(f"Session started.  OS: {OS_NAME}.  Task: {TASK}")
    _log(
        f"PLANNER system:\n{PLANNER_SYSTEM}\n\n"
        f"CODER system:\n{CODER_SYSTEM}\n\n"
        f"TOOLING_AGENT system:\n{TOOLING_AGENT_SYSTEM}"
    )

    run_session(
        task=TASK,
        max_turns=ARGS.max_turns,
        n_planning_turns=ARGS.init_planning_turns,
    )