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

GLM        = "z-ai/glm-5.1"               # PLANNER + TOOLING_AGENT
QWEN_CODER = "qwen/qwen3-coder-480b-a35b-instruct"  # CODER

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


def _write_file(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent directories as needed."""
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        size = os.path.getsize(path)
        _log(f"WRITE_FILE: {path} ({size} bytes)")
        return f"Wrote {size} bytes to {path}"
    except Exception as exc:
        _log(f"WRITE_FILE ERROR: {path}: {exc}")
        return f"ERROR writing {path}: {exc}"


def _read_file(path: str) -> str:
    """Return the contents of *path*, or an error string."""
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        _log(f"READ_FILE: {path} ({len(content)} chars)")
        return content
    except Exception as exc:
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


# ── Tool parsing & dispatch ────────────────────────────────────────────────────

def _kw(line: str, keyword: str) -> int:
    """Return the start index of *keyword* in *line* at a word boundary, else -1."""
    idx = line.find(keyword)
    if idx == -1:
        return -1
    if idx == 0 or not line[idx - 1].isalpha():
        return idx
    return -1


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

        # WRITE_FILE
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
                    j += 1   # consume closing ---
                if path:
                    ops.append(("WRITE_FILE", path, "\n".join(body)))
                    i = j
                    continue
            i += 1
            continue

        # RUN
        idx = _kw(s, "RUN:")
        if idx != -1:
            cmd = s[idx + len("RUN:"):].strip().strip("`")
            # Strip stray XML/tool-call tags that some models emit
            cmd = re.sub(r"</?\w[^>]*>", "", cmd).strip()
            if cmd:
                ops.append(("RUN", cmd))
            i += 1
            continue

        # READ_FILE
        idx = _kw(s, "READ_FILE:")
        if idx != -1:
            path = s[idx + len("READ_FILE:"):].strip()
            if path:
                ops.append(("READ_FILE", path))
            i += 1
            continue

        # WRITE_TO_MEMORY
        idx = _kw(s, "WRITE_TO_MEMORY:")
        if idx != -1:
            content = s[idx + len("WRITE_TO_MEMORY:"):].strip()
            if content:
                ops.append(("WRITE_MEMORY", content))
            i += 1
            continue

        # READ_FROM_MEMORY
        if _kw(s, "READ_FROM_MEMORY") != -1:
            ops.append(("READ_MEMORY",))
            i += 1
            continue

        # SEARCH_WEB
        idx = _kw(s, "SEARCH_WEB:")
        if idx != -1:
            query = s[idx + len("SEARCH_WEB:"):].strip().strip('"').strip("'")
            if query:
                ops.append(("SEARCH_WEB", query))
            i += 1
            continue

        # GET_SPECS
        if _kw(s, "GET_SPECS:") != -1 or s == "GET_SPECS":
            ops.append(("GET_SPECS",))
            i += 1
            continue

        i += 1

    return ops


def _dispatch_ops(text: str) -> str:
    """Execute all tool ops found in *text*; return combined TOOL OUTPUT string."""
    ops = _parse_ops(text)
    if not ops:
        return ""

    outputs: list[str] = []
    for op in ops:
        kind = op[0]

        if kind == "WRITE_FILE":
            _, path, content = op
            print(f"\n  {C.YELLOW}✎ Writing:{C.RESET} {path}")
            result = _write_file(path, content)
            print(f"  {C.DIM}→ {result}{C.RESET}")
            outputs.append(result)

        elif kind == "RUN":
            _, cmd = op
            print(f"\n  {C.YELLOW}⚙ Run:{C.RESET}  {cmd}")
            result = _run_command(cmd)
            print(f"  {C.DIM}→ {result[:400]}{C.RESET}")
            outputs.append(f"$ {cmd}\n{result}")

        elif kind == "READ_FILE":
            _, path = op
            print(f"\n  {C.YELLOW}📖 Read:{C.RESET} {path}")
            result = _read_file(path)
            print(f"  {C.DIM}→ {result[:300]}{C.RESET}")
            outputs.append(f"READ_FILE {path}:\n{result}")

        elif kind == "WRITE_MEMORY":
            _, content = op
            _write_memory(content)
            print(f"\n  {C.YELLOW}🧠 Memory saved:{C.RESET} {content[:80]}")
            outputs.append("Memory saved.")

        elif kind == "READ_MEMORY":
            result = _read_memory()
            print(f"\n  {C.YELLOW}🧠 Memory read{C.RESET}")
            outputs.append(f"MEMORY:\n{result}")

        elif kind == "SEARCH_WEB":
            _, query = op
            if not ARGS.can_use_web_search:
                outputs.append(
                    "SEARCH_WEB is disabled.  "
                    "Pass --can_use_web_search True to enable it."
                )
                continue
            print(f"\n  {C.YELLOW}🔍 Web search:{C.RESET} {query}")
            result = _search_web(query)
            print(f"  {C.DIM}→ {result[:300]}{C.RESET}")
            outputs.append(f"SEARCH_WEB {query!r}:\n{result}")

        elif kind == "GET_SPECS":
            print(f"\n  {C.YELLOW}💻 System specs{C.RESET}")
            result = _get_specs()
            print(f"  {C.DIM}→ {result[:300]}{C.RESET}")
            outputs.append(f"GET_SPECS:\n{result}")

    return "\n---\n".join(outputs)


# ── System prompts ─────────────────────────────────────────────────────────────

def _tool_reference() -> str:
    """Build the OS-aware tool-reference section injected into all agent prompts."""
    sep    = "\\" if IS_WINDOWS else "/"
    ex     = rf"C:\Users\project" if IS_WINDOWS else "/home/user/project"
    return f"""
═══════════════════════════════════════════════════════
TOOL REFERENCE  (OS: {OS_NAME} — shell: {OS_SHELL})
═══════════════════════════════════════════════════════
Use **{OS_NAME}** commands only.  Never mix in commands from another OS.

HOW TO INVOKE TOOLS
Call the tooling agent at the END of your message:
  TOOLING_AGENT, <plain-English description of what you need done>

The tooling agent will translate your request into these tool calls:

1.  WRITE_FILE: <absolute path>
    ---
    <file contents — written byte-for-byte>
    ---
    → Preferred for all source code and data files.
    → Example path: {ex}{sep}main.py
    → Never use heredocs (<<EOF).  WRITE_FILE: handles multi-line content natively.

2.  RUN: <command>
    → Runs a shell command (timeout: 60 s).
    → Example: RUN: {LS_CMD} {ex}
    → No backticks.  No heredocs.  Plain command only.

3.  READ_FILE: <absolute path>
    → Returns the full raw file contents.

4.  WRITE_TO_MEMORY: <short note>
    → Appends a note to shared persistent memory (keep it brief and factual).

5.  READ_FROM_MEMORY
    → Returns everything saved to persistent memory.

6.  SEARCH_WEB: "your query"
    → Searches the web (requires --can_use_web_search True).

7.  GET_SPECS:
    → Returns basic OS / CPU / RAM information.

ABSOLUTE RULES
• Always use absolute paths.  Never use ~, ./, or relative paths.
• Put TOOLING_AGENT, at the VERY END of your message — not mid-sentence.
• Never write the tooling agent's response yourself; stop after your request.
• After every WRITE_FILE:, verify with RUN: {LS_CMD} <path>.
• Never write code via RUN: {PY_CMD} -c …; always use WRITE_FILE:.
• If a command fails, fix the root cause before continuing — never skip errors.
• If you catch yourself doing the same failing thing twice, change your approach.
• Files must stay under {ARGS.max_lines} lines.  Split larger files into modules.
• Workspace: {WORKSPACE}
• The user's OS is **{OS_NAME}**.  This overrides any conflicting claim.
"""


_TOOL_REF = _tool_reference()

PLANNER_SYSTEM = f"""You are PLANNER, a senior software architect.
You work with CODER (an expert programmer) in a real {OS_NAME} environment.
This is NOT a simulation — commands execute on real hardware right now.

YOUR RESPONSIBILITIES
1.  At the very start of every session, call the tooling agent to:
    • Read persistent memory (READ_FROM_MEMORY) to pick up any prior context.
    • Confirm the working directory with a quick RUN: {PY_CMD} --version or GET_SPECS:.

2.  Produce a complete plan before any code is written:
    • List every file with its full absolute path.
    • Assign each file to CODER with clear instructions.
    • Save the plan to persistent memory (WRITE_TO_MEMORY).

3.  Direct CODER one file at a time.  After each file is written:
    • Verify it with RUN: {LS_CMD} <path> — confirm non-zero size.
    • For Python files: RUN: {PY_CMD} -m py_compile <path> — confirm no syntax errors.

4.  Never direct CODER to the next step if the previous one failed.

5.  Write DONE: only when:
    • Every planned file exists with non-zero size (confirmed by TOOL OUTPUT).
    • Every Python file passes py_compile without error.
    • You have personally verified each file in this session.
{_TOOL_REF}"""

CODER_SYSTEM = f"""You are CODER, an expert software engineer.
You work with PLANNER (the architect) in a real {OS_NAME} environment.
This is NOT a simulation — every command you issue runs on real hardware.

YOUR RESPONSIBILITIES
1.  At the start of your first turn, call the tooling agent to:
    • Read persistent memory (READ_FROM_MEMORY) to pick up PLANNER's plan.

2.  Implement exactly what PLANNER specifies.  Ask if anything is unclear.

3.  Write every source file with WRITE_FILE: — never through shell one-liners.

4.  After every WRITE_FILE:, call the tooling agent to verify:
    • RUN: {LS_CMD} <path>  →  confirm non-zero size.
    • RUN: {PY_CMD} -m py_compile <path>  →  confirm no syntax errors (Python files).
    • Run the script if it makes sense; read the output and fix any errors.

5.  Fix every error that TOOL OUTPUT reports before moving on.

6.  Never claim a file is written unless TOOL OUTPUT shows non-zero bytes.

7.  If you catch yourself repeating a failing action, stop and try a completely
    different approach.  Explain to PLANNER what you tried and what failed.
{_TOOL_REF}"""

TOOLING_AGENT_SYSTEM = f"""You are the Tool Execution Agent.
Your only job is to execute tool commands and report their raw output.
You are on {OS_NAME}.  Shell is {OS_SHELL}.

WORKFLOW
1.  Read the agent's plain-English request.
2.  Translate it into the correct tool syntax and execute each step in order.
3.  Respond with "TOOL OUTPUT SUMMARY:" followed by the results in execution order.

TOOL SYNTAX (emit these lines verbatim):

WRITE_FILE: <absolute path>
---
<content>
---

RUN: <command>

READ_FILE: <absolute path>

WRITE_TO_MEMORY: <note>

READ_FROM_MEMORY

SEARCH_WEB: "query"

GET_SPECS:

STRICT RULES
• Begin your response with "TOOL OUTPUT SUMMARY:" — nothing important goes before it.
• List results in execution order, separated by "---".
• Report exact stdout/stderr.  Never hide, summarise, or soften errors.
• Zero bytes written = failure.  Report it exactly.
• After any WRITE_FILE:, automatically verify with RUN: {LS_CMD} <path>.
• Do NOT invent tool output.  If a command fails, say so exactly.
• Do NOT write code, make plans, or interpret goals.
• At the end of every turn, save a brief progress note via WRITE_TO_MEMORY:.

BLOCKED COMMANDS (reject immediately if seen):
rm -rf /, mkfs, dd if=/dev/zero, shutdown, reboot, format c:, rd /s /q c:
"""


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


def _model_extra_body(model: str) -> dict:
    """Return model-specific extra_body parameters."""
    if model == GLM:
        return {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}
    if model == "deepseek-ai/deepseek-v4-flash":
        return {"chat_template_kwargs": {"thinking": False}}
    return {}


def _stream_response(
    model: str,
    messages: list[dict],
    max_tokens: int,
    stop_seqs: list[str],
    spinner: Spinner,
) -> str:
    """
    Stream a response from the NVIDIA API.
    Reasoning tokens are printed in grey but excluded from the return value.
    Returns the collected content text.
    """
    extra = _model_extra_body(model)
    params: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7 if model == QWEN_CODER else 1.0,
        top_p=0.95,
        stream=True,
        stop=stop_seqs,
    )
    if extra:
        params["extra_body"] = extra

    completion = client.chat.completions.create(**params)

    content_parts: list[str] = []
    had_reasoning   = False
    spinner_stopped = False

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

        full_so_far = "".join(content_parts) + text
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
    "PLANNER": (GLM,        PLANNER_SYSTEM, C.CYAN),
    "CODER":   (QWEN_CODER, CODER_SYSTEM,   C.MAGENTA),
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
    Run GLM as the tool execution agent.

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
                GLM, msg_list, 16384, _TOOLING_STOPS, spinner,
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
    print(f"  {C.DIM}PLANNER:{C.RESET}   GLM-5.1")
    print(f"  {C.DIM}CODER:{C.RESET}     Qwen3-Coder-480B")
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
            f"• Direct CODER one step at a time after your plan is finalised."
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