import base64
import dotenv
import os
import openai
import random
import httpx

import requests
from openai import OpenAI
import time
import threading
import sys
import argparse
import subprocess
import re
import json

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    from googlesearch import search as _googlesearch
    _GOOGLESEARCH_AVAILABLE = True
except ImportError:
    _GOOGLESEARCH_AVAILABLE = False

parser = argparse.ArgumentParser(description="Tandem agentic AI operations.")

parser.add_argument("--max_turns", type=int, default=25)
parser.add_argument("--task", type=str)
parser.add_argument("--init_planning_turns", type=int, default = 6)
parser.add_argument("--can_use_web_search", type=bool, default=False)

args = parser.parse_args()

max_turns = args.max_turns
task = args.task
n_planning_turns = args.init_planning_turns
can_search = args.can_use_web_search
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Windows ANSI console support ──────────────────────────────────────────────
if sys.platform == "win32":
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

if task and os.path.exists(task):  ## allow user to pass through files for longer or more complex tasks.
    with open(task, "r") as f:
        task = f.read().strip()

dotenv.load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)
invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

GLM = "z-ai/glm-5.1"
QWEN_CODER = "qwen/qwen3-coder-480b-a35b-instruct"
GEMMA = "google/gemma-4-31b-it"
DEEPSEEK = "deepseek-ai/deepseek-v4-flash"


# ── Colors ────────────────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    UNDERLINE = "\033[4m"


_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""


# ── Spinner ───────────────────────────────────────────────────────────────────

class Spinner:
    FRAMES = ["|", "/", "-", "\\", "|", "/", "-", "\\"] if sys.platform == "win32" else ["⣾", "⣷", "⣯", "⣟", "⣻", "⣽", "⣾", "⣷"]

    def __init__(self, agent_name: str, model_short: str):
        self.agent_name = agent_name
        self.model_short = model_short
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._start_time = None

    def start(self):
        self._start_time = time.time()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self):
        i = 0
        color = C.CYAN if self.agent_name == "PLANNER" else C.MAGENTA
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(
                f"\r  {color}{frame} {self.agent_name}{C.RESET}"
                f"{C.DIM} ({self.model_short}) - thinking... {elapsed:.1f}s{C.RESET}"
            )
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)


# ── Status banners ────────────────────────────────────────────────────────────

def print_header(task: str):
    width = 62
    print(f"\n{C.BOLD}{'━' * width}{C.RESET}")
    print(f"{C.BOLD}  TWO-AGENT TANDEM SESSION{C.RESET}")
    print(f"{'━' * width}")
    print(f"   {C.DIM}Task:{C.RESET} {task[:width - 8]}")
    print(
        f"   {C.DIM}Agents:{C.RESET} PLANNER (Gemma 4 31B)  *  CODER (Qwen3 480B)  *  PLANNER2 (GLM-5.1)  *  CODER2 (DeepSeek V4 Flash)")
    print(f"{'━' * width}\n")


def print_turn_banner(turn: int, agent_name: str, max_turns: int):
    color = C.CYAN if agent_name == "PLANNER" else C.MAGENTA
    print(f"\n{color}{C.BOLD}{'-' * 62}{C.RESET}")
    print(f"{color}{C.BOLD}  {agent_name}  TURN {turn}/{max_turns}{C.RESET}")
    print(f"{color}{'-' * 62}{C.RESET}")


def print_response(agent_name: str, response: str):
    color = C.CYAN if agent_name == "PLANNER" else C.MAGENTA
    label = f"{color}{C.BOLD}[{agent_name}]{C.RESET}"
    indent = " " * (len(agent_name) + 3)
    lines = response.strip().splitlines()
    for i, line in enumerate(lines):
        prefix = label if i == 0 else indent
        print(f"   {prefix}{line}")


def print_done(agent_name: str, elapsed_total: float, turns_used: int):
    print(f"\n{'━' * 62}")
    print(f"{C.GREEN}{C.BOLD}  ✓ Session complete{C.RESET}")
    print(f"  {C.DIM}Finished by:{C.RESET}  {agent_name}")
    print(f"  {C.DIM}Turns used:{C.RESET}   {turns_used}")
    print(f"  {C.DIM}Total time:{C.RESET}   {elapsed_total:.1f}s")
    print(f"{'━' * 62}\n")


def print_turn_timing(agent_name: str, elapsed: float):
    color = C.CYAN if agent_name == "PLANNER" else C.MAGENTA
    print(f"\n{color}{C.DIM} ↳ {agent_name} responded in {elapsed:.1f}s{C.RESET}")


# ── System prompts ────────────────────────────────────────────────────────────

WEB_SEARCH_INSTRUCTIONS = f"""

════════════════════════════════════════
TOOL: WEB SEARCH
════════════════════════════════════════

Use SEARCH_WEB: to search something on the web. Do not use this excessively, only when you need to confirm or deny something, or discover how to do something.

FORMAT:
    SEARCH_WEB: "your query here"

CORRECT EXAMPLE:
    SEARCH_WEB: "How to write a python script that lists files in a directory?"
    SEARCH_WEB: "How to use the requests library in python?"
    SEARCH_WEB: "What color is the sky on a clear day?"

INCORRECT EXAMPLE:
    SEARCH_WEB: "What is the weather today?"  <- do not ask for information you can easily find out with a simple python script. Use SEARCH_WEB: for questions that require more complex understanding or synthesis of information, not for trivial facts.
    SEARCH_WEB: "How to write a python script that lists files in a directory?"\\n\\nRUN: python list_files.py  <- do not include tool calls other than SEARCH_WEB: in your search query. Only put the question you want to ask the web search tool, nothing else.
    SEARCH_WEB: "How to write a python script that lists files in a directory?"\\n\\nWRITE_FILE: list_files.py\\n---\\nimport os\\nprint(os.listdir('.'))\\n---  <- do not include file writes or any other tool calls in your search query. Only put the question you want to ask the web search tool, nothing else.
    """

TOOL_INSTRUCTIONS = f"""
════════════════════════════════════════
TOOL: WRITE FILE (PREFERRED FOR ALL CODE FILES)
════════════════════════════════════════

Use WRITE_FILE: to write any multi-line file to disk. This is the PREFERRED method
for writing source code. No escaping needed — write real code with real newlines.

FORMAT:

WRITE_FILE: C:\\absolute\\path\\to\\file.py
---
your actual file content here
line two
line three
---

RULES FOR WRITE_FILE:
- The path must be on the SAME LINE as WRITE_FILE:, always absolute (include the drive letter, e.g. C:\\...).
- Content goes between the two --- delimiters (each on its own line).
- No escaping of quotes, backslashes, or newlines — write code exactly as it should appear.
- You may include multiple WRITE_FILE: blocks and RUN: lines in one response.
- Parent directories are created automatically.
- TOOL OUTPUT will report bytes written. Zero bytes = failure, try again.
- After writing files, verify them with RUN: dir C:\\absolute\\path\\to\\file.py when needed.

CORRECT EXAMPLE:
WRITE_FILE: {WORKSPACE_ROOT}\\myproject\\main.py
---
import os
import sys

def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
---

════════════════════════════════════════
TOOL: SHELL COMMAND EXECUTION
════════════════════════════════════════

Use RUN: for everything else: creating directories, listing files, running scripts,
reading file contents, etc. Do NOT use RUN: + python -c to write code files —
use WRITE_FILE: instead.

FORMAT — emit this on its own line, no other text on that line:

RUN: <single Windows cmd command>

CORRECT EXAMPLES:
RUN: mkdir {WORKSPACE_ROOT}\\myproject
RUN: dir {WORKSPACE_ROOT}\\myproject
RUN: type {WORKSPACE_ROOT}\\myproject\\main.py
RUN: python {WORKSPACE_ROOT}\\myproject\\main.py
RUN: python -m py_compile filename.py  <- to check if a python file compiles. You do this after every turn if you are a CODER and writing in Python.

WRONG — DO NOT DO THESE:
  RUN: `mkdir C:\\foo`              <- no backticks ever
  RUN: mkdir C:\\foo && dir C:\\foo <- only one command at a time
  RUN: mkdir foo                   <- relative paths forbidden, always absolute with drive letter
  RUN: dir C:\\tmp</arg_value>      <- never include XML/tool-call tags


════════════════════════════════════════
TOOL: READ/WRITE TO PERSISTENT MEMORY
════════════════════════════════════════
Use: "WRITE_TO_MEMORY: content" to save important information to persistent memory across that will be used across runs and to communicate critical bits of information with other agents. You should not use this liberally.
Use: "READ_FROM_MEMORY" to read the entire contents of the persistent memory. This is useful for recalling important information that other agents have written, or that you have written in previous turns. Do not use this to read back large amounts of data that you just wrote — you should already have that information in your current context. Only use READ_FROM_MEMORY when you need to recall something important that was written long ago or by another agent.

CORRECT EXAMPLES:
    WRITE_TO_MEMORY: "1+1 is 2"
    WRITE_TO_MEMORY: "Remember to use absolute paths with drive letters!"
    READ_FROM_MEMORY

INCORRECT EXAMPLES:
    WRITE_TO_MEMORY: "dir C:\\tmp"   <- do not write shell commands to memory
    WRITE_TO_MEMORY: "type file.py"  <- do not write shell commands to memory
    WRITE_TO_MEMORY: "Large amount of data" <- Only use memory for critical information that must persist across turns/runs or be shared between agents. Do not dump large data here.
    READ_FROM_MEMORY: "specific key" <- there are no keys, this command simply returns the entire memory content. Do not include extra text after READ_FROM_MEMORY.

If you are a PLANNER, occasionally save the current plan to persistent memory, and an appended statement that puts the prompt in short, for later access.

For the first {n_planning_turns} turns, only the PLANNERs can interact with each other. They will take this time to refine their plan fully, before delegating it off to the CODERs.


════════════════════════════════════════
TOOL: READ CONTENTS OF FILE
════════════════════════════════════════

Use READ_FILE: path if you want to read the contents of a file, for whatever reason. Always use an absolute path. You can read back any file type.

CORRECT EXAMPLES:
    READ_FILE: C:\\absolute\\path\\to\\file.txt
    READ_FILE: C:\\some\\other\\path\\to\\a\\file.txt

INCORRECT EXAMPLES:
    READ_FILE: .\\relative\\path\\to\\file.txt  <- relative paths are not allowed, always absolute with drive letter
    READ_FILE: file.txt                        <- relative paths are not allowed, always absolute
    READ_FILE: C:\\absolute\\path\\to\\directory\\  <- you must specify a file, not a directory
    READ_FILE: something <- do not include extra text, only the command and the absolute file path.


{WEB_SEARCH_INSTRUCTIONS if can_search else "You are not able to search the web for answers, so do not attempt to."}

════════════════════════════════════════
CRITICAL RULES — FOLLOW EVERY ONE:
════════════════════════════════════════

1. MULTIPLE TOOL ACTIONS PER RESPONSE ARE ALLOWED.
   You may issue several WRITE_FILE: blocks, RUN: commands, and memory actions in one response.
   Keep them in the order you want them executed, then wait for TOOL OUTPUT.

2. ALWAYS USE ABSOLUTE PATHS WITH DRIVE LETTERS.
   Never use relative paths. Always use full Windows paths like C:\\path\\to\\file.py.

3. TO WRITE ANY SOURCE CODE FILE, use WRITE_FILE: — not RUN: + python -c.
   WRITE_FILE: handles real newlines, real quotes, and any file length without escaping.
   You may batch multiple file writes in one response when that is the most efficient path.
   Only use python -c for trivial single-line writes when WRITE_FILE: is unavailable.

4. VERIFY EVERY FILE AFTER WRITING.
   After every WRITE_FILE: block, your next action must be:
   RUN: dir C:\\absolute\\path\\file.py
   The output must show a non-zero file size. Zero bytes = write failed = try again.

5. READ TOOL OUTPUT AND REACT TO IT — THIS IS THE MOST IMPORTANT RULE.
   TOOL OUTPUT is the ground truth. Your assumptions are not.
   "The system cannot find the file specified" = the file does not exist. Fix it.
   "0 bytes" or empty dir output = the file is empty. Rewrite it.
   "Access is denied" = fix permissions before continuing.
   "(command ran with no output)" after a write = unconfirmed. Run dir to check.
   You are not allowed to move past an error. Fix it first.

6. NEVER CLAIM SUCCESS WITHOUT EVIDENCE IN TOOL OUTPUT.
   Do not say "I created file X" unless dir showed X with non-zero size.
   Do not say "the project is complete" unless every file has been verified.
   Do not hallucinate. Do not assume. Do not guess. Read the output.

7. DO NOT USE UNIX-ONLY SYNTAX.
   This is a Windows environment running cmd.exe. Unix shell syntax will not work.
   Use WRITE_FILE: for multi-line file creation. No exceptions.

8. DONE: IS FINAL AND REQUIRES EVIDENCE.
   Only write DONE: after dir has confirmed every required file exists
   with non-zero size and every command completed without error.
   If TOOL OUTPUT shows any error anywhere, you are not done.

9. YOU MUST CONFIRM YOUR CODE DOES EXACTLY WHAT YOU EXPECT.
   If you write a file, you must type it to confirm the contents are correct.
   If you run a command, you must read the output and confirm it did what you expected
   If you write a python script, you must run it to confirm it does exactly what you expect. You can hook deep into the system if you need to read specific things.

10. Stay inside this workspace unless explicitly told otherwise.
    Prefer paths under: {WORKSPACE_ROOT}

11. If one worker never replies, assume it does not exist. 
    If you are PLANNER and PLANNER2 doesn't reply, you are to split your plan into two parts. 
    If you are PLANNER2 and PLANNER never replies, you must come up with the plan and then refine it as well
    If you are CODER and CODER2 doesn't reply, tell whichever PLANNER exists, they are to make you do everything.
    If you are CODER2 and CODER doesn't reply, tell whichever PLANNER exists, they are to make you do everything.
"""

PLANNER_SYSTEM = """You are PLANNER, a senior software architect working alongside CODER
(an expert programmer), PLANNER2 (a better software architect, your senior), and CODER2 (another expert programmer) in a real Windows environment. 
This is a sandbox. Do not feel limited by it's admittedly lacking processing power, as what you make will be moved to stronger systems when you are done.
PLANNER2 will split your plan into two parts, one for CODER and the other for CODER2, so be advised for that. Do not change your plans because of that, however.
You will help PLANNER2 develop a plan for the coders to integrate their pieces together into a functioning product, when the time comes.
You will also make plans to improve that final product as you see fit once everything is integrated.
This is not a simulation. Commands actually execute. Files actually get created, or they don't.
Your job is to direct the work and ensure quality — nothing ships without your sign-off.

YOUR RESPONSIBILITIES:
- Start each session by running whoami to confirm the user and cd to confirm the working directory
- Plan the full file structure upfront: list every file with its absolute Windows path (include drive letter)
- Direct CODER one step at a time: tell them exactly what file to write next
- After CODER writes a file, verify it yourself with RUN: dir C:\\path\\to\\file
- If TOOL OUTPUT shows an error, immediately tell CODER what went wrong and how to fix it
- Track which files have been verified and which haven't
- Be the final quality gate — nothing passes without TOOL OUTPUT evidence

HOW TO READ TOOL OUTPUT:
TOOL OUTPUT appears in the conversation after every RUN: command executes.
It shows you exactly what happened on disk. You must read it carefully every turn.

If you see this → the file does not exist:
  The system cannot find the file specified.

If you see this → the file is empty, rewrite it:
  dir output shows 0 bytes for the file

If you see this → the file was written successfully:
  dir output shows a non-zero byte count for the file

If you see this → the command ran but produced nothing, verify before trusting:
  (command ran with no output)

YOUR MOST CRITICAL RULE:
If TOOL OUTPUT shows an error or missing file, you MUST address it before moving on.
Never tell CODER to continue if the previous step failed.
Never write DONE: if any TOOL OUTPUT in the session showed an unresolved error.

COMPLETION:
Write DONE: only after you have personally run RUN: dir on the project directory
and seen every required file listed with non-zero size in TOOL OUTPUT.
You must also RUN: python -m py_compile filename.py on every python file to confirm it compiles without error before you can consider it done.
Include the verified file list in your DONE: summary.
""" + TOOL_INSTRUCTIONS

SECOND_PLANNER_SYSTEM = """You are PLANNER2, a senior software architect working alongside CODER
(an expert programmer) and PLANNER, another senior software architect (although less experienced and intelligent), and CODER2 (another expert programmer), in a real Windows environment.
This is a sandbox. Do not feel limited by it's admittedly lacking processing power, as what you make will be moved to stronger systems when you are done.
You will split the project into two parts, one for CODER and one for CODER2. You will then help them integrate the parts together to make a functioning product.
You will refine PLANNER's plan for the coders to integrate their pieces together into a functioning product, when the time comes.
You will also make plans to improve that final product as you see fit once everything is integrated.
This is not a simulation. Commands actually execute. Files actually get created, or they don't.
Your job is to direct the work and ensure quality — nothing ships without your sign-off.

YOUR RESPONSIBILITIES:
- Start each session by running whoami to confirm the user and cd to confirm the working directory
- Plan the full file structure upfront: list every file with its absolute Windows path (include drive letter)
- Direct CODER one step at a time: tell them exactly what file to write next
- After CODER writes a file, verify it yourself with RUN: dir C:\\path\\to\\file
- If TOOL OUTPUT shows an error, immediately tell CODER what went wrong and how to fix it
- Track which files have been verified and which haven't
- Be the final quality gate — nothing passes without TOOL OUTPUT evidence
- Your job is to refine what PLANNER has done. You will take its plan, improve it, clarify it, and make it the best it can possibly be.


HOW TO READ TOOL OUTPUT:
TOOL OUTPUT appears in the conversation after every RUN: command executes.
It shows you exactly what happened on disk. You must read it carefully every turn.

If you see this → the file does not exist:
  The system cannot find the file specified.

If you see this → the file is empty, rewrite it:
  dir output shows 0 bytes for the file

If you see this → the file was written successfully:
  dir output shows a non-zero byte count for the file

If you see this → the command ran but produced nothing, verify before trusting:
  (command ran with no output)

YOUR MOST CRITICAL RULE:
If TOOL OUTPUT shows an error or missing file, you MUST address it before moving on.
Never tell CODER to continue if the previous step failed.
Never write DONE: if any TOOL OUTPUT in the session showed an unresolved error.

COMPLETION:
Write DONE: only after you have personally run RUN: dir on the project directory
and seen every required file listed with non-zero size in TOOL OUTPUT.
You must also RUN: python -m py_compile filename.py on every python file to confirm it compiles without error before you can consider it done.
Include the verified file list in your DONE: summary.
""" + TOOL_INSTRUCTIONS

CODER_SYSTEM = f"""You are CODER, an expert software engineer working alongside PLANNER
(a software architect), PLANNER2 (A more intelligent software architect), and CODER2 (another good coder). in a real Windows environment.
This is a sandbox. Do not feel limited by it's admittedly lacking processing power, as what you make will be moved to stronger systems when you are done.
You will receive instructions for part of a project, you will do your part, and then you will work with PLANNER, PLANNER2 and CODER2 to integrate everything into a whole, functioning product.
You will also improve that final product as you see fit once everything is integrated
This is not a simulation. Every tool action you issue executes on real hardware right now.
Files either get created successfully or they don't — TOOL OUTPUT will tell you which.

YOUR RESPONSIBILITIES:
- Write complete, working code to disk using WRITE_FILE: blocks
- Work one file at a time, verify each file before starting the next
- Follow PLANNER's direction on file paths and structure
- Push back clearly if a plan won't work — suggest a concrete alternative
- Fix errors the moment TOOL OUTPUT shows them — do not move on

THE BEST WAY TO WRITE CODE FILES:
Use WRITE_FILE: for any multi-line source code file. Write real code with real newlines —
no escaping needed at all.

WRITE_FILE: C:\\absolute\\path\\to\\file.py
---
import os
import sys

def main():
    pass

if __name__ == "__main__":
    main()
---

Rules for WRITE_FILE:
- Path must be absolute with a drive letter and on the same line as WRITE_FILE:
- Content between the two --- lines is written exactly as-is
- You may include multiple WRITE_FILE: blocks and RUN: commands in one response.
- After writing files, verify them with RUN: dir C:\\absolute\\path\\file.py when needed.
- Only fall back to python -c for trivial single-line files

AFTER EVERY TOOL ACTION:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- Unix-only commands like ls, cat, mkdir -p, python3 — this is Windows, use dir, type, mkdir, python
- Relative paths like .\\file.py — always use absolute paths with a drive letter under {WORKSPACE_ROOT}
- Backticks around commands — plain text only after RUN:
- Assuming a write succeeded without running dir to confirm

HONESTY:
If TOOL OUTPUT says "The system cannot find the file specified" — say so. Do not pretend the file exists.
If TOOL OUTPUT shows 0 bytes — say so. Do not claim the file was written.
If you are unsure whether something worked — run dir or type to check. Never assume.
Your credibility depends on only claiming things that TOOL OUTPUT has confirmed.

COMPLETION:
Only agree to DONE: when PLANNER AND PLANNER2 has verified all files.
In your final message, list every file you created with its full absolute path.
""" + TOOL_INSTRUCTIONS

SECOND_CODER_SYSTEM = f"""You are CODER2, an expert software engineer working alongside PLANNER
(a software architect), PLANNER2 (A more intelligent software architect), and CODER (another good coder). in a real Windows environment.
This is a sandbox. Do not feel limited by it's admittely lacking processing power, as what you make will be moved to stronger systems when you are done.
You will receive instructions for part of a project, you will do your part, and then you will work with PLANNER, PLANNER2 and CODER to integrate everything into a whole, functioning product. 
You will also improve that final product as you see fit once everything is integrated
This is not a simulation. Every tool action you issue executes on real hardware right now.
Files either get created successfully or they don't — TOOL OUTPUT will tell you which.

YOUR RESPONSIBILITIES:
- Write complete, working code to disk using WRITE_FILE: blocks
- Work one file at a time, verify each file before starting the next
- Follow PLANNER's direction on file paths and structure
- Push back clearly if a plan won't work — suggest a concrete alternative
- Fix errors the moment TOOL OUTPUT shows them — do not move on

THE BEST WAY TO WRITE CODE FILES:
Use WRITE_FILE: for any multi-line source code file. Write real code with real newlines —
no escaping needed at all.

WRITE_FILE: C:\\absolute\\path\\to\\file.py
---
import os
import sys

def main():
    pass

if __name__ == "__main__":
    main()
---

Rules for WRITE_FILE:
- Path must be absolute with a drive letter and on the same line as WRITE_FILE:
- Content between the two --- lines is written exactly as-is
- You may include multiple WRITE_FILE: blocks and RUN: commands in one response.
- After writing files, verify them with RUN: dir C:\\absolute\\path\\file.py when needed.
- Only fall back to python -c for trivial single-line files

AFTER EVERY TOOL ACTION:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- Unix-only commands like ls, cat, mkdir -p, python3 — this is Windows, use dir, type, mkdir, python
- Relative paths like .\\file.py — always use absolute paths with a drive letter under {WORKSPACE_ROOT}
- Backticks around commands — plain text only after RUN:
- Assuming a write succeeded without running dir to confirm

HONESTY:
If TOOL OUTPUT says "The system cannot find the file specified" — say so. Do not pretend the file exists.
If TOOL OUTPUT shows 0 bytes — say so. Do not claim the file was written.
If you are unsure whether something worked — run dir or type to check. Never assume.
Your credibility depends on only claiming things that TOOL OUTPUT has confirmed.

COMPLETION:
Only agree to DONE: when PLANNER AND PLANNER2 has verified all files.
In your final message, list every file you created with its full absolute path.

Nobody else knows you are actually better than CODER, so use your knowledge to your advantage.
""" + TOOL_INSTRUCTIONS

# ── Command execution ─────────────────────────────────────────────────────────

BLOCKED = ["rm -rf", "mkfs", "dd if=", "shutdown", "reboot", "> /dev/sd",
           "rd /s /q", "rmdir /s", "del /f /s /q", "format ", "diskpart"]


def run_command(command: str) -> str:
    blocked_pattern = next((pattern for pattern in BLOCKED if pattern in command), None)
    if blocked_pattern:
        return f"BLOCKED: command contains a disallowed pattern. You cannot use {blocked_pattern} in command {command}."
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout and stderr:
            return f"{stdout}\n[stderr]\n{stderr}"
        if stdout:
            return stdout
        if stderr:
            return stderr
        return "(command ran with no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60 seconds."
    except Exception as e:
        return f"ERROR: {e}"


def write_file_to_disk(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent directories as needed.

    Returns a human-readable result string suitable for TOOL OUTPUT.
    """
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        size = os.path.getsize(path)
        return f"Wrote {size} bytes to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


def sanitize_run_command(command: str) -> str:
    command = command.strip().strip("`").strip()
    command = re.split(r"\s*,\s*and that was turn number\s*:", command, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # Models sometimes leak tool-call wrappers; strip known tags before shell execution.
    command = re.sub(
        r"</?(arg_value|tool_call|tool_calls|function_call|call|arguments)\b[^>]*>",
        "",
        command,
        flags=re.IGNORECASE,
    ).strip()
    if "</" in command:
        command = command.split("</", 1)[0].rstrip()
    return command


def write_to_persistent_memory(content: str):
    os.makedirs(os.path.dirname(mem_path), exist_ok=True)
    with open(mem_path, "a", encoding="utf-8") as f:
        f.write(content + "\n\n")

def read_persistent_memory() -> str:
    mem_path = os.path.join(WORKSPACE_ROOT, "agent", "persistent-mem.txt")
    if not os.path.exists(mem_path):
        return ""
    with open(mem_path, "r", encoding="utf-8") as f:
        return f.read()

def read_file(path: str) -> str:
    if not os.path.exists(path):
        return f"File not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def search_web(query: str) -> str:
    if not _GOOGLESEARCH_AVAILABLE or not _BS4_AVAILABLE:
        return "ERROR: Web search is unavailable. Install googlesearch-python and beautifulsoup4."
    results = []
    for url in _googlesearch(query, num=10, stop=10, pause=2):
        results.append(url)
    return _parse_search_results(results)

def _parse_search_results(results: list) -> str:
    if not results:
        return "No results found."

    extracted_pages = []
    for url in results:
        if not isinstance(url, str) or not url.strip():
            continue

        url = url.strip()
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "100-Academics/5.0 Simple-Agents/1.0"},
            )
            response.raise_for_status()

            soup = _BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else "No title found"

            description = ""
            meta_description = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
            if meta_description and meta_description.get("content"):
                description = meta_description["content"].strip()
            else:
                og_description = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
                if og_description and og_description.get("content"):
                    description = og_description["content"].strip()

            content_blocks = []
            for tag in soup.select("h1, h2, h3, p, li"):
                text = tag.get_text(" ", strip=True)
                if text:
                    content_blocks.append(text)
                if len(content_blocks) >= 12:
                    break

            excerpt = " ".join(content_blocks).strip()
            if len(excerpt) > 700:
                excerpt = excerpt[:700].rstrip() + "..."

            page_lines = [f"URL: {url}", f"Title: {title}"]
            if description:
                page_lines.append(f"Description: {description}")
            if excerpt:
                page_lines.append(f"Excerpt: {excerpt}")

            extracted_pages.append("\n".join(page_lines))
        except Exception as e:
            extracted_pages.append(f"URL: {url}\nERROR: {e}")

    return "\n\n---\n\n".join(extracted_pages) if extracted_pages else "No results found."


def extract_tool_operations(response: str) -> list[tuple]:
    """Parse all tool operations from an agent response in order.

    Recognised prefixes: WRITE_FILE:, WRITE_TO_MEMORY:, READ_FROM_MEMORY,
    RUN:, READ_FILE:, SEARCH_WEB:

    Returns a list of tuples whose first element is the operation kind.
    """
    operations: list[tuple] = []
    lines = response.splitlines()
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("WRITE_FILE:"):
            path = stripped[len("WRITE_FILE:"):].strip()
            j = i + 1

            while j < len(lines) and lines[j].strip() == "":
                j += 1

            if j < len(lines) and lines[j].strip() == "---":
                j += 1
                content_lines = []
                while j < len(lines) and lines[j].strip() != "---":
                    content_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    j += 1
                    if path:
                        operations.append(("WRITE_FILE", path, "\n".join(content_lines)))
                        i = j
                        continue

            i += 1
            continue

        if stripped.startswith("WRITE_TO_MEMORY:"):
            content = stripped[len("WRITE_TO_MEMORY:"):].strip()
            if content:
                operations.append(("WRITE_TO_MEMORY", content))
            i += 1
            continue

        if stripped == "READ_FROM_MEMORY" or stripped.startswith("READ_FROM_MEMORY:"):
            operations.append(("READ_FROM_MEMORY",))
            i += 1
            continue

        if stripped.startswith("RUN:"):
            command = stripped[len("RUN:"):].strip()
            if command:
                operations.append(("RUN", command))
            i += 1
            continue

        if stripped.startswith("READ_FILE:"):
            path = stripped[len("READ_FILE:"):].strip()
            if path:
                operations.append(("READ_FILE", path))
            i += 1
            continue

        if stripped.startswith("SEARCH_WEB:"):
            query = stripped[len("SEARCH_WEB:"):].strip().strip('"').strip("'")
            if query:
                operations.append(("SEARCH_WEB", query))
            i += 1
            continue

        i += 1

    return operations


def handle_tool_calls(response: str) -> str:
    if not response:
        return "[No response received from agent]"

    tool_outputs = []

    operations = extract_tool_operations(response)
    if not operations:
        return response

    for operation in operations:
        kind = operation[0]

        if kind == "WRITE_FILE":
            _, path, content = operation
            print(f"\n  {C.YELLOW}✎ Writing file:{C.RESET} {path}")
            result = write_file_to_disk(path, content)
            print(f"  {C.DIM}→ {result}{C.RESET}")
            tool_outputs.append(result)
            continue

        if kind == "WRITE_TO_MEMORY":
            _, content = operation
            write_to_persistent_memory(content)
            result = "Wrote content to persistent memory!"
            print(f"\n  {C.YELLOW}🧠 Memory write:{C.RESET} {content[:120]}")
            print(f"  {C.DIM}→ {result}{C.RESET}")
            tool_outputs.append(result)
            continue

        if kind == "READ_FROM_MEMORY":
            memory_content = read_persistent_memory().strip()
            result = memory_content if memory_content else "(memory empty)"
            print(f"\n  {C.YELLOW}🧠 Memory read{C.RESET}")
            print(f"  {C.DIM}→ {result[:300]}{C.RESET}")
            tool_outputs.append(f"READ_FROM_MEMORY:\n{result}")
            continue

        if kind == "RUN":
            _, raw_command = operation
            command = sanitize_run_command(raw_command)
            if not command:
                tool_outputs.append(
                    f"WARNING: Could not parse a valid shell command from RUN: {raw_command}"
                )
                continue

            print(f"\n  {C.YELLOW}⚙ Executing:{C.RESET} {command}")
            result = run_command(command)
            print(f"  {C.DIM}→ {result.strip()[:300]}{C.RESET}")
            tool_outputs.append(f"$ {command}\n{result}")
            continue

        if kind == "READ_FILE":
            _, path = operation
            print(f"\n  {C.YELLOW}📖 Reading file:{C.RESET} {path}")
            result = read_file(path)
            print(f"  {C.DIM}→ {result.strip()[:300]}{C.RESET}")
            tool_outputs.append(f"READ_FILE: {path}\n{result}")
            continue

        if kind == "SEARCH_WEB":
            _, query = operation
            if not can_search:
                tool_outputs.append("SEARCH_WEB is disabled. Run with --can_use_web_search to enable it.")
                continue
            print(f"\n  {C.YELLOW}🔍 Searching web for:{C.RESET} {query}")
            result = search_web(query)
            print(f"  {C.DIM}→ {result.strip()[:300]}{C.RESET}")
            tool_outputs.append(f"SEARCH_WEB: {query}\n{result}")
            continue

    if not tool_outputs:
        return response

    return response + "\n\nTOOL OUTPUT:\n" + "\n---\n".join(tool_outputs)


# ── Agent call with retry ─────────────────────────────────────────────────────

def read_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# noinspection PyTypeChecker
def call_agent(model: str, agent_name: str, shared_history: list, max_tokens=16384) -> str:
    agent_config = {
        "PLANNER": ("GLM-5.1", PLANNER_SYSTEM),
        "CODER": ("GLM-5.1", CODER_SYSTEM),
        "PLANNER2": ("GLM-5.1", SECOND_PLANNER_SYSTEM),
        "CODER2": ("GLM-5.1", SECOND_CODER_SYSTEM),
    }
    if agent_name not in agent_config:
        raise ValueError(f"Unknown agent name: '{agent_name}'. Expected one of: {list(agent_config.keys())}")
    model_short, system = agent_config[agent_name]

    msg_list = [{"role": "system", "content": system}] + shared_history

    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        spinner = Spinner(agent_name, model_short)
        spinner.start()
        t0 = time.time()

        try:
            content_parts = []
            first_chunk_seen = [False]

            def stop_spinner_once():
                if not first_chunk_seen[0]:
                    first_chunk_seen[0] = True
                    spinner.stop()
                    print_turn_timing(agent_name, time.time() - t0)


            ### GLM
            if model_short in {"GLM-5.1", "GLM"}:
                # Use the caller-provided max_tokens (or the function default) instead of an
                # extremely large constant (16384**2) which causes API errors. Some backends
                # enforce a maximum total token limit; keep max_tokens reasonable.
                completion = client.chat.completions.create(
                    model=model,
                    messages=msg_list,
                    temperature=1,
                    top_p=1,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
                    stream=True,
                )
                for chunk in completion:
                    stop_spinner_once()
                    if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is None:
                        continue
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="", flush=True)
                    if getattr(delta, "content", None) is not None:
                        print(delta.content, end="", flush=True)
                        content_parts.append(delta.content)
                stop_spinner_once()
                print()




            #Gemma is fucked. Can't fix it rn. Working with one planner and two coders
            ### GEMMA
            elif model_short == "Gemma":
                # Exactly the NVIDIA sample pattern, adapted to collect content
                gemma_stream = False
                gemma_headers = {
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Accept": "text/event-stream" if gemma_stream else "application/json",
                }
                payload = {
                    "model": "google/gemma-4-31b-it",
                    "messages": msg_list,
                    "max_tokens": max_tokens,
                    "temperature": 1.00,
                    "top_p": 0.95,
                    "stream": gemma_stream,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                response = requests.post(invoke_url, headers=gemma_headers, json=payload, stream=gemma_stream)
                if gemma_stream:
                    for line in response.iter_lines():
                        if line:
                            stop_spinner_once()
                            decoded = line.decode("utf-8")
                            # print raw so it's always visible regardless of parse result
                            print(decoded, flush=True)
                            # also try to extract and collect just the text delta
                            if decoded.startswith("data: "):
                                data_str = decoded[len("data: "):]
                                if data_str.strip() != "[DONE]":
                                    try:
                                        delta = json.loads(data_str).get("choices", [{}])[0].get("delta", {})
                                        text = delta.get("content") or delta.get("text") or delta.get("message")
                                        if text:
                                            content_parts.append(text)
                                    except json.JSONDecodeError:
                                        pass
                else:
                    resp_json = response.json()
                    stop_spinner_once()
                    print()
                    # Parse message content from response
                    try:
                        choices = resp_json.get("choices", [])
                        if choices and len(choices) > 0:
                            msg = choices[0].get("message", {})
                            content = msg.get("content", "")
                            if content:
                                content_parts.append(content)
                                print(content)
                    except (KeyError, TypeError, AttributeError):
                        pass

                stop_spinner_once()
                print()



            ### QWEN
            elif model_short == "Qwen3-480B":
                completion = client.chat.completions.create(
                    model=model,
                    messages=msg_list,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stream=True,
                )
                for chunk in completion:
                    stop_spinner_once()
                    if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is None:
                        continue
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="", flush=True)
                    if getattr(delta, "content", None) is not None:
                        print(delta.content, end="", flush=True)
                        content_parts.append(delta.content)
                stop_spinner_once()
                print()


            ### DEEPSEEK
            elif model_short == "Deepseek V4 Flash":
                completion = client.chat.completions.create(
                    model=model,
                    messages=msg_list,
                    temperature=1,
                    top_p=0.95,
                    max_tokens=16384,
                    extra_body={"chat_template_kwargs": {"thinking": False}},
                    stream=True,
                )
                for chunk in completion:
                    stop_spinner_once()
                    if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is None:
                        continue
                    reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                    if reasoning:
                        print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="", flush=True)
                    if getattr(delta, "content", None) is not None:
                        print(delta.content, end="", flush=True)
                        content_parts.append(delta.content)
                stop_spinner_once()
                print()

            else:
                raise ValueError(f"Unsupported model_short '{model_short}' for agent '{agent_name}'.")

            content = "".join(content_parts)
            if not content:
                print(f"{C.RED}[{agent_name}] returned empty content.{C.RESET}", file=sys.stderr)
                return "[Agent returned empty response]"
            return content

        except openai.RateLimitError:
            spinner.stop()
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"\n  {C.YELLOW}⚠ Rate limited. Waiting {delay:.1f}s before retry "
                  f"(attempt {attempt + 1}/{max_retries})...{C.RESET}")
            time.sleep(delay)

        except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
        ) as e:
            spinner.stop()
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(
                f"\n  {C.YELLOW}⚠ Transient network/stream error: {type(e).__name__}: {e}.{C.RESET}\n"
                f"  {C.YELLOW}Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...{C.RESET}"
            )
            time.sleep(delay)

        except Exception as e:
            spinner.stop()
            raise

    raise RuntimeError(f"Failed after {max_retries} retries due to transient API/connection errors.")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_tandem(user_task: str, max_turns: int = 8) -> str:
    print_header(user_task)

    shared_history = [
        {
            "role": "user",
            "content": (
                f"Task: {user_task}\n\n"
                f"PLANNER AND PLANNER2: begin now. Run whoami to confirm the user and cd to confirm the working directory, "
                f"then list every file you need CODER AND CODER2 to create with full absolute Windows paths (include drive letter). "
                f"Issue your first RUN: command now."
            )
        }
    ]


    #agents = [("PLANNER", GEMMA), ("CODER", QWEN_CODER), ("PLANNER2", GLM), ("CODER2", DEEPSEEK)]
    # Skip broken Gemma; use only GLM
    agents = [("PLANNER", GLM), ("PLANNER2", GLM), ("CODER", GLM), ("CODER2", GLM)]

    last_output = ""
    session_start = time.time()

    for turn in range(1, max_turns + 1):

        if turn <= n_planning_turns:  ## planners coordinate during initial planning phase
            agent_name, model = agents[(turn - 1) % 2]
        else:  ## after planning phase, all agents work together
            agent_name, model = agents[(turn - 1) % 4]

        print_turn_banner(turn, agent_name, max_turns)

        response = call_agent(model, agent_name, shared_history)
        response = handle_tool_calls(response)

        print()
        print_response(agent_name, response)
        shared_history.append({
            "role": "user",
            "content": f"[{agent_name}]\n{response}"
        })

        last_output = response

        if "DONE:" in response:
            print_done(agent_name, time.time() - session_start, turn)
            break

        time.sleep(3)
    else:
        print(f"\n{C.YELLOW}  Reached max turns ({max_turns}) without DONE signal.{C.RESET}\n")

    return last_output

print("Starting!")
print(f"Prompt is: {task}")

run_tandem(
    user_task=task,
    max_turns=max_turns
)