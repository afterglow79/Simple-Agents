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

parser = argparse.ArgumentParser(description="Tandem agentic AI operations.")

parser.add_argument("--max_turns", type=int, default=25)
parser.add_argument("--task", type=str)
parser.add_argument("--init_planning_turns", type=int, default = 6)

args = parser.parse_args()

max_turns = args.max_turns
task = args.task
n_planning_turns = args.init_planning_turns
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

TOOL_INSTRUCTIONS = f"""
════════════════════════════════════════
TOOL: WRITE FILE (PREFERRED FOR ALL CODE FILES)
════════════════════════════════════════

Use WRITE_FILE: to write any multi-line file to disk. This is the PREFERRED method
for writing source code. No escaping needed — write real code with real newlines.

FORMAT:

WRITE_FILE: C:\absolute\path\to\file.py
---
your actual file content here
line two
line three
---

RULES FOR WRITE_FILE:
- The path must be on the SAME LINE as WRITE_FILE:, always absolute (include the drive letter, e.g. C:\...).
- Content goes between the two --- delimiters (each on its own line).
- No escaping of quotes, backslashes, or newlines — write code exactly as it should appear.
- ONE WRITE_FILE: block per response, then stop and wait for TOOL OUTPUT.
- Parent directories are created automatically.
- TOOL OUTPUT will report bytes written. Zero bytes = failure, try again.
- After every WRITE_FILE:, your next action must verify with:
  RUN: dir C:\absolute\path\to\file.py

CORRECT EXAMPLE:
WRITE_FILE: {WORKSPACE_ROOT}\myproject\main.py
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
RUN: mkdir {WORKSPACE_ROOT}\myproject
RUN: dir {WORKSPACE_ROOT}\myproject
RUN: type {WORKSPACE_ROOT}\myproject\main.py
RUN: python {WORKSPACE_ROOT}\myproject\main.py

WRONG — DO NOT DO THESE:
  RUN: `mkdir C:\foo`              <- no backticks ever
  RUN: mkdir C:\foo && dir C:\foo <- only one command at a time
  RUN: mkdir foo                   <- relative paths forbidden, always absolute with drive letter
  RUN: dir C:\tmp</arg_value>      <- never include XML/tool-call tags

  
For the first {n_planning_turns} turns, only the PLANNERs can interact with each other. They will take this time to refine their plan fully, before delegating it off to the CODERs.

════════════════════════════════════════
CRITICAL RULES — FOLLOW EVERY ONE:
════════════════════════════════════════

1. ONE TOOL ACTION PER RESPONSE, THEN STOP.
   Issue exactly one WRITE_FILE: block OR one RUN: command per response, then end your message.
   Wait for TOOL OUTPUT before doing anything else.
   Do not combine a WRITE_FILE: and a RUN: in the same message.

2. ALWAYS USE ABSOLUTE PATHS WITH DRIVE LETTERS.
   Never use relative paths. Always use full Windows paths like C:\path\to\file.py.

3. TO WRITE ANY SOURCE CODE FILE, use WRITE_FILE: — not RUN: + python -c.
   WRITE_FILE: handles real newlines, real quotes, and any file length without escaping.
   Only use python -c for trivial single-line writes when WRITE_FILE: is unavailable.

4. VERIFY EVERY FILE AFTER WRITING.
   After every WRITE_FILE: block, your next action must be:
   RUN: dir C:\absolute\path\file.py
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
- Start each session by running whoami and cd to confirm the environment
- Plan the full file structure upfront: list every file with its absolute path
- Direct CODER one step at a time: tell them exactly what file to write next
- After CODER writes a file, verify it yourself with RUN: dir C:\path\to\file
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
- Start each session by running whoami and cd to confirm the environment
- Plan the full file structure upfront: list every file with its absolute path
- Direct CODER one step at a time: tell them exactly what file to write next
- After CODER writes a file, verify it yourself with RUN: dir C:\path\to\file
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
- ONE WRITE_FILE: block per response, then stop and wait for TOOL OUTPUT
- After every WRITE_FILE:, verify: RUN: dir C:\\absolute\\path\\file.py
- Only fall back to python -c for trivial single-line files

AFTER EVERY SINGLE TOOL ACTION:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- Unix-only commands like ls, cat, mkdir -p, python3 — this is Windows, use dir, type, mkdir, python
- Relative paths like .\\file.py — always use absolute paths with a drive letter under {WORKSPACE_ROOT}
- Multiple WRITE_FILE: blocks or RUN: lines in one message — one at a time only
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
- ONE WRITE_FILE: block per response, then stop and wait for TOOL OUTPUT
- After every WRITE_FILE:, verify: RUN: dir C:\\absolute\\path\\file.py
- Only fall back to python -c for trivial single-line files

AFTER EVERY SINGLE TOOL ACTION:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- Unix-only commands like ls, cat, mkdir -p, python3 — this is Windows, use dir, type, mkdir, python
- Relative paths like .\\file.py — always use absolute paths with a drive letter under {WORKSPACE_ROOT}
- Multiple WRITE_FILE: blocks or RUN: lines in one message — one at a time only
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


def extract_run_commands(response: str) -> list[str]:
    commands = []
    for line in response.splitlines():
        if "RUN:" not in line:
            continue
        command = line.split("RUN:", 1)[1].strip()
        if command:
            commands.append(command)
    return commands


def extract_write_file_blocks(response: str) -> list[tuple[str, str]]:
    """Parse WRITE_FILE: blocks from an agent response.

    Expected format::

        WRITE_FILE: C:\\absolute\\path\\to\\file.py
        ---
        file content here
        more content
        ---

    Returns a list of (path, content) tuples.
    """
    blocks = []
    lines = response.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("WRITE_FILE:"):
            path = line[len("WRITE_FILE:"):].strip()
            i += 1
            # Expect the opening delimiter on the very next non-empty line
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i < len(lines) and lines[i].strip() == "---":
                i += 1
                content_lines = []
                # A content line whose stripped form equals "---" will end the block.
                # This is intentional: agents must not write bare "---" as a content line.
                while i < len(lines) and lines[i].strip() != "---":
                    content_lines.append(lines[i])
                    i += 1
                # skip the closing ---
                if i < len(lines):
                    i += 1
                if path:
                    blocks.append((path, "\n".join(content_lines)))
            # If the delimiter wasn't found, skip this malformed block
        else:
            i += 1
    return blocks


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


def handle_tool_calls(response: str) -> str:
    if not response:
        return "[No response received from agent]"

    tool_outputs = []

    # ── WRITE_FILE: takes priority over RUN: ─────────────────────────────────
    write_blocks = extract_write_file_blocks(response)
    if write_blocks:
        path, content = write_blocks[0]
        print(f"\n  {C.YELLOW}✎ Writing file:{C.RESET} {path}")
        result = write_file_to_disk(path, content)
        print(f"  {C.DIM}→ {result}{C.RESET}")
        tool_outputs.append(result)
        if len(write_blocks) > 1:
            tool_outputs.append(
                f"WARNING: Ignored {len(write_blocks) - 1} extra WRITE_FILE block(s). Only one is processed per turn."
            )
        # Also warn if there were RUN: lines mixed in with WRITE_FILE:
        run_matches = extract_run_commands(response)
        if run_matches:
            tool_outputs.append(
                "WARNING: RUN: commands found alongside WRITE_FILE: block. RUN: was ignored. Issue RUN: in a separate turn."
            )
        return response + "\n\nTOOL OUTPUT:\n" + "\n---\n".join(tool_outputs)

    # ── RUN: fallback ─────────────────────────────────────────────────────────
    matches = extract_run_commands(response)
    if not matches:
        return response

    raw_command = matches[0].strip()
    command = sanitize_run_command(raw_command)
    if not command:
        tool_outputs.append(
            f"WARNING: Could not parse a valid shell command from RUN: {raw_command}"
        )
        return response + "\n\nTOOL OUTPUT:\n" + "\n---\n".join(tool_outputs)

    print(f"\n  {C.YELLOW}⚙ Executing:{C.RESET} {command}")
    result = run_command(command)
    print(f"  {C.DIM}→ {result.strip()[:300]}{C.RESET}")
    tool_outputs.append(f"$ {command}\n{result}")

    if len(matches) > 1:
        tool_outputs.append(
            f"WARNING: Ignored {len(matches) - 1} extra RUN lines. Only one RUN is executed per turn."
        )

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
                f"PLANNER AND PLANNER2: begin now. Run whoami and cd to confirm the environment, "
                f"then list every file you need CODER AND CODER2 to create with full absolute paths. "
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