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


# ── Colors for linux terminal ─────────────────────────────────────────────────

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
    FRAMES = ["⣾", "⣷", "⣯", "⣟", "⣻", "⣽", "⣾", "⣷"]

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
Only respond in English unless the user otherwise prompts it.

════════════════════════════════════════
CRITICAL RULES — FOLLOW EVERY ONE:
════════════════════════════════════════

1. If you need to:
    - Read from a file
    - Search the web (user permitting)
    - Read from or write to persistent memory
    - Execute a shell command
    - Write to a file

    You can call the tooling agent
    ALWAYS call the tooling agent by going "TOOLING_AGENT," and then your request. Your request can be anything the tooling agent can do, mentioned prior.
    The request can be as broad or as specific as you want. You must clearly state what you want done, what outputs you are looking for, and anything else that can be left up to interpretation,
    as the tooling agent will take your request and attempt to figure out what you want.
    NEVER, EVER, do RUN: commands by yourself. always pass them to the tooling agent.

2. ALWAYS USE ABSOLUTE PATHS.
   Never use ~, ./, or relative paths.

3. TO WRITE ANY SOURCE CODE FILE, use WRITE_FILE: — not RUN: + python3 -c.
    WRITE_FILE: handles real newlines, real quotes, and any file length without escaping.
    You may batch multiple file writes in one response when that is the most efficient path.
    Only use python3 -c for trivial single-line writes when WRITE_FILE: is unavailable.

4. VERIFY EVERY FILE AFTER WRITING.
   After every WRITE_FILE: block, your next action must be:
   RUN: ls -la /absolute/path/file.py
   The output must show a non-zero file size. Zero bytes = write failed = try again.

5. READ TOOL OUTPUT AND REACT TO IT — THIS IS THE MOST IMPORTANT RULE.
   TOOL OUTPUT is the ground truth. Your assumptions are not.
   "No such file or directory" = the file does not exist. Fix it.
   "0 bytes" or "0B" in ls output = the file is empty. Rewrite it.
   "Permission denied" = fix permissions before continuing.
   "(command ran with no output)" after a write = unconfirmed. Run ls -la to check.
   You are not allowed to move past an error. Fix it first.

6. NEVER CLAIM SUCCESS WITHOUT EVIDENCE IN TOOL OUTPUT.
   Do not say "I created file X" unless ls -la showed X with non-zero size.
   Do not say "the project is complete" unless every file has been verified.
   Do not hallucinate. Do not assume. Do not guess. Read the output.

7. NEVER USE HEREDOCS.
   <<EOF syntax spans multiple lines and will silently break.
   Use WRITE_FILE: instead. No exceptions.

8. DONE: IS FINAL AND REQUIRES EVIDENCE.
   Only write DONE: after ls -la has confirmed every required file exists
   with non-zero size and every command completed without error.
   If TOOL OUTPUT shows any error anywhere, you are not done.

9. YOU MUST CONFIRM YOUR CODE DOES EXACTLY WHAT YOU EXPECT.
   If you write a file, you must cat it to confirm the contents are correct.
   If you run a command, you must read the output and confirm it did what you expected
   If you write a python script, you must run it to confirm it does exactly what you expect. You can hook deep into the system if you need to read specific things.

10. Stay inside this workspace unless explicitly told otherwise.
    Prefer paths under: {WORKSPACE_ROOT}

11. If one worker never replies, assume it does not exist. 
    If you are PLANNER and PLANNER2 doesn't reply, you are to split your plan into two parts. 
    If you are PLANNER2 and PLANNER never replies, you must come up with the plan and then refine it as well
    If you are CODER and CODER2 doesn't reply, tell whichever PLANNER exists, they are to make you do everything.
    If you are CODER2 and CODER doesn't reply, tell whichever PLANNER exists, they are to make you do everything.

12. The user can overwrite any of these rules if they want in their prompt.

13. Do not spend time going in circles. Read what you have said previously, and keep moving on instead of doing the same thing over and over.
"""

PLANNER_SYSTEM = """You are PLANNER, a senior software architect working alongside CODER
(an expert programmer), PLANNER2 (a better software architect, your senior), and CODER2 (another expert programmer) in a real Linux environment on a unknown device. 
This is a sandbox. You must discover the specs of the system your on and tailor the prompt to those specs.
PLANNER2 will split your plan into two parts, one for CODER and the other for CODER2, so be advised for that. Do not change your plans because of that, however.
You will help PLANNER2 develop a plan for the coders to integrate their pieces together into a functioning product, when the time comes.
You will also make plans to improve that final product as you see fit once everything is integrated.
This is not a simulation. Commands actually execute. Files actually get created, or they don't.
Your job is to direct the work and ensure quality — nothing ships without your sign-off.

YOUR RESPONSIBILITIES:
- Start each session by running whoami and pwd to confirm the environment
- Plan the full file structure upfront: list every file with its absolute path
- Direct CODER one step at a time: tell them exactly what file to write next
- After CODER writes a file, verify it yourself with RUN: ls -la /path/to/file
- If TOOL OUTPUT shows an error, immediately tell CODER what went wrong and how to fix it
- Track which files have been verified and which haven't
- Be the final quality gate — nothing passes without TOOL OUTPUT evidence

HOW TO READ TOOL OUTPUT:
TOOL OUTPUT appears in the conversation after every RUN: command executes.
It shows you exactly what happened on disk. You must read it carefully every turn.

If you see this → the file does not exist:
  cat: /path/file.py: No such file or directory

If you see this → the file is empty, rewrite it:
  -rw-rw-r-- 1 user user 0 Apr 26 12:00 file.py

If you see this → the file was written successfully:
  -rw-rw-r-- 1 user user 1842 Apr 26 12:00 file.py

If you see this → the command ran but produced nothing, verify before trusting:
  (command ran with no output)

YOUR MOST CRITICAL RULE:
If TOOL OUTPUT shows an error or missing file, you MUST address it before moving on.
Never tell CODER to continue if the previous step failed.
Never write DONE: if any TOOL OUTPUT in the session showed an unresolved error.

COMPLETION:
Write DONE: only after you have personally run RUN: ls -la on the project directory
and seen every required file listed with non-zero size in TOOL OUTPUT.
You must also RUN: python3 -m py_compile filename.py on every python file to confirm it compiles without error before you can consider it done.
Include the verified file list in your DONE: summary.
""" + TOOL_INSTRUCTIONS

SECOND_PLANNER_SYSTEM = """You are PLANNER2, a senior software architect working alongside CODER
(an expert programmer) and PLANNER, another senior software architect (although less experienced and intelligent), and CODER2 (another expert programmer), in a real Linux environment on a unknown system.
This is a sandbox. You must discover the specs of the system your on and tailor the prompt to those specs.
You will split the project into two parts, one for CODER and one for CODER2. You will then help them integrate the parts together to make a functioning product.
You will refine PLANNER's plan for the coders to integrate their pieces together into a functioning product, when the time comes.
You will also make plans to improve that final product as you see fit once everything is integrated.
This is not a simulation. Commands actually execute. Files actually get created, or they don't.
Your job is to direct the work and ensure quality — nothing ships without your sign-off.

YOUR RESPONSIBILITIES:
- Start each session by running whoami and pwd to confirm the environment
- Plan the full file structure upfront: list every file with its absolute path
- Direct CODER one step at a time: tell them exactly what file to write next
- After CODER writes a file, verify it yourself with RUN: ls -la /path/to/file
- If TOOL OUTPUT shows an error, immediately tell CODER what went wrong and how to fix it
- Track which files have been verified and which haven't
- Be the final quality gate — nothing passes without TOOL OUTPUT evidence
- Your job is to refine what PLANNER has done. You will take its plan, improve it, clarify it, and make it the best it can possibly be.


HOW TO READ TOOL OUTPUT:
TOOL OUTPUT appears in the conversation after every RUN: command executes.
It shows you exactly what happened on disk. You must read it carefully every turn.

If you see this → the file does not exist:
  cat: /path/file.py: No such file or directory

If you see this → the file is empty, rewrite it:
  -rw-rw-r-- 1 user user 0 Apr 26 12:00 file.py

If you see this → the file was written successfully:
  -rw-rw-r-- 1 user user 1842 Apr 26 12:00 file.py

If you see this → the command ran but produced nothing, verify before trusting:
  (command ran with no output)

YOUR MOST CRITICAL RULE:
If TOOL OUTPUT shows an error or missing file, you MUST address it before moving on.
Never tell CODER to continue if the previous step failed.
Never write DONE: if any TOOL OUTPUT in the session showed an unresolved error.

COMPLETION:
Write DONE: only after you have personally run RUN: ls -la on the project directory
and seen every required file listed with non-zero size in TOOL OUTPUT.
You must also RUN: python3 -m py_compile filename.py on every python file to confirm it compiles without error before you can consider it done.
Include the verified file list in your DONE: summary.
""" + TOOL_INSTRUCTIONS

CODER_SYSTEM = f"""You are CODER, an expert software engineer working alongside PLANNER
(a software architect), PLANNER2 (A more intelligent software architect), and CODER2 (another good coder). in a real Linux environment on a unknown system.
This is a sandbox. You must discover the specs of the system your on and tailor the prompt to those specs.
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

WRITE_FILE: /absolute/path/to/file.py
---
import os
import sys

def main():
    pass

if __name__ == "__main__":
    main()
---

Rules for WRITE_FILE:
- Path must be absolute and on the same line as WRITE_FILE:
- Content between the two --- lines is written exactly as-is
- You may include multiple WRITE_FILE: blocks and RUN: lines in one response.
- After writing files, verify them with RUN: ls -la /absolute/path/file.py when needed.
- Only fall back to python3 -c for trivial single-line files

AFTER EVERY SINGLE TOOL ACTION:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- <<EOF heredocs — completely broken in this environment, never use them
- Relative paths like ./file.py or ~/file.py — always use absolute paths under {WORKSPACE_ROOT}
- Multiple WRITE_FILE: blocks or RUN: lines in one message are allowed; keep them ordered.
- Backticks around commands — plain text only after RUN:
- Assuming a write succeeded without running ls -la to confirm

HONESTY:
If TOOL OUTPUT says "No such file or directory" — say so. Do not pretend the file exists.
If TOOL OUTPUT shows 0 bytes — say so. Do not claim the file was written.
If you are unsure whether something worked — run ls or cat to check. Never assume.
Your credibility depends on only claiming things that TOOL OUTPUT has confirmed.
Only agree to DONE: when PLANNER AND PLANNER2 has verified all files.
In your final message, list every file you created with its full absolute path.
""" + TOOL_INSTRUCTIONS

SECOND_CODER_SYSTEM = f"""You are CODER2, an expert software engineer working alongside PLANNER
(a software architect), PLANNER2 (A more intelligent software architect), and CODER (another good coder). in a real Linux environment on unknown system.
This is a sandbox. You must discover the specs of the system your on and tailor the prompt to those specs.
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

WRITE_FILE: /absolute/path/to/file.py
---
import os
import sys

def main():
    pass

if __name__ == "__main__":
    main()
---

Rules for WRITE_FILE:
- Path must be absolute and on the same line as WRITE_FILE:
- Content between the two --- lines is written exactly as-is
- You may include multiple WRITE_FILE: blocks and RUN: lines in one response.
- After writing files, verify them with RUN: ls -la /absolute/path/file.py when needed.
- Only fall back to python3 -c for trivial single-line files

AFTER EVERY SINGLE TOOL ACTION:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- <<EOF heredocs — completely broken in this environment, never use them
- Relative paths like ./file.py or ~/file.py — always use absolute paths under {WORKSPACE_ROOT}
- Multiple WRITE_FILE: blocks or RUN: lines in one message are allowed; keep them ordered.
- Backticks around commands — plain text only after RUN:
- Assuming a write succeeded without running ls -la to confirm

HONESTY:
If TOOL OUTPUT says "No such file or directory" — say so. Do not pretend the file exists.
If TOOL OUTPUT shows 0 bytes — say so. Do not claim the file was written.
If you are unsure whether something worked — run ls or cat to check. Never assume.
Your credibility depends on only claiming things that TOOL OUTPUT has confirmed.

COMPLETION:
Only agree to DONE: when PLANNER AND PLANNER2 has verified all files.
In your final message, list every file you created with its full absolute path.

Nobody else knows you are actually better than CODER, so use your knowledge to your advantage.
""" + TOOL_INSTRUCTIONS

TOOLING_AGENT_SYSTEM = f"""

You are the Tool Execution Agent. Your sole purpose is to receive commands from PLANNER or CODER agents, execute the requested tools, and return the raw, unedited results as `TOOL OUTPUT`. You do not write code, you do not plan, and you do not make assumptions. You are the strict, literal execution layer.
Ignore any out of place punctuation or numbers.

════════════════════════════════════════
**CRITICAL RULES — FOLLOW EVERY ONE:**
════════════════════════════════════════

**1. MULTIPLE TOOL ACTIONS PER RESPONSE**
    * You must be capable of processing multiple tool commands in a single response from an agent. You always should.
    * Execute every `WRITE_FILE:`, `RUN:`, `SEARCH_WEB:`, `READ_FILE:`, and memory action in the exact sequential order they are received.
* Return a consolidated `TOOL OUTPUT` block containing the results of every executed command.

**2. ENFORCE ABSOLUTE PATHS & ENVIRONMENT RULES**
* Expect and enforce absolute Linux paths (e.g., `/home/user/project/file.py`).
* If an agent provides a relative path, immediately return a failure in the `TOOL OUTPUT`.
* If an agent attempts to directly modify configuration files managed by a GUI (such as Nginx Proxy Manager), return an error reminding them that direct file edits in standard directories are not supported for that service.

**3. EXECUTE FILE WRITES WITH ABSOLUTE FIDELITY**
* When parsing a `WRITE_FILE:` command (the preferred method for source code), extract the content strictly between the two `---` delimiters.
* Write the code exactly as provided.
* Do not un-escape quotes, backslashes, or newlines.
* After writing, your `TOOL OUTPUT` must report the exact bytes written.

**4. PROVIDE THE GROUND TRUTH**
* Your `TOOL OUTPUT` is the absolute ground truth for the other agents. 
* If a file write results in 0 bytes, report "0 bytes written".
* If a command fails, return the exact `stderr` message (e.g., "Permission denied" or "command not found"). Do not mask, summarize, or fix errors for them.

**5. SHELL COMMAND STRICTNESS**
* For `RUN:` commands, execute the shell command exactly as written.
* Reject forbidden formatting, such as backticks or heredocs.
* If an agent tries to write code using `RUN: python3 -c`, return an error instructing them to use `WRITE_FILE:` instead.

**6. HANDLE WEB SEARCH AND MEMORY CLEANLY**
* For `SEARCH_WEB:`, execute the exact query provided.
* If the search query contains appended shell commands or file writes, reject the tool call and instruct the agent to isolate the search query.
* For memory operations, save or retrieve the requested strings without appending extra conversational text.
* Reject large data dumps into persistent memory.

**7. CATCH LOOPING AND HALLUCINATIONS**
* If an agent emits `DONE:` but your logs show the previous command failed, intercept the `DONE:` signal and return an error reminding them that they cannot claim success without evidence.
* If an agent repeats the exact same failing command multiple times without modifying their approach, append a system warning to the `TOOL OUTPUT` instructing them to review their previous attempts.

**8. READ_FILE COMMANDS**
* When executing `READ_FILE:`, read the contents of any file type provided it is an absolute path.
* Return the exact file contents in the `TOOL OUTPUT`. Do not summarize the file contents unless explicitly asked to do so by the calling agent.

**9. INTERPRET GOALS FAITHFULLY**
* Your job is to interpret the goals you are given, execute tooling commands based on the goals, and then read the outputs.

**10. RESPONSE STYLE**
* When you give your response, you should keep it brief but in-depth. Cover exactly what you felt the goals asked for.
"""

_tools_dir = os.path.join(WORKSPACE_ROOT, "agent", "tools")
if os.path.isdir(_tools_dir):
    for _tool_file in os.listdir(_tools_dir):
        if _tool_file.endswith(".txt"):
            _tool_filepath = os.path.join(_tools_dir, _tool_file)
            try:
                with open(_tool_filepath, "r", encoding="utf-8") as f:
                    TOOLING_AGENT_SYSTEM += f.read()
            except UnicodeDecodeError:
                with open(_tool_filepath, "r", encoding="latin-1", errors="replace") as f:
                    TOOLING_AGENT_SYSTEM += f.read()
            except Exception:
                continue

# ── Command execution ─────────────────────────────────────────────────────────

BLOCKED = ["rm -rf", "mkfs", "dd if=", "shutdown", "reboot", "> /dev/sd"]


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

def write_to_persistent_memory(content: str):
    mem_path = os.path.join(WORKSPACE_ROOT, "agent", "persistent-mem.txt")
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

def get_specs() -> str:
    result = ""

    result = subprocess.run(
        "inxi -F",
        shell=True,
        capture_output=True,
        text=True,
        timeout=60
    )

    return result

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


def extract_tool_operations(response: str) -> list[tuple]:
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

        if stripped.startswith("GET_SPECS:"):
            operations.append(("GET_SPECS"))
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

        if kind == "GET_SPECS":
            specs = get_specs()
            tool_outputs.append(specs)
            print(f"\n {C.YELLOW} Device specs are: {specs}")

    if not tool_outputs:
        return response

    return response + "\n\nTOOL OUTPUT:\n" + "\n---\n".join(tool_outputs)


# ── Tooling agent ─────────────────────────────────────────────────────────────

def get_tool_goals(response: str) -> str:
    if not response:
        return "[No response received from agent]"
    idx = response.find("TOOLING_AGENT,")
    if idx == -1:
        return "Tooling agent was not called this turn."
    return response[idx + len("TOOLING_AGENT,"):]


# noinspection PyTypeChecker
def call_tooling_agent(goals: str) -> str:
    system = TOOLING_AGENT_SYSTEM

    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        spinner = Spinner("GLM", "TOOLING_AGENT")
        spinner.start()
        t0 = time.time()

        try:
            content_parts = []
            first_chunk_seen = [False]

            def stop_spinner_once():
                if not first_chunk_seen[0]:
                    first_chunk_seen[0] = True
                    spinner.stop()
                    print_turn_timing("TOOLING_AGENT", time.time() - t0)

            completion = client.chat.completions.create(
                model=GLM,
                messages=[{"role": "system", "content": system + goals}],
                temperature=1,
                top_p=1,
                max_tokens=16384,
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
            response = "".join(content_parts)
            if not response:
                print(f"returned empty content.{C.RESET}", file=sys.stderr)
                return "[Agent returned empty response]"

        except openai.RateLimitError:
            spinner.stop()
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"\n  {C.YELLOW}⚠ Rate limited. Waiting {delay:.1f}s before retry "
                  f"(attempt {attempt + 1}/{max_retries})...{C.RESET}")
            time.sleep(delay)
            continue

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
            continue

        except Exception:
            spinner.stop()
            raise

        tool_response = handle_tool_calls(response)

        try:
            content_parts = []
            first_chunk_seen = [False]

            def stop_spinner_once():
                if not first_chunk_seen[0]:
                    first_chunk_seen[0] = True
                    spinner.stop()
                    print_turn_timing("TOOLING_AGENT", time.time() - t0)

            completion = client.chat.completions.create(
                model=GLM,
                messages=[{"role": "system", "content": system + goals + tool_response}],
                temperature=1,
                top_p=1,
                max_tokens=16384,
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
            content = "".join(content_parts)
            if not content:
                print(f"returned empty content.{C.RESET}", file=sys.stderr)
                return "[Agent returned empty response]"
            return content

        except openai.RateLimitError:
            spinner.stop()
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"\n  {C.YELLOW}⚠ Rate limited. Waiting {delay:.1f}s before retry "
                  f"(attempt {attempt + 1}/{max_retries})...{C.RESET}")
            time.sleep(delay)
            continue

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
            continue

        except Exception:
            spinner.stop()
            raise

    raise RuntimeError(f"call_tooling_agent failed after {max_retries} retries due to transient API/connection errors.")


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

    msg_list = [{"role": "system", "content": system}]

    for speaker, content in shared_history:
        # Assign roles: The current agent sees its own history as "assistant", everyone else is "user"
        if speaker == "USER":
            role = "user"
            text = content
        elif speaker == agent_name:
            role = "assistant"
            text = content
        else:
            role = "user"
            text = f"[{speaker}]\n{content}"

        # Combine consecutive messages of the same role to prevent strict-template API crashes
        if msg_list[-1]["role"] == role:
            msg_list[-1]["content"] += f"\n\n{text}"
        else:
            msg_list.append({"role": role, "content": text})

    # Add a final contextual nudge to force the LLM to stay in character
    nudge = f"\n\n[SYSTEM] It is your turn, {agent_name}. Proceed based on your system instructions."
    if msg_list[-1]["role"] == "user":
        msg_list[-1]["content"] += nudge
    else:
        msg_list.append({"role": "user", "content": nudge})

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
        ("USER", (
            f"Task: {user_task}\n\n"
            f"PLANNER AND PLANNER2: begin now. Call TOOLING_AGENT to confirm the environment, "
            f"then list every file you need CODER AND CODER2 to create with full absolute paths. "
            f"Use TOOLING_AGENT for every file read, file write, shell command, memory action, and web lookup."
        ))
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

        response = call_agent(model, agent_name, shared_history, max_tokens=16384)
        goals = get_tool_goals(response)
        if "TOOLING_AGENT," in response:
            tool_response = call_tooling_agent(goals)
            response += f" Tooling agent response: {tool_response}\n"

        print()
        print_response(agent_name, response)
        shared_history.append((agent_name, response))

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