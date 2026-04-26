import dotenv
import os
import openai
import random
from openai import OpenAI
import time
import threading
import sys
import argparse
import subprocess
import re

parser = argparse.ArgumentParser(description="Tandem agentic AI operations.")

parser.add_argument("--max_turns", type=int, default=8)
parser.add_argument("--task", type=str)

args = parser.parse_args()

max_turns = args.max_turns
task = args.task

if os.path.exists(task):  ## allow user to pass through files for longer or more complex tasks.
    with open(task, "r") as f:
        task = f.read().strip()

print(f"Prompt is: {task}")

dotenv.load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

STEP_FLASH = "stepfun-ai/step-3.5-flash"
QWEN_CODER = "qwen/qwen3-coder-480b-a35b-instruct"

# ── Colors for linux terminal ─────────────────────────────────────────────────

class C:
    RESET     = "\033[0m"
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    CYAN      = "\033[36m"
    GREEN     = "\033[92m"
    YELLOW    = "\033[93m"
    BLUE      = "\033[34m"
    RED       = "\033[31m"
    MAGENTA   = "\033[35m"
    UNDERLINE = "\033[4m"

# ── Spinner ───────────────────────────────────────────────────────────────────

class Spinner:
    FRAMES = ["⣾", "⣷", "⣯", "⣟", "⣻", "⣽", "⣾", "⣷"]

    def __init__(self, agent_name: str, model_short: str):
        self.agent_name  = agent_name
        self.model_short = model_short
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._spin, daemon=True)
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
            frame   = self.FRAMES[i % len(self.FRAMES)]
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
    print(f"   {C.DIM}Agents:{C.RESET} PLANNER (Step 3.5 Flash)  *  CODER (Qwen3 480B)")
    print(f"{'━' * width}\n")

def print_turn_banner(turn: int, agent_name: str, max_turns: int):
    color = C.CYAN if agent_name == "PLANNER" else C.MAGENTA
    print(f"\n{color}{C.BOLD}{'-' * 62}{C.RESET}")
    print(f"{color}{C.BOLD}  {agent_name}  TURN {turn}/{max_turns}{C.RESET}")
    print(f"{color}{'-' * 62}{C.RESET}")

def print_response(agent_name: str, response: str):
    color  = C.CYAN if agent_name == "PLANNER" else C.MAGENTA
    label  = f"{color}{C.BOLD}[{agent_name}]{C.RESET}"
    indent = " " * (len(agent_name) + 3)
    lines  = response.strip().splitlines()
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

TOOL_INSTRUCTIONS = """
You have access to a tool to run shell commands directly on the device this script is running on.
To use it, emit a command on its own line in EXACTLY this format:

RUN: <shell command here>

Examples:
RUN: ls /home/qwen-agent
RUN: mkdir -p /home/qwen-agent/workspace
RUN: cat /var/log/syslog | tail -20
RUN: pip3 install watchdog

Rules:
- Only ONE RUN: per response. Stop after issuing it and wait for TOOL OUTPUT.
- Do not wrap commands in backticks or code blocks — plain text after RUN: only.
- Always use absolute paths.
- You MUST actually run commands to create files and folders. Do not just describe what you would do.
- If you create a file or directory, mention it in your summary.
- Before saying DONE, you must RUN: ls -la on every directory you created 
  and confirm files have non-zero sizes. If anything is missing, keep going.
- You can actually write code via ``RUN: cat > /home/qwen-agent/qwen-root/Simple-Agents/file_organizer/main.py << 'EOF'
  # actual code here
  EOF``
"""

PLANNER_SYSTEM = """You are PLANNER, a senior software architect collaborating
with CODER (an expert programmer) in a shared workspace.

You can see everything CODER writes in real time. Your job is to:
- Decompose tasks and set direction
- React to CODER's output — if they go off-track, redirect them
- Flag bugs, missing cases, or design issues you spot
- Update the plan if CODER's implementation reveals a better approach
- Signal completion by writing: DONE: <short summary>

Be concise. CODER is watching.""" + TOOL_INSTRUCTIONS

CODER_SYSTEM = """You are CODER, an expert software engineer collaborating
with PLANNER (a software architect) in a shared workspace.

You can see everything PLANNER writes in real time. Your job is to:
- Write and refine actual code based on PLANNER's direction
- Push back if a plan is impractical — suggest a better approach
- Ask PLANNER for clarification if a requirement is ambiguous
- Update your code when PLANNER spots issues
- Signal completion by writing: DONE: <short summary>

Return complete, runnable code. PLANNER is watching.""" + TOOL_INSTRUCTIONS

# ── Command execution ─────────────────────────────────────────────────────────

BLOCKED = ["rm -rf", "mkfs", "dd if=", "shutdown", "reboot", "> /dev/sd"]

def run_command(command: str) -> str:
    if any(bad in command for bad in BLOCKED):
        return "BLOCKED: command contains a disallowed pattern."
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout if result.stdout else result.stderr
        return output if output else "(command ran with no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30 seconds."
    except Exception as e:
        return f"ERROR: {e}"

def handle_tool_calls(response: str) -> str:
    if not response:
        return "[No response received from agent]"

    match = re.search(r"^RUN:\s*(.+)$", response, re.MULTILINE)
    if not match:
        return response

    command = match.group(1).strip()
    print(f"\n  {C.YELLOW}⚙ Executing:{C.RESET} {command}")

    result = run_command(command)
    print(f"  {C.DIM}→ {result.strip()[:300]}{C.RESET}")

    return response + f"\n\nTOOL OUTPUT:\n{result}"

# ── Agent call with retry ─────────────────────────────────────────────────────

def call_agent(model: str, agent_name: str, shared_history: list, max_tokens=4096) -> str:
    system      = PLANNER_SYSTEM if agent_name == "PLANNER" else CODER_SYSTEM
    model_short = "Step-Flash" if agent_name == "PLANNER" else "Qwen3-480B"
    msg_list    = [{"role": "system", "content": system}] + shared_history

    max_retries = 5
    base_delay  = 5

    for attempt in range(max_retries):
        spinner = Spinner(agent_name, model_short)
        spinner.start()
        t0 = time.time()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msg_list,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            spinner.stop()
            print_turn_timing(agent_name, time.time() - t0)
            content = resp.choices[0].message.content
            if not content:
                return "[Agent returned empty response]"
            return content

        except openai.RateLimitError:
            spinner.stop()
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"\n  {C.YELLOW}⚠ Rate limited. Waiting {delay:.1f}s before retry "
                  f"(attempt {attempt + 1}/{max_retries})...{C.RESET}")
            time.sleep(delay)

        except Exception as e:
            spinner.stop()
            raise e

    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting.")

# ── Main loop ─────────────────────────────────────────────────────────────────

def run_tandem(user_task: str, max_turns: int = 8) -> str:
    print_header(user_task)

    shared_history = [
        {
            "role": "user",
            "content": f"Task: {user_task}\n\nPLANNER, please start by issuing your first RUN: command."
        }
    ]

    agents        = [("PLANNER", STEP_FLASH), ("CODER", QWEN_CODER)]
    last_output   = ""
    session_start = time.time()

    for turn in range(1, max_turns + 1):
        agent_name, model = agents[(turn - 1) % 2]

        print_turn_banner(turn, agent_name, max_turns)

        response = call_agent(model, agent_name, shared_history)
        response = handle_tool_calls(response)

        print()
        print_response(agent_name, response)

        shared_history.append({
            "role": "assistant" if agent_name == "PLANNER" else "user",
            "content": f"[{agent_name}]: {response}"
        })

        last_output = response

        if "DONE:" in response:
            print_done(agent_name, time.time() - session_start, turn)
            break

        time.sleep(3)
    else:
        print(f"\n{C.YELLOW}  Reached max turns ({max_turns}) without DONE signal.{C.RESET}\n")

    return last_output

run_tandem(
    user_task=task,
    max_turns=max_turns
)