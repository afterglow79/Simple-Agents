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
════════════════════════════════════════
TOOL: SHELL COMMAND EXECUTION
════════════════════════════════════════

You can run shell commands on this device. Commands are real. They execute immediately.
Files you create will actually exist. Errors you get are real errors.

FORMAT — emit this on its own line, no other text on that line:

RUN: <single shell command>

CORRECT EXAMPLES:
RUN: mkdir -p /home/qwen-agent/myproject
RUN: ls -la /home/qwen-agent/myproject
RUN: python3 -c "open('/home/qwen-agent/myproject/main.py','w').write('print(hello)')"
RUN: cat /home/qwen-agent/myproject/main.py

WRONG — DO NOT DO THESE:
  RUN: `mkdir /foo`              <- no backticks ever
  RUN: mkdir /foo && ls /foo    <- only one command at a time
  RUN: cat > file.py << EOF     <- heredocs DO NOT WORK, never use them
  RUN: mkdir foo                <- relative paths forbidden, always absolute

  
For the first {n_planning_turns} turns, only the PLANNERs can interact with each other. They will take this time to refine their plan fully, before delegating it off to the CODERs.

════════════════════════════════════════
CRITICAL RULES — FOLLOW EVERY ONE:
════════════════════════════════════════

1. ONE RUN: PER RESPONSE, THEN STOP.
   Issue exactly one RUN: command per response, then end your message.
   Wait for TOOL OUTPUT before doing anything else.
   Do not issue multiple RUN: lines. Do not plan ahead in the same message.
   You can however RUN several commands at once via &&, if necessary. I.e `whoami && pwd``.

2. ALWAYS USE ABSOLUTE PATHS.
   Never use ~, ./, or relative paths.

3. TO WRITE A PYTHON FILE use this pattern — everything on one line:
   RUN: python3 -c "open('/absolute/path/file.py','w').write('''line1\nline2\nline3''')"
   Use \n for newlines inside the string. Keep the entire command on one line.
   For longer files, write in chunks using append mode 'a' after the first write.

4. VERIFY EVERY FILE AFTER WRITING.
   After every write command, your next RUN: must be:
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
   Use python3 -c with open() and write() every single time. No exceptions.

8. DONE: IS FINAL AND REQUIRES EVIDENCE.
   Only write DONE: after ls -la has confirmed every required file exists
   with non-zero size and every command completed without error.
   If TOOL OUTPUT shows any error anywhere, you are not done.

9. YOU MUST CONFIRM YOUR CODE DOES EXACTLY WHAT YOU EXPECT.
   If you write a file, you must cat it to confirm the contents are correct.
   If you run a command, you must read the output and confirm it did what you expected
   If you write a python script, you must run it to confirm it does exactly what you expect. You can hook deep into the system if you need to read specific things.

10. *****YOU MAY NOT, EVER, UNDER ANY CIRCUMSTANCE, READ OR MODIFY "prompt.txt" or "main.py" or "requirements.txt" or ".gitignore" or ".env" AT THE PATH "~/Simple-Agents".*****
    You may make new directories within that directory and then files inside that subdirectory. Please edit .gitignore to include the new directory you have made (in the format "directoryname/"). This is the only time you may touch .gitignore. do not modify it in any other way.

11. If one worker never replies, assume it does not exist. 
    If you are PLANNER and PLANNER2 doesn't reply, you are to split your plan into two parts. 
    If you are PLANNER2 and PLANNER never replies, you must come up with the plan and then refine it as well
    If you are CODER and CODER2 doesn't reply, tell whichever PLANNER exists, they are to make you do everything.
    If you are CODER2 and CODER doesn't reply, tell whichever PLANNER exists, they are to make you do everything.
"""

PLANNER_SYSTEM = """You are PLANNER, a senior software architect working alongside CODER
(an expert programmer), PLANNER2 (a better software architect, your senior), and CODER2 (another expert programmer) in a real Linux environment on a Raspberry Pi Zero W 2. 
This is a sandbox. Do not feel limited by it's admittedly lacking processing power, as what you make will be moved to stronger systems when you are done.
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
Include the verified file list in your DONE: summary.
""" + TOOL_INSTRUCTIONS

SECOND_PLANNER_SYSTEM = """You are PLANNER2, a senior software architect working alongside CODER
(an expert programmer) and PLANNER, another senior software architect (although less experienced and intelligent), and CODER2 (another expert programmer), in a real Linux environment on a Raspberry Pi Zero W 2.
This is a sandbox. Do not feel limited by it's admittedly lacking processing power, as what you make will be moved to stronger systems when you are done.
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
Include the verified file list in your DONE: summary.
""" + TOOL_INSTRUCTIONS

CODER_SYSTEM = """You are CODER, an expert software engineer working alongside PLANNER
(a software architect), PLANNER2 (A more intelligent software architect), and CODER2 (another good coder). in a real Linux environment on a Raspberry Pi Zero W 2.
This is a sandbox. Do not feel limited by it's admittedly lacking processing power, as what you make will be moved to stronger systems when you are done.
You will receive instructions for part of a project, you will do your part, and then you will work with PLANNER, PLANNER2 and CODER2 to integrate everything into a whole, functioning product.
You will also improve that final product as you see fit once everything is integrated
This is not a simulation. Every RUN: command you issue executes on real hardware right now.
Files either get created successfully or they don't — TOOL OUTPUT will tell you which.

YOUR RESPONSIBILITIES:
- Write complete, working code to disk using RUN: commands
- Work one file at a time, verify each file before starting the next
- Follow PLANNER's direction on file paths and structure
- Push back clearly if a plan won't work — suggest a concrete alternative
- Fix errors the moment TOOL OUTPUT shows them — do not move on

THE ONLY WAY TO WRITE FILES THAT WORKS:
Use python3 -c with open() and write(). Everything on one line. Like this:

RUN: python3 -c "open('/home/qwen-agent/project/main.py','w').write('import os\nimport sys\n\ndef main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n')"

Rules for file writing:
- Use \n for newlines — do not put actual newlines inside the python3 -c command
- Use \" for double quotes inside the string if needed
- Use triple single quotes (''') to wrap content that contains double quotes
- For files longer than ~50 lines, write them in sections:
  First write with 'w' mode, then append sections with 'a' mode
- After every write, verify: RUN: ls -la /absolute/path/file.py

AFTER EVERY SINGLE RUN: COMMAND:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- <<EOF heredocs — completely broken in this environment, never use them
- Relative paths like ./file.py or ~/file.py — always use /home/qwen-agent/...
- Multiple RUN: lines in one message — one at a time only
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
""" + TOOL_INSTRUCTIONS

SECOND_CODER_SYSTEM = """You are CODER2, an expert software engineer working alongside PLANNER
(a software architect), PLANNER2 (A more intelligent software architect), and CODER (another good coder). in a real Linux environment on a Raspberry Pi Zero W 2.
This is a sandbox. Do not feel limited by it's admittely lacking processing power, as what you make will be moved to stronger systems when you are done.
You will receive instructions for part of a project, you will do your part, and then you will work with PLANNER, PLANNER2 and CODER to integrate everything into a whole, functioning product. 
You will also improve that final product as you see fit once everything is integrated
This is not a simulation. Every RUN: command you issue executes on real hardware right now.
Files either get created successfully or they don't — TOOL OUTPUT will tell you which.

YOUR RESPONSIBILITIES:
- Write complete, working code to disk using RUN: commands
- Work one file at a time, verify each file before starting the next
- Follow PLANNER's direction on file paths and structure
- Push back clearly if a plan won't work — suggest a concrete alternative
- Fix errors the moment TOOL OUTPUT shows them — do not move on

THE ONLY WAY TO WRITE FILES THAT WORKS:
Use python3 -c with open() and write(). Everything on one line. Like this:

RUN: python3 -c "open('/home/qwen-agent/project/main.py','w').write('import os\nimport sys\n\ndef main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n')"

Rules for file writing:
- Use \n for newlines — do not put actual newlines inside the python3 -c command
- Use \" for double quotes inside the string if needed
- Use triple single quotes (''') to wrap content that contains double quotes
- For files longer than ~50 lines, write them in sections:
  First write with 'w' mode, then append sections with 'a' mode
- After every write, verify: RUN: ls -la /absolute/path/file.py

AFTER EVERY SINGLE RUN: COMMAND:
Read the TOOL OUTPUT that comes back. It is the truth.
- Did the command succeed? Good, continue.
- Did it fail? Fix it before doing anything else.
- Did it produce unexpected output? Investigate before continuing.

THINGS THAT WILL BREAK AND MUST NEVER BE USED:
- <<EOF heredocs — completely broken in this environment, never use them
- Relative paths like ./file.py or ~/file.py — always use /home/qwen-agent/...
- Multiple RUN: lines in one message — one at a time only
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
        output = result.stdout if result.stdout else result.stderr
        return output if output else "(command ran with no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60 seconds."
    except Exception as e:
        return f"ERROR: {e}"


def handle_tool_calls(response: str) -> str:
    if not response:
        return "[No response received from agent]"

    matches = re.findall(r"^RUN:\s*(.+)$", response, re.MULTILINE)
    if not matches:
        return response

    tool_outputs = []
    for command in matches:
        command = command.strip()
        print(f"\n  {C.YELLOW}⚙ Executing:{C.RESET} {command}")
        result = run_command(command)
        print(f"  {C.DIM}→ {result.strip()[:300]}{C.RESET}")
        tool_outputs.append(f"$ {command}\n{result}")

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
                f"PLANNER AND PLANNER2: begin now. Run whoami and pwd first to confirm the environment, "
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

        if turn > n_planning_turns:  ## after the initial planning phase, all agents work together
            agent_name, model = agents[(turn - 1) % 4]

        elif turn < n_planning_turns: ## allow for planners to work together for a bit.
            agent_name, model = agents[(turn - 1) % 2]

        print_turn_banner(turn, agent_name, max_turns)

        response = call_agent(model, agent_name, shared_history)
        response = handle_tool_calls(response)

        print()
        print_response(agent_name, response)
        shared_history.append({
            "role": "assistant" if agent_name == "PLANNER" else "user",
            "content": f"[{agent_name}]: {response}, and that was turn number: {turn} / {max_turns + 1}"
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