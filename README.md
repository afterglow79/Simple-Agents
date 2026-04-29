# Simple-Agents (Windows Fork)

> **⚠️ This is a Windows-only fork.** All agents operate inside a real `cmd.exe` environment using Windows paths and Windows commands. If you are on Linux/macOS, see the [original repository](https://github.com/afterglow79/Simple-Agents).

A multi-agent AI framework that runs four specialized AI agents in tandem using the [NVIDIA build API](https://build.nvidia.com/). Two planners and two coders collaborate in a shared Windows environment to plan, write, verify, and deliver working software.

---

## How it works

Agents operate in two phases:

1. **Planning phase** — PLANNER and PLANNER2 collaborate for the first N turns (default: 6) to refine a full plan: file structure (with full Windows paths), responsibilities, and integration strategy.
2. **Coding phase** — CODER and CODER2 receive the plan, each implements their assigned portion, then all four agents integrate the pieces into a finished product.

Every agent has access to the same set of real tools (write files, run `cmd.exe` commands, read/write shared memory, and optionally search the web). Agents can see each other's responses and tool output, so they coordinate naturally through the conversation.

### Agents

| Role     | Model                                   | Color   | Responsibility                                      |
|----------|-----------------------------------------|---------|-----------------------------------------------------|
| PLANNER  | `google/gemma-4-31b-it`                 | Cyan    | Initial architecture and task breakdown             |
| PLANNER2 | `z-ai/glm-5.1`                          | Cyan    | Refines the plan; splits work between the coders    |
| CODER    | `qwen/qwen3-coder-480b-a35b-instruct`   | Magenta | Implements first half of the project                |
| CODER2   | `deepseek-ai/deepseek-v4-flash`         | Magenta | Implements second half of the project               |

### Agent tools

Each agent can use the following tools in any turn:

| Tool               | Description                                                                    |
|--------------------|--------------------------------------------------------------------------------|
| `WRITE_FILE:`      | Write a multi-line file to an absolute Windows path on disk                    |
| `RUN:`             | Execute a `cmd.exe` command (60 s timeout)                                     |
| `READ_FILE:`       | Read the contents of any file by absolute Windows path                         |
| `WRITE_TO_MEMORY:` | Append a note to the shared persistent memory file                             |
| `READ_FROM_MEMORY` | Read the entire shared persistent memory                                       |
| `SEARCH_WEB:`      | Google search *(only when `--can_use_web_search` is set)*                      |
| `GET_SPECS:`       | Retrieve device and OS specifications via `systeminfo`                         |

Dangerous commands (`rd /s /q C:\`, `format`, `shutdown`, `del /f /s /q`) are blocked.

> **Note:** Agents are instructed to use Windows-native commands only (`dir`, `type`, `mkdir`, `python`, etc.). Unix commands such as `ls`, `cat`, `python3`, or `mkdir -p` will not work.

---

## Requirements

- **Windows 10 / 11** (this fork is Windows-only)
- Python 3.9+
- An [NVIDIA build API key](https://build.nvidia.com/)

---

## Setup

1. **Clone the repository**

   ```cmd
   git clone https://github.com/afterglow79/Simple-Agents.git
   cd Simple-Agents
   ```

2. **Create and activate a virtual environment** *(recommended)*

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```cmd
   pip install -r requirements.txt
   ```

4. **Add your API key**

   Create a `.env` file in the project root:

   ```env
   NVIDIA_API_KEY=your_api_key_here
   ```

---

## Usage

```cmd
python main.py --task "your task description here"
```

You can also pass a path to a text file for longer or more complex task descriptions:

```cmd
python main.py --task path\to\task.txt
```

### CLI arguments

| Argument                | Type  | Default | Description                                                  |
|-------------------------|-------|---------|--------------------------------------------------------------|
| `--task`                | str   | —       | Task description or path to a `.txt` file containing it      |
| `--max_turns`           | int   | `25`    | Maximum number of turns before the session ends              |
| `--init_planning_turns` | int   | `6`     | Turns reserved for planners before coders are allowed to act |
| `--can_use_web_search`  | bool  | `False` | Allow agents to issue `SEARCH_WEB:` queries                  |

### Examples

```cmd
:: Build a simple CLI todo app
python main.py --task "Build a command-line todo app in Python that saves tasks to a JSON file."

:: Give more turns for a complex project
python main.py --task "Write a REST API in Python using only the standard library." --max_turns 50

:: Let agents search the web
python main.py --task "Write a web scraper for Hacker News top stories." --can_use_web_search True
```

---

## Project structure

```
Simple-Agents\
├── main.py           # Entry point: agent loop, tool execution, system prompts
├── requirements.txt  # Python dependencies
├── agent\
│   └── persistent-mem.txt   # Shared memory written and read by agents at runtime
└── .env              # NVIDIA API key (not committed)
```
