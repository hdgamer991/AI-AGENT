import os
import re
import json
import ast
import operator as op
import subprocess
import requests
from openai import OpenAI

# local model via Ollama (OpenAI-compat endpoint), no API key needed
client = OpenAI(api_key="ollama",
                 base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))

MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

# ---------- tools ----------

_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
}

def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("bad expr")

def calculator(expression: str):
    try:
        expression = expression.replace("^", "**")
        tree = ast.parse(expression, mode="eval").body
        return {"result": _eval_node(tree)}
    except Exception as e:
        return {"error": f"invalid expression '{expression}': {e}. retry with valid Python math syntax, e.g. 23**2"}

def file_read(path: str):
    try:
        with open(path, "r") as f:
            return {"content": f.read()[:5000]}
    except Exception as e:
        return {"error": str(e)}

def write_file(path: str = None, content: str = None, **kwargs):
    # normalize common key aliases small models use instead of 'path'/'content'
    if path is None:
        path = kwargs.get("filename") or kwargs.get("file_path") or kwargs.get("filepath") or "output.txt"
    if content is None:
        content = kwargs.get("text") or kwargs.get("data") or kwargs.get("code") or ""
    try:
        # small models sometimes double-escape newlines/tabs as literal \n \t
        if "\\n" in content or "\\t" in content:
            content = content.replace("\\n", "\n").replace("\\t", "\t")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return {"status": "written", "path": path, "bytes": len(content)}
    except Exception as e:
        return {"error": str(e)}

def run_script(path: str, timeout: int = 10):
    try:
        result = subprocess.run(
            ["python3", path], capture_output=True, text=True, timeout=timeout
        )
        return {
            "stdout": result.stdout[:3000],
            "stderr": result.stderr[:1000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"script timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}

def list_dir(path: str = "."):
    try:
        entries = os.listdir(path)
        return {"path": path, "entries": entries}
    except Exception as e:
        return {"error": str(e)}

def append_file(path: str = None, content: str = None, **kwargs):
    if path is None:
        path = kwargs.get("filename") or kwargs.get("file_path") or "output.txt"
    if content is None:
        content = kwargs.get("text") or kwargs.get("data") or ""
    try:
        if "\\n" in content or "\\t" in content:
            content = content.replace("\\n", "\n").replace("\\t", "\t")
        with open(path, "a") as f:
            f.write(content)
        return {"status": "appended", "path": path, "bytes": len(content)}
    except Exception as e:
        return {"error": str(e)}

def get_timestamp(fmt: str = "%Y-%m-%d %H:%M:%S"):
    from datetime import datetime
    return {"timestamp": datetime.now().strftime(fmt)}

def run_command(command: str, timeout: int = 15):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "stdout": result.stdout[:3000],
            "stderr": result.stderr[:1000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}

def http_request(url: str, method: str = "GET", body: str = None):
    try:
        resp = requests.request(method.upper(), url, data=body, timeout=10)
        return {
            "status_code": resp.status_code,
            "body": resp.text[:3000],
        }
    except Exception as e:
        return {"error": str(e)}

def pip_install(package: str):
    try:
        result = subprocess.run(
            ["pip", "install", "--break-system-packages", package],
            capture_output=True, text=True, timeout=60
        )
        return {
            "stdout": result.stdout[-1000:],
            "stderr": result.stderr[-500:],
            "returncode": result.returncode,
        }
    except Exception as e:
        return {"error": str(e)}

def finish_task(summary: str):
    return {"status": "done", "summary": summary}

TOOL_IMPLS = {
    "calculator": calculator,
    "file_read": file_read,
    "write_file": write_file,
    "run_script": run_script,
    "list_dir": list_dir,
    "append_file": append_file,
    "get_timestamp": get_timestamp,
    "run_command": run_command,
    "http_request": http_request,
    "pip_install": pip_install,
    "finish_task": finish_task,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read contents of a local file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a local file, creating it or overwriting if it exists",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": "Execute a local Python script and return its stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to the end of an existing file, or create it if missing",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timestamp",
            "description": "Get the current real date/time as a formatted string. Always use this instead of guessing a timestamp.",
            "parameters": {
                "type": "object",
                "properties": {"fmt": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute any shell command (not just python scripts) and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an HTTP request to a URL and return status code and response body",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Install a Python package via pip so it can be used in scripts",
            "parameters": {
                "type": "object",
                "properties": {"package": {"type": "string"}},
                "required": ["package"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Call this when goal fully achieved, with a summary of what was done",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]

# ---------- planning ----------

def _try_parse_fake_tool_call(content: str):
    """
    Small local models emit tool calls as raw text in wildly inconsistent
    shapes instead of using the real tool_calls field. Rather than matching
    exact wrapper formats, scan for any embedded {...} dict and check for
    known arg keys.
    """
    if not content:
        return None

    for m in re.finditer(r'\{[^{}]*\}', content):
        fixed = m.group(0).replace("'", '"').replace('\\"', '"')
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', fixed)
        try:
            obj = json.loads(fixed)
        except Exception:
            continue
        if "parameters" in obj and isinstance(obj["parameters"], dict):
            obj = obj["parameters"]
        if "expression" in obj:
            return "calculator", {"expression": obj["expression"]}
        if "path" in obj:
            return "file_read", {"path": obj["path"]}
        if "summary" in obj:
            return "finish_task", {"summary": obj["summary"]}

    # fn_name("...") style, anywhere in text
    m = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\(\s*(?:\w+\s*=\s*)?["\'](.*?)["\']\s*\)', content, re.DOTALL)
    if m:
        name, arg_str = m.group(1), m.group(2)
        if name == "finish_task":
            return name, {"summary": arg_str}
        if name == "calculator":
            return name, {"expression": arg_str}
        if name == "file_read":
            return name, {"path": arg_str}

    return None


def plan(goal: str):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Break the user's goal into a short numbered list of concrete subtasks. Output only the list."},
            {"role": "user", "content": goal},
        ],
    )
    return resp.choices[0].message.content

# ---------- agent loop ----------

def run_agent(goal: str, max_steps: int = 20, verbose: bool = True):
    subtasks = plan(goal)
    if verbose:
        print("PLAN:\n", subtasks, "\n")

    history = [
        {"role": "system", "content": (
            "You are an autonomous agent. You have tools: calculator, file_read, finish_task. "
            "Work through the plan step by step. "
            "Always use the actual function-calling mechanism to invoke tools — "
            "never write a tool call as plain text or JSON in your message content. "
            "When the goal is fully achieved, call finish_task with a summary. "
            "Do not call finish_task early. "
            "When calling write_file, always include the complete code or text as the content argument — never leave it empty. "
            "Default to Python (.py) for code-writing tasks unless another language is explicitly requested. "
            "The calculator tool is ONLY for evaluating numeric math expressions — never use it to run print statements, console.log, or any code. "
            "write_file requires exactly two arguments named 'path' and 'content' — never 'filename', 'text', or any other name."
        )},
        {"role": "user", "content": f"Goal: {goal}\n\nPlan:\n{subtasks}"},
    ]

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
        history.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # some small models emit a fake tool call as plain JSON text
            # instead of using the real tool_calls field. try to catch that.
            fallback = _try_parse_fake_tool_call(msg.content)
            if fallback:
                fn_name, args = fallback
                if verbose:
                    print(f"[step {step}] fallback tool_call: {fn_name}({args})")
                impl = TOOL_IMPLS.get(fn_name)
                if not impl:
                    result = {"error": f"unknown tool {fn_name}"}
                else:
                    try:
                        result = impl(**args)
                    except TypeError as e:
                        result = {"error": f"bad arguments for {fn_name}: {e}. retry with correct parameter names"}
                history.append({
                    "role": "user",
                    "content": f"Tool result for {fn_name}: {json.dumps(result)}",
                })
                if fn_name == "finish_task":
                    return result.get("summary", "done")
                continue

            if verbose:
                print(f"[step {step}] no tool call, model said:", msg.content)
            if step < max_steps - 1:
                history.append({
                    "role": "user",
                    "content": "Continue the task by calling the appropriate tool now. If fully done, call finish_task with a summary."
                })
                continue
            return msg.content

        for call in msg.tool_calls:
            fn_name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"[step {step}] tool_call: {fn_name}({args})")

            impl = TOOL_IMPLS.get(fn_name)
            if not impl:
                result = {"error": f"unknown tool {fn_name}"}
            else:
                try:
                    result = impl(**args)
                except TypeError as e:
                    result = {"error": f"bad arguments for {fn_name}: {e}. check the correct parameter names for this tool and retry"}

            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

            if fn_name == "finish_task":
                return result.get("summary", "done")

    return "max steps hit, no finish_task called"


if __name__ == "__main__":
    goal = input("Goal: ")
    output = run_agent(goal)
    print("\nFINAL:\n", output)
