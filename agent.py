import os
import re
import json
import ast
import operator as op
from openai import OpenAI

# local model via Ollama (OpenAI-compat endpoint), no API key needed
client = OpenAI(api_key="ollama",
                 base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))

MODEL = os.environ.get("LLM_MODEL", "llama3.2:3")

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

def write_file(path: str, content: str):
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

def finish_task(summary: str):
    return {"status": "done", "summary": summary}

TOOL_IMPLS = {
    "calculator": calculator,
    "file_read": file_read,
    "write_file": write_file,
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
        fixed = m.group(0).replace("'", '"').replace('\\"', '"').replace('\\"', '"').replace('\\', '')
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

def run_agent(goal: str, max_steps: int = 12, verbose: bool = True):
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
            "The calculator tool is ONLY for evaluating numeric math expressions — never use it to run print statements, console.log, or any code."
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
