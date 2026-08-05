"""
Minimal AI agent w/ tool use (Anthropic API).
Tools: calculator, file_read
"""
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic math expression. Input e.g. '2 + 2 * 3'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "file_read",
        "description": "Read contents of a local text file by path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"}
            },
            "required": ["path"]
        }
    }
]


def run_calculator(expression: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def run_file_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:5000]
    except Exception as e:
        return f"Error: {e}"


TOOL_FUNCS = {
    "calculator": lambda inp: run_calculator(inp["expression"]),
    "file_read": lambda inp: run_file_read(inp["path"]),
}


def run_agent(user_input: str, messages=None, max_turns: int = 8):
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": user_input})

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts), messages

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_FUNCS.get(block.name)
                result = fn(block.input) if fn else f"Error: unknown tool {block.name}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return "Max turns hit w/o final answer.", messages


if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.")
    history = []
    while True:
        user_input = input("> ")
        if user_input.strip().lower() == "exit":
            break
        answer, history = run_agent(user_input, history)
        print(answer)