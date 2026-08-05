import os
from openai import OpenAI
import json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a local text file by path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    }
]

def run_calculator(expression):
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: invalid chars"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def run_file_read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:5000]
    except Exception as e:
        return f"Error: {e}"

TOOL_FUNCS = {
    "calculator": run_calculator,
    "file_read": run_file_read,
}

def run_agent(user_input, messages=None, max_turns=8):
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": user_input})
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL, tools=TOOLS, messages=messages
        )
        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content, messages
        for tc in msg.tool_calls:
            fn = TOOL_FUNCS.get(tc.function.name)
            args = json.loads(tc.function.arguments)
            result = fn(list(args.values())[0]) if fn else f"Error: unknown tool"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Max turns hit.", messages

if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.")
    history = []
    while True:
        user_input = input("> ")
        if user_input.strip().lower() == "exit":
            break
        answer, history = run_agent(user_input, history)
        print(answer)
