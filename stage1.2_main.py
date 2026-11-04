import json
import signal
import sys, os, readline
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取项目根目录的 .env

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")

# ---------- 配置 ----------
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
MODEL = "deepseek-flash"
MAX_STEPS = 8
TIMEOUT_SECONDS = 30
VERBOSE = True  # True 时打印工具调用轨迹


# ---------- 工具定义 ----------
def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:2000]
    except Exception as e:
        return f"文件读取错误: {e}"

def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content) 
            return "文件写入成功"
    except Exception as e:
        return f"文件写入错误: {e}"


def search(query: str) -> str:
    return f"[搜索结果] 关于 '{query}' 的模拟信息。"


TOOL_MAP: dict[str, Callable] = {
    "calculator": calculator,
    "read_file": read_file,
    "write_file": write_file,
    "search": search,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
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
            "name": "read_file",
            "description": "读取文本文件内容",
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
            "description": "写入文本文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索信息",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


# ---------- 工具执行 ----------
def execute_tool(name: str, args: dict) -> str:
    func = TOOL_MAP.get(name)
    if not func:
        return f"错误：未知工具 '{name}'"

    def _handler(signum, frame):
        raise TimeoutError(f"工具 '{name}' 执行超时")

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(TIMEOUT_SECONDS)
        result = func(**args)
        signal.alarm(0)
        return str(result)
    except TimeoutError as e:
        return str(e)
    except Exception as e:
        return f"工具执行错误: {e}"
    finally:
        signal.alarm(0)


# ---------- Agent（多轮） ----------
class Agent:
    def __init__(self, system_prompt: str, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def chat(self, user_input: str, on_token=None) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for step in range(self.max_steps):
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=self.messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0,
                    stream=True,
                )
            except Exception as e:
                self.messages.pop()
                return f"[错误] 模型调用失败（第 {step+1} 步）: {e}"

            content = ""
            tool_calls: list[dict] = []

            # 逐块读取流式响应，实时累积文本与工具调用
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    content += delta.content
                    if on_token:
                        on_token(delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls[tc.index]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls[tc.index]["arguments"] += tc.function.arguments

            # 记录本轮 assistant 消息
            assistant_msg = {"role": "assistant", "content": content or None}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            if not tool_calls:
                return content or "（模型返回空内容）"

            for tc in tool_calls:
                name = tc["name"]
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"错误：参数不是合法 JSON，收到: {tc['arguments']}",
                    })
                    continue

                result = execute_tool(name, args)
                if VERBOSE:
                    print(f"  [tool] {name}({args}) -> {result[:120]}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        return f"[警告] 已达到最大步数 ({self.max_steps})，任务未完成。"


# ---------- 命令行 REPL ----------
HELP_TEXT = """
可用命令:
  /help    显示帮助
  /reset   清空对话历史（保留 system prompt）
  /exit    退出（也可用 Ctrl-D / Ctrl-C）
  /quit    同 /exit
直接输入内容即可与 agent 对话。
"""


def repl():
    agent = Agent("你是一个助手。需要计算、读文件或搜索时调用工具，否则直接回答。")
    print("=== 最小 Agent 命令行对话 ===")
    print("输入 /help 查看命令，/exit 退出。\n")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            print("再见。")
            break
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/reset":
            agent = Agent("你是一个助手。需要计算、读文件或搜索时调用工具，否则直接回答。")
            print("[已重置对话历史]\n")
            continue

        buf = {"started": False}

        def on_token(t):
            if not buf["started"]:
                print("助手 > ", end="", flush=True)
                buf["started"] = True
            print(t, end="", flush=True)

        try:
            answer = agent.chat(user_input, on_token=on_token)
            if buf["started"]:
                print()
            else:
                print(f"助手 > {answer}\n")
        except Exception as e:
            # 兜底：任何未预期异常都不让 REPL 崩掉
            if buf["started"]:
                print()  # 流式输出被打断时补一个换行，避免顶到下一行输入
            print(f"[错误] {e}\n")
            continue


if __name__ == "__main__":
    repl()
