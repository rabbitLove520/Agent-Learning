import json
import signal
import sys, os
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


def search(query: str) -> str:
    return f"[搜索结果] 关于 '{query}' 的模拟信息。"


TOOL_MAP: dict[str, Callable] = {
    "calculator": calculator,
    "read_file": read_file,
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

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for step in range(self.max_steps):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=self.messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0,
                )
            except Exception as e:
                self.messages.pop()
                return f"[错误] 模型调用失败（第 {step+1} 步）: {e}"

            msg = response.choices[0].message
            self.messages.append(msg)

            if not msg.tool_calls:
                return msg.content or "（模型返回空内容）"

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"错误：参数不是合法 JSON，收到: {tc.function.arguments}",
                    })
                    continue

                result = execute_tool(name, args)
                if VERBOSE:
                    print(f"  [tool] {name}({args}) -> {result[:120]}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
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

        try:
            answer = agent.chat(user_input)
        except Exception as e:
            # 兜底：任何未预期异常都不让 REPL 崩掉
            print(f"[错误] {e}\n")
            continue

        print(f"助手 > {answer}\n")


if __name__ == "__main__":
    repl()
