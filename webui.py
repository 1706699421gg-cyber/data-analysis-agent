import os

# 用本地缓存的 embedding 模型；必须在 import gradio/huggingface_hub 之前设置才生效
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
import sqlite3
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_app
from memory.vector_store import save_conclusion

DEFAULT_CSV = "data/sales.csv"
PROMPT_VERSION = "v3"
RECURSION_LIMIT = 15

# 长连接：不能用 with 短连接，否则每次请求后 checkpointer 失效、记忆丢失
_conn = sqlite3.connect("agent.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)

# 按 prompt 版本缓存各自的图，切换版本时复用，不用每次重建
_APPS = {}


def get_app(version: str):
    if version not in _APPS:
        _APPS[version] = build_app(DEFAULT_CSV, checkpointer=checkpointer, prompt_version=version)
    return _APPS[version]


get_app(PROMPT_VERSION)

IMG_PATTERN = re.compile(r"outputs/[^\s，,、）)]+\.png")


def new_thread_id():
    """每个浏览器会话独立 uuid，作为 langgraph thread_id 实现会话内多轮记忆"""
    return str(uuid.uuid4())


def extract_images(answer: str):
    """从回答里抽出 outputs/*.png 路径，转绝对路径去重后返回"""
    paths = []
    for p in IMG_PATTERN.findall(answer or ""):
        abs_path = Path(p).resolve()
        if abs_path.exists() and str(abs_path) not in paths:
            paths.append(str(abs_path))
    return paths


def chat(message, history, thread_id, version):
    history = history or []
    if not thread_id:
        thread_id = new_thread_id()
    if not message or not message.strip():
        return "", history

    history = history + [{"role": "user", "content": message}]

    handler = CallbackHandler()
    try:
        result = get_app(version or PROMPT_VERSION).invoke(
            {"messages": [HumanMessage(content=message)]},
            config={
                "configurable": {"thread_id": thread_id},
                "callbacks": [handler],
                "recursion_limit": RECURSION_LIMIT,
                "run_name": f"webui-{version or PROMPT_VERSION}",
                "metadata": {"langfuse_tags": ["webui", version or PROMPT_VERSION]},
            },
        )
        answer = result["messages"][-1].content
        if not isinstance(answer, str):
            answer = str(answer)
        try:
            save_conclusion(thread_id, DEFAULT_CSV, answer)
        except Exception:
            pass
    except Exception as e:
        answer = f"（运行异常: {type(e).__name__}）"
    finally:
        client = get_client()
        if client is not None:
            client.flush()

    images = extract_images(answer)
    if images:
        content = [{"type": "text", "text": answer}]
        content += [{"type": "image", "path": p} for p in images]
    else:
        content = answer

    history = history + [{"role": "assistant", "content": content}]
    return "", history


with gr.Blocks(title="CSV 数据分析 Agent") as demo:
    gr.Markdown(
        "# CSV 数据分析 Agent\n"
        f"当前数据文件：`{DEFAULT_CSV}`（产品 / 销售额 / 月份）\n\n"
        "**使用说明**\n"
        "- 同一会话内多轮对话共享记忆，可以直接追问（如「那按月份拆分呢」）\n"
        "- 需要图表时直接说「画柱状图」，生成的图片会显示在对话中\n"
        "- 回答中的数字均来自工具真实返回，不会编造"
    )

    thread_state = gr.State(new_thread_id())
    chatbot = gr.Chatbot(label="对话", height=520)
    with gr.Row():
        version_dd = gr.Dropdown(
            choices=["v3", "v2", "v1"], value=PROMPT_VERSION,
            label="Prompt 版本", scale=1, interactive=True,
        )
        new_btn = gr.Button("新会话", scale=0)
    status = gr.Markdown(f"当前 **{PROMPT_VERSION}** · 切换版本或点「新会话」会清空对话并新建会话")
    msg = gr.Textbox(placeholder="输入你的问题，例如：总销售额是多少？", label="你的问题")
    send_btn = gr.Button("发送", variant="primary")

    gr.Examples(
        examples=["总销售额是多少？", "哪个产品的销售额最高？", "按产品画一个销售额柱状图"],
        inputs=msg,
        label="示例问题",
    )

    def switch(version):
        return new_thread_id(), [], f"已切到 **{version}** · 新会话已开始，上下文已清空"

    demo.load(new_thread_id, outputs=thread_state)

    version_dd.change(switch, [version_dd], [thread_state, chatbot, status])
    new_btn.click(switch, [version_dd], [thread_state, chatbot, status])
    send_btn.click(chat, [msg, chatbot, thread_state, version_dd], [msg, chatbot])
    msg.submit(chat, [msg, chatbot, thread_state, version_dd], [msg, chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=7860 ,allowed_paths=[str(Path("outputs").resolve())])
