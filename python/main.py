from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage

from dotenv import load_dotenv

load_dotenv()  # .envファイルから環境変数を読み込む

# ツールとモデルの初期化
tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)
model = ChatOpenAI(temperature=0, streaming=True).bind_tools(tools)

# 状態定義
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 継続条件判定
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return END if not last_message.tool_calls else "tools"

# エージェントノード
def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# ツールを実行する関数を定義
def call_tool(state):
    messages = state['messages']
    # 継続条件に基づき、最後のメッセージが関数呼び出しを含まれています
    last_message = messages[-1]
    # 関数呼び出しから ToolInvocation を構築します
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # tool_executorを呼び出し、レスポンスを返します
    response = tool_node.invoke(action)
    # 応答を使って FunctionMessage を作成します
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # 既存のリストに追加されるので、リストを返します。
    return {"messages": [function_message]}

# グラフ構築
workflow = StateGraph(AgentState)

# 循環する二つのノードを定義
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# エントリーポイントとして `agent` を設定
# これはこのノードが最初に呼ばれることを意味します
workflow.set_entry_point("agent")

# 条件付きエッジを追加します
workflow.add_conditional_edges(
    # 最初に、開始ノードを定義します。`agent` を使用します。
    # これは `agent` ノードが呼び出された後に取られるエッジを意味します。
    "agent",
    # 次に、次に呼び出されるノードを決定する関数を渡します。
    should_continue,
    # 最後に、マッピングを渡します。
    # キーは文字列で、値は他のノードです。
    # END はグラフが終了することを示す特別なノードです。
    # `should_continue` を呼び出し、その出力がこのマッピングのキーに一致するものに基づいて、
    # 次に呼び出されるノードが決定されます。
    {
        # `continue` の場合は `action` ノードを呼び出します。
        "continue": "action",
        # それ以外の場合は終了します。
        "end": END
    }
)

# `action` から `agent` への通常のエッジを追加します。
# これは `action` が呼び出された後、次に `agent` ノードが呼ばれることを意味します。
workflow.add_edge('action', 'agent')

# 最後に、コンパイルします。
# これを LangChain Runnable にコンパイルし、他の runnable と同じように使用できるようにします
app = workflow.compile()

if __name__ == "__main__":
    # 実行例
    inputs = {"messages": [HumanMessage(content="東京の天気は？")]}
    for output in app.stream(inputs):
        for message in output.get("messages", []):
            print(f"{message.type}: {message.content}")

    # グラフ構造の可視化（オプション）
    # app.get_graph().draw_mermaid("./graph.svg")
    app.get_graph().print_ascii()
