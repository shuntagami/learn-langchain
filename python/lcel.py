from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# チェーンの準備
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("{topic}についてジョークを言ってください")
chain = prompt | model

# ストリーム
print("ストリーム結果:\n")
for s in chain.stream({"topic": "人工知能"}):
    print(s.content, end="", flush=True)

print("\n")

result = chain.invoke({"topic": "ロボット"})
print("Invoke結果:\n", result.content)  # .content でテキスト内容を取得

print("\n")

# batch の結果を表示
batch_results = chain.batch([{"topic": "人工知能"}, {"topic": "ロボット"}])
print("\nBatch結果:\n")
for i, result in enumerate(batch_results):
    print(f"結果 {i+1}: {result.content}")
