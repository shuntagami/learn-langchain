from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()  # .envファイルから環境変数を読み込む

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangchainのRunnableインターフェースを使ったシンプルなAPIサーバー",
)

add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo"),  # モデル名を明示的に指定
    path="/openai",
)
add_routes(
    app,
    ChatAnthropic(model_name="claude-3-haiku-20240307"),  # モデル名を明示的に指定
    path="/anthropic",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
