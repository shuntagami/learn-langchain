import os
import chainlit as cl

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@cl.on_chat_start
async def start():
    """チャットセッション開始時に実行される関数"""
    await cl.Message(content="こんにちは！何かお手伝いできることはありますか？").send()

@cl.on_message
async def main(message: cl.Message):
    """ユーザーメッセージを受け取った時に実行される関数"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
            {"role": "user", "content": message.content}
        ],
        temperature=0.7
    )

    await cl.Message(content=response.choices[0].message.content).send()
