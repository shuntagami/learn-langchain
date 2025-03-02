import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import "dotenv/config";

// テンプレート文章を定義し、プロンプトを作成
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "あなたは優秀な校正者です。"],
  [
    "user",
    "次の文章に誤字があれば訂正してください。\n{sentences_before_check}",
  ],
]);

// OpenAIのモデルのインスタンスを作成
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0,
});

// 出力パーサーの作成
const outputParser = new StringOutputParser();

// OpenAIのAPIにこのプロンプトを送信するためのチェーンを作成
// LCELのパイプ演算子を使用
const chain = prompt.pipe(model).pipe(outputParser);

// チェーンを実行し、結果を表示
const main = async () => {
  const result = await chain.invoke({
    sentences_before_check: "こんんんちわ、真純です。",
  });
  console.log(result);
};

main();
