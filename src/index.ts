import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import 'dotenv/config';

const llmchain_invoke = async (sentences_before_check: string) => {
  // Azure OpenAIのモデルのインスタンスを作成
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env["OPENAI_API_KEY"],
  });

  // プロンプトのテンプレート文章を定義
  const template = `
次の文章に誤字がないか調べて。誤字があれば訂正してください。
{sentences_before_check}
`;
  // テンプレート文章にあるチェック対象の単語を変数化
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "あなたは優秀な校正者です。"],
    ["user", template],
  ]);

  // チャットメッセージを文字列に変換するための出力解析インスタンスを作成
  const outputParser = new StringOutputParser();

  // Azure OpenAIのAPIにこのプロンプトを送信するためのチェーンを作成
  const llmChain = prompt.pipe(chatModel).pipe(outputParser);

  // 関数を実行
  return await llmChain.invoke({
    sentences_before_check: sentences_before_check,
  });
};

// 使用例
llmchain_invoke("こんんんちわ、真純です。").then(console.log);
