import { config } from 'dotenv';
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// .envから環境変数を読み込む
config();

async function main() {
  // チェーンの準備
  const model = new ChatOpenAI();
  const prompt = ChatPromptTemplate.fromTemplate("{topic}についてジョークを言ってください");
  const chain = prompt.pipe(model);

  // ストリーム
  console.log("ストリーム結果:\n");
  const stream = await chain.stream({topic: "人工知能"});
  for await (const chunk of stream) {
    process.stdout.write(chunk.content);
  }
  console.log("\n");

  // invoke
  const result = await chain.invoke({topic: "ロボット"});
  console.log("Invoke結果:\n", result.content);
  console.log("\n");

  // batch
  console.log("Batch結果:\n");
  const batchResults = await chain.batch([
    {topic: "人工知能"},
    {topic: "ロボット"}
  ]);

  batchResults.forEach((result, i) => {
    console.log(`結果 ${i+1}: ${result.content}`);
  });
}

main().catch(console.error);
