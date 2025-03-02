import { ChatOpenAI } from "@langchain/openai";
import 'dotenv/config';

import { SerpAPI } from "@langchain/community/tools/serpapi";
import { Calculator } from "@langchain/community/tools/calculator";
import { initializeAgentExecutor } from "langchain/agents";
import { AgentExecutor } from "langchain/agents";


async function main() {
  const llm = new ChatOpenAI({
    temperature: 0,
    openAIApiKey: process.env["OPENAI_API_KEY"],
  });

  // ツールとしてSerpAPIとCalculatorを用意
  const serpTool = new SerpAPI(process.env["SERPAPI_API_KEY"], {
    location: "Tokyo,Japan",
    hl: "ja",
    gl: "jp",
  });
  const calculatorTool = new Calculator();

  // Agent（Zero-shot React）を初期化
  const tools = [serpTool, calculatorTool];
  const agentExecutor: AgentExecutor = await initializeAgentExecutor(
    tools,
    llm,
    "zero-shot-react-description",
    // ログを表示させたい場合は true にする
    true
  );

  const inputs = [
    "現在の日本の内閣総理大臣と一個前の総理大臣の名前は？二人のうち年齢が高いのはどちら？",
  ];

  async function runAgent(agent: AgentExecutor, input: string) {
    try {
      const response = await agent.call({ input });
      return response['output'];
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  }

  const results = await Promise.all(
    inputs.map((question) => runAgent(agentExecutor, question))
  );

  console.log("=== Agent Results ===");
  results.forEach((result, index) => {
    console.log(`Q: ${inputs[index]}`);
    console.log(`A: ${result}\n`);
  });
}

main().catch((err) => {
  console.error(err);
});
