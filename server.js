import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// const openai = new OpenAI({
//   apiKey: process.env.OPENAI_API_KEY
// });
const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
});

// 聊天接口
app.post("/api/chat", async (req, res) => {
  const { question } = req.body;
  const context = knowledgeBase.join("\n");
  const prompt = `
    请基于以下内容回答问题：

    内容：
    ${context}

    问题：
    ${question}

    如果无法从内容中找到答案，请回答“不确定”。
    `;

  // try {
    // const completion = await openai.chat.completions.create({
    //   model: "gpt-4o-mini",
    //   messages: [
    //     { role: "system", content: "你是一个专业的知识助手" },
    //     { role: "user", content: question }
    //   ]
    // });
    const completion = await openai.chat.completions.create({
      model: "qwen-turbo", // 或 qwen-plus
      messages: [
        { role: "system", content: "你是一个专业的知识助手" },
        { role: "user", content: prompt }
      ]
    });

    res.json({
      answer: completion.choices[0].message.content
    });

  // } catch (err) {
  //   console.error(err);
  //   res.status(500).json({ error: "AI调用失败" });
  // }
});

// upload接口
let knowledgeBase = []; // 临时存储

app.post("/api/upload", (req, res) => {
  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: "文本不能为空" });
  }

  knowledgeBase.push(text);
  console.log("当前知识库:", knowledgeBase);
  res.json({ message: "上传成功" });
});

app.listen(3001, () => {
  console.log("server running on http://localhost:3001");
});