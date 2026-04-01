import express from "express";
import cors from "cors";

import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
});

const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.send("OK");
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true });
});

app.post("/api/test", (req, res) => {
  res.json({ received: req.body });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, "0.0.0.0", () => {
  console.log("Server running on", PORT);
});

app.get("/api/ai-check", async (req, res) => {
  try {
    const completion = await openai.chat.completions.create({
      model: "qwen-turbo",
      messages: [{ role: "user", content: "只回复：连接成功" }]
    });

    res.json({
      ok: true,
      answer: completion.choices[0].message.content
    });
  } catch (error) {
    console.error("AI check error:", error);
    res.status(500).json({
      ok: false,
      message: error.message
    });
  }
});

console.log("DASHSCOPE key exists:", !!process.env.DASHSCOPE_API_KEY);