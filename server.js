import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import { splitText, cosineSimilarity } from "./common.ts";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Configuration
const EMBEDDING_MODEL = "text-embedding-v1";
const CHAT_MODEL = "qwen-turbo";
const TOP_K = 3;

const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
});

let knowledgeBase: Array<{ text: string; embedding: number[] }> = [];

// Chat endpoint - ask questions based on uploaded knowledge base
app.post("/api/chat", async (req, res) => {
  try {
    const { question } = req.body;

    // Input validation
    if (!question || typeof question !== "string" || question.trim().length === 0) {
      return res.status(400).json({ error: "Question is required and must be a non-empty string" });
    }

    if (knowledgeBase.length === 0) {
      return res.status(400).json({ error: "Knowledge base is empty. Please upload documents first." });
    }

    // 1️⃣ Get question embedding
    const qEmb = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: question
    });

    const qVector = qEmb.data[0].embedding;

    // 2️⃣ Calculate similarity and retrieve top K relevant chunks
    const topChunks = knowledgeBase
      .map(item => ({
        text: item.text,
        score: cosineSimilarity(qVector, item.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    const context = topChunks.map(c => c.text).join("\n");

    // 3️⃣ Prepare prompt with context
    const prompt = `请基于以下内容回答问题：

      内容：
      ${context}

      问题：
      ${question}

      如果无法从内容中找到答案，请回答"不确定"。`;

    // 4️⃣ Get completion from LLM
    const completion = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: "你是一个专业的知识助手" },
        { role: "user", content: prompt }
      ]
    });

    res.json({
      answer: completion.choices[0].message.content
    });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "An error occurred while processing your request" });
  }
});

// Upload endpoint - process and embed document chunks
app.post("/api/upload", async (req, res) => {
  try {
    const { text } = req.body;

    // Input validation
    if (!text || typeof text !== "string" || text.trim().length === 0) {
      return res.status(400).json({ error: "Text is required and must be a non-empty string" });
    }

    // Split text into chunks
    const chunks = splitText(text);

    if (chunks.length === 0) {
      return res.status(400).json({ error: "Failed to create chunks from text" });
    }

    // Batch embedding creation in parallel for better performance
    const embeddings = await Promise.all(
      chunks.map(chunk =>
        openai.embeddings.create({
          model: EMBEDDING_MODEL,
          input: chunk
        })
      )
    );

    // Store chunks with their embeddings in knowledge base
    knowledgeBase = chunks.map((chunk, i) => ({
      text: chunk,
      embedding: embeddings[i].data[0].embedding
    }));

    res.json({
      message: `Successfully processed ${knowledgeBase.length} chunks`,
      chunksCount: knowledgeBase.length
    });
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ error: "An error occurred while processing the upload" });
  }
});

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    knowledgeBaseSize: knowledgeBase.length
  });
});

const PORT = process.env.PORT || 3001;

app.listen(PORT, () => {
  console.log(`✓ Server running on http://localhost:${PORT}`);
});
