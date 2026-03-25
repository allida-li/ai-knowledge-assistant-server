import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import multer from "multer";
import OpenAI from "openai";
import { PDFParse } from "pdf-parse";

import { cosineSimilarity, splitText } from "./common.js";

dotenv.config();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json());

const EMBEDDING_MODEL = "text-embedding-v1";
const CHAT_MODEL = "qwen-turbo";
const TOP_K = 3;
const PDF_MIME_TYPE = "application/pdf";
const TEXT_MIME_PREFIX = "text/";
const UTF16LE_BOM = Buffer.from([0xff, 0xfe]);
const UTF8_BOM = Buffer.from([0xef, 0xbb, 0xbf]);

const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
});

let knowledgeBase = [];

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function createPrompt(context, question) {
  return [
    "你是一个企业级知识助手，请严格按照以下规则回答：",
    "1. 只能基于提供的内容回答",
    "2. 回答要结构清晰（分点）",
    "3. 如果没有答案，说“无法从文档中找到”",
    "",
    "【内容】",
    context,
    "",
    "【问题】",
    question
  ].join("\n");
}

function createHttpError(status, message) {
  const error = new Error(message);
  error.status = status;
  return error;
}

function getErrorMessage(error) {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "Unknown error";
}

function stripBom(text) {
  return text.replace(/^\uFEFF/, "");
}

function decodeWithEncoding(buffer, encoding) {
  try {
    return new TextDecoder(encoding).decode(buffer);
  } catch {
    return null;
  }
}

function looksMisdecoded(text) {
  if (!text) {
    return true;
  }

  const replacementCharCount = (text.match(/\uFFFD/g) || []).length;
  const controlCharCount = (text.match(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g) || []).length;

  return replacementCharCount > 0 || controlCharCount > Math.max(2, Math.floor(text.length * 0.05));
}

function looksLikeUtf16Le(buffer) {
  if (buffer.length >= 2 && buffer.subarray(0, 2).equals(UTF16LE_BOM)) {
    return true;
  }

  let zeroByteCount = 0;
  const sampleSize = Math.min(buffer.length, 64);

  for (let index = 1; index < sampleSize; index += 2) {
    if (buffer[index] === 0) {
      zeroByteCount += 1;
    }
  }

  return zeroByteCount >= Math.floor(sampleSize / 4);
}

function decodeTextBuffer(buffer) {
  if (buffer.length >= 3 && buffer.subarray(0, 3).equals(UTF8_BOM)) {
    return stripBom(buffer.toString("utf-8"));
  }

  if (looksLikeUtf16Le(buffer)) {
    return stripBom(buffer.toString("utf16le"));
  }

  const utf8Text = stripBom(buffer.toString("utf-8"));

  if (!looksMisdecoded(utf8Text)) {
    return utf8Text;
  }

  const gb18030Text = decodeWithEncoding(buffer, "gb18030");

  if (gb18030Text && !looksMisdecoded(gb18030Text)) {
    return stripBom(gb18030Text);
  }

  return utf8Text;
}

function normalizeText(text) {
  return text.replace(/\r\n/g, "\n").trim();
}

function createPreview(text, maxLength = 200) {
  if (!isNonEmptyString(text)) {
    return "";
  }

  return text.slice(0, maxLength);
}

function writeStreamMessage(res, payload) {
  res.write(`${JSON.stringify(payload)}\n`);
}

async function createEmbedding(input) {
  const response = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input
  });

  return response.data[0].embedding;
}

async function getRelevantContext(question) {
  const questionEmbedding = await createEmbedding(question);

  const topChunks = knowledgeBase
    .map((item) => ({
      text: item.text,
      score: cosineSimilarity(questionEmbedding, item.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K);

  return {
    context: topChunks.map((chunk) => chunk.text).join("\n"),
    topChunks
  };
}

async function createChatStream(question, context) {
  return openai.chat.completions.create({
    model: CHAT_MODEL,
    stream: true,
    messages: [
      { role: "system", content: "你是一个企业级知识助手" },
      { role: "user", content: createPrompt(context, question) }
    ]
  });
}

async function createChatCompletion(question, context) {
  return openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "system", content: "你是一个企业级知识助手" },
      { role: "user", content: createPrompt(context, question) }
    ]
  });
}

async function addTextToKnowledgeBase(text) {
  const normalizedText = normalizeText(text);
  const chunks = splitText(normalizedText);

  if (chunks.length === 0) {
    throw new Error("Failed to create chunks from text");
  }

  const embeddings = await Promise.all(chunks.map(createEmbedding));

  const entries = chunks.map((chunk, index) => ({
    text: chunk,
    embedding: embeddings[index]
  }));

  knowledgeBase = knowledgeBase.concat(entries);

  return entries.length;
}

async function extractTextFromFile(file) {
  if (!file?.buffer?.length) {
    throw createHttpError(400, `Uploaded file "${file?.originalname ?? "unknown"}" is empty`);
  }

  if (file.mimetype === PDF_MIME_TYPE) {
    const parser = new PDFParse({ data: new Uint8Array(file.buffer) });

    try {
      const pdfData = await parser.getText();

      if (!isNonEmptyString(pdfData.text)) {
        throw createHttpError(400, `Uploaded PDF "${file.originalname}" does not contain readable text`);
      }

      return normalizeText(pdfData.text);
    } finally {
      await parser.destroy();
    }
  }

  if (file.mimetype?.startsWith(TEXT_MIME_PREFIX) || !file.mimetype) {
    return normalizeText(decodeTextBuffer(file.buffer));
  }

  throw createHttpError(
    400,
    `Unsupported file type "${file.mimetype}" for "${file.originalname}". Please upload PDF or text files`
  );
}

function sendErrorResponse(res, error, fallbackMessage) {
  const status = Number.isInteger(error?.status) ? error.status : 500;

  res.status(status).json({
    error: status >= 500 ? fallbackMessage : getErrorMessage(error),
    details: getErrorMessage(error)
  });
}

app.post("/api/chat", async (req, res) => {
  try {
    const { question, stream } = req.body;

    if (!isNonEmptyString(question)) {
      return res.status(400).json({
        error: "Question is required and must be a non-empty string"
      });
    }

    if (knowledgeBase.length === 0) {
      return res.status(400).json({
        error: "Knowledge base is empty. Please upload documents first."
      });
    }

    const { context, topChunks } = await getRelevantContext(question);

    if (stream === true) {
      res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders?.();

      const completionStream = await createChatStream(question, context);
      let answer = "";

      writeStreamMessage(res, {
        type: "start",
        question,
        topChunks: topChunks.map((chunk) => ({
          score: chunk.score,
          preview: createPreview(chunk.text, 120)
        }))
      });

      for await (const chunk of completionStream) {
        const delta = chunk.choices[0]?.delta?.content ?? "";

        if (!delta) {
          continue;
        }

        answer += delta;
        writeStreamMessage(res, {
          type: "delta",
          delta
        });
      }

      writeStreamMessage(res, {
        type: "done",
        answer: answer || "无法从文档中找到"
      });
      res.end();
      return;
    }

    const completion = await createChatCompletion(question, context);

    res.json({
      answer: completion.choices[0]?.message?.content ?? "无法从文档中找到",
      topChunks: topChunks.map((chunk) => ({
        score: chunk.score,
        preview: createPreview(chunk.text, 120)
      }))
    });
  } catch (error) {
    console.error("Chat error:", error);

    if (res.headersSent) {
      writeStreamMessage(res, {
        type: "error",
        error: getErrorMessage(error)
      });
      res.end();
      return;
    }

    sendErrorResponse(res, error, "An error occurred while processing your request");
  }
});

app.post("/api/upload", upload.none(), async (req, res) => {
  try {
    const { text } = req.body;

    if (!isNonEmptyString(text)) {
      return res.status(400).json({
        error: "Text is required and must be a non-empty string"
      });
    }

    const addedChunks = await addTextToKnowledgeBase(text);

    res.json({
      message: `Successfully processed ${addedChunks} chunks`,
      chunksAdded: addedChunks,
      chunksCount: knowledgeBase.length,
      status: "200"
    });
  } catch (error) {
    console.error("Upload error:", error);
    sendErrorResponse(res, error, "An error occurred while processing the upload");
  }
});

app.post("/api/uploadFiles", upload.array("files"), async (req, res) => {
  try {
    const files = Array.isArray(req.files) ? req.files : [];

    if (files.length === 0) {
      return res.status(400).json({
        error: "At least one file is required"
      });
    }

    let totalChunksAdded = 0;
    const filePreviews = [];

    for (const file of files) {
      const text = await extractTextFromFile(file);

      if (!isNonEmptyString(text)) {
        filePreviews.push({
          fileName: file.originalname,
          mimetype: file.mimetype,
          preview: "",
          chunksAdded: 0
        });
        continue;
      }

      const chunksAdded = await addTextToKnowledgeBase(text);
      totalChunksAdded += chunksAdded;
      filePreviews.push({
        fileName: file.originalname,
        mimetype: file.mimetype,
        preview: createPreview(text),
        chunksAdded
      });
    }

    if (totalChunksAdded === 0) {
      return res.status(400).json({
        error: "No readable content was found in the uploaded files"
      });
    }

    res.json({
      message: "Files processed successfully",
      filesCount: files.length,
      chunksAdded: totalChunksAdded,
      chunksCount: knowledgeBase.length,
      filePreviews,
      status: "200"
    });
  } catch (error) {
    console.error("Upload files error:", error);
    sendErrorResponse(res, error, "An error occurred while processing the files");
  }
});

app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    knowledgeBaseSize: knowledgeBase.length
  });
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      error: "File upload failed",
      details: error.message
    });
  }

  next(error);
});

const PORT = process.env.PORT || 3001;

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
