import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import multer from "multer";
import OpenAI from "openai";
import { PDFParse } from "pdf-parse";

import { cosineSimilarity, splitText } from "./common.js";

dotenv.config();

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 20 * 1024 * 1024,
    files: 5
  }
});

const allowedOrigins = [
  "http://localhost:3000",      // 本地开发
  "https://aida-assistant.vercel.app" // 上线后的前端
];

app.disable("x-powered-by");
app.set("trust proxy", 1);

app.use(cors(
  {
    origin(origin, callback) {
      // 允许没有 origin（比如 curl / postman）
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error("Not allowed by CORS"));
      }
    }
  }
));
app.use(express.json({
  limit: process.env.JSON_BODY_LIMIT || "2mb"
}));

const HOST = "0.0.0.0";
const PORT = Number.parseInt(process.env.PORT ?? "3001", 10) || 3001;
const EMBEDDING_MODEL = "text-embedding-v1";
const CHAT_MODEL = "qwen-turbo";
const PDF_OCR_MODEL = process.env.PDF_OCR_MODEL || "qwen-vl-max-latest";
const PDF_OCR_MAX_PAGES = Number.parseInt(process.env.PDF_OCR_MAX_PAGES ?? "5", 10);
const PDF_OCR_SCALE = Number.parseFloat(process.env.PDF_OCR_SCALE ?? "1.5");
const MIN_EXTRACTED_TEXT_LENGTH = 20;
const TOP_K = 6;
const MIN_RELEVANCE_SCORE = 0.2;
const PDF_MIME_TYPE = "application/pdf";
const TEXT_MIME_PREFIX = "text/";
const UTF16LE_BOM = Buffer.from([0xff, 0xfe]);
const UTF8_BOM = Buffer.from([0xef, 0xbb, 0xbf]);
const DEFAULT_TEXT_SOURCE = "手动输入";
const CHAT_TIMEOUT_MS = Number.parseInt(process.env.CHAT_TIMEOUT_MS ?? "90000", 10) || 90000;
const EMBEDDING_TIMEOUT_MS = Number.parseInt(process.env.EMBEDDING_TIMEOUT_MS ?? "45000", 10) || 45000;
const OCR_TIMEOUT_MS = Number.parseInt(process.env.OCR_TIMEOUT_MS ?? "120000", 10) || 120000;
const EMBEDDING_BATCH_SIZE = Math.max(1, Number.parseInt(process.env.EMBEDDING_BATCH_SIZE ?? "2", 10) || 2);
const STREAM_HEARTBEAT_MS = Math.max(
  10000,
  Number.parseInt(process.env.STREAM_HEARTBEAT_MS ?? "15000", 10) || 15000
);
const SHUTDOWN_TIMEOUT_MS = 10000;

const openai = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
});

let knowledgeBase = [];

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function createPrompt(topChunks, question) {
  const context = topChunks
    .map((chunk, index) => [
      `[片段${index + 1}]`,
      `来源: ${chunk.source}`,
      `相关度: ${chunk.score.toFixed(4)}`,
      chunk.text
    ].join("\n"))
    .join("\n\n---\n\n");

  return [
    "你是一个企业级知识助手，请严格遵守以下规则：",
    "1. 只能根据给定片段回答，禁止补充片段里没有的信息。",
    "2. 如果片段不足以支持答案，直接回答“无法从文档中找到”。",
    "3. 优先引用原文中的关键表述，不要改写出片段里不存在的事实。",
    "",
    "【文档片段】",
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

function logError(label, error) {
  console.error(`[${label}]`, error);
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

function buildUploadMarkdown({ preview }) {
  if (isNonEmptyString(preview)) {
    return `> ${preview.replace(/\n+/g, "\n> ")}`;
  }

  return "";
}

function buildUploadFilesMarkdown(files) {
  return files
    .filter((file) => file.chunksAdded > 0 && isNonEmptyString(file.content))
    .map((file) => [
      file.content
        .split("\n")
        .map((line) => `> ${line}`)
        .join("\n")
    ].join("\n"))
    .join("\n\n");
}

function writeStreamMessage(res, payload) {
  if (!res.writableEnded) {
    res.write(`${JSON.stringify(payload)}\n`);
  }
}

function buildTopChunkPayload(chunk) {
  return {
    score: chunk.score,
    source: chunk.source,
    chunkIndex: chunk.chunkIndex,
    preview: createPreview(chunk.text, 160)
  };
}

function resetKnowledgeBase() {
  knowledgeBase = [];
}

function replaceKnowledgeBase(entries) {
  knowledgeBase = entries;
}

function cleanExtractedPdfText(text) {
  return normalizeText(
    text
      .replace(/\u0000/g, "")
      .replace(/\u00AD/g, "")
      .replace(/[ \t]+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
  );
}

function scoreExtractedPdfText(text) {
  if (!isNonEmptyString(text)) {
    return Number.NEGATIVE_INFINITY;
  }

  const compactText = text.replace(/\s+/g, "");
  const replacementCharCount = (text.match(/\uFFFD/g) || []).length;
  const controlCharCount = (text.match(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g) || []).length;
  const cjkCharCount = (text.match(/[\u3400-\u9FFF]/g) || []).length;

  return compactText.length + cjkCharCount * 2 - replacementCharCount * 20 - controlCharCount * 10;
}

function isUsableExtractedPdfText(text) {
  if (!isNonEmptyString(text)) {
    return false;
  }

  const compactText = text.replace(/\s+/g, "");
  if (compactText.length < MIN_EXTRACTED_TEXT_LENGTH) {
    return false;
  }

  return !looksMisdecoded(text);
}

function pickBestExtractedPdfText(...texts) {
  return texts
    .map((text) => cleanExtractedPdfText(text || ""))
    .sort((left, right) => scoreExtractedPdfText(right) - scoreExtractedPdfText(left))[0] ?? "";
}

function extractCompletionText(content) {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((item) => {
      if (typeof item === "string") {
        return item;
      }

      if (item && typeof item === "object" && "text" in item && typeof item.text === "string") {
        return item.text;
      }

      return "";
    })
    .join("\n");
}

async function withTimeout(task, timeoutMs, label) {
  let timer;

  try {
    return await Promise.race([
      task,
      new Promise((_, reject) => {
        timer = setTimeout(() => {
          reject(createHttpError(504, `${label} timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      })
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function ensureApiKey() {
  if (!isNonEmptyString(process.env.DASHSCOPE_API_KEY)) {
    throw createHttpError(503, "DASHSCOPE_API_KEY is not configured");
  }
}

function isClientAbortError(error) {
  return error?.code === "ECONNRESET" || error?.code === "EPIPE";
}

function startStreamHeartbeat(res) {
  return setInterval(() => {
    writeStreamMessage(res, { type: "heartbeat" });
  }, STREAM_HEARTBEAT_MS);
}

function prepareNdjsonStream(res) {
  res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();
}

async function extractPdfTextWithParser(buffer) {
  const parser = new PDFParse({ data: new Uint8Array(buffer) });

  try {
    const defaultResult = await withTimeout(parser.getText({
      lineEnforce: true,
      pageJoiner: "\n\n",
      itemJoiner: ""
    }), OCR_TIMEOUT_MS, "PDF parser");

    const rawResult = await withTimeout(parser.getText({
      lineEnforce: true,
      disableNormalization: true,
      pageJoiner: "\n\n",
      itemJoiner: ""
    }), OCR_TIMEOUT_MS, "PDF raw parser");

    return pickBestExtractedPdfText(defaultResult.text, rawResult.text);
  } finally {
    await parser.destroy().catch(() => {});
  }
}

async function extractPdfTextWithOcr(buffer, originalname) {
  ensureApiKey();

  const parser = new PDFParse({ data: new Uint8Array(buffer) });

  try {
    const screenshotResult = await withTimeout(parser.getScreenshot({
      first: PDF_OCR_MAX_PAGES,
      scale: PDF_OCR_SCALE,
      imageBuffer: false,
      imageDataUrl: true
    }), OCR_TIMEOUT_MS, "PDF screenshot");

    if (!Array.isArray(screenshotResult.pages) || screenshotResult.pages.length === 0) {
      throw createHttpError(400, `Uploaded PDF "${originalname}" does not contain renderable pages`);
    }

    const userContent = [
      {
        type: "text",
        text: [
          "请对这些 PDF 页面做 OCR，只输出页面中的原文内容。",
          "要求：",
          "1. 不要总结，不要解释，不要补充。",
          "2. 尽量保留段落、标题、列表和换行。",
          "3. 如果看到表格，按阅读顺序输出每一行。",
          "4. 按页顺序输出内容。"
        ].join("\n")
      },
      ...screenshotResult.pages.flatMap((page) => ([
        {
          type: "text",
          text: `第 ${page.pageNumber} 页`
        },
        {
          type: "image_url",
          image_url: {
            url: page.dataUrl
          }
        }
      ]))
    ];

    const completion = await withTimeout(openai.chat.completions.create({
      model: PDF_OCR_MODEL,
      messages: [
        {
          role: "user",
          content: userContent
        }
      ]
    }), OCR_TIMEOUT_MS, "PDF OCR");

    const ocrText = cleanExtractedPdfText(
      extractCompletionText(completion.choices[0]?.message?.content)
    );

    if (!isUsableExtractedPdfText(ocrText)) {
      throw createHttpError(400, `Uploaded PDF "${originalname}" could not be OCR processed`);
    }

    return ocrText;
  } finally {
    await parser.destroy().catch(() => {});
  }
}

async function extractPdfText(file) {
  const parserText = await extractPdfTextWithParser(file.buffer);

  if (isUsableExtractedPdfText(parserText)) {
    return parserText;
  }

  try {
    return await extractPdfTextWithOcr(file.buffer, file.originalname);
  } catch (ocrError) {
    if (isUsableExtractedPdfText(parserText)) {
      return parserText;
    }

    if (isNonEmptyString(parserText)) {
      return parserText;
    }

    throw createHttpError(
      400,
      `Uploaded PDF "${file.originalname}" does not contain readable text. OCR fallback also failed: ${getErrorMessage(ocrError)}`
    );
  }
}

async function createEmbedding(input) {
  ensureApiKey();

  const response = await withTimeout(openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input
  }), EMBEDDING_TIMEOUT_MS, "Embedding request");

  return response.data[0].embedding;
}

async function createEmbeddingsInBatches(chunks) {
  const embeddings = [];

  for (let index = 0; index < chunks.length; index += EMBEDDING_BATCH_SIZE) {
    const batch = chunks.slice(index, index + EMBEDDING_BATCH_SIZE);
    const batchEmbeddings = await Promise.all(batch.map((chunk) => createEmbedding(chunk)));
    embeddings.push(...batchEmbeddings);
  }

  return embeddings;
}

async function getRelevantContext(question) {
  const questionEmbedding = await createEmbedding(question);

  const rankedChunks = knowledgeBase
    .map((item) => ({
      ...item,
      score: cosineSimilarity(questionEmbedding, item.embedding)
    }))
    .sort((left, right) => right.score - left.score);

  const topChunks = rankedChunks
    .filter((chunk) => chunk.score >= MIN_RELEVANCE_SCORE)
    .slice(0, TOP_K);

  return {
    topChunks,
    fallbackTopChunks: rankedChunks.slice(0, TOP_K)
  };
}

async function createChatStream(question, topChunks) {
  ensureApiKey();

  return withTimeout(openai.chat.completions.create({
    model: CHAT_MODEL,
    stream: true,
    messages: [
      {
        role: "system",
        content: "你是一个严谨的企业知识库问答助手。只能依据提供的文档片段回答，禁止使用片段之外的知识。"
      },
      { role: "user", content: createPrompt(topChunks, question) }
    ]
  }), CHAT_TIMEOUT_MS, "Chat stream");
}

async function createChatCompletion(question, topChunks) {
  ensureApiKey();

  return withTimeout(openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: "system",
        content: "你是一个严谨的企业知识库问答助手。只能依据提供的文档片段回答，禁止使用片段之外的知识。"
      },
      { role: "user", content: createPrompt(topChunks, question) }
    ]
  }), CHAT_TIMEOUT_MS, "Chat completion");
}

async function buildKnowledgeEntries(text, source = DEFAULT_TEXT_SOURCE) {
  const normalizedText = normalizeText(text);
  const chunks = splitText(normalizedText);

  if (chunks.length === 0) {
    throw createHttpError(400, `No readable content found for "${source}"`);
  }

  const embeddings = await createEmbeddingsInBatches(chunks);

  return chunks.map((chunk, index) => ({
    text: chunk,
    embedding: embeddings[index],
    source,
    chunkIndex: index + 1
  }));
}

async function addTextToKnowledgeBase(text, source = DEFAULT_TEXT_SOURCE) {
  const entries = await buildKnowledgeEntries(text, source);
  knowledgeBase = knowledgeBase.concat(entries);
  return entries.length;
}

async function extractTextFromFile(file) {
  if (!file?.buffer?.length) {
    throw createHttpError(400, `Uploaded file "${file?.originalname ?? "unknown"}" is empty`);
  }

  if (file.mimetype === PDF_MIME_TYPE) {
    return extractPdfText(file);
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

  if (res.headersSent || res.writableEnded) {
    return;
  }

  res.status(status).json({
    error: status >= 500 ? fallbackMessage : getErrorMessage(error),
    details: getErrorMessage(error)
  });
}

function sendNoMatchResponse(res, question, stream, fallbackTopChunks = []) {
  const answer = "无法从文档中找到";
  const topChunks = fallbackTopChunks.map(buildTopChunkPayload);

  if (stream === true) {
    prepareNdjsonStream(res);
    writeStreamMessage(res, {
      type: "start",
      question,
      topChunks
    });
    writeStreamMessage(res, {
      type: "done",
      answer
    });
    res.end();
    return;
  }

  res.json({
    answer,
    topChunks
  });
}

app.get("/", (req, res) => {
  res.status(200).send("ok");
});

app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    knowledgeBaseSize: knowledgeBase.length,
    uptime: Math.round(process.uptime()),
    timestamp: new Date().toISOString()
  });
});

app.post("/api/chat", async (req, res) => {
  try {
    const { question, stream } = req.body ?? {};

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

    const { topChunks, fallbackTopChunks } = await getRelevantContext(question);

    if (topChunks.length === 0) {
      return sendNoMatchResponse(res, question, stream, fallbackTopChunks);
    }

    if (stream === true) {
      prepareNdjsonStream(res);

      const completionStream = await createChatStream(question, topChunks);
      const heartbeat = startStreamHeartbeat(res);
      let answer = "";
      let clientClosed = false;

      req.on("close", () => {
        clientClosed = true;
        clearInterval(heartbeat);
      });

      writeStreamMessage(res, {
        type: "start",
        question,
        topChunks: topChunks.map(buildTopChunkPayload)
      });

      try {
        for await (const chunk of completionStream) {
          if (clientClosed || res.writableEnded) {
            break;
          }

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
      } finally {
        clearInterval(heartbeat);
      }

      if (!clientClosed && !res.writableEnded) {
        writeStreamMessage(res, {
          type: "done",
          answer: answer || "无法从文档中找到"
        });
        res.end();
      }

      return;
    }

    const completion = await createChatCompletion(question, topChunks);
    const finalAnswer = completion.choices[0]?.message?.content ?? "无法从文档中找到";

    res.json({
      answer: finalAnswer,
      topChunks: topChunks.map(buildTopChunkPayload)
    });
  } catch (error) {
    if (isClientAbortError(error)) {
      if (!res.writableEnded) {
        res.end();
      }
      return;
    }

    logError("chat_error", error);

    if (res.headersSent) {
      if (!res.writableEnded) {
        writeStreamMessage(res, {
          type: "error",
          error: getErrorMessage(error)
        });
        res.end();
      }
      return;
    }

    sendErrorResponse(res, error, "An error occurred while processing your request");
  }
});

app.post("/api/upload", upload.none(), async (req, res) => {
  try {
    const { text } = req.body ?? {};

    if (!isNonEmptyString(text)) {
      return res.status(400).json({
        error: "Text is required and must be a non-empty string"
      });
    }

    const entries = await buildKnowledgeEntries(text, DEFAULT_TEXT_SOURCE);
    replaceKnowledgeBase(entries);

    const markdown = buildUploadMarkdown({
      preview: normalizeText(text)
    });

    res.json({
      message: `Successfully processed ${entries.length} chunks`,
      markdown,
      chunksAdded: entries.length,
      chunksCount: knowledgeBase.length,
      status: "200"
    });
  } catch (error) {
    logError("upload_error", error);
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
    const nextKnowledgeBase = [];

    for (const file of files) {
      const text = await extractTextFromFile(file);

      if (!isNonEmptyString(text)) {
        filePreviews.push({
          mimetype: file.mimetype,
          content: "",
          chunksAdded: 0
        });
        continue;
      }

      const entries = await buildKnowledgeEntries(text, file.originalname);
      nextKnowledgeBase.push(...entries);
      totalChunksAdded += entries.length;

      filePreviews.push({
        mimetype: file.mimetype,
        content: normalizeText(text),
        chunksAdded: entries.length
      });
    }

    if (totalChunksAdded === 0) {
      return res.status(400).json({
        error: "No readable content was found in the uploaded files"
      });
    }

    replaceKnowledgeBase(nextKnowledgeBase);

    res.json({
      message: "Files processed successfully",
      markdown: buildUploadFilesMarkdown(filePreviews),
      filesCount: files.length,
      chunksAdded: totalChunksAdded,
      chunksCount: knowledgeBase.length,
      filePreviews,
      status: "200"
    });
  } catch (error) {
    logError("upload_files_error", error);
    sendErrorResponse(res, error, "An error occurred while processing the files");
  }
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      error: "File upload failed",
      details: error.message
    });
  }

  logError("unhandled_express_error", error);

  if (res.headersSent) {
    return next(error);
  }

  return sendErrorResponse(res, error, "An unexpected server error occurred");
});

const server = app.listen(PORT, HOST, () => {
  console.log(`Server running on http://${HOST}:${PORT}`);
  console.log("DASHSCOPE key exists:", Boolean(process.env.DASHSCOPE_API_KEY));
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.requestTimeout = 120000;

server.on("error", (error) => {
  logError("server_error", error);
});

function shutdown(signal) {
  console.log(`Received ${signal}, shutting down gracefully`);

  server.close((error) => {
    if (error) {
      logError("shutdown_error", error);
      process.exit(1);
      return;
    }

    process.exit(0);
  });

  setTimeout(() => {
    console.error("Forced shutdown after timeout");
    process.exit(1);
  }, SHUTDOWN_TIMEOUT_MS).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("uncaughtException", (error) => {
  logError("uncaught_exception", error);
});
process.on("unhandledRejection", (reason) => {
  logError("unhandled_rejection", reason);
});

resetKnowledgeBase();
