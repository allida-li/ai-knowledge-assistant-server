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
    .map((file) => ([
      file.content
        .split("\n")
        .map((line) => `> ${line}`)
        .join("\n")
    ].join("\n")))
    .join("\n\n");
}

function writeStreamMessage(res, payload) {
  res.write(`${JSON.stringify(payload)}\n`);
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

async function extractPdfTextWithParser(buffer) {
  const parser = new PDFParse({ data: new Uint8Array(buffer) });

  try {
    const defaultResult = await parser.getText({
      lineEnforce: true,
      pageJoiner: "\n\n",
      itemJoiner: ""
    });
    const rawResult = await parser.getText({
      lineEnforce: true,
      disableNormalization: true,
      pageJoiner: "\n\n",
      itemJoiner: ""
    });

    return pickBestExtractedPdfText(defaultResult.text, rawResult.text);
  } finally {
    await parser.destroy();
  }
}

async function extractPdfTextWithOcr(buffer, originalname) {
  const parser = new PDFParse({ data: new Uint8Array(buffer) });

  try {
    const screenshotResult = await parser.getScreenshot({
      first: PDF_OCR_MAX_PAGES,
      scale: PDF_OCR_SCALE,
      imageBuffer: false,
      imageDataUrl: true
    });

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

    const completion = await openai.chat.completions.create({
      model: PDF_OCR_MODEL,
      messages: [
        {
          role: "user",
          content: userContent
        }
      ]
    });

    const ocrText = cleanExtractedPdfText(
      extractCompletionText(completion.choices[0]?.message?.content)
    );

    if (!isUsableExtractedPdfText(ocrText)) {
      throw createHttpError(400, `Uploaded PDF "${originalname}" could not be OCR processed`);
    }

    return ocrText;
  } finally {
    await parser.destroy();
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
  const response = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input
  });

  return response.data[0].embedding;
}

async function getRelevantContext(question) {
  const questionEmbedding = await createEmbedding(question);

  const rankedChunks = knowledgeBase
    .map((item) => ({
      ...item,
      score: cosineSimilarity(questionEmbedding, item.embedding)
    }))
    .sort((a, b) => b.score - a.score);

  const topChunks = rankedChunks
    .filter((chunk) => chunk.score >= MIN_RELEVANCE_SCORE)
    .slice(0, TOP_K);

  return {
    topChunks,
    fallbackTopChunks: rankedChunks.slice(0, TOP_K)
  };
}

async function createChatStream(question, topChunks) {
  return openai.chat.completions.create({
    model: CHAT_MODEL,
    stream: true,
    messages: [
      {
        role: "system",
        content: "你是一个严谨的企业知识库问答助手。只能依据提供的文档片段回答，禁止使用片段之外的知识。"
      },
      { role: "user", content: createPrompt(topChunks, question) }
    ]
  });
}

async function createChatCompletion(question, topChunks) {
  return openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: "system",
        content: "你是一个严谨的企业知识库问答助手。只能依据提供的文档片段回答，禁止使用片段之外的知识。"
      },
      { role: "user", content: createPrompt(topChunks, question) }
    ]
  });
}

async function addTextToKnowledgeBase(text, source = DEFAULT_TEXT_SOURCE) {
  const normalizedText = normalizeText(text);
  const chunks = splitText(normalizedText);

  if (chunks.length === 0) {
    throw new Error("Failed to create chunks from text");
  }

  const embeddings = await Promise.all(chunks.map(createEmbedding));

  const entries = chunks.map((chunk, index) => ({
    text: chunk,
    embedding: embeddings[index],
    source,
    chunkIndex: index + 1
  }));

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

  res.status(status).json({
    error: status >= 500 ? fallbackMessage : getErrorMessage(error),
    details: getErrorMessage(error)
  });
}

function sendNoMatchResponse(res, question, stream, fallbackTopChunks = []) {
  const answer = "无法从文档中找到";
  const topChunks = fallbackTopChunks.map(buildTopChunkPayload);

  if (stream === true) {
    res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();

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

    const { topChunks, fallbackTopChunks } = await getRelevantContext(question);

    if (topChunks.length === 0) {
      return sendNoMatchResponse(res, question, stream, fallbackTopChunks);
    }

    if (stream === true) {
      res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders?.();

      const completionStream = await createChatStream(question, topChunks);
      let answer = "";

      writeStreamMessage(res, {
        type: "start",
        question,
        topChunks: topChunks.map(buildTopChunkPayload)
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

    const completion = await createChatCompletion(question, topChunks);
    const finalAnswer = completion.choices[0]?.message?.content ?? "无法从文档中找到";

    res.json({
      answer: finalAnswer,
      topChunks: topChunks.map(buildTopChunkPayload)
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

    resetKnowledgeBase();
    const addedChunks = await addTextToKnowledgeBase(text, DEFAULT_TEXT_SOURCE);
    const message = `Successfully processed ${addedChunks} chunks`;
    const markdown = buildUploadMarkdown({
      preview: normalizeText(text)
    });

    res.json({
      message,
      markdown,
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

    resetKnowledgeBase();

    let totalChunksAdded = 0;
    const filePreviews = [];

    for (const file of files) {
      const text = await extractTextFromFile(file);

      if (!isNonEmptyString(text)) {
        filePreviews.push({
          fileName: file.originalname,
          mimetype: file.mimetype,
          preview: "",
          chunksAdded: 0,
          content: ""
        });
        continue;
      }

      const chunksAdded = await addTextToKnowledgeBase(text, file.originalname);
      totalChunksAdded += chunksAdded;
      filePreviews.push({
        fileName: file.originalname,
        mimetype: file.mimetype,
        preview: createPreview(text),
        chunksAdded,
        content: text
      });
    }

    if (totalChunksAdded === 0) {
      return res.status(400).json({
        error: "No readable content was found in the uploaded files"
      });
    }

    const markdown = buildUploadFilesMarkdown(filePreviews);

    res.json({
      message: "Files processed successfully",
      markdown,
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
