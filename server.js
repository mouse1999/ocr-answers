/**
 * Image / Text -> Questions -> Answers backend
 */

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const OpenAI = require("openai");

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const PORT = process.env.PORT || 8080;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash";

// ANSWER_PROVIDER = "openai" | "grok"
const ANSWER_PROVIDER = (process.env.ANSWER_PROVIDER || "openai").toLowerCase();
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY; // OpenRouter key for Grok
const ANSWER_MODEL =
  process.env.ANSWER_MODEL ||
  (ANSWER_PROVIDER === "grok" ? "x-ai/grok-2-1212" : "gpt-4o-mini");

const FRONTEND_URL = process.env.FRONTEND_URL || "";
const ALLOWED_ORIGINS = FRONTEND_URL.split(",").map((s) => s.trim()).filter(Boolean);

const DEFAULT_SYSTEM_PROMPT =
  process.env.DEFAULT_SYSTEM_PROMPT ||
  [
    "You are an expert candidate completing a strict 60-second timed coding assessment.",
"Your goal is to provide accurate, high-scoring answers that are extremely short and easy to manually type in under 30 seconds.",
"STRICT CONSTRAINTS:",
"- Maximum output length: Under 120 characters total (15 words max).",
"- Direct plain text only — NO markdown, NO bullet points, NO code blocks, NO headings, and NO backticks.",
"- Lead directly with the solution on line 1 with zero preamble or conversational filler.",
"- For code explanation questions: Explain what the code does using dense technical terms in 1 concise sentence.",
"- For coding or output questions: Return only the exact minimal code line or requested output value on a single line.",
"- Output must be ready to read and type instantly."
  ].join(" ");

if (!GEMINI_API_KEY) {
  console.warn("[warn] GEMINI_API_KEY is not set — image extraction will fail.");
}
if (ANSWER_PROVIDER === "openai" && !OPENAI_API_KEY) {
  console.warn("[warn] OPENAI_API_KEY is not set.");
}
if (ANSWER_PROVIDER === "grok" && !OPENROUTER_API_KEY) {
  console.warn("[warn] OPENROUTER_API_KEY is not set.");
}

function getValidSiteUrl() {
  const envUrl = process.env.SITE_URL || process.env.FRONTEND_URL;
  if (!envUrl) return "http://localhost:8080";
  try {
    return new URL(envUrl.startsWith("http") ? envUrl : `https://${envUrl}`).toString();
  } catch (_e) {
    return "http://localhost:8080";
  }
}

// ---------------------------------------------------------------------------
// Clients
// ---------------------------------------------------------------------------

const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;

const answerClient =
  ANSWER_PROVIDER === "grok"
    ? new OpenAI({
        apiKey: OPENROUTER_API_KEY,
        baseURL: "https://openrouter.ai/api/v1",
        defaultHeaders: {
          "HTTP-Referer": getValidSiteUrl(),
          "X-Title": process.env.SITE_NAME || "Image Question Solver",
        },
      })
    : new OpenAI({ apiKey: OPENAI_API_KEY });

// ---------------------------------------------------------------------------
// App setup
// ---------------------------------------------------------------------------

const app = express();
app.set("trust proxy", 1);

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || ALLOWED_ORIGINS.length === 0 || ALLOWED_ORIGINS.includes(origin)) {
        return callback(null, true);
      }
      console.warn(`[CORS] Blocked request from unauthorized origin: ${origin}`);
      return callback(new Error(`Origin ${origin} not allowed by CORS`));
    },
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);

app.use(express.json({ limit: "1mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter(_req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image files are allowed"));
    }
    cb(null, true);
  },
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EXTRACTION_PROMPT = [
  "You are an expert OCR and text extraction assistant.",
  "Your sole task is to extract the exact text of the question and its associated code snippet from the provided image.",
  "STRICT CONSTRAINTS:",
  "- DO NOT answer, solve, or attempt to resolve the question.",
  "- DO NOT add any commentary, explanations, introductions, or conversational filler.",
  "- DO NOT summarize or rephrase; transcribe the text verbatim.",
  "EXTRACTION RULES:",
  "1. Question: Transcribe the full problem statement, including all text, prompt details, or options directly attached to the question.",
  "2. Code Snippet: Transcribe any code block, pseudo-code, output snippet, or syntax attached to the question. Retain exact indentation, casing, and line breaks.",
  "OUTPUT FORMAT:",
  "Output ONLY the extracted content using the following markdown structure and nothing else:",
  "### Question\n[Insert verbatim question text here]\n\n### Code Snippet\n```[language]\n[Insert verbatim code snippet here]\n```\n(If no code snippet is present, write \"None\" under the Code Snippet heading.)",
].join(" ");

function stripMarkdown(text) {
  return text
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```/g, ""))
    .replace(/`([^`]*)`/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s*/gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/_(.*?)_/g, "$1")
    .replace(/^\s{0,3}[-*+]\s+/gm, "")
    .replace(/^\s{0,3}>\s?/gm, "")
    .replace(/\[(.*?)\]\((.*?)\)/g, "$1")
    .trim();
}

async function extractQuestionsFromImage(buffer, mimeType) {
  if (!genAI) throw new Error("Gemini is not configured (missing GEMINI_API_KEY).");
  console.log(`[Gemini] Starting extraction using model: ${GEMINI_MODEL}`);
  const model = genAI.getGenerativeModel({ model: GEMINI_MODEL });

  const result = await model.generateContent([
    { text: EXTRACTION_PROMPT },
    { inlineData: { data: buffer.toString("base64"), mimeType } },
  ]);

  const text = result.response.text();
  if (!text || !text.trim()) {
    throw new Error("Gemini returned no text for this image.");
  }
  return text.trim();
}

async function answerQuestions(questionsText, systemPrompt) {
  console.log(`[${ANSWER_PROVIDER}] Generating answer using model: ${ANSWER_MODEL}`);
  const completion = await answerClient.chat.completions.create({
    model: ANSWER_MODEL,
    temperature: 0.3,
    max_tokens: 1000,
    messages: [
      { role: "system", content: systemPrompt || DEFAULT_SYSTEM_PROMPT },
      { role: "user", content: questionsText },
    ],
  });

  const raw = completion.choices?.[0]?.message?.content || "";
  return stripMarkdown(raw);
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/", (_req, res) => res.json({ ok: true, service: "image-question-solver-backend" }));
app.get("/health", (_req, res) => res.status(200).send("ok"));

// Endpoint for image processing
app.post("/api/solve", upload.single("image"), async (req, res) => {
  const startedAt = Date.now();
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded. Use field name 'image'." });

    const customSystemPrompt = typeof req.body?.systemPrompt === "string" && req.body.systemPrompt.trim()
      ? req.body.systemPrompt.trim() : null;

    const t1 = Date.now();
    const questions = await extractQuestionsFromImage(req.file.buffer, req.file.mimetype);
    const extractMs = Date.now() - t1;

    const t2 = Date.now();
    const answer = await answerQuestions(questions, customSystemPrompt);
    const answerMs = Date.now() - t2;

    return res.json({
      questions,
      answer,
      timingMs: { extract: extractMs, answer: answerMs, total: Date.now() - startedAt },
      provider: { extraction: `gemini:${GEMINI_MODEL}`, answering: `${ANSWER_PROVIDER}:${ANSWER_MODEL}` },
    });
  } catch (err) {
    console.error("[/api/solve] Error:", err.message);
    return res.status(500).json({ error: err.message || "Something went wrong." });
  }
});

// Endpoint for text questions directly
app.post("/api/solve-text", async (req, res) => {
  const startedAt = Date.now();
  try {
    const { questionsText, systemPrompt } = req.body || {};
    if (!questionsText || !questionsText.trim()) {
      return res.status(400).json({ error: "No question text provided." });
    }

    const t2 = Date.now();
    const answer = await answerQuestions(questionsText, systemPrompt);
    const answerMs = Date.now() - t2;

    return res.json({
      questions: questionsText,
      answer,
      timingMs: { extract: 0, answer: answerMs, total: Date.now() - startedAt },
      provider: { extraction: "text-input", answering: `${ANSWER_PROVIDER}:${ANSWER_MODEL}` },
    });
  } catch (err) {
    console.error("[/api/solve-text] Error:", err.message);
    return res.status(500).json({ error: err.message || "Something went wrong." });
  }
});

app.use((err, _req, res, _next) => {
  res.status(err.status || 500).json({ error: err.message || "Server error" });
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
