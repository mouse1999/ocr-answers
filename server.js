/**
 * Image -> Questions -> Answers backend
 * -------------------------------------
 * Flow:
 *   1. Client uploads an image (multipart/form-data, field name "image").
 *   2. Gemini (vision) reads the image and extracts the raw question text.
 *   3. The extracted text is handed to a second model (OpenAI or Grok via OpenRouter,
 *      both use the OpenAI-compatible chat API) with a system prompt that
 *      forces a clean, plain-text, copy-paste-ready answer.
 *   4. The server strips any leftover markdown as a safety net and returns
 *      plain text (plus the extracted questions, and timing info).
 *
 * Kept deliberately small and dependency-light so cold starts on Render
 * stay fast: no image resizing library, no queues, no extra middleware.
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

// Comma separated list of allowed frontend origins, e.g.
// "https://my-app.vercel.app,http://localhost:5500"
const FRONTEND_URL = process.env.FRONTEND_URL || "";
const ALLOWED_ORIGINS = FRONTEND_URL.split(",").map((s) => s.trim()).filter(Boolean);

const DEFAULT_SYSTEM_PROMPT =
  process.env.DEFAULT_SYSTEM_PROMPT ||
  [
    "You are an expert candidate completing a 60-second timed assessment.",
    "You will be given technical coding questions, code snippets, or conceptual prompts.",
    "Answer every question directly, accurately, and as concisely as possible with zero fluff.",
    "Number your answers to match the question numbers/letters given to you.",
    "STRICT OUTPUT RULES:",
    "- Plain text only.",
    "- No markdown of any kind: no asterisks, no underscores, no backticks, no '#' headings, no bullet symbols like '-' or '*'.",
    "- No code fences.",
    "- No preamble or conversational filler — lead directly with the answer on line 1.",
    "- If code is requested, output valid code using minimal boilerplate, top-level statements, or expression-bodied syntax.",
    "- If an explanation is needed, write 1 to 2 sentences max in natural human phrasing.",
    "- Format requested outputs directly on a single line (example: Output: 25).",
    "- The output must be ready to copy and paste as-is.",
  ].join(" ");

if (!GEMINI_API_KEY) {
  console.warn("[warn] GEMINI_API_KEY is not set — /api/solve will fail until it is.");
}
if (ANSWER_PROVIDER === "openai" && !OPENAI_API_KEY) {
  console.warn("[warn] OPENAI_API_KEY is not set — /api/solve will fail until it is.");
}
if (ANSWER_PROVIDER === "grok" && !OPENROUTER_API_KEY) {
  console.warn("[warn] OPENROUTER_API_KEY is not set — /api/solve will fail until it is.");
}

// Helper to ensure OpenRouter HTTP-Referer is always a valid absolute URL
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

// Trust Render's proxy (needed for correct protocol/IP handling)
app.set("trust proxy", 1);

app.use(
  cors({
    origin(origin, callback) {
      // Allow server-to-server / curl calls with no Origin header,
      // and allow everything if no allowlist was configured (dev mode).
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
  limits: { fileSize: 8 * 1024 * 1024 }, // 8MB cap keeps upload+inference fast
  fileFilter(_req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      console.warn(`[Multer] Rejected file upload with unsupported mimetype: ${file.mimetype}`);
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

/** Strips common markdown artifacts as a safety net, in case a model still
 * slips some in despite the system prompt. */
function stripMarkdown(text) {
  const cleaned = text
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```/g, "")) // fenced code
    .replace(/`([^`]*)`/g, "$1") // inline code
    .replace(/^\s{0,3}#{1,6}\s*/gm, "") // headings
    .replace(/\*\*(.*?)\*\*/g, "$1") // bold
    .replace(/__(.*?)__/g, "$1") // bold (underscore)
    .replace(/\*(.*?)\*/g, "$1") // italic
    .replace(/_(.*?)_/g, "$1") // italic (underscore)
    .replace(/^\s{0,3}[-*+]\s+/gm, "") // bullet list markers
    .replace(/^\s{0,3}>\s?/gm, "") // blockquotes
    .replace(/\[(.*?)\]\((.*?)\)/g, "$1") // links -> text only
    .trim();

  return cleaned;
}

async function extractQuestionsFromImage(buffer, mimeType) {
  if (!genAI) throw new Error("Gemini is not configured (missing GEMINI_API_KEY).");

  console.log(`[Gemini] Starting extraction using model: ${GEMINI_MODEL}`);
  const model = genAI.getGenerativeModel({ model: GEMINI_MODEL });

  const result = await model.generateContent([
    { text: EXTRACTION_PROMPT },
    {
      inlineData: {
        data: buffer.toString("base64"),
        mimeType,
      },
    },
  ]);

  const text = result.response.text();
  if (!text || !text.trim()) {
    throw new Error("Gemini returned no text for this image.");
  }
  console.log("[Gemini] Successfully extracted text from image.");
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
  console.log(`[${ANSWER_PROVIDER}] Received raw answer. Cleaning markdown formatting...`);

  const cleaned = stripMarkdown(raw);
  console.log(`[${ANSWER_PROVIDER}] Answer processing complete.`);
  return cleaned;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/", (_req, res) => {
  res.json({ ok: true, service: "image-question-solver-backend" });
});

// Cheap health check for Render
app.get("/health", (_req, res) => res.status(200).send("ok"));

app.post("/api/solve", upload.single("image"), async (req, res) => {
  const startedAt = Date.now();
  console.log(`\n[Incoming Request] POST /api/solve from IP: ${req.ip}`);

  try {
    if (!req.file) {
      console.warn("[/api/solve] Bad Request: No image file attached.");
      return res.status(400).json({ error: "No image uploaded. Use field name 'image'." });
    }

    console.log(`[File Received] Size: ${req.file.size} bytes, MimeType: ${req.file.mimetype}`);

    const customSystemPrompt =
      typeof req.body?.systemPrompt === "string" && req.body.systemPrompt.trim()
        ? req.body.systemPrompt.trim()
        : null;

    if (customSystemPrompt) {
      console.log("[/api/solve] Using custom system prompt provided in request body.");
    }

    const t1 = Date.now();
    const questions = await extractQuestionsFromImage(req.file.buffer, req.file.mimetype);
    const extractMs = Date.now() - t1;

    const t2 = Date.now();
    const answer = await answerQuestions(questions, customSystemPrompt);
    const answerMs = Date.now() - t2;

    const totalMs = Date.now() - startedAt;
    console.log(`[/api/solve] Success | Extract: ${extractMs}ms | Answer: ${answerMs}ms | Total: ${totalMs}ms`);

    return res.json({
      questions,
      answer,
      timingMs: {
        extract: extractMs,
        answer: answerMs,
        total: totalMs,
      },
      provider: {
        extraction: `gemini:${GEMINI_MODEL}`,
        answering: `${ANSWER_PROVIDER}:${ANSWER_MODEL}`,
      },
    });
  } catch (err) {
    console.error("[/api/solve] Error encountered:", err.message);
    if (err.stack) {
      console.error(err.stack);
    }
    return res.status(500).json({ error: err.message || "Something went wrong." });
  }
});

// Multer / generic error handler (keeps CORS headers on error responses too)
app.use((err, _req, res, _next) => {
  console.error("[Global Error Handler]", err.message);
  if (err.stack) {
    console.error(err.stack);
  }
  res.status(err.status || 500).json({ error: err.message || "Server error" });
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log(`Answer provider: ${ANSWER_PROVIDER} (${ANSWER_MODEL})`);
  console.log(`Extraction model: gemini (${GEMINI_MODEL})`);
});
