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
// gemini-2.5-flash / gemini-3.7-flash are optimized for low-latency multimodal OCR
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash";

// ANSWER_PROVIDER = "openai" | "grok"
const ANSWER_PROVIDER = (process.env.ANSWER_PROVIDER || "openai").toLowerCase();
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const ANSWER_MODEL =
  process.env.ANSWER_MODEL ||
  (ANSWER_PROVIDER === "grok" ? "x-ai/grok-2-1212" : "gpt-4o-mini");

const FRONTEND_URL = process.env.FRONTEND_URL || "";
const ALLOWED_ORIGINS = FRONTEND_URL.split(",").map((s) => s.trim()).filter(Boolean);

const DEFAULT_SYSTEM_PROMPT =
  process.env.DEFAULT_SYSTEM_PROMPT ||
  [
    "You are an expert exam and homework solver.",
    "You will be given a list of questions extracted from an image.",
    "Answer every question correctly, clearly, and as concisely as possible while staying complete.",
    "Number your answers to match the question numbers/letters given to you.",
    "STRICT OUTPUT RULES:",
    "- Plain text only.",
    "- No markdown of any kind: no asterisks, no underscores, no backticks, no '#' headings, no bullet symbols like '-' or '*'.",
    "- No code fences.",
    "- No preamble or conversational filler — lead directly with the answer on line 1.",
    "- If code is requested, output valid code using minimal boilerplate, top-level statements, or expression-bodied syntax.",
    "- Format requested outputs directly on a single line (example: Output: 25).",
    "- Output must be written as if by a human, concise, direct, and copy-paste ready.",
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

// ---------------------------------------------------------------------------
// Clients
// ---------------------------------------------------------------------------

const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;

const answerClient =
  ANSWER_PROVIDER === "grok"
    ? new OpenAI({
        apiKey: OPENROUTER_API_KEY,
        baseURL: "[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
        defaultHeaders: {
          "HTTP-Referer": process.env.SITE_URL || "http://localhost:8080",
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
      console.warn(`[Multer] Rejected file upload with unsupported mimetype: ${file.mimetype}`);
      return cb(new Error("Only image files are allowed"));
    }
    cb(null, true);
  },
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Stripped-down OCR prompt targeting ultra-low output tokens for speed
const EXTRACTION_PROMPT =
  "Transcribe all text, code snippets, and questions verbatim from this image. Do not answer them. Output raw extracted text directly with zero headers or commentary.";

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

  console.log(`[Gemini] Starting fast OCR extraction using model: ${GEMINI_MODEL}`);
  
  // Adding generationConfig parameters to reduce generation latency
  const model = genAI.getGenerativeModel({
    model: GEMINI_MODEL,
    generationConfig: {
      temperature: 0.0,      // Deterministic extraction
      maxOutputTokens: 1024, // Prevents excessive/runaway generation
    },
  });

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
  console.log("[Gemini] Successfully extracted raw text.");
  return text.trim();
}

async function answerQuestions(questionsText, systemPrompt) {
  console.log(`[${ANSWER_PROVIDER}] Generating answer using model: ${ANSWER_MODEL}`);

  const completion = await answerClient.chat.completions.create({
    model: ANSWER_MODEL,
    temperature: 0.1,    // Low temp for concise/deterministic answers
    max_tokens: 500,     // Keeps answer generation short and fast
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

app.get("/", (_req, res) => {
  res.json({ ok: true, service: "image-question-solver-backend" });
});

app.get("/health", (_req, res) => res.status(200).send("ok"));

app.post("/api/solve", upload.single("image"), async (req, res) => {
  const startedAt = Date.now();
  console.log(`\n[Incoming Request] POST /api/solve from IP: ${req.ip}`);

  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded. Use field name 'image'." });
    }

    const customSystemPrompt =
      typeof req.body?.systemPrompt === "string" && req.body.systemPrompt.trim()
        ? req.body.systemPrompt.trim()
        : null;

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
    return res.status(500).json({ error: err.message || "Something went wrong." });
  }
});

app.use((err, _req, res, _next) => {
  console.error("[Global Error Handler]", err.message);
  res.status(err.status || 500).json({ error: err.message || "Server error" });
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log(`Answer provider: ${ANSWER_PROVIDER} (${ANSWER_MODEL})`);
  console.log(`Extraction model: gemini (${GEMINI_MODEL})`);
});
