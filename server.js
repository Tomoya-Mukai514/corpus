import express from "express";

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const DATA_URL = process.env.DATA_URL;
const API_KEY = process.env.API_KEY || "";

if (!DATA_URL) {
  console.error("DATA_URL is not set.");
  process.exit(1);
}

function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[　\s]+/g, " ")
    .trim();
}

function tokenize(text) {
  const normalized = normalizeText(text);

  const latinTokens = normalized.match(/[a-z0-9][a-z0-9._-]*/g) || [];
  const japaneseChunks = normalized
    .replace(/[a-z0-9._-]+/g, " ")
    .split(/[\s,，、。．・:：;；()[\]{}"“”'‘’!?！？/]+/)
    .map((x) => x.trim())
    .filter(Boolean);

  const merged = [...latinTokens, ...japaneseChunks]
    .filter((x) => x.length >= 1);

  return merged;
}

function buildDocText(doc) {
  return [
    doc.title,
    doc.subtitle,
    doc.keywords,
    doc.abstract
  ].join(" ");
}

function bm25Build(docs) {
  const built = docs.map((doc, idx) => {
    const text = buildDocText(doc);
    const tokens = tokenize(text);
    const tf = new Map();

    for (const token of tokens) {
      tf.set(token, (tf.get(token) || 0) + 1);
    }

    return {
      idx,
      doc,
      tokens,
      tf,
      length: tokens.length
    };
  });

  const N = built.length;
  const df = new Map();
  let totalLength = 0;

  for (const item of built) {
    totalLength += item.length;
    const seen = new Set(item.tokens);
    for (const term of seen) {
      df.set(term, (df.get(term) || 0) + 1);
    }
  }

  const avgdl = N > 0 ? totalLength / N : 0;

  return { built, df, N, avgdl };
}

function bm25Search(index, query, topK = 5, k1 = 1.5, b = 0.75) {
  const qTokens = [...new Set(tokenize(query))];
  if (qTokens.length === 0) return [];

  const results = [];

  for (const item of index.built) {
    let score = 0;
    const matchedTerms = [];

    for (const term of qTokens) {
      const freq = item.tf.get(term) || 0;
      if (freq === 0) continue;

      const df = index.df.get(term) || 0;
      const idf = Math.log(1 + (index.N - df + 0.5) / (df + 0.5));
      const denom = freq + k1 * (1 - b + b * (item.length / (index.avgdl || 1)));
      score += idf * ((freq * (k1 + 1)) / denom);
      matchedTerms.push(term);
    }

    if (score > 0) {
      results.push({
        score,
        matched_terms: matchedTerms,
        document: item.doc
      });
    }
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}

function buildSnippet(abstract, queryTerms) {
  const text = String(abstract || "").replace(/\s+/g, " ").trim();
  if (!text) return "";

  for (const term of queryTerms) {
    const idx = text.toLowerCase().indexOf(term.toLowerCase());
    if (idx >= 0) {
      const start = Math.max(0, idx - 80);
      const end = Math.min(text.length, idx + 160);
      return text.slice(start, end);
    }
  }

  return text.slice(0, 240);
}

let corpus = [];
let index = null;
let lastLoadedAt = null;

async function loadCorpus() {
  const res = await fetch(DATA_URL, {
    headers: { "User-Agent": "jjlawpsychology-bm25-api" }
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch DATA_URL: ${res.status}`);
  }

  const data = await res.json();
  if (!Array.isArray(data)) {
    throw new Error("Corpus JSON must be an array.");
  }

  corpus = data;
  index = bm25Build(corpus);
  lastLoadedAt = new Date().toISOString();

  console.log(`Loaded ${corpus.length} records at ${lastLoadedAt}`);
}

function authMiddleware(req, res, next) {
  if (!API_KEY) return next();

  const auth = req.headers.authorization || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : "";

  if (token !== API_KEY) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  next();
}

app.get("/health", (req, res) => {
  res.json({
    ok: true,
    records: corpus.length,
    loaded_at: lastLoadedAt
  });
});

app.post("/reload", authMiddleware, async (req, res) => {
  try {
    await loadCorpus();
    res.json({
      ok: true,
      records: corpus.length,
      loaded_at: lastLoadedAt
    });
  } catch (err) {
    res.status(500).json({
      ok: false,
      error: String(err.message || err)
    });
  }
});

app.post("/search", authMiddleware, (req, res) => {
  try {
    const query = String(req.body.query || "").trim();
    const top_k = Number(req.body.top_k || 5);

    if (!query) {
      return res.status(400).json({ error: "query is required" });
    }

    if (!index) {
      return res.status(503).json({ error: "index not loaded" });
    }

    const results = bm25Search(index, query, top_k);
    const queryTerms = [...new Set(tokenize(query))];

    const formatted = results.map((r, i) => ({
      rank: i + 1,
      score: Number(r.score.toFixed(4)),
      matched_terms: r.matched_terms,
      id: r.document.id,
      title: r.document.title,
      subtitle: r.document.subtitle,
      authors: r.document.authors,
      publication_year: r.document.publication_year,
      volume: r.document.volume,
      issue: r.document.issue,
      pages: r.document.pages,
      doi: r.document.doi,
      article_url: r.document.article_url,
      keywords: r.document.keywords,
      abstract: r.document.abstract,
      snippet: buildSnippet(r.document.abstract, queryTerms)
    }));

    res.json({
      query,
      top_k,
      total_records: corpus.length,
      results: formatted
    });
  } catch (err) {
    res.status(500).json({
      error: String(err.message || err)
    });
  }
});

loadCorpus()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Server listening on port ${PORT}`);
    });
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
