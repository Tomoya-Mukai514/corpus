import express from "express";

const app = express();
app.use(express.json());

app.use((req, res, next) => {
  if (req.method === "GET" && req.url === "/health") {
    return next();
  }

  console.log(new Date().toISOString(), req.method, req.url);
  next();
});

const PORT = process.env.PORT || 3000;
const DATA_URL = process.env.DATA_URL;
const API_KEY = process.env.API_KEY || "";

if (!DATA_URL) {
  console.error("DATA_URL is not set.");
  process.exit(1);
}

const LOW_SIGNAL_QUERY_TERM_WEIGHTS = new Map([
  ["関連", 0.3],
  ["関係", 0.3],
  ["相関", 0.3],
  ["影響", 0.3],
  ["関連性", 0.3],
  ["連関", 0.3],
  ["関連する", 0.3],
  ["関係する", 0.3],

  ["規定", 0.3],
  ["規定因", 0.3],
  ["規定要因", 0.3],
  ["予測", 0.3],
  ["予測因", 0.3],
  ["予測要因", 0.3],
  ["説明", 0.3],

  ["検討", 0.3],
  ["分析", 0.3],
  ["研究", 0.3],
  ["実証", 0.3],
  ["比較", 0.3],
  ["目的", 0.3],
  ["方法", 0.3],
  ["結果", 0.3],
  ["考察", 0.3],

  ["効果", 0.5],
  ["有効性", 0.5],

  ["association", 0.3],
  ["associations", 0.3],
  ["relationship", 0.3],
  ["relationships", 0.3],
  ["correlation", 0.3],
  ["correlations", 0.3],
  ["impact", 0.3],
  ["impacts", 0.3],

  ["predict", 0.3],
  ["predictor", 0.3],
  ["predictors", 0.3],
  ["determinant", 0.3],
  ["determinants", 0.3],

  ["examine", 0.3],
  ["examined", 0.3],
  ["analysis", 0.3],
  ["study", 0.3],
  ["studies", 0.3],
  ["research", 0.3],
  ["investigation", 0.3],

  ["effect", 0.5],
  ["effects", 0.5],
  ["effectiveness", 0.5],
]);

const ORIGINAL_TERM_BOOST = 10.0;
const EXPANDED_TERM_WEIGHT = 1.0;
const LOW_SIGNAL_ORIGINAL_CAP = 0.5;

function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[　\s]+/g, " ")
    .trim();
}

function hasJapanese(text) {
  return /[ぁ-ゖァ-ヺ一-龠々ー]/u.test(String(text || ""));
}

function makeJapaneseNgrams(text, minN = 2, maxN = 4) {
  const compact = String(text || "")
    .replace(/\s+/g, "")
    .replace(/[^ぁ-ゖァ-ヺ一-龠々ー]/gu, "");

  if (!compact) return [];

  const grams = [compact];
  const upper = Math.min(maxN, compact.length);

  for (let n = minN; n <= upper; n++) {
    for (let i = 0; i <= compact.length - n; i++) {
      grams.push(compact.slice(i, i + n));
    }
  }

  return grams;
}

function tokenize(text) {
  const normalized = normalizeText(text);

  const latinTokens = normalized.match(/[a-z0-9][a-z0-9._-]*/g) || [];

  const rawChunks = normalized
    .replace(/[a-z0-9._-]+/g, " ")
    .split(/[\s|,，、。．・:：;；()[\]{}"“”'‘’!?！？/]+/)
    .map((x) => x.trim())
    .filter(Boolean);

  const tokens = [...latinTokens];

  for (const chunk of rawChunks) {
    if (hasJapanese(chunk)) {
      tokens.push(...makeJapaneseNgrams(chunk, 2, 4));
    } else {
      tokens.push(chunk);
    }
  }

  return [...new Set(tokens.filter((x) => x.length >= 1))];
}

function buildDocText(doc) {
  return [
    doc.title,
    doc.subtitle,
    doc.authors,
    doc.publication_year,
    doc.doi,
    doc.keywords,
    doc.abstract,
    `${doc.authors || ""} ${doc.publication_year || ""}`
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

function buildQueryWeights(query, userQuery = "") {
  const expandedTokens = tokenize(query);
  const originalTokenSet = new Set(tokenize(userQuery));
  const weights = new Map();

  for (const token of expandedTokens) {
    const isOriginalTerm = originalTokenSet.has(token);
    const lowSignalWeight = LOW_SIGNAL_QUERY_TERM_WEIGHTS.get(token) || 1.0;

    let finalWeight;
    if (isOriginalTerm) {
      finalWeight = ORIGINAL_TERM_BOOST * lowSignalWeight;

      if (lowSignalWeight < 1.0) {
        finalWeight = Math.min(finalWeight, LOW_SIGNAL_ORIGINAL_CAP);
      }
    } else {
      finalWeight = EXPANDED_TERM_WEIGHT * lowSignalWeight;
    }

    weights.set(token, (weights.get(token) || 0) + finalWeight);
  }

  return weights;
}

function bm25Search(index, query, topK = 5, userQuery = "", k1 = 1.5, b = 0.75) {
  const qWeights = buildQueryWeights(query, userQuery);
  const qTerms = [...qWeights.keys()];
  if (qTerms.length === 0) return [];

  const results = [];

  for (const item of index.built) {
    let score = 0;
    const matchedTerms = [];

    for (const term of qTerms) {
      const freq = item.tf.get(term) || 0;
      if (freq === 0) continue;

      const df = index.df.get(term) || 0;
      const idf = Math.log(1 + (index.N - df + 0.5) / (df + 0.5));
      const denom =
        freq + k1 * (1 - b + b * (item.length / (index.avgdl || 1)));
      const baseScore = idf * ((freq * (k1 + 1)) / denom);
      const termWeight = qWeights.get(term) || 1.0;

      score += termWeight * baseScore;
      matchedTerms.push(term);
    }

    if (score > 0) {
      results.push({
        score,
        matched_terms: [...new Set(matchedTerms)],
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
    console.log("headers:", req.headers);
    console.log("body:", req.body);

    const query = String(req.body?.query || "").trim();
    const user_query = String(req.body?.user_query || "").trim();
    let top_k = Number(req.body?.top_k || 5);

    if (!query) {
      return res.status(400).json({ error: "query is required" });
    }

    if (!user_query) {
      return res.status(400).json({ error: "user_query is required" });
    }

    if (!Number.isFinite(top_k)) {
      top_k = 5;
    }
    top_k = Math.max(1, Math.min(20, Math.floor(top_k)));

    if (!index) {
      return res.status(503).json({ error: "index not loaded" });
    }

    const results = bm25Search(index, query, top_k, user_query);
    const queryTerms = [...new Set([
      ...tokenize(query),
      ...tokenize(user_query)
    ])];

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
      user_query,
      top_k,
      total_records: corpus.length,
      results: formatted
    });
  } catch (err) {
    console.error("search error:", err);
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
