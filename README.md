# iOS Risk Sentinel

## Group Members

- **Dhriti Sood** — LinkedIn: <https://www.linkedin.com/in/dhriti-sood/>`
- **Kashish Tandon** — LinkedIn: `<https://www.linkedin.com/in/ktandon18/>`
- **Hazel Lin** — LinkedIn: `<http://www.linkedin.com/in/linhazelin/>`
- **Parth Drona**
- **Akshitha Chinthareddy** """

---

## What's this about

TransUnion uses FingerprintJS iOS — a Swift SDK — to identify devices for fraud detection. It works by reading subtle signals from a phone (screen dimensions, storage info, OS version, identifiers, etc.) and combining them into a stable fingerprint. That fingerprint is how they know whether the device logging into your account is the same one you've used before.

The issue is Apple. Every year, iOS gets more privacy-restrictive, and signals that worked before quietly stop working or get degraded. There was no automated way to track this — someone had to manually read Apple's release notes and figure out what broke. That's a slow, reactive process that doesn't scale.

This project replaces that manual process. It monitors Apple's security bulletins, classifies every Swift file in FingerprintJS by how at-risk it is, explains the reasoning in plain language, and surfaces everything through a dashboard. No one has to manually read anything.

---

## Architecture

At a high level the system has three layers — data ingestion, agentic processing, and the dashboard. Here's how they connect:

```
Apple Bulletins  ──┐
                   ├──► Scraper Agent ──► Supabase Vector DB
FingerprintJS iOS ─┘         │
                              ▼
                       Sentinel Agent
                       ├── Nomic bi-encoder (embed Swift summaries)
                       ├── pgvector cosine similarity retrieval
                       ├── DeBERTa zero-shot NLI (High/Medium/Low)
                       └── LLaMA 3.1 70B (rationale generation)
                              │
                              ▼
                      Coordinator Agent
                      ├── Escalation detection
                      ├── De-escalation detection
                      ├── Similarity trending (early warning)
                      ├── New bulletin coverage alerts
                      └── Cross-file cluster analysis
                              │
                              ▼
                       Streamlit Dashboard
                       ├── Risk KPIs & charts
                       ├── File-level drill-down
                       └── RAG chatbot (LLaMA 3.1 + Nomic + pgvector)
```

The **ingestion layer** handles getting data into the system. Apple bulletins get scraped, chunked, embedded with Nomic, and stored in Supabase. Swift files go through a separate LLaMA summarization step before they're embedded — more on why below. Both live in the same vector database.

The **agentic layer** is where the reasoning happens. The Sentinel Agent looks at each Swift file and decides whether it's at risk based on what Apple bulletins are relevant to it. The Coordinator Agent then runs on top of that and looks for changes across runs — escalations, trending scores, cluster-level issues. Neither of these needs a human to trigger them.

The **dashboard layer** surfaces everything the agents produce. It's a Streamlit app where analysts can explore risk by category, drill into individual files, and ask freeform questions through a chatbot that's grounded in the actual codebase and bulletins.

One thing worth noting about the agent split: Sentinel and Coordinator answer different questions. Sentinel asks "what is the risk right now?" Coordinator asks "is this getting better or worse over time?" Keeping them separate also means you can rerun just the Coordinator after a new bulletin drops without re-triaging the entire codebase from scratch.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Vector Database | Supabase (pgvector) |
| Embeddings | Nomic `nomic-embed-text-v1.5` (768-dim, bi-encoder) |
| Risk Classification | DeBERTa `cross-encoder/nli-deberta-v3-small` (zero-shot NLI) |
| LLM — Summarization & RAG | Meta LLaMA 3.1 70B via NVIDIA NIM |
| LLM — Structured Classification | Gemma 2 |
| Scraping | BeautifulSoup4 + Requests |
| ML Framework | Hugging Face Transformers + Sentence Transformers |

---

## How risk gets calculated

**First — summarization**

We don't embed raw Swift code. We tried it early on and retrieval was noticeably worse — the model struggles to extract semantic meaning from syntax. Instead, every Swift file goes through LLaMA 3.1 70B first, which reads the code and produces a plain-language summary:

```json
{
  "file_role": "...",
  "content_summary": "...",
  "fingerprint_contribution": "..."
}
```

Everything downstream — embedding, classification, retrieval — works with these summaries, not the raw code.

**Second — zero-shot NLI classification**

The summary gets scored by a DeBERTa NLI model against three candidate labels at the same time:

- `"high risk: privacy restriction or signal blocked by Apple"`
- `"medium risk: API behavior change or partial degradation"`
- `"low risk: stable, minimal disruption expected"`

DeBERTa is a cross-encoder — it reads the summary and each label together and scores how strongly one entails the other. The highest confidence score wins. No training data, no labeled examples — the intelligence is in how the labels are written.

**Third — bulletin validation**

Zero-shot alone isn't enough. A file can be intrinsically sensitive but have no Apple bulletin actually threatening it right now. So the Sentinel embeds the file summary with Nomic and retrieves the top matching Apple bulletins from Supabase. If any of them score above cosine similarity **0.55**, they qualify. If nothing clears that bar, the verdict flips to **No Impact** — regardless of what zero-shot said. Both signals have to agree.

**Fourth — rationale generation**

For High and Medium files only, LLaMA 3.1 writes a plain-language explanation of why the file is at risk, grounded in the retrieved bulletin. This is what makes the output actually usable — an analyst can read it and verify the reasoning in under a minute.

---

## Embedding model evaluation

We tested four models before settling on Nomic. The evaluation wasn't just "which one scores highest on a benchmark" — we cared specifically about cross-source retrieval, since the system needs to match natural language queries against both Apple documentation and Swift code summaries. Those are pretty different text styles and a lot of models fall apart on one or the other.

| Model | Dimensions | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | Fast, but too lossy at 384 dimensions for this use case |
| `intfloat/e5-base-v2` | 768 | Solid general performance, decent recall |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Comparable to E5, slightly weaker on code-adjacent queries |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Best across the board |

**What we measured:**

- **Precision** — of the top-k results returned, how many were actually relevant to what was asked
- **Recall** — of all the relevant documents in the database, how many made it into top-k
- **Cross-source retrieval quality** — whether the model degraded when queries crossed domains (e.g. a natural language question about device identifiers matching a Swift code summary)

**Why Nomic:**

Nomic had the highest precision on our test queries and held recall better than the others when queries were phrased loosely or used domain vocabulary that didn't appear verbatim in the stored text. BGE-small dropped too much quality at 384 dimensions. E5 and MPNet were both fine but consistently a step behind Nomic on the mixed-source queries that matter most for this system.

Nomic also enforces explicit prefixes — `search_query:` for queries and `search_document:` for documents at indexing time. That might sound like a minor detail but it actually pushed retrieval quality up by making the model's internal representation cleanly distinguish between "I'm indexing this" and "I'm searching for this."

---

## Project structure

```
apple_ai_risk/
├── src/
│   ├── app.py                   # Streamlit dashboard
│   ├── agents/
│   │   ├── scraper_agent.py     # Scrapes Apple security bulletins
│   │   ├── sentinel_agent.py    # Per-file risk triage
│   │   └── coordinator_agent.py # Risk change detection & alerts
│   ├── embedders.py             # Nomic bi-encoder wrapper
│   ├── retriever.py             # Supabase pgvector retrieval
│   ├── similarity.py            # Cosine similarity reranking
│   ├── zero_shot.py             # DeBERTa NLI classification
│   ├── llm_clients.py           # NVIDIA NIM LLM client
│   ├── prompts.py               # Prompt templates
│   ├── code_summarizer.py       # LLaMA-based Swift file summarizer
│   ├── config.py                # Environment settings
│   ├── run_agents.py            # Pipeline orchestrator CLI
│   └── assets/
│       ├── uic_logo.png
│       └── transunion_logo.png
├── repos/
│   └── fingerprintjs-ios/       # FingerprintJS iOS source code
├── code_summaries/              # LLM-generated Swift file summaries
├── requirements.txt
└── .env                         # API keys (not committed)
```

---

## Getting it running

**1. Set up the environment**
```bash
git clone <repo-url>
cd apple_ai_risk
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Add your API keys**

Create a `.env` file in the project root:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
NVIDIA_API_KEY=your_nvidia_nim_key
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
```

**3. Run the dashboard**
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501`

---

## Running the agent pipeline

```bash
# Full pipeline — scrape, triage, coordinate
python -m src.run_agents

# Run just one agent
python -m src.run_agents --only scraper
python -m src.run_agents --only sentinel
python -m src.run_agents --only coordinator

# Skip scraping if bulletins are already in Supabase
python -m src.run_agents --skip scraper
```

---

## Summarizing Swift files

This runs once upfront, and again whenever new Swift files are added:
```bash
python -m src.code_summarizer
```
Summaries land in `code_summaries/` as JSON files. Push them to Supabase before running the agents.

---

## What each agent actually does

**Scraper Agent**

Crawls Apple's security bulletin pages, pulls the content, chunks it, embeds it with Nomic, and writes it to Supabase. This is the system's knowledge base of Apple ecosystem changes — everything the Sentinel retrieves comes from here.

**Sentinel Agent**

The main workhorse. For every Swift file:
1. Runs zero-shot NLI to get an intrinsic risk level from the file summary
2. Embeds the file and retrieves the top matching Apple bulletins from Supabase
3. Reranks by exact cosine similarity and drops anything below 0.55
4. If nothing qualifies → verdict is No Impact, regardless of zero-shot score
5. If something qualifies → zero-shot label becomes the effective risk
6. Calls LLaMA to generate a rationale for High and Medium files only (keeps costs down)
7. Writes the full verdict to `triage_results` in Supabase

**Coordinator Agent**

Runs after Sentinel and looks at each file's history across runs. It raises alerts for:
- **Escalation** — risk went up since last run, e.g. Low → High
- **De-escalation** — risk went down (positive signal worth knowing about too)
- **Trending** — cosine similarity has been creeping up across 3+ consecutive runs even if still below 0.55 — early warning before a file tips over
- **New bulletin coverage** — a file that had zero matching bulletins now has some
- **Cluster alerts** — two or more files in the same signal category are both High risk, meaning the whole category may be under threat

---

## Dashboard

| Tab | What's there |
|---|---|
| Risk Overview | KPI counts, donut chart by risk level, stacked bar by signal category, file cards |
| File Detail | Drill into any file — bulletin evidence, rationale, similarity scores |
| File Risk Table | Full filterable table of all Swift files with risk levels |
| AI Chatbot | Ask anything — answers are grounded in the actual codebase and bulletins via RAG |

---

## A few decisions worth explaining

**Why 0.55 as the similarity threshold?**
We tested a range. Below 0.55, unrelated Apple bulletins were clearing the bar and inflating risk scores — things about Safari or WebKit showing up for hardware signal files. Above 0.60, niche signals with slightly different vocabulary started getting missed. 0.55 was the point where false positives dropped without losing meaningful recall.

**Why zero-shot NLI instead of training a classifier?**
We had no labeled examples and no time to annotate. Zero-shot NLI with carefully written label descriptions gave surprisingly good results. The key is that the label text encodes domain knowledge — you're not just saying "High" or "Low", you're describing what those mean in the context of Apple privacy restrictions.

**Why not use a cross-encoder for reranking?**
It would be more accurate but too slow for this use case. Cross-encoders need one forward pass per candidate pair, which adds up quickly when you're retrieving 20 candidates per file across 30+ files. We get most of the benefit from exact cosine reranking on top of pgvector's approximate search. A cross-encoder reranker is on the future roadmap.

---

## What's next

- Fine-tune the embedding model on iOS-specific security data for higher precision
- Add a cross-encoder reranking step between retrieval and generation
- Move to real-time bulletin ingestion triggered by Apple's RSS feed
- Build a proper labeled evaluation set for formal precision/recall benchmarking
- Extend the same framework to Android and browser APIs — the architecture is already platform-agnostic

---

## Team

**UIC — College of Business Administration**
Dhriti Sood · In partnership with TransUnion
