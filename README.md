# BSDI Agentic AI

An agentic AI system built on the **PMTS Projects List** — 4,083 real public
infrastructure projects across 39 districts of Balochistan (~PKR 51.5B) —
covering all three tracks of the assignment brief:

- **Track A — Data Assistant**: natural-language Q&A over the dataset. The
  agent chooses which tool to call (`query_projects`, `group_projects`,
  `rank_projects`, `filter_projects`), runs it, and answers from the result.
- **Track B — Risk Audit**: given a goal, the agent plans which checks to
  run (not hardcoded), executes them, and produces a ranked risk report.
- **Track C — Multi-Agent Review Board**: Finance, Delivery and Equity
  agents independently evaluate the `Not Started` portfolio; a Coordinator
  merges their evidence, resolves conflicts, and produces a ranked funding
  shortlist for an extra PKR 2 billion.

The whole thing runs on a **local, open-source LLM via Ollama** (`qwen3:4b`)
— no API keys, no external LLM calls — served through a Streamlit UI.

---

## Architecture

```
data/Projects.xlsx
        │
        ▼
src/ingestion/excel_loader.py   — skips the 3-row banner, coerces numeric cols
        │
        ▼
src/tools/                      — pure functions the agents call as tools
  project_tools.py              — filter / aggregate / group / rank
  audit_tools.py                — 5 red-flag checks (missing data, cost
                                   outliers, budget concentration, etc.)
  review_tools.py                — Finance / Delivery / Equity evidence builders
        │
        ▼
src/graph/                      — LangGraph agent loops (Track A & B)
  track_a.py                    — plan → call tool → observe → answer
  track_b.py                    — plan checks → run checks → rank → report
src/agents/                     — Track C specialists + coordinator
  finance_agent.py / delivery_agent.py / equity_agent.py
  coordinator_agent.py          — merges evidence, resolves trade-offs, ranks
        │
        ▼
app.py                          — Streamlit UI (Dashboard, 3 agent pages,
                                   Projects Explorer), shows every tool call
                                   and the agent's plan, not just the final
                                   answer.
```

**How hallucination is prevented:** every number shown to the user comes
from a tool call against the real dataframe — the LLM is only used to
*decide which tool to call and to phrase the final answer*, never to
invent a figure. Track A's system prompt explicitly forbids guessing, and
the UI's "Agent reasoning" panel shows the exact tool + arguments + raw
result behind every answer, so a wrong number is traceable and auditable
rather than hidden inside a paragraph of prose.

**Messy/missing data handling:** `Cost (M)` and `Progress %` are coerced
with `errors="coerce"` and dropped from aggregations rather than crashing;
blank contractor/XEN/date fields are treated as missing and flagged as
findings (that's literally what Track B's audit checks are looking for),
never silently filled in with an invented value.

---

## Running locally

### 1. Install Ollama and pull the model

```bash
# https://ollama.com/download
ollama serve                 # starts the local LLM server on :11434
ollama pull qwen3:4b         # one-time download of the model
```

### 2. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. The sidebar shows an **Ollama Online / Offline**
indicator so you always know whether the agent pages will work — the
Dashboard and Projects Explorer pages work even with Ollama offline, since
they don't call the LLM.

---

## Deploying on a server

### Option A — Docker Compose (recommended)

Ships the Streamlit app and an Ollama server as two containers on one box.

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen3:4b   # one-time
```

Visit `http://<server-ip>:8501`. Put this behind an nginx/Caddy reverse
proxy with TLS for a real public URL — Streamlit itself doesn't handle
HTTPS.

> GPU: if the server has an NVIDIA GPU, uncomment the `deploy.resources`
> block for the `ollama` service in `docker-compose.yml` (requires the
> NVIDIA Container Toolkit) — inference is much faster.

### Option B — systemd, no Docker

```bash
# as a service user, in the project directory
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `/etc/systemd/system/bsdi-ollama.service`:

```ini
[Unit]
Description=Ollama server
After=network.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=always

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/bsdi-app.service`:

```ini
[Unit]
Description=BSDI Agentic AI (Streamlit)
After=bsdi-ollama.service

[Service]
WorkingDirectory=/path/to/BSDI_Agentic_AI
Environment=OLLAMA_BASE_URL=http://localhost:11434
Environment=OLLAMA_MODEL=qwen3:4b
ExecStart=/path/to/BSDI_Agentic_AI/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bsdi-ollama bsdi-app
sudo ollama pull qwen3:4b
```

Then reverse-proxy `:8501` through nginx with TLS.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the agents/UI look for Ollama |
| `OLLAMA_MODEL` | `qwen3:4b` | Model name used by both Track A and Track B |

---

## Project structure

```
app.py                    Streamlit UI — Dashboard, Data Assistant,
                           Risk Audit, Review Board, Projects Explorer
data/Projects.xlsx         The dataset
src/ingestion/              Excel loading + cleaning
src/tools/                  Tool functions the agents call
src/graph/                  Track A / Track B LangGraph loops
src/agents/                 Track C specialist + coordinator agents
tests/                      Exploratory scripts used during development
                             (call Ollama directly / exercise individual
                             tools — run with Ollama serving locally)
.streamlit/config.toml      Server + theme config
Dockerfile, docker-compose.yml   Container deployment
requirements.txt
```

---

## Notes for the write-up

- **Track A & B** are genuine agent loops: the LLM decides which tool to
  call (or which audit checks to run) at each turn; nothing is hardcoded
  end-to-end. The Streamlit UI surfaces the full trace (tool name,
  arguments, raw result) for every answer/report so the reasoning is
  auditable, satisfying the "transparency" rubric item.
- **Track C** is intentionally a deterministic, rule-based multi-agent
  system rather than three separate LLM calls: each specialist
  (`finance_agent.py`, `delivery_agent.py`, `equity_agent.py`) grounds its
  claims directly in queried data, and the Coordinator's conflict
  resolution (`detect_tradeoffs` / `make_project_decision`) is transparent
  and reproducible. If you want to push this further toward "genuine
  agent-to-agent communication" for the rubric, a natural extension is to
  give the Coordinator an LLM call that reads the three agents' structured
  outputs and writes the final narrative in natural language instead of
  the current templated report — the data plumbing for that is already in
  place in `run_coordinator()`.
- A failure worth documenting in the write-up: the original `app.py` had
  a broken indentation block that silently turned the page router into a
  Python `SyntaxError` (an `elif` with no matching `if`), so the app
  never actually started before this pass. It's worth walking through
  *why* that happened (a stray dedent introduced while editing the charts
  section) as the "one failure I hit and how I fixed it" part of the
  write-up.

## Tests

`tests/` contains exploratory scripts from development (they call Ollama
directly or exercise a single tool) rather than a pytest suite. With
Ollama running, you can sanity-check the core tools without the UI:

```bash
python3 -c "
from src.tools.project_tools import query_projects
print(query_projects(district='Kech', category='water', status='completed', operation='count'))
"
```