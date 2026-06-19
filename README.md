<img width="1672" height="941" alt="Featured display of multimodal input flow using qdrand edge and gemini embedding 2" src="https://github.com/user-attachments/assets/1294f9f8-e043-46af-8f06-73f80f2a2144" />

# Life Memorizer

A privacy-first, **offline** memory assistant for smart glasses (or any video clip).
It indexes what you **see**, **hear**, and **read** using **Gemini Embedding 2**, stores
it locally in **Qdrant Edge**, and lets you recall moments instantly — with **no cloud**
needed for storage or search.

> "Where did I leave my keys?" · "What did Sarah say to buy?" · "Show me that cafe menu."

# Step-by-Step Run Instructions

This guide provides step-by-step instructions for running the **Life Memorizer** project in both of its operational modes:
1. **Scenario 1: Mock / Offline Mode** (`LIFE_MEMORIZER_FAKE_EMBEDDINGS=1`) — Uses pre-scripted mock data and requires no API keys.
2. **Scenario 2: Live Video Ingestion Mode** (`LIFE_MEMORIZER_FAKE_EMBEDDINGS=0`) — Processes actual video files using OpenCV, FFmpeg, Tesseract OCR, and embeds the frames/audio using the Gemini API. Mock data is automatically excluded from search results in this mode.

## Highlights

- **One model for everything** — images, audio and text share one semantic space (Gemini Embedding 2). No separate CLIP/Whisper pipelines.
- **Runs on the edge** — Qdrant Edge in a local file, no server, no internet.
- **Small footprint** — 3072→768 Matryoshka downscaling + int8/binary quantization.
- **Works with zero setup** — a built-in offline stub embedder runs the whole demo without an API key.

**Why this Tech Stack?**
- Gemini Embedding 2: Processes images, audio, and text into a unified semantic space natively—no need to manage separate CLIP, Whisper, and text-embedding pipelines.
- Qdrant Edge: Run a lightweight, blazing-fast vector database directly on an edge device (like an NVIDIA Jetson, Raspberry Pi, or local mobile device) without needing internet connectivity.

## Repository Structure

The repository follows a clean, modular python structure optimized for edge execution:

```
life-memorizer/
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git exclude patterns
├── CONTRIBUTING.md               # Contribution guidelines for developers
├── LICENSE                       # MIT License
├── README.md                     # Main documentation & quickstart guide
├── pyproject.toml                # Build system, metadata, and dependencies
├── requirements.txt              # Pinned requirements file
├── ARCHITECTURE-DOCUMENTATION.md # System architecture, codebase knowledge graph and architectural explanation of each code file
├── samples/                      # Sample video files (user-supplied or downloaded)
│   ├── pov-urban-bike-ride-through-city-streets.mp4
│   ├── vibrant-city-street-with-shops-and-pedestrians.mp4
├── life_memorizer/               # Core source package
│   ├── __init__.py               # Package initializer exporting modules
│   ├── cli.py                    # Command-line interface definitions
│   ├── config.py                 # Configuration settings loader & validator
│   ├── embeddings.py             # Multi-modal embedding (Gemini / Matryoshka)
│   ├── ingest.py                 # Ingestion pipeline coordinating media processing
│   ├── media.py                  # Media processing utils (OpenCV, ffmpeg, PyTesseract)
│   ├── mock_data.py              # Mock dataset for quick seeding and testing
│   ├── models.py                 # Core Pydantic data schemas & enums
│   ├── rag.py                    # Local Retrieval-Augmented Generation flows
│   ├── recall.py                 # Recall engine for vector & hybrid queries
│   └── store.py                  # Qdrant Edge vector store wrapper
└── tests/                        # Test suite directory
    ├── __init__.py               # Test package initializer
    ├── conftest.py               # Shared pytest fixtures (mock store / mock embedder)
    ├── test_embeddings.py        # Unit tests for real and fake embedding layers
    ├── test_rag.py               # Unit tests for LocalRAG pipeline
    ├── test_step5.py             # Unit tests for Step 5 (quantization & TTL pruning)
    └── test_store_and_recall.py  # Unit tests for Qdrant storage and retrieval
```
Please refer to [ARCHITECTURE-DOCUMENTATION.md](ARCHITECTURE-DOCUMENTATION.md) for a detailed explanation of the system architecture, codebase knowledge graph, component interactions, ingestion and retrieval pipeline knowledge graph and architectural explanations of each code file used in this project.

## Requirements

- **Python 3.10+**
- (Optional) a **Gemini API key** for real embeddings — not needed for the offline demo.
- (Optional) **ffmpeg** + **Tesseract** only if you ingest real video.

## Install

```bash
# Clone
git clone https://github.com/satyam671/Life-Memorizer-With-Gemini-Embedding-2-And-Qdrant-Edge.git life-memorizer
cd life-memorizer

## 📥 Installation

From the project root directory, set up your virtual environment and install the package with dependencies:

### For Windows PowerShell:
```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 3. Install the project in editable mode with development & media tools
pip install -e .[dev,media]
```

### For Bash (macOS/Linux):
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install the project in editable mode with development & media tools
pip install -e .[dev,media]
```

---

## 🧪 Scenario 1: Mock / Offline Mode (No API Key Required)

In this mode, we use a deterministic, local feature-hashing embedder that simulates Gemini embeddings. It runs fully offline and queries the mock data seeded from `life_memorizer/mock_data.py`.

### 1. Set the Environment Variables
Configure the session to run in fake/offline mode:

- **Windows PowerShell**:
  ```powershell
  $env:LIFE_MEMORIZER_FAKE_EMBEDDINGS="1"
  $env:LIFE_MEMORIZER_FAKE_RAG="1"
  ```
- **Bash (macOS/Linux)**:
  ```bash
  export LIFE_MEMORIZER_FAKE_EMBEDDINGS=1
  export LIFE_MEMORIZER_FAKE_RAG=1
  ```

### 2. Initialize the Database
Verify or create the local Qdrant collection:
```bash
life-memorizer init
```

### 3. Seed the Mock Data
Load the simulated memory log (lost keys, conversation with Sarah, cafe menu, etc.):
```bash
life-memorizer seed
```

### 4. Query the Mock Data
Run queries to recall specific moments:
```bash
# Scenario A (Visual Search): Search image/video frames
life-memorizer recall "where did I leave my keys?" --modality image
```
#### Output:
<img width="1090" height="337" alt="image" src="https://github.com/user-attachments/assets/bb18f38a-0a69-46fb-89cf-f3c75ce7cb70" />

```bash
# Scenario B (Audio Search): Search conversation transcripts
life-memorizer recall "what did Sarah say to buy?" --modality audio
```
#### Output:
<img width="1090" height="264" alt="image" src="https://github.com/user-attachments/assets/50c6bcde-b6bf-4298-8613-9ec6d907240b" />

```bash
# Scenario C (Hybrid Search): Fuses OCR text & vision, filtered to Cafe
life-memorizer recall "the cafe menu" --location Cafe --hybrid
```
#### Output:
<img width="1090" height="228" alt="image" src="https://github.com/user-attachments/assets/2f45819f-2651-4d53-9520-b376327f075c" />

That's it — you now have a working offline memory assistant.

> **Note on the offline embedder.** With `LIFE_MEMORIZER_FAKE_EMBEDDINGS=1` there is no
> AI model: a deterministic, lexical (word-overlap) embedder stands in for Gemini so the
> demo runs with zero setup. It ranks results by shared words, which is good enough for the
> sample queries but is **not** true semantic understanding. For real meaning-based recall,
> use the Gemini API (below).

> **Where are the sample images/audio?** There aren't any — and you don't need them.
> `seed` builds its moments from *text descriptions* in `life_memorizer/mock_data.py`. The
> `media_file_path` values (e.g. `media_cache/home/hallway_table_keys.jpg`) are just metadata
> showing where a real device *would* save a frame; no files are written to disk.


### 5. Ask Questions (Local RAG)
Go beyond search: ask a question and get a spoken-style answer **grounded only in your
recalled memories** (no hallucination). This retrieves the most relevant moments and feeds
them to a local Gemma model.

Run it offline with the deterministic RAG stub (no model, no key):

```bash
export LIFE_MEMORIZER_FAKE_EMBEDDINGS=1
export LIFE_MEMORIZER_FAKE_RAG=1
life-memorizer seed
life-memorizer ask "where did I leave my keys?"
life-memorizer ask "what did Sarah ask me to buy?"
```
#### Output
<img width="1246" height="221" alt="image" src="https://github.com/user-attachments/assets/d71ee58f-f597-4e16-81c9-bb7159a7b4d4" />

---

## 📹 Scenario 2: Live Video Ingestion Mode (Real Gemini API)

In this mode, the system utilizes the actual **Gemini API** to process, slice, and embed real video frames, OCR logs, and audio. It automatically **excludes** the mock data from your queries to prevent wrong results and incorrect scores.

### 1. Set up the `.env` Configuration
Copy the template `.env` file to `.env`:

- **Windows PowerShell / Bash**:
  ```bash
  cp .env.example .env
  ```

Open `.env` in a text editor and configure your Gemini API Key:
```env
GEMINI_API_KEY=AIzaSy...YourActualGeminiKeyHere
LIFE_MEMORIZER_FAKE_EMBEDDINGS=0
LIFE_MEMORIZER_FAKE_RAG=0
```

*Note: Make sure `LIFE_MEMORIZER_FAKE_EMBEDDINGS` is either omitted or set to `0`.*

### 2. Clear / Reset Environment Variables
Ensure the offline environment flags are disabled:

- **Windows PowerShell**:
  ```powershell
  $env:LIFE_MEMORIZER_FAKE_EMBEDDINGS="0"
  $env:LIFE_MEMORIZER_FAKE_RAG="0"
  ```
- **Bash (macOS/Linux)**:
  ```bash
  export LIFE_MEMORIZER_FAKE_EMBEDDINGS=0
  export LIFE_MEMORIZER_FAKE_RAG=0
  ```

### 3. Initialize the Database
Initialize (or verify) the local database. If you want to delete any previous mock collection and start completely fresh, you can remove the folder `./life_memorizer_db` or reinitialize:
```bash
life-memorizer init
```

### 4. Ingest Your Video File
Provide a video file on your system to analyze. The video will be sampled into image frames, audio chunks will be extracted, OCR will be run, and everything will be embedded:
```bash
# Point to any video file on your machine (e.g. samples/walk.mp4)
life-memorizer ingest --video samples/pov-urban-bike-ride-through-city-streets.mp4 --location Home
```
#### Output
<img width="618" height="16" alt="image" src="https://github.com/user-attachments/assets/22f92e1f-d8db-4c5d-8105-9caa0d547776" />
<img width="1164" height="255" alt="image" src="https://github.com/user-attachments/assets/8205de30-09d6-4c62-83e9-8cec4f239267" />
All the ingested frames are stored locally whose path is specified in the output.

---

You supply your own media for `ingest`; nothing is bundled. Ingesting real video also needs
the media extras: `pip install -e ".[media]"` (plus system `ffmpeg`, and `tesseract` if you
want OCR). With the real API, semantic queries like "where did I leave my keys?" rank the
correct moment first because Gemini understands meaning, not just words.

---

### 5. Query Only the Ingested Video Frames
Ask questions related to your ingested video:

```bash
life-memorizer recall "where did i see the red car today while i was cycling?" --modality image
```
#### Output
<img width="1461" height="760" alt="ingested video 1" src="https://github.com/user-attachments/assets/0dc9d6f2-478d-42ee-9329-631f5eb41a8d" />
---

```bash
life-memorizer recall "where did i spotted a white truck today while i was cycling?" --modality image
```
#### Output
<img width="1452" height="762" alt="ingested video 2" src="https://github.com/user-attachments/assets/7232edb2-8ff1-469c-b245-3f2581b680f1" />
---

```bash
life-memorizer recall "when did a couple crossed  me while I was walking on the city streets?" --modality image
```
#### Output
<img width="1453" height="588" alt="ingested video 3" src="https://github.com/user-attachments/assets/50c628b8-d4da-4f3a-8de0-b607279ddbbc" />
---


*Thanks to our query filter, the search will **only** return matches from your ingested video files. Mock data seeded previously will be automatically excluded, ensuring accurate scores and correct moments are returned.*

### 6. Ask grounded questions (Local RAG)
To ask questions and get answers grounded on your real video memories:

For a **fully local** answer with real Gemma via [Ollama](https://ollama.com):

```bash
# Make sure to run Ollama locally or set LIFE_MEMORIZER_RAG_BACKEND=gemini in your .env
# Option A: Ollama backend (default)
# (Run `ollama pull gemma2:2b` first)
life-memorizer ask "where did i spotted a white truck today while i was cycling?"

# Option B: Gemini API backend
# (Requires GEMINI_API_KEY set in .env)
# Powershell: $env:LIFE_MEMORIZER_RAG_BACKEND="gemini"
# Bash: export LIFE_MEMORIZER_RAG_BACKEND=gemini
life-memorizer ask "when did a couple crossed  me while I was walking on the city streets?"
```


The answer cites the memories it used (`--show-sources`, on by default). If nothing relevant
is found, it says it has no memory of that instead of guessing.


## Commands

| Command | What it does |
|---------|--------------|
| `life-memorizer init` | Create/verify the local memory database. |
| `life-memorizer seed` | Load the realistic sample day. |
| `life-memorizer ingest --video PATH [--location TAG]` | Index a real video clip. |
| `life-memorizer recall "QUERY" [--modality image\|audio\|text] [--location TAG] [--hybrid]` | Recall moments. |
| `life-memorizer ask "QUESTION" [--location TAG] [--hybrid] [--limit N]` | Answer a question from your memories (local RAG). |
| `life-memorizer stats` | Count stored memories. |
| `life-memorizer prune [--ttl-days N] [--summarize]` | Age out / summarize old memories. |

Run `life-memorizer --help` for everything.



## Configuration

Settings come from environment variables (or a `.env` file). The defaults work out of the box.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | *(empty)* | Your Gemini key (real embeddings only). |
| `LIFE_MEMORIZER_FAKE_EMBEDDINGS` | `0` | `1` = offline stub, no API calls. |
| `LIFE_MEMORIZER_EMBED_MODEL` | `gemini-embedding-2` | Embedding model id. |
| `LIFE_MEMORIZER_EMBED_DIM` | `768` | Downscaled vector size (native is up to 3072). |
| `LIFE_MEMORIZER_DB_PATH` | `./life_memorizer_db` | Where the local database lives. |
| `LIFE_MEMORIZER_QUANTIZATION` | `scalar` | `none` \| `scalar` \| `binary`. |
| `LIFE_MEMORIZER_TTL_DAYS` | `90` | Auto-prune age (`0` = never). |

## How it works

```
 video clip  ->  sample frames / audio / OCR text
             ->  Gemini Embedding 2  (one aligned space, 3072 -> 768)
             ->  Qdrant Edge         (one "moment" point, 3 named vectors)
             ->  recall              (visual / audio / hybrid + filters)
```

Each moment stores up to three named vectors — `video_frame`, `ambient_audio`, `ocr_log` —
plus light metadata (timestamp, location, media path). Raw media never enters the database.

---

## 📊 Useful Utilities

### Check Database Statistics
To see how many moments are currently stored in Qdrant Edge:
```bash
life-memorizer stats
```

## Production & privacy

**Quantization** — set `LIFE_MEMORIZER_QUANTIZATION` to `scalar` (int8, default) or `binary`
for up to ~4× less memory on small devices. Combined with the 3072→768 Matryoshka downscale,
this keeps the index small enough for a Jetson / Raspberry Pi.

**TTL + summarization** — old memories are aged out so storage stays roughly constant:

```bash
life-memorizer prune                 # summarize expired moments into compact digests (default)
life-memorizer prune --no-summarize  # just delete expired moments
life-memorizer prune --ttl-days 30   # override the TTL window
```

Summarization mean-pools the named vectors of expired moments (per location) into a single
searchable "digest" point and keeps an extractive text gist, so the past is still recallable
without keeping every frame. Enable automatic, cadence-limited pruning with
`LIFE_MEMORIZER_AUTO_PRUNE=1` (runs at most every `LIFE_MEMORIZER_AUTO_PRUNE_INTERVAL_HOURS`).

**Privacy-first** — raw media never enters the database; only embeddings and light metadata are
stored, locally. For maximum privacy set `LIFE_MEMORIZER_STORE_MEDIA_PATH=0` so not even the
file path is persisted. Nothing biometric or raw leaves the edge container.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest        # runs fully offline
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
