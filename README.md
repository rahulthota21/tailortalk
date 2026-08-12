# TailorTalk — Saree Visual Similarity Search Agent

A chat agent that finds visually similar sarees from a catalogue. You chat
naturally, share a photo (upload or URL), and the agent decides when to run
a similarity search, calls a vector-search tool behind the scenes, and shows
you the closest matches with scores.

## How it works (architecture)

```
Streamlit chat UI  →  LangChain tool-calling agent (gpt-4o-mini)
                              │
                              ▼
                 search_similar_sarees(image_ref, top_k)
                              │
                              ▼
        Chroma vector store  ── stage 1: ANN over DINOv2 embeddings
                              ── stage 2: re-rank by DINOv2 + colour histogram
```

- **Vector DB:** ChromaDB (local, persistent, zero infra to run).
- **Agent framework:** LangChain `create_tool_calling_agent`, one
  `StructuredTool` (`search_similar_sarees`) with a strict pydantic schema.
- **Frontend:** Streamlit chat interface (`app.py`).
- **Embeddings:** DINOv2 (`facebook/dinov2-small`) + an HSV colour histogram.

## Why these model choices (search quality)

The brief calls out the core difficulty directly: every image is the same
garment, so generic embeddings return "loose" results. Two decisions target
that:

1. **DINOv2 instead of CLIP.** CLIP is trained to align images with text
   captions, so its embedding space is optimized for *semantic* matching
   ("this is a saree") — on a single-category dataset that signal is close
   to useless. DINOv2 is trained with a self-supervised, image-only
   objective and is well known to produce embeddings that are much more
   sensitive to texture, weave, print and fine structure — exactly the
   fabric/border/pallu differences that actually distinguish sarees.

2. **Colour-histogram re-ranking.** Backbones like DINOv2 are trained with
   colour-jitter augmentation, so they're deliberately somewhat
   colour-invariant — but colour combination is one of the strongest cues a
   shopper uses. Rather than concatenate colour into the indexed vector
   (which would make the DINO signal harder to tune later), the app does a
   **two-stage retrieve-then-rerank**: ANN-retrieve the top ~30 candidates
   by DINOv2 cosine similarity, then re-rank those by
   `0.65 × DINO_similarity + 0.35 × HSV_histogram_similarity`, and return
   the top-k. This keeps ANN search fast while sharpening final ranking on
   colour/pattern.

3. **Cheap test-time augmentation.** Each embedding is the average of the
   full image and an 80%-centre-crop, which makes matches more robust to
   backgrounds, margins, and model-shot vs. flat-lay product photos.

4. **The `dino_weight` (0.65) and `candidate_k` (30) are the two knobs to
   tune first if you want to shift results more toward texture/pattern vs.
   colour** — see `core/vector_store.py::search`.

## Why the agent design is the way it is

LLM tool-call arguments are plain text/JSON — passing a full image as
base64 through a tool call is wasteful and unreliable. Instead:

- The moment a user uploads or links an image, the Streamlit app saves it
  to disk and mints a short opaque id (`img_3f9a1c2b`), stored in an
  in-session `image_registry: {ref: path}`.
- That id is appended to the user's chat message as a system note, so the
  LLM sees it in context.
- The LLM's only job is to notice "the user wants a similarity search" and
  call `search_similar_sarees(image_ref=..., top_k=...)` with the right id
  — it never touches raw pixels. This is also what keeps the agent from
  firing a search on plain chit-chat ("hi", "what colours suit dark skin
  tones?") — the system prompt and tool description both gate the tool to
  "user explicitly wants matches for a shared image."
- Actual pixels and scoring stay entirely inside `core/vector_store.py`.

## Project structure

```
tailortalk/
├── app.py                     # Streamlit chat UI
├── core/
│   ├── embeddings.py          # DINOv2 + colour histogram
│   ├── vector_store.py        # Chroma wrapper + two-stage search
│   ├── search_tool.py         # LangChain tool schema
│   └── agent.py                # Tool-calling agent + system prompt
├── scripts/
│   ├── download_dataset.py    # pulls the assignment's Google Drive dataset
│   └── build_index.py         # embeds data/raw/ into chroma_db/
├── chroma_db/                 # prebuilt vector index (COMMIT this)
├── requirements.txt
└── .streamlit/secrets.toml.example
```

## Local setup

```bash
git clone <your-repo-url>
cd tailortalk
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Get the dataset (from the assignment's Google Drive link)
python scripts/download_dataset.py
# If gdown fails on the Drive link (Drive sometimes rate-limits automated
# downloads), just download the zip manually from the link in the
# assignment PDF and unzip it into data/raw/ yourself.

# 2. Build the vector index (embeds every image, one-time — a few minutes
#    on CPU depending on dataset size)
python scripts/build_index.py

# 3. Add your OpenAI key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your OPENAI_API_KEY

# 4. Run
streamlit run app.py
```

Then in the app: upload a saree photo (or paste an image URL) in the
sidebar, and type something like *"find me similar sarees"* in the chat.

## Deployment

**Recommended: Hugging Face Spaces (Streamlit SDK).** It gives more RAM/CPU
on the free tier than Streamlit Community Cloud, which matters because
`torch` + `transformers` + DINOv2 need more headroom than a typical
Streamlit app.

1. Create a new Space → SDK: Streamlit.
2. Push this repo's contents to the Space's git repo (**including the
   `chroma_db/` folder** — it's a prebuilt index, so the app needs no
   build step at startup and works "out of the box" as required).
3. In Space Settings → Repository secrets, add `OPENAI_API_KEY` (and
   optionally `OPENAI_MODEL`).
4. The Space will install `requirements.txt` and launch `app.py`
   automatically.

**Alternative: Streamlit Community Cloud.** Same idea — connect the GitHub
repo, set `app.py` as the entrypoint, add `OPENAI_API_KEY` under app
Secrets. Community Cloud's free tier is more RAM-constrained, so if you hit
memory limits, switch `DINO_MODEL_NAME` in `core/embeddings.py` to an even
smaller backbone or move to HF Spaces.

Either way: **commit `chroma_db/`** (it's small — a few MB for a
few-thousand-image catalogue) so the reviewer's deployment doesn't need to
download the dataset or re-embed anything at runtime.

## Assumptions & trade-offs

- **Prebuilt index over runtime indexing.** Embedding thousands of images
  takes minutes on CPU; doing that on every cold start would violate "must
  work out of the box." So the index is built once locally and committed.
  Trade-off: if you change the dataset, you must re-run
  `scripts/build_index.py` and re-commit `chroma_db/`.
- **`dinov2-small` over `dinov2-large`.** Smaller model = fast enough for
  free-tier CPU hosting with a small quality cost. Swappable via
  `DINO_MODEL_NAME` if you have GPU hosting.
- **OpenAI (`gpt-4o-mini`) for the agent LLM**, since it's the most common
  LangChain function-calling setup; swapping to Anthropic/another provider
  only requires changing `core/agent.py`'s `llm =` line to
  `ChatAnthropic(...)`.
- **Single flat category assumption.** The colour-histogram re-rank weight
  (0.65/0.35) was chosen for a homogeneous "all sarees" catalogue — if the
  dataset also contains very different garment types, this weighting and
  the tool description should probably change.
- **Session-only image registry.** Uploaded query images aren't persisted
  or added to the searchable index — they're used purely as queries. This
  matches the assignment's "search behind the scenes" framing rather than
  "let users contribute to the catalogue."
