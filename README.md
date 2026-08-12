# TailorTalk — Saree Visual Similarity Search Agent

**Live app:** https://tailortalking.streamlit.app/
**Code:** https://github.com/rahulthota21/tailortalk

A chat agent that finds visually similar sarees from a catalogue. You chat
naturally with it, share a saree photo (upload or paste a URL), and the
agent figures out on its own when you're asking for a similarity search,
calls a vector-search tool behind the scenes, and shows you the closest
matches with similarity scores.

---

## 1. The assignment, in plain words

The brief (TailorTalk take-home) asked for:

1. An AI agent that **chats naturally**, understands **when** the user is
   asking for a similarity search (not every message), takes in an image,
   and searches a **vector index** behind the scenes.
2. A **vector database** (Pinecone / Qdrant / ChromaDB / FAISS) storing
   embeddings I compute myself from a given saree image dataset.
3. The search exposed as a **callable tool with a schema** via an agent
   framework (LangChain / LlamaIndex / similar function-calling setup).
4. A **Streamlit or Gradio** chat frontend.
5. The hard part, called out explicitly in the brief: every image in the
   dataset is *the same category* (a saree), so a naive/generic embedding
   search returns loose, non-specific matches. The real differences are
   fine-grained — fabric, weave, print, colour combination, border, pallu
   work — and results need to hold up across repeated tests with different
   query images, not just look good on one lucky example.
6. Deployed somewhere that works with **no local setup** for the reviewer,
   plus a GitHub repo and this README.

---

## 2. What I actually built (architecture)

```
Streamlit chat UI  (app.py)
        │
        ▼
LangChain tool-calling agent (Groq: llama-3.3-70b-versatile)
        │
        ▼
search_similar_sarees(image_ref, top_k)   ← the tool, with a pydantic schema
        │
        ▼
ChromaDB vector store
   stage 1: ANN search over DINOv2 embeddings  (fast, texture/pattern-aware)
   stage 2: re-rank candidates by DINOv2-sim + HSV colour-histogram-sim
```

- **Frontend:** Streamlit chat interface (`app.py`) — file upload + URL
  input in the sidebar, chat box, images with scores rendered inline.
- **Agent framework:** LangChain's `create_tool_calling_agent`, one
  `StructuredTool` (`search_similar_sarees`) with a strict pydantic input
  schema (`image_ref: str`, `top_k: int`).
- **LLM for the agent:** Groq (`llama-3.3-70b-versatile`), via
  `langchain-groq`. I originally set this project up against OpenAI, but
  switched to Groq (free/fast tier, still full tool-calling support via
  `ChatGroq`) — the LangChain agent code is identical either way, only the
  `llm = ...` line changes.
- **Vector DB:** ChromaDB, running locally/embedded (no external service to
  provision), persisted to a `chroma_db/` folder that's committed to the
  repo so the deployed app needs zero build step.
- **Embeddings:** DINOv2 (`facebook/dinov2-small`) for the indexed vector +
  an HSV colour histogram used only for re-ranking. Why, in detail, below.

---

## 3. The dataset (and a real-world detour)

The assignment linked a Google Drive file for the dataset. When I actually
opened it, it turned out to be a **CSV of ~1,074 saree products**
(`data/byrappa_tejas_31july.csv`) from a real saree retailer's catalogue,
with columns:

```
Name, SKU, Stock, Retail Price, Discounted Price, image_url, Website Link
```

So "process, embed and index this dataset yourself" in practice meant:

1. **`scripts/download_from_csv.py`** — reads the CSV, downloads every
   `image_url` (mostly `.webp`), saves each to `data/raw/<SKU>.<ext>`.
   Ran once locally: **1070/1074 images downloaded successfully** (4 dead
   links, safely skipped — logged, not fatal).
2. **`scripts/build_index.py`** — walks `data/raw/`, embeds each image, and
   writes it into Chroma. It also re-reads the CSV to attach each image's
   original public `image_url` as metadata (see §6 — this turned out to
   matter a lot for deployment).
3. Final indexed catalogue: **650 unique images** (fewer than 1070 because
   several products share the same underlying photo/SKU pattern, and those
   duplicate IDs were naturally deduplicated by Chroma's `id`-based `add`).

I deliberately kept `data/raw/` and the CSV **out of git**
(`.gitignore`) — only the prebuilt `chroma_db/` vector index is committed,
so the deployed app never needs to re-download or re-embed anything.

---

## 4. Why DINOv2 instead of CLIP (the main "search quality" decision)

This is the crux of the assignment, so worth spelling out precisely.

**CLIP** is trained to align images with *text captions*. Its embedding
space is optimized for semantic/category-level matching — "this looks like
a saree." On a dataset where **every single image is a saree**, that signal
is close to worthless: CLIP will call two completely different sarees
"similar" simply because they're both sarees. That's exactly the "loose,
generic results" failure mode the assignment brief warns about.

**DINOv2** is trained with a self-supervised, image-only objective (no
text supervision at all — it's never seen a caption). It's well documented
to produce embeddings that are much more sensitive to *texture, weave,
print structure and fine visual detail* — precisely the fabric/border/pallu
differences that actually separate one saree from another.

So: swap CLIP → DINOv2 as the backbone for the indexed vector
(`core/embeddings.py::get_dino_embedding`).

**One extra trick — test-time augmentation (TTA):** each embedding is the
*average* of the embedding of the full image and an 80%-centre-crop. This
makes matches more robust to product-photo variation (mannequin vs.
flat-lay, background clutter, framing).

---

## 5. Why a colour-histogram re-ranking stage on top

DINOv2 (like most modern self-supervised vision backbones) is trained with
colour-jitter augmentation, so it's *deliberately* somewhat colour-invariant.
That's normally a feature — but colour combination is one of the strongest
cues a shopper actually uses to judge "similar" sarees, so under-weighting
it hurts here.

Rather than bake colour into the single indexed vector (which would make
the DINO signal harder to tune independently later), the search does a
**two-stage retrieve-then-rerank** (`core/vector_store.py::search`):

1. **Stage 1 — ANN retrieval:** pull the top ~30 candidates by DINOv2
   cosine similarity from Chroma. Fast, and already texture/pattern-aware.
2. **Stage 2 — re-rank:** for those 30 candidates, compute an HSV colour
   histogram similarity (`cv2.compareHist`, correlation method) against the
   query image, and blend:

   ```
   final_score = 0.65 * dino_similarity + 0.35 * colour_similarity
   ```
3. Sort by `final_score`, return the top-k.

`dino_weight` (currently `0.65`) is the single knob to turn if results feel
too texture-driven vs. too colour-driven — it lives in
`core/vector_store.py::search()` and needs **no re-indexing** to change,
since both raw similarity components are stored/recomputed at query time.

Each returned result includes `score`, `dino_similarity`, and
`color_similarity` separately, which was very useful for debugging during
development (you can literally see which signal is driving a given match).

---

## 6. Agent design — why it doesn't pass raw images to the LLM

LLM tool-calling arguments are plain JSON/text. Passing a full image as
base64 through a tool call is wasteful and unreliable — the model would
have to faithfully reproduce potentially megabytes of base64 just to make
one tool call.

Instead:

- The moment the user uploads or pastes an image URL in the Streamlit
  sidebar, the app immediately saves it and mints a short opaque id, e.g.
  `img_3f9a1c2b`, stored in a session-scoped `image_registry: {ref: path}`.
- That id is silently appended to the user's next chat message as a system
  note (`[system note: user just shared an image, image_ref=img_3f9a1c2b]`).
- The agent's **only** job is to notice "the user wants a similarity
  search" from natural conversation and call
  `search_similar_sarees(image_ref="img_3f9a1c2b", top_k=5)` — it never
  touches raw pixels. This is also what stops the tool from firing on
  plain chit-chat ("hi", "what colours suit dark skin tones?") — both the
  system prompt and the tool's own description explicitly gate it to "user
  is asking for matches to a shared image."
- The actual pixel-level work (embedding, ANN search, re-ranking) happens
  entirely inside `core/vector_store.py`, outside the LLM's control loop.

---

## 7. A real deployment bug I hit, and the fix (worth knowing)

After deploying to Streamlit Community Cloud, the search worked (verbose
agent logs showed correct tool calls and real scores) but **rendering the
result images crashed** with `MediaFileStorageError`. Root cause: the app
correctly does *not* commit `data/raw/` (650 raw images, unnecessary repo
bloat) — but that means the deployed server has no local files to point
`st.image()` at when it tries to open `res["path"]`.

**Fix:** since the CSV already has each product's original public
`image_url`, `build_index.py` now stores that URL as Chroma metadata
alongside every embedding, and `app.py` renders `res["url"] or res["path"]`
— i.e. prefer the public URL (works anywhere), fall back to local disk path
(useful for local dev/offline testing). No images need to be committed to
git at all.

---

## 8. Project structure

```
tailortalk/
├── app.py                          # Streamlit chat UI
├── core/
│   ├── embeddings.py                # DINOv2 + HSV colour histogram
│   ├── vector_store.py              # Chroma wrapper, 2-stage search
│   ├── search_tool.py               # LangChain StructuredTool schema
│   └── agent.py                     # Tool-calling agent + system prompt
├── scripts/
│   ├── download_from_csv.py         # pulls all images from the CSV's image_url column
│   └── build_index.py               # embeds data/raw/ + attaches image_url → chroma_db/
├── chroma_db/                       # prebuilt vector index (committed)
├── requirements.txt
└── .streamlit/secrets.toml.example  # GROQ_API_KEY template
```

## 9. Local setup (reproducing this from scratch)

```bash
git clone https://github.com/rahulthota21/tailortalk
cd tailortalk
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt

# 1. Get the dataset
python scripts/download_from_csv.py     # needs data/byrappa_tejas_31july.csv present

# 2. Build the vector index (one-time, a few minutes on CPU)
python scripts/build_index.py

# 3. Add your Groq key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit it: GROQ_API_KEY = "gsk_..."

# 4. Run
streamlit run app.py
```

Then: upload a saree photo (or paste an image URL) in the sidebar, and type
something like *"find me similar sarees"* in the chat.

## 10. Deployment notes

Deployed on **Streamlit Community Cloud** (`tailortalking.streamlit.app`),
connected directly to the GitHub repo, with `GROQ_API_KEY` set under app
Secrets. Because `chroma_db/` is committed and result images are served
from their original public `image_url`s, the app needs **no build step and
no local files** to work — matching the assignment's "must work out of the
box" requirement.

**Known Windows dev-environment gotcha (documented in case you hit it
again):** `torch==2.4.0`'s CPU wheel failed to load on Windows
(`OSError ... fbgemm.dll`) even after installing the VC++ Redistributable.
Downgrading to `torch==2.2.2+cpu` (via
`--index-url https://download.pytorch.org/whl/cpu`) resolved it. Pinned in
`requirements.txt`.

## 11. Assumptions & trade-offs

- **Prebuilt index over runtime indexing.** Embedding hundreds of images
  takes several minutes on CPU; doing that on every cold start would break
  "works out of the box." So the index is built once locally and
  committed — re-run `scripts/build_index.py` and re-commit `chroma_db/` if
  the catalogue changes.
- **`dinov2-small` over larger DINOv2 variants** — enough quality for
  free-tier CPU hosting; swappable via `DINO_MODEL_NAME` in
  `core/embeddings.py` if you have GPU hosting and want more accuracy.
- **Groq (`llama-3.3-70b-versatile`) instead of OpenAI/Anthropic** for the
  agent LLM — same LangChain tool-calling pattern, just a different
  provider; swapping providers is a one-line change in `core/agent.py`.
- **`dino_weight = 0.65`** was chosen for a homogeneous "all sarees"
  catalogue by eyeballing results across several query images; it's the
  first thing to retune if results skew too texture- or too colour-driven.
- **Images displayed from their original public URLs**, not re-hosted —
  keeps the repo small and the app has no storage dependency, at the cost
  of depending on the retailer's CDN staying up.
- **Session-only image registry** — uploaded *query* images are used
  purely as search input, never added to the searchable catalogue itself.