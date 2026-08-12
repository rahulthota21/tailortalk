# TailorTalk

Live app: https://tailortalking.streamlit.app/  
Code: https://github.com/rahulthota21/tailortalk

TailorTalk is a chat assistant that finds sarees similar to a photo you share. Upload a picture or paste a link, chat normally, and it returns the closest matching sarees from a catalogue with confidence scores.

This README has two parts. The first explains the project simply. The second is the technical walkthrough.

## Part One: The Project

### What problem this solves

You see a saree you like (a friend's, a photo online, a shop window) and want something similar from a catalogue. Typing a description rarely works well for clothing. Exact shade, weave, border pattern, and pallu style are hard to put into words. Showing a picture and asking "find me something like this" is much easier. That is what TailorTalk does.

### Why this is harder than it sounds

If a shop mixed shirts, shoes, bags, and sarees, simple visual search could lean on category differences. This catalogue is only sarees. Every image is the same broad category, so the search must notice fine details: fabric, print, colour palette, border work. Getting that right was the main focus.

### How the assistant behaves

No special commands. Just chat. Greetings or general styling questions get normal replies. It searches the catalogue only when you clearly want similar items for a photo you shared. If you ask for a search without an image, it asks you to upload one first.

When you upload a photo, the app quietly gives the assistant a short reference to that image. The assistant only needs to recognise intent and call the search with that reference. It never handles the actual picture, which keeps things fast and reliable.

### What was used to build it

Catalogue images were indexed with a modern image recognition model and stored in ChromaDB. The chat assistant runs on a Groq large language model through LangChain, which lets the model call a similarity search tool. The interface is Streamlit and is deployed publicly so anyone can try it.

### Where the dataset came from

A real saree retailer's product listing of just over a thousand items, each with name, price, and photo link. Every photo was downloaded, examined, and indexed.

### What a good result looks like

A strong match shares the same colour family, similar print or pattern style, and overall look and feel. It should do this consistently across many queries, not just one lucky demo image. That reliability was the real bar for the project.

## Part Two: Technical Deep Dive

### Architecture

```
Streamlit chat interface (app.py)
        │
        ▼
LangChain tool calling agent (Groq llama-3.3-70b-versatile)
        │
        ▼
search_similar_sarees(image_ref, top_k)
        │
        ▼
ChromaDB vector store
   stage one: nearest neighbour over DINOv2 embeddings
   stage two: re-rank with DINOv2 similarity + colour histogram similarity
```

### Dataset preparation

The source was a Google Drive CSV of about 1,074 saree products (data/byrappa_tejas_31july.csv) with name, SKU, stock, prices, image URL, and website link.

scripts/download_from_csv.py downloads every image_url into data/raw/<SKU>.<extension>. 1,070 of 1,074 images downloaded successfully. Four 404 links were skipped and logged.

scripts/build_index.py walks data/raw/, embeds each image, and writes to ChromaDB. It looks up each SKU in the CSV and attaches the original public image URL as metadata. After deduplication (some SKUs shared identical photos), the final catalogue holds 650 unique images.

Raw images and the CSV are excluded via .gitignore. Only the finished chroma_db/ index is committed, so the deployed app needs no build step or dataset download at startup.

### Why DINOv2 instead of CLIP

CLIP aligns images with text captions, so its space is good for category matching ("this looks like a saree"). On a dataset of only sarees that signal is almost useless. CLIP can call two very different sarees similar just because both are sarees, which matches the loose generic results the assignment warned about.

DINOv2 is self supervised and image only. It is more sensitive to texture, weave, print structure, and other fine visual detail, which is what separates one saree from another.

The project uses facebook/dinov2-small (core/embeddings.py, get_dino_embedding). Each embedding averages two passes: the full image and an 80 percent centre crop. This test time augmentation helps with background clutter and different photo framing.

### Why colour gets a separate re-ranking stage

DINOv2 and similar models are trained with colour jitter, so they are somewhat colour invariant. For sarees, colour combination is one of the strongest signals shoppers use, so under weighting it hurts usefulness.

Colour is not merged into the indexed vector. Search runs in two stages (core/vector_store.py):

1. Retrieve top thirty candidates from ChromaDB by DINOv2 cosine similarity.
2. Re-rank those thirty with a weighted blend:

```
final_score = 0.65 × dino_similarity + 0.35 × colour_similarity
```

Colour similarity uses HSV histogram comparison (OpenCV compareHist, correlation method). The blend weight (dino_weight) is a single constant in vector_store.py::search() and can be changed without re-indexing because both components are computed at query time.

Every result includes the blended score and its two components separately, which helped debug matches during development.

### How the agent decides when to search

LLM tool calls take plain text or JSON. Passing a full image as base64 would be wasteful and unreliable.

When a user uploads a photo or pastes an image URL, the app saves it and generates a short opaque reference such as img_3f9a1c2b, stored in the session image_registry. That reference is appended to the next chat message as a system note. The agent only needs to recognise that the user wants a similarity search and call search_similar_sarees with the reference and a count. It never touches raw pixels.

The system prompt and tool description both restrict calls to genuine similarity search requests, so ordinary conversation does not trigger search.

### Deployment issue and fix

After the first Streamlit Community Cloud deploy, search worked (correct tool calls and scores in logs) but rendering result images failed with MediaFileStorageError. data/raw/ was correctly excluded from the repo to avoid 650 images, but that left the deployed server with no local files for st.image().

The CSV already had each product's public image URL. The fix was to store that URL as Chroma metadata in build_index.py, then have the app render res["url"] first and fall back to a local path only when needed. No images need to live in the repository.

### Project layout

```
tailortalk/
  app.py                          Streamlit chat interface
  core/
    embeddings.py                  DINOv2 embedding + HSV colour histogram
    vector_store.py                ChromaDB wrapper and two stage search
    search_tool.py                 LangChain tool schema
    agent.py                       Tool calling agent and system prompt
  scripts/
    download_from_csv.py           downloads images from the CSV
    build_index.py                 embeds images and attaches public URL
  chroma_db/                       prebuilt vector index (committed)
  requirements.txt
  .streamlit/secrets.toml.example  template for the Groq API key
```

### Running locally

```bash
git clone https://github.com/rahulthota21/tailortalk
cd tailortalk
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

python scripts/download_from_csv.py
python scripts/build_index.py

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit the file and add your GROQ_API_KEY

streamlit run app.py
```

### Deployment notes

Deployed on Streamlit Community Cloud from the GitHub repo, with GROQ_API_KEY set as an app secret. Because chroma_db/ is committed and result images load from original public URLs, there is no build step and no need for local image files. It works out of the box for a reviewer.

On Windows, torch 2.4.0 CPU wheel failed with an fbgemm.dll error even after installing the Visual C++ Redistributable. Downgrading to torch 2.2.2 with the official CPU wheel index fixed it. That version is pinned in requirements.txt.

### Assumptions and trade offs

The vector index is built once locally and committed. Embedding hundreds of images takes several minutes on CPU, so building on every cold start would break the "works out of the box" requirement. If the catalogue changes, re-run build_index.py and commit the updated chroma_db/.

dinov2-small was chosen over larger DINOv2 variants for free tier CPU hosting. The model name is a single constant in embeddings.py and can be swapped if GPU hosting becomes available.

Groq llama-3.3-70b-versatile was chosen for cost and speed on the free tier. The LangChain tool calling pattern is provider agnostic. Switching to OpenAI or Anthropic is a one line change in agent.py.

The 0.65 / 0.35 blend (DINO vs colour) was chosen by eye across several query images. It is the first parameter to revisit if results feel too driven by texture or too driven by colour.

Result images are served from the retailer's own hosted URLs. This keeps the repo small and removes any need for the app to manage image storage, at the cost of depending on that external hosting.

Uploaded query images exist only for the session and are used only as search input. They are never added to the searchable catalogue.
