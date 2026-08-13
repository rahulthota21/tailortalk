# TailorTalk

A chat agent that finds visually similar sarees from a catalogue. Upload a photo or paste a link, and it searches a vector index of real saree products to return the closest matches with similarity scores.

Every image in the catalogue belongs to the same category, a saree, so the real differences are fine grained: fabric, weave, print, border, and colour combination. TailorTalk is built to notice those details instead of returning generic matches.

**Live app:** https://tailortalking.streamlit.app/
**Repo:** https://github.com/rahulthota21/tailortalk

## Description

You chat with TailorTalk the way you would talk to a shop assistant. Share a saree photo, ask it to find similar ones, and it replies with a short message plus a row of matching sarees and their scores. If you just want to chat, it responds normally. It only searches when you are actually asking for one.

The catalogue is embedded using DINOv2, a vision model trained purely on images with no text supervision, which makes it more sensitive to texture and fine visual detail than a general model like CLIP. Search happens in two stages: a fast nearest neighbour lookup over DINOv2 embeddings, followed by a re rank of the shortlist using a colour histogram comparison, since colour is a strong cue that DINOv2 alone tends to underweight.

## Features

* Natural chat interface, built with Streamlit
* Accepts an image by upload or by URL
* Detects on its own when a similarity search is being requested
* Two stage search: DINOv2 embeddings followed by colour aware re ranking
* Returns ranked matches with similarity scores
* Vector search exposed as a single schema defined tool for the agent

## Tech Stack

* Vector database: ChromaDB
* Embeddings: DINOv2 (facebook/dinov2 small) and HSV colour histogram
* Agent framework: LangChain
* LLM: Groq (llama 3.3 70b versatile)
* Frontend: Streamlit
* Deployment: Streamlit Community Cloud

## Requirements

* Python 3.10 or later
* A Groq API key (free at console.groq.com)

## Installation

```
git clone https://github.com/rahulthota21/tailortalk
cd tailortalk
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Build the vector index once, before first run.

```
python scripts/download_from_csv.py
python scripts/build_index.py
```

Add your API key.

```
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Open `.streamlit\secrets.toml` and set `GROQ_API_KEY` to your key.

## Usage

Run the app.

```
streamlit run app.py
```

Open the app in your browser, upload a saree photo or paste an image URL in the sidebar, then type a message such as:

```
find similar sarees
```

The assistant replies with a short message and a row of matching sarees, each with a similarity score.

## Project Structure

```
app.py                          Streamlit chat interface
core/embeddings.py              DINOv2 embedding and colour histogram
core/vector_store.py            Chroma wrapper and two stage search
core/search_tool.py             LangChain tool schema
core/agent.py                   Agent setup and system prompt
scripts/download_from_csv.py    downloads catalogue images
scripts/build_index.py          builds the vector index
chroma_db/                      prebuilt vector index, committed to the repo
```

## Dataset

The catalogue is a set of about 650 unique saree product images, sourced from a CSV of roughly 1,070 products with public image links. `download_from_csv.py` downloads the images, and `build_index.py` embeds them into the vector index.

## Deployment

The app is deployed on Streamlit Community Cloud, connected directly to this repository, with `GROQ_API_KEY` set as a secret. The vector index is prebuilt and committed, and result images are served from their original catalogue URLs, so the app needs no build step and no local files to run on first load.

## Notes

The vector index is built once locally and committed, rather than rebuilt on every server start, since embedding several hundred images takes a few minutes on CPU. If the catalogue changes, rerun `build_index.py` and recommit `chroma_db`.

The balance between texture and colour in the re ranking step was tuned by testing across several query images, and is the first setting worth adjusting if results feel too texture driven or too colour driven. It lives in `core/vector_store.py`.

## License

MIT
