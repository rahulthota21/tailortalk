import os
import uuid
import requests
import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

# Push secrets (Streamlit Cloud / HF Spaces "Repo secrets") into env vars
# so langchain_openai.ChatOpenAI can pick them up automatically.
for key in ("GROQ_API_KEY", "GROQ_MODEL"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from core.agent import build_agent  # noqa: E402  (import after env setup)

st.set_page_config(page_title="TailorTalk - Saree Similarity Search", page_icon="🧵")
st.title("🧵 TailorTalk")
st.caption("Chat naturally, share a saree photo, and I'll find visually similar ones from the catalogue.")

if "image_registry" not in st.session_state:
    st.session_state.image_registry = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_image_from_upload(file) -> str:
    ref = f"img_{uuid.uuid4().hex[:8]}"
    path = os.path.join(UPLOAD_DIR, f"{ref}.png")
    Image.open(file).convert("RGB").save(path)
    st.session_state.image_registry[ref] = path
    return ref


def save_image_from_url(url: str) -> str:
    ref = f"img_{uuid.uuid4().hex[:8]}"
    path = os.path.join(UPLOAD_DIR, f"{ref}.png")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    Image.open(path).convert("RGB").save(path)
    st.session_state.image_registry[ref] = path
    return ref


with st.sidebar:
    st.subheader("Share an image")
    uploaded = st.file_uploader("Upload a saree photo", type=["png", "jpg", "jpeg", "webp"])
    url_input = st.text_input("...or paste an image URL")

    if uploaded is not None and st.session_state.get("last_upload_name") != uploaded.name:
        ref = save_image_from_upload(uploaded)
        st.session_state.last_upload_name = uploaded.name
        st.session_state.pending_ref = ref
        st.image(uploaded, caption=f"Ready to search ({ref})", width=200)

    if url_input and st.session_state.get("last_url") != url_input:
        try:
            ref = save_image_from_url(url_input)
            st.session_state.last_url = url_input
            st.session_state.pending_ref = ref
            st.image(st.session_state.image_registry[ref], caption=f"Ready to search ({ref})", width=200)
        except Exception as e:
            st.error(f"Could not load image from that URL: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            cols = st.columns(len(msg["images"]))
            for c, res in zip(cols, msg["images"]):
                with c:
                    st.image(res["path"], use_column_width=True)
                    st.caption(f"score: {res['score']}")

user_input = st.chat_input("Ask me to find similar sarees, or just chat...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    pending_ref = st.session_state.pop("pending_ref", None)
    agent_input = user_input
    if pending_ref:
        agent_input += f"\n\n[system note: user just shared an image, image_ref={pending_ref}]"

    executor = build_agent(st.session_state.image_registry)
    with st.spinner("Thinking..."):
        result = executor.invoke({
            "input": agent_input,
            "chat_history": st.session_state.chat_history,
        })

    reply = result["output"]
    images_found = []
    for action, output in result.get("intermediate_steps", []):
        if action.tool == "search_similar_sarees" and isinstance(output, list):
            images_found = [r for r in output if "error" not in r]

    with st.chat_message("assistant"):
        st.markdown(reply)
        if images_found:
            cols = st.columns(len(images_found))
            for c, res in zip(cols, images_found):
                with c:
                    st.image(res["url"] or res["path"], use_column_width=True)
                    st.caption(f"score: {res['score']}")

    st.session_state.messages.append({"role": "assistant", "content": reply, "images": images_found})
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=reply))
