"""
Exposes the vector search as a callable, schema'd tool for the agent.

DESIGN NOTE - why the tool takes an `image_ref`, not raw image bytes
----------------------------------------------------------------------
LLM function-calling arguments are plain JSON/text. Passing a full image
(base64) as a tool argument is wasteful and unreliable - the model would
have to faithfully reproduce potentially megabytes of base64 in its tool
call. Instead, the Streamlit app saves any uploaded/linked image to disk
the moment the user shares it and hands the LLM a short opaque id
(e.g. "img_3f9a1c2b"). The model's job is purely to (a) notice the user
wants a similarity search and (b) call the tool with that id and a top_k -
i.e. exactly the "understand when a similarity search is being asked for"
requirement from the brief. The actual pixels never touch the LLM.
"""
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from PIL import Image

from core.vector_store import get_collection, search as vector_search


class SareeSearchInput(BaseModel):
    image_ref: str = Field(
        description=(
            "The internal reference id of the image to search with, e.g. "
            "'img_3f9a1c2b'. This id is given to you in the conversation "
            "whenever the user uploads or links an image - use that exact "
            "id, never a description of the image."
        )
    )
    top_k: int = Field(default=5, description="How many similar sarees to return (1-20).")


def make_search_tool(image_registry: dict):
    """
    image_registry maps image_ref -> local file path for this session.
    The Streamlit app owns and populates this dict.
    """
    collection = get_collection()

    def _run(image_ref: str, top_k: int = 5):
        if image_ref not in image_registry:
            return [{"error": f"Unknown image_ref '{image_ref}'. Ask the user to upload or link an image first."}]
        top_k = max(1, min(top_k, 20))
        img = Image.open(image_registry[image_ref])
        return vector_search(collection, img, top_k=top_k)

    return StructuredTool.from_function(
        func=_run,
        name="search_similar_sarees",
        description=(
            "Search the saree catalogue for images visually similar to an "
            "image the user has uploaded or linked in this conversation. "
            "Call this ONLY when the user is asking for similar/matching/"
            "comparable items to a specific shared image. Returns a ranked "
            "list of matches, each with a similarity score."
        ),
        args_schema=SareeSearchInput,
    )
