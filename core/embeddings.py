"""
Embedding utilities for TailorTalk.

WHY DINOv2 INSTEAD OF CLIP
---------------------------
CLIP is trained to align images with text captions, so its embedding space
is optimized for semantic / category-level matching ("this is a saree").
On a dataset where EVERY image is a saree, that semantic signal is useless
- CLIP will happily tell you two totally different sarees are "similar"
because they're both, well, sarees. That's the "loose, generic results"
the assignment warns about.

DINOv2 is trained purely on images with a self-distillation objective (no
text supervision at all). It's known to produce embeddings that are far
more sensitive to texture, weave, print structure and fine visual detail -
exactly the kind of fine-grained difference we need here.

WHY WE ALSO USE A COLOUR HISTOGRAM
------------------------------------
DINOv2 (like most modern SSL/contrastive vision backbones) is trained with
colour-jitter augmentation, so it is deliberately somewhat colour-invariant.
That's normally desirable but it's a problem here, because colour
combination is one of the strongest cues a shopper uses to judge "similar"
sarees. Rather than bake colour into the indexed vector (which would make
DINO's texture signal harder to tune), we compute a separate HSV colour
histogram and use it purely as a second-stage re-ranking signal - see
vector_store.py.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import cv2

DINO_MODEL_NAME = "facebook/dinov2-small"  # 384-dim, CPU-friendly
_device = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        _model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(_device).eval()
    return _processor, _model


def _dino_embed_single(img: Image.Image) -> np.ndarray:
    processor, model = _load_model()
    inputs = processor(images=img, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # CLS token
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.astype(np.float32)


def get_dino_embedding(img: Image.Image, tta: bool = True) -> np.ndarray:
    """
    L2-normalised DINOv2 embedding for an image.

    tta=True averages the embedding of the full image with a centre crop
    (80% of the shorter side) - cheap test-time augmentation that makes the
    embedding more robust to backgrounds/margins/model-vs-flat-lay shots
    that vary across an e-commerce catalogue.
    """
    img = img.convert("RGB")
    vecs = [_dino_embed_single(img)]
    if tta:
        w, h = img.size
        side = int(min(w, h) * 0.8)
        left, top = (w - side) // 2, (h - side) // 2
        crop = img.crop((left, top, left + side, top + side))
        vecs.append(_dino_embed_single(crop))
    vec = np.mean(vecs, axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.astype(np.float32)


def get_color_histogram(img: Image.Image, bins=(8, 8, 8)) -> np.ndarray:
    """Normalised 3D HSV colour histogram, flattened. Used only for re-ranking."""
    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([img_cv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    score = cv2.compareHist(h1.reshape(-1, 1), h2.reshape(-1, 1), cv2.HISTCMP_CORREL)
    return float(max(0.0, score))
