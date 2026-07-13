#!/usr/bin/env python3
"""
- Lightweight Manga Translation API
- NO LOCAL AI MODELS (No ONNX, No LaMa, No YOLO, No Qwen)
- OCR: Google Lens Cloud API ONLY
- Translation: OpenRouter Cloud API ONLY
- NPU: Detects and initializes NPU/XPU hardware via PyTorch
- API: /health /version /meta /warmup /console
       /v1/translate /v1/translate/{id} /v1/translate/{id}/image
       /SetFont /GetFont /GetFonts /SetModelType /GetModelType /SetOpenRouterModel
"""

import asyncio
import base64
import bisect
import io
import os
import pathlib
import time
import traceback
import urllib.request
import uuid
import logging
import threading
import functools
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# --- FastAPI ---------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query, Request, Form
from fastapi.responses import JSONResponse, Response, HTMLResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- GPU / NPU / Device helpers --------------------------------------------
import torch

# Attempt to import Huawei Ascend NPU support
try:
    import torch_npu
    _HAS_NPU = torch.npu.is_available()
except ImportError:
    _HAS_NPU = False

# Attempt to import Intel NPU/XPU support
try:
    import intel_extension_for_pytorch as ipex
    _HAS_XPU = torch.xpu.is_available()
except ImportError:
    _HAS_XPU = False

def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def has_npu() -> bool:
    return _HAS_NPU

def has_xpu() -> bool:
    return _HAS_XPU

def get_torch_device() -> str:
    if has_cuda():
        return "cuda"
    if has_npu():
        return "npu"
    if has_xpu():
        return "xpu"
    return "cpu"

logging.info(f"[Device] CUDA: {has_cuda()}, NPU: {has_npu()}, XPU: {has_xpu()} -> device='{get_torch_device()}'")

# --- Optional deps ---------------------------------------------------------
try:
    from chrome_lens_py import LensAPI
except Exception:
    LensAPI = None

# --- Sanitization ----
import re

_ALLOWED_RANGES = (
    (0x0020, 0x007E), (0x00A0, 0x00FF), (0x0100, 0x017F), (0x0180, 0x024F),
    (0x0400, 0x04FF), (0x0500, 0x052F), (0x2000, 0x206F), (0x3000, 0x303F),
    (0x3040, 0x309F), (0x30A0, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF), (0xFF00, 0xFFEF),
)

_ALLOWED_LOWS  = tuple(r[0] for r in _ALLOWED_RANGES)
_ALLOWED_HIGHS = tuple(r[1] for r in _ALLOWED_RANGES)

_PUNCT_MAP = {
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2013: '-', 0x2014: '-',
    0x2026: '...', 0x00A0: ' ', 0x2022: '*', 0x2122: '(TM)', 0x00A9: '(c)', 0x00AE: '(R)',
}

def _is_allowed_cp(cp: int) -> bool:
    idx = bisect.bisect_right(_ALLOWED_LOWS, cp) - 1
    return idx >= 0 and cp <= _ALLOWED_HIGHS[idx]

def clean_text_for_font(text: str) -> str:
    if not text: return ""
    if not hasattr(clean_text_for_font, '_trans_table'):
        clean_text_for_font._punct_table = str.maketrans({chr(cp): rep for cp, rep in _PUNCT_MAP.items()})
        clean_text_for_font._re_space = re.compile(r'[ \t]+')
        clean_text_for_font._re_nl   = re.compile(r'\n+')
    out = text.translate(clean_text_for_font._punct_table)
    out = ''.join(ch for ch in out if (ch in '\t\n') or (0x20 <= ord(ch) and _is_allowed_cp(ord(ch))))
    out = clean_text_for_font._re_space.sub(' ', out)
    out = clean_text_for_font._re_nl.sub(' ', out)
    return out.strip()

# --- Config ----------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FONT_DIR = ROOT_DIR / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = FONT_DIR / "NotoCJK.ttc"
FONT_URL = "https://github.com/Kirogii/MangaAMTL/releases/download/Packages/NotoCJK.ttc"

if not FONT_PATH.exists():
    try:
        logging.info(f"Downloading font from {FONT_URL}")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    except Exception as e:
        logging.warning(f"Failed to download font: {e}. Falling back.")
        FONT_PATH = pathlib.Path("NotoCJK.ttf")

DEFAULT_LANG       = "en"
BUILD_ID           = "manga-cloud-npu-v1"

# --- Logging / Console -----------------------------------------------------
class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 2000):
        super().__init__()
        self.logs = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self.logs.append(self.format(record))

    def get_logs(self) -> List[str]:
        return list(self.logs)

log_handler = MemoryLogHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_handler)
logging.getLogger("uvicorn").addHandler(log_handler)
logging.getLogger("uvicorn.access").addHandler(log_handler)

# --- Globals ---------------------------------------------------------------
app = FastAPI(title="Manga Cloud Translation API", version=BUILD_ID)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*", allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"], expose_headers=["*"],
)

# --- Model Type Configuration Globals ---
_current_model_type: str = "openrouter" # Hardcoded to openrouter
_openrouter_api_key: Optional[str] = None
_openrouter_model: str = "openai/gpt-4o-mini"
_model_type_lock = threading.Lock()

_lens_api = None
_lens_lock = threading.Lock()

_current_font_path: pathlib.Path = FONT_PATH
_current_stroke_width: int = 0
_font_config_lock = threading.Lock()

_jobs: Dict[str, Dict[str, Any]] = {}
_job_lock = asyncio.Lock()

_inpaint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inpaint")

# ===========================================================================
# Image utils
# ===========================================================================
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

# ===========================================================================
# Google Lens OCR (cloud)
# ===========================================================================
def get_lens_api():
    global _lens_api
    if LensAPI is None:
        raise RuntimeError("chrome-lens-py not installed: pip install chrome-lens-py")
    if _lens_api is None:
        with _lens_lock:
            if _lens_api is None:
                _lens_api = LensAPI()
                logging.info("[Google Lens] LensAPI initialized.")
    return _lens_api

def _geometry_to_bbox(geometry, img_w, img_h):
    if not geometry: return None
    if isinstance(geometry, dict):
        try:
            cx, cy, bw, bh = geometry.get("center_x"), geometry.get("center_y"), geometry.get("width"), geometry.get("height")
            if None in (cx, cy, bw, bh): return None
            x1, y1 = max(0, int((cx - bw / 2) * img_w)), max(0, int((cy - bh / 2) * img_h))
            x2, y2 = min(img_w - 1, int((cx + bw / 2) * img_w)), min(img_h - 1, int((cy + bh / 2) * img_h))
            return (x1, y1, x2, y2) if (x2 - x1 >= 5 and y2 - y1 >= 5) else None
        except: return None
    if isinstance(geometry, list) and len(geometry) >= 2:
        try:
            xs, ys = [p[0] for p in geometry], [p[1] for p in geometry]
            x1, y1 = max(0, int(min(xs) * img_w)), max(0, int(min(ys) * img_h))
            x2, y2 = min(img_w - 1, int(max(xs) * img_w)), min(img_h - 1, int(max(ys) * img_h))
            return (x1, y1, x2, y2) if (x2 - x1 >= 5 and y2 - y1 >= 5) else None
        except: return None
    return None

def _merge_close_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(blocks) <= 1:
        for b in blocks:
            if "bboxes" not in b: b["bboxes"] = [b["bbox"]]
        return blocks

    parent = list(range(len(blocks)))
    def find(i):
        root = i
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj

    for i in range(len(blocks)):
        x1_i, y1_i, x2_i, y2_i = blocks[i]["bbox"]
        w_i, h_i = max(1, x2_i - x1_i), max(1, y2_i - y1_i)
        exp_x1_i, exp_x2_i = x1_i - w_i * 0.5, x2_i + w_i * 0.5

        for j in range(i + 1, len(blocks)):
            x1_j, y1_j, x2_j, y2_j = blocks[j]["bbox"]
            w_j, h_j = max(1, x2_j - x1_j), max(1, y2_j - y1_j)
            exp_x1_j, exp_x2_j = x1_j - w_j * 0.5, x2_j + w_j * 0.5

            if (min(exp_x2_i, exp_x2_j) - max(exp_x1_i, exp_x1_j)) <= 0: continue
            if (min(y2_i, y2_j) - max(y1_i, y1_j)) < 0.3 * min(h_i, h_j): continue
            union(i, j)

    groups = {}
    for i in range(len(blocks)): groups.setdefault(find(i), []).append(blocks[i])

    merged_blocks = []
    for group in groups.values():
        x1, y1 = min(b["bbox"][0] for b in group), min(b["bbox"][1] for b in group)
        x2, y2 = max(b["bbox"][2] for b in group), max(b["bbox"][3] for b in group)
        group.sort(key=lambda b: (b["bbox"][0] * -1, b["bbox"][1]))
        merged_blocks.append({
            "text": " ".join(b["text"] for b in group),
            "bbox": (x1, y1, x2, y2),
            "bboxes": [b["bbox"] for b in group]
        })
    return merged_blocks

async def google_lens_ocr(pil_img: Image.Image, ocr_lang: str = "ja") -> List[Dict[str, Any]]:
    api = get_lens_api()
    w, h = pil_img.size
    logging.info(f"[Google Lens] Running OCR on {w}x{h} image (lang={ocr_lang})...")
    lens_lang_map = {"ja": "ja", "ko": "ko", "en": "en", "zh": "zh", "ru": "ru", "es": "es", "id": "id", "cz": "zh"}
    lens_lang = lens_lang_map.get(ocr_lang, ocr_lang)
    try:
        result = await api.process_image(image_path=pil_img, ocr_language=lens_lang, output_format='blocks')
    except Exception as e:
        logging.error(f"[Google Lens] OCR failed: {e}")
        return []
    if not isinstance(result, dict): return []
    
    out = []
    for block in result.get("text_blocks", []):
        if not isinstance(block, dict): continue
        text = (block.get("text") or "").strip()
        if not text: continue
        bbox = _geometry_to_bbox(block.get("geometry", []), w, h)
        if bbox is None: continue
        out.append({"text": text, "bbox": bbox})
    
    merged = _merge_close_blocks(out)
    logging.info(f"[Google Lens] Found {len(out)} raw blocks -> merged to {len(merged)} blocks.")
    return merged

async def get_ocr_results(pil_img: Image.Image, ocr_lang: str = "ja") -> List[Dict[str, Any]]:
    return await google_lens_ocr(pil_img, ocr_lang)

# ===========================================================================
# OpenRouter Translation (Cloud)
# ===========================================================================
LANG_MAP = {"en": "English", "ja": "Japanese", "ko": "Korean", "id": "Indonesian", "ru": "Russian", "es": "Spanish", "zh": "Chinese", "cz": "Chinese"}
SRC_LANG_MAP = {"ja": "Japanese", "ko": "Korean", "en": "English", "zh": "Chinese", "ru": "Russian", "es": "Spanish", "id": "Indonesian", "cz": "Chinese"}

def _script_hint(lang_name: str) -> str:
    if lang_name in ("Japanese", "Korean", "Chinese"):
        return (f"Write the translation using the native {lang_name} writing system. Do NOT romanize.")
    return ""

SYSTEM_PROMPT = (
    "You are a professional manga translator. "
    "Translate the user's text from its original language into {lang}. "
    "Output ONLY the {lang} translation. {script_hint}"
)

def _looks_like_target(trans: str, target_lang: str) -> bool:
    if not trans: return False
    lang_code = "zh" if target_lang == "cz" else target_lang
    cjk = sum(1 for c in trans if 0x3000 <= ord(c) <= 0x9FFF or 0xAC00 <= ord(c) <= 0xD7AF or 0xFF00 <= ord(c) <= 0xFFEF)
    if lang_code in ("en", "es", "id", "ru") and cjk > len(trans) * 0.3: return False
    if lang_code in ("ja", "ko", "zh") and cjk == 0: return False
    return True

async def openrouter_translate_batch(texts: List[str], target_lang: str = "en", ocr_lang: str = "ja", max_retries: int = 2) -> List[str]:
    import aiohttp, random

    with _model_type_lock:
        api_key = _openrouter_api_key
        model = _openrouter_model
    if not api_key: return [""] * len(texts)

    indexed_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not indexed_texts: return [""] * len(texts)

    lang_name = LANG_MAP.get(target_lang, "English")
    src_lang_name = SRC_LANG_MAP.get(ocr_lang, "the original language")
    max_tok = max(256, min(4096, sum(len(t) for _, t in indexed_texts) + (len(indexed_texts) * 20)))

    batch_text = f"[Source language: {src_lang_name}]\n" + "\n".join(f"{idx + 1}. {text.replace(chr(10), ' ')}" for idx, (_, text) in enumerate(indexed_texts))
    sys_prompt = (f"You are a professional manga translator. Translate each numbered line from {src_lang_name} into {lang_name}. "
                  f"CRITICAL: Actually translate, do NOT copy. Output ONLY the translated list. {_script_hint(lang_name)}").strip()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "http://localhost:8000", "X-Title": "Manga API"}
    payload = {"model": model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": batch_text}], "max_tokens": max_tok, "temperature": 0.2, "top_p": 0.9}
    QUOTE_CHARS = "\"'“”‘’"

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
            payload["messages"][0]["content"] = (f"YOUR PREVIOUS RESPONSE WAS INVALID. You MUST translate into {lang_name}. "
                                                 f"Do NOT repeat original text. {_script_hint(lang_name)}").strip()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status == 429:
                        await asyncio.sleep(float(response.headers.get("Retry-After", 10.0))); continue
                    if response.status != 200: continue

                    data = await response.json()
                    raw = data.get("choices", [{}])[0].get("message", {}).get("content")
                    if not raw or not isinstance(raw, str): continue

                    raw = raw.strip()
                    if raw.startswith("```"):
                        lines = raw.split('\n')
                        if lines[0].startswith("```"): lines = lines[1:]
                        if lines and lines[-1].startswith("```"): lines = lines[:-1]
                        raw = '\n'.join(lines).strip()

                    results = [""] * len(texts)
                    parsed_lines = [ln.strip() for ln in raw.split('\n') if ln.strip()]
                    matched_count = 0

                    for line in parsed_lines:
                        match = re.match(r"^\s*[\[\(]?(\d+)[\]\)]?[\.\)\-:]\s*(.*)$", line)
                        if match:
                            num = int(match.group(1)) - 1
                            trans = match.group(2).strip()
                            if len(trans) >= 2 and trans[0] in QUOTE_CHARS and trans[-1] in QUOTE_CHARS: trans = trans[1:-1].strip()
                            if 0 <= num < len(indexed_texts):
                                results[indexed_texts[num][0]] = clean_text_for_font(trans)
                                matched_count += 1

                    if matched_count == 0 and len(parsed_lines) == len(indexed_texts):
                        for i, line in enumerate(parsed_lines):
                            trans = line.strip()
                            if len(trans) >= 2 and trans[0] in QUOTE_CHARS and trans[-1] in QUOTE_CHARS: trans = trans[1:-1].strip()
                            results[indexed_texts[i][0]] = clean_text_for_font(trans)
                            matched_count += 1

                    valid_count = sum(1 for r in results if r and _looks_like_target(r, target_lang))
                    if valid_count > 0: return results
                    else: continue
        except: continue

    return [""] * len(texts)

async def openrouter_translate(text: str, target_lang: str = "en", ocr_lang: str = "ja", max_retries: int = 5) -> str:
    import aiohttp, random

    with _model_type_lock:
        api_key = _openrouter_api_key
        model = _openrouter_model
    if not api_key or not text.strip(): return ""

    lang_name = LANG_MAP.get(target_lang, "English")
    src_lang_name = SRC_LANG_MAP.get(ocr_lang, "the original language")
    sys_prompt = SYSTEM_PROMPT.format(lang=lang_name, script_hint=_script_hint(lang_name))
    user_prompt = f"[Source language: {src_lang_name}]\n{text}"
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "http://localhost:8000", "X-Title": "Manga API"}
    payload = {"model": model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], "max_tokens": max(16, min(96, len(text) + 16)), "temperature": 0.2, "top_p": 0.9}

    for attempt in range(1, max_retries + 1):
        if attempt > 1: await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status == 429:
                        await asyncio.sleep(float(response.headers.get("Retry-After", 10.0))); continue
                    if response.status != 200: continue
                    data = await response.json()
                    raw = data.get("choices", [{}])[0].get("message", {}).get("content")
                    if not raw or not isinstance(raw, str): continue
                    return clean_text_for_font(raw)
        except: continue
    return ""

# ===========================================================================
# CV2 Inpainting Fallback (Strictly CPU/OpenCV, no AI models)
# ===========================================================================
INPAINT_RADIUS_CV2 = 7

async def inpaint_image_async(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_inpaint_executor, cv2_inpaint_fallback, img_bgr, mask)
    return result

def cv2_inpaint_fallback(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(img_bgr, mask, INPAINT_RADIUS_CV2, cv2.INPAINT_TELEA)

def build_inpaint_mask(img_shape: Tuple[int, int, int], bboxes: List[Tuple[int, int, int, int]], padding: int = 2, dilate_kernel: int = 3) -> np.ndarray:
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[max(0, y1 - padding):min(h, y2 + padding), max(0, x1 - padding):min(w, x2 + padding)] = 255
    if dilate_kernel > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)), iterations=1)
    return mask

# ===========================================================================
# Text color detection
# ===========================================================================
def detect_text_and_bg_colors(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    pad_x, pad_y = max(3, (x2 - x1) // 6), max(3, (y2 - y1) // 6)
    ex_x1, ex_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ex_x2, ex_y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    region = img_bgr[ex_y1:ex_y2, ex_x1:ex_x2]
    if region.size == 0: return (255, 255, 255), (0, 0, 0)

    rh, rw = region.shape[:2]
    if rh > 180 or rw > 180:
        scale = 180 / max(rh, rw)
        region = cv2.resize(region, (max(8, int(rw * scale)), max(8, int(rh * scale))), interpolation=cv2.INTER_AREA)

    region_lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    pixels_lab = np.ascontiguousarray(region_lab.reshape(-1, 3).astype(np.float32))
    pixels_bgr = region.reshape(-1, 3).astype(np.float32)
    n_pixels = int(pixels_bgr.shape[0])
    if n_pixels < 8: return (255, 255, 255), (0, 0, 0)

    K = 3
    try:
        _, labels, centers_lab = cv2.kmeans(pixels_lab, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0), 10, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return ((0, 0, 0), (255, 255, 255)) if float(gray.mean()) > 127 else ((255, 255, 255), (0, 0, 0))

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=K)
    sorted_idx = np.argsort(-counts)

    bg_idx = int(sorted_idx[0])
    bg_lab = centers_lab[bg_idx]
    bg_mask = (labels == bg_idx)
    bg_bgr = np.median(pixels_bgr[bg_mask], axis=0)

    min_text_count = max(5, int(n_pixels * 0.04))
    best_text_idx, best_text_dist = None, -1.0
    for i in range(K):
        if i == bg_idx or counts[i] < min_text_count: continue
        d = float(np.linalg.norm(centers_lab[i] - bg_lab))
        if d > best_text_dist: best_text_dist, best_text_idx = d, i

    if best_text_idx is not None:
        text_bgr = np.median(pixels_bgr[(labels == best_text_idx)], axis=0)
    else:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]) if (rh >= 2 and rw >= 2) else gray.flatten()
        text_sel = (otsu.flatten() == 0) if float(border.mean() if border.size else float(gray.mean())) > 127 else (otsu.flatten() != 0)
        text_bgr = np.median(pixels_bgr[text_sel], axis=0) if int(text_sel.sum()) > 0 else (np.array([0, 0, 0]) if float(bg_bgr.mean()) > 127 else np.array([255, 255, 255]))

    def gentle_snap(c): 
        c = np.asarray(c, dtype=np.float32)
        return np.array([0,0,0]) if np.all(c <= 20) else (np.array([255,255,255]) if np.all(c >= 235) else c)
    
    text_bgr, bg_bgr = gentle_snap(text_bgr), gentle_snap(bg_bgr)
    if float(np.linalg.norm(text_bgr - bg_bgr)) < 60:
        text_bgr = np.array([0,0,0]) if float(bg_bgr.mean()) > 127 else np.array([255,255,255])

    return (int(text_bgr[2]), int(text_bgr[1]), int(text_bgr[0])), (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))

def detect_text_colors_batch(img_bgr: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    results = [detect_text_and_bg_colors(img_bgr, bbox) for bbox in bboxes]
    light_votes = sum(1 for t, b in results if (t[0]+t[1]+t[2])/3.0 > (b[0]+b[1]+b[2])/3.0)
    dark_votes = len(results) - light_votes
    if light_votes + dark_votes == 0: return results

    force_light = light_votes >= 2 * dark_votes and light_votes >= 2
    force_dark = dark_votes >= 2 * light_votes and dark_votes >= 2
    if not force_light and not force_dark: return results

    final_results = []
    for text_rgb, bg_rgb in results:
        if force_light:
            if (text_rgb[0]+text_rgb[1]+text_rgb[2])/3.0 <= (bg_rgb[0]+bg_rgb[1]+bg_rgb[2])/3.0:
                final_results.append(((255, 255, 255), bg_rgb if (bg_rgb[0]+bg_rgb[1]+bg_rgb[2])/3.0 < 90 else (0, 0, 0)))
            else: final_results.append((text_rgb, bg_rgb))
        else:
            if (text_rgb[0]+text_rgb[1]+text_rgb[2])/3.0 >= (bg_rgb[0]+bg_rgb[1]+bg_rgb[2])/3.0:
                final_results.append(((0, 0, 0), bg_rgb if (bg_rgb[0]+bg_rgb[1]+bg_rgb[2])/3.0 > 165 else (255, 255, 255)))
            else: final_results.append((text_rgb, bg_rgb))
    return final_results

# ===========================================================================
# Text wrapping & rendering
# ===========================================================================
@functools.lru_cache(maxsize=256)
def _get_font_cached(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try: return ImageFont.truetype(font_path, size)
    except: return ImageFont.load_default()

def clear_font_cache() -> None: _get_font_cached.cache_clear()
def get_font(font_path, size: int) -> ImageFont.FreeTypeFont: return _get_font_cached(str(font_path), size)
def get_current_font(size: int) -> ImageFont.FreeTypeFont:
    with _font_config_lock: return get_font(_current_font_path, size)

def wrap_text(draw, text, font, max_width, allow_break=False, is_vertical=False):
    if is_vertical: return [text] if text else [""]
    words = text.split()
    if not words: return [""]
    lines, cur = [], ""
    for word in words:
        test = (cur + " " + word) if cur else word
        if draw.textlength(test, font=font) <= max_width: cur = test
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    return lines

def _measure_block(draw, lines, font):
    try: ascent, descent = font.getmetrics(); line_h = int(ascent + descent)
    except: line_h = int(font.size * 1.2)
    line_h = max(line_h, int(font.size * 1.1))
    return [line_h] * len(lines), line_h * len(lines), max([draw.textlength(ln, font=font) for ln in lines] + [0.0])

def fit_font_and_wrap(draw, text, box_w, box_h, font_path=None, max_size=96, min_size=8, is_vertical=False):
    if font_path is None:
        with _font_config_lock: font_path = str(_current_font_path)
    if not text.strip(): return min_size, [""], [0]
    if not hasattr(fit_font_and_wrap, '_cache'): fit_font_and_wrap._cache = {}
    cache = fit_font_and_wrap._cache

    lo, hi, best_size, best_lines, best_heights = min_size, max_size, None, None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        key = (font_path, mid)
        if key not in cache:
            try: cache[key] = ImageFont.truetype(font_path, mid)
            except: cache[key] = ImageFont.load_default()
        font = cache[key]
        lines = wrap_text(draw, text, font, box_w - 4, allow_break=False, is_vertical=False)
        heights, total_h, max_w = _measure_block(draw, lines, font)
        if max_w <= box_w - 4 and total_h <= box_h - 4:
            best_size, best_lines, best_heights = mid, lines, heights; lo = mid + 1
        else: hi = mid - 1
        
    if best_lines is None:
        key = (font_path, min_size)
        if key not in cache:
            try: cache[key] = ImageFont.truetype(font_path, min_size)
            except: cache[key] = ImageFont.load_default()
        font = cache[key]
        best_lines = wrap_text(draw, text, font, box_w - 4, allow_break=True, is_vertical=False) or [text]
        heights, _, _ = _measure_block(draw, best_lines, font)
        best_size, best_heights = min_size, heights
    return best_size, best_lines, best_heights

def draw_text_with_config(draw, position, text, font, fill, stroke_fill=None, anchor=None):
    with _font_config_lock: stroke_width = _current_stroke_width
    if stroke_width > 0 and stroke_fill is not None:
        draw.text(position, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill, anchor=anchor)
    else:
        draw.text(position, text, font=font, fill=fill, anchor=anchor)

# ===========================================================================
# Font Endpoints
# ===========================================================================
def list_available_fonts() -> List[Dict[str, Any]]:
    fonts = []
    if FONT_DIR.exists():
        for ext in ('*.ttf', '*.otf', '*.ttc'):
            for f in sorted(FONT_DIR.glob(ext)):
                fonts.append({"name": f.stem, "filename": f.name, "path": str(f), "size_kb": round(f.stat().st_size / 1024, 1)})
    return fonts

class SetFontRequest(BaseModel):
    font_path: Optional[str] = None
    font_url: Optional[str] = None
    font_name: Optional[str] = None
    stroke_width: int = 0

@app.post("/SetFont")
async def set_font(req: SetFontRequest):
    global _current_font_path, _current_stroke_width
    with _font_config_lock:
        if sum(1 for p in [req.font_path, req.font_url, req.font_name] if p) > 1:
            raise HTTPException(400, "Provide either font_path, font_url, or font_name, not multiple")
        if req.font_url:
            filename = pathlib.Path(req.font_url).name
            if not filename.lower().endswith(('.ttf', '.otf', '.ttc')): filename += '.ttf'
            new_path = FONT_DIR / filename
            urllib.request.urlretrieve(req.font_url, new_path)
            _current_font_path = new_path
            clear_font_cache()
        elif req.font_path:
            p = pathlib.Path(req.font_path).resolve()
            if not p.exists(): raise HTTPException(400, "Font file not found")
            _current_font_path = p
            clear_font_cache()
        elif req.font_name:
            matched = next((f for f in list_available_fonts() if f["filename"].lower() == req.font_name.lower() or f["name"].lower() == req.font_name.lower()), None)
            if not matched: raise HTTPException(404, "Font not found")
            _current_font_path = pathlib.Path(matched["path"])
            clear_font_cache()
        _current_stroke_width = max(0, min(20, req.stroke_width))
    return {"status": "ok", "font_path": str(_current_font_path), "stroke_width": _current_stroke_width}

@app.get("/GetFont")
async def get_font_config():
    with _font_config_lock: return {"font_path": str(_current_font_path), "stroke_width": _current_stroke_width}

@app.get("/GetFonts")
async def get_fonts():
    fonts = list_available_fonts()
    return {"fonts": fonts, "count": len(fonts)}

@app.get("/v1/font")
async def get_font_file():
    with _font_config_lock: path = pathlib.Path(_current_font_path)
    if not path.exists(): raise HTTPException(404, "Font file not found")
    return FileResponse(str(path), media_type={"ttf": "font/ttf", "otf": "font/otf", "ttc": "font/collection"}.get(path.suffix.lower(), "application/octet-stream"), filename=path.name)

# ===========================================================================
# Model Type Endpoints (OpenRouter only)
# ===========================================================================
class SetModelTypeRequest(BaseModel):
    model_type: str
    api_key: Optional[str] = None
    model: Optional[str] = None

@app.post("/SetModelType")
async def set_model_type(req: SetModelTypeRequest):
    global _current_model_type, _openrouter_api_key, _openrouter_model
    if req.model_type.lower().strip() != "openrouter":
        raise HTTPException(400, "This lightweight build only supports 'openrouter'.")
    with _model_type_lock:
        _current_model_type = "openrouter"
        if req.api_key: _openrouter_api_key = req.api_key
        if not _openrouter_api_key: raise HTTPException(400, "OpenRouter API key is required.")
        if req.model: _openrouter_model = req.model
    return {"status": "ok", "model_type": _current_model_type, "openrouter_model": _openrouter_model}

@app.get("/GetModelType")
async def get_model_type():
    with _model_type_lock:
        return {"model_type": _current_model_type, "openrouter_model": _openrouter_model, "api_key_set": _openrouter_api_key is not None}

class SetOpenRouterModelRequest(BaseModel):
    model: str
    api_key: Optional[str] = None

@app.post("/SetOpenRouterModel")
async def set_openrouter_model(req: SetOpenRouterModelRequest):
    global _openrouter_model, _openrouter_api_key
    with _model_type_lock:
        _openrouter_model = req.model.strip()
        if req.api_key: _openrouter_api_key = req.api_key
    return {"status": "ok", "openrouter_model": _openrouter_model, "api_key_set": _openrouter_api_key is not None}

# ===========================================================================
# Health / Warmup (Triggers NPU init)
# ===========================================================================
@app.get("/health")
async def health(): return {"status": "ok"}

@app.get("/version")
async def version(): return {"version": BUILD_ID}

@app.get("/meta")
async def meta():
    with _model_type_lock: mt, om = _current_model_type, _openrouter_model
    return {
        "version": BUILD_ID,
        "device": get_torch_device(),
        "npu_available": has_npu(),
        "xpu_available": has_xpu(),
        "lens_available": LensAPI is not None,
        "model_type": mt,
        "openrouter_model": om,
        "font_path": str(_current_font_path),
        "stroke_width": _current_stroke_width,
    }

@app.post("/warmup")
async def warmup():
    """Initializes NPU hardware and Cloud APIs."""
    errors = []
    device = get_torch_device()
    try:
        if device != "cpu":
            logging.info(f"[Warmup] Initializing {device} hardware...")
            # Create a dummy tensor to physically wake up the NPU/XPU/CUDA device
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            logging.info(f"[Warmup] {device} hardware test successful. Tensor sum: {z.sum().item()}")
        else:
            logging.info("[Warmup] No NPU/XPU/CUDA device found. Running on CPU.")
    except Exception as e:
        errors.append(f"NPU/XPU Init: {e}")

    try:
        if LensAPI is not None:
            get_lens_api()
        else:
            errors.append("chrome-lens-py not installed")
    except Exception as e:
        errors.append(f"Google Lens: {e}")

    return {"status": "warmed" if not errors else "partial", "device": device, "errors": errors}

# ===========================================================================
# Console / Logs
# ===========================================================================
@app.get("/console")
async def console():
    html = """<!DOCTYPE html><html><head><title>Console</title><style>body{background:#1a1a2e;color:#e0e0e0;font-family:monospace;padding:20px}.log-line{padding:2px 8px;border-bottom:1px solid #2a2a4a}.level-INFO{color:#a0d0ff}.level-WARNING{color:#ffd060}.level-ERROR{color:#ff6060}button{background:#2a4a8a;color:white;border:1px solid #4080c0;padding:8px 16px;cursor:pointer}</style></head><body><h1>Backend Console</h1><button onclick="fetchLogs()">Refresh</button><div id="logs"></div><script>async function fetchLogs(){const r=await fetch('/console/json');const logs=await r.json();document.getElementById('logs').innerHTML=logs.map(l=>`<div class="log-line level-${(l.match(/\\b(INFO|WARNING|ERROR)\\b/)||['','INFO'])[1]}">${l.replace(/</g,'&lt;')}</div>`).join('')}fetchLogs();</script></body></html>"""
    return HTMLResponse(content=html)

@app.get("/console/json")
async def console_json(): return JSONResponse(content=log_handler.get_logs())

# ===========================================================================
# Translation Job endpoints
# ===========================================================================
@app.post("/v1/translate")
async def create_translate_job(image: UploadFile = File(...), target_lang: str = Form(DEFAULT_LANG), ocr_lang: str = Form("ja"), inpaint: bool = Form(True)):
    job_id = str(uuid.uuid4())[:8]
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    async with _job_lock:
        _jobs[job_id] = {"id": job_id, "status": "pending", "image": pil_img, "target_lang": target_lang, "ocr_lang": ocr_lang, "inpaint": inpaint, "result": None, "error": None}
    asyncio.create_task(_process_job(job_id))
    return {"job_id": job_id, "status": "pending", "inpaint": inpaint}

async def _process_job(job_id: str):
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job: return
        job["status"] = "processing"

    try:
        pil_img = job["image"]
        ocr_results = await get_ocr_results(pil_img, job["ocr_lang"])

        if not ocr_results:
            async with _job_lock:
                job["status"] = "completed"
                job["result"] = {"boxes": [], "translations": []}
            return

        texts_to_translate = [item["text"] for item in ocr_results]
        logging.info(f"[Job {job_id}] Using OpenRouter BATCH strategy for {len(texts_to_translate)} boxes.")
        
        batch_results = await openrouter_translate_batch(texts_to_translate, job["target_lang"], job["ocr_lang"])
        
        translations = []
        for idx, text in enumerate(texts_to_translate):
            translated = batch_results[idx]
            if not translated and text.strip():
                logging.warning(f"[Job {job_id}] Box {idx+1} missed in batch, retrying individually...")
                translated = await openrouter_translate(text, job["target_lang"], job["ocr_lang"])
                await asyncio.sleep(1.0)
            
            ocr_bbox = ocr_results[idx]["bbox"]
            translations.append({
                "text": text,
                "translation": translated,
                "bbox": ocr_bbox,
                "bboxes": ocr_results[idx].get("bboxes", [ocr_bbox]),
            })

        async with _job_lock:
            job["status"] = "completed"
            job["result"] = {"boxes": ocr_results, "translations": translations}

    except Exception as e:
        logging.error(f"[Job {job_id}] Failed: {e}\n{traceback.format_exc()}")
        async with _job_lock:
            job["status"] = "failed"
            job["error"] = str(e)

@app.get("/v1/translate/{job_id}")
async def get_translate_job(job_id: str):
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job: raise HTTPException(404, "Job not found")
        result = {"id": job["id"], "status": job["status"], "target_lang": job["target_lang"], "ocr_lang": job["ocr_lang"], "inpaint": job.get("inpaint", True)}
        if job["status"] == "completed": result["result"] = job["result"]
        elif job["status"] == "failed": result["error"] = job["error"]
        return result

@app.post("/v1/translate/{job_id}/image")
async def get_translated_image(job_id: str):
    async with _job_lock:
        job = _jobs.get(job_id)
        if not job: raise HTTPException(404, "Job not found")
        if job["status"] != "completed": raise HTTPException(400, "Job not completed")
        pil_img = job["image"]
        translations = job["result"].get("translations", [])
        do_inpaint = job.get("inpaint", True)

    if not translations:
        buf = io.BytesIO(); pil_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    img_bgr = pil_to_cv2(pil_img)
    boxes_to_inpaint, items_to_draw = [], []

    for item in translations:
        if not item.get("translation", "").strip(): continue
        bbox = item.get("bbox")
        if not bbox: continue
        for bx in item.get("bboxes", [bbox]):
            if (bx[2] - bx[0]) >= 10 and (bx[3] - bx[1]) >= 10: boxes_to_inpaint.append(bx)
        if (bbox[2] - bbox[0]) >= 10 and (bbox[3] - bbox[1]) >= 10:
            items_to_draw.append({"translation": item["translation"], "bbox": bbox})

    if do_inpaint and boxes_to_inpaint:
        mask = build_inpaint_mask(img_bgr.shape, boxes_to_inpaint, padding=2, dilate_kernel=3)
        img_bgr = await inpaint_image_async(img_bgr, mask)

    orig_bgr = pil_to_cv2(pil_img)
    out_pil = cv2_to_pil(img_bgr)
    draw = ImageDraw.Draw(out_pil)

    with _font_config_lock: fp = str(_current_font_path)
    all_box_colors = detect_text_colors_batch(orig_bgr, [item["bbox"] for item in items_to_draw])

    HARD_MAX_SIZE, HARD_MIN_SIZE = 30, 15
    LENS_OVERFLOW_PX, LENS_SMALL_BOX_THRESHOLD = 1, 24
    is_lens = True # Always true in this pruned build

    for item_idx, item in enumerate(items_to_draw):
        text, bbox = item["translation"], item["bbox"]
        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        text_color, bg_color = all_box_colors[item_idx]

        really_small = (box_w < LENS_SMALL_BOX_THRESHOLD or box_h < LENS_SMALL_BOX_THRESHOLD)
        if really_small:
            font_size = HARD_MIN_SIZE
            font = get_font(fp, font_size)
            lines = wrap_text(draw, text, font, max_width=max(box_w, font_size * 3), allow_break=True, is_vertical=False) or [text]
            heights, _, _ = _measure_block(draw, lines, font)
        else:
            font_size, lines, heights = fit_font_and_wrap(draw, text, box_w + LENS_OVERFLOW_PX * 2, box_h + LENS_OVERFLOW_PX * 2, font_path=fp, max_size=HARD_MAX_SIZE, min_size=HARD_MIN_SIZE)
            font = get_font(fp, font_size)

        if font_size > HARD_MAX_SIZE or font_size < HARD_MIN_SIZE:
            font_size = max(HARD_MIN_SIZE, min(HARD_MAX_SIZE, font_size))
            font = get_font(fp, font_size)
            lines = wrap_text(draw, text, font, max_width=(box_w + LENS_OVERFLOW_PX * 2) - 4, allow_break=False, is_vertical=False)
            heights, _, _ = _measure_block(draw, lines, font)

        total_text_h = sum(heights) if heights else font_size * len(lines)
        start_y = (y1 - LENS_OVERFLOW_PX) + ((box_h + LENS_OVERFLOW_PX * 2) - total_text_h) // 2 if not really_small else y1 + (box_h - total_text_h) // 2

        current_y = start_y
        for i, line in enumerate(lines):
            if not line:
                current_y += heights[i] if i < len(heights) else font_size
                continue
            line_w = draw.textlength(line, font=font)
            line_x = ((x1 - LENS_OVERFLOW_PX) + ((box_w + LENS_OVERFLOW_PX * 2) - line_w) / 2) if not really_small else (x1 + (box_w - line_w) / 2)
            
            # FIX IS ON THIS LINE:
            draw_text_with_config(draw, (line_x, current_y), line, font=font, fill=text_color, stroke_fill=bg_color)
            
            current_y += heights[i] if i < len(heights) else font_size

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
