"""Shared visual-primitive format (Objects365 + Pixmo-Points). See docs/FORMAT.md.

Coordinates are integers in [0, 999], normalized by image width/height.
Boxes are [x1, y1, x2, y2] (top-left, bottom-right), sorted left->right (x1, then y1).
"""
from __future__ import annotations

import json
import re

COORD_MAX = 999

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two",
             "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
             "twenty-eight", "twenty-nine", "thirty"]


def _q(v: float, size: float) -> int:
    return max(0, min(COORD_MAX, round(v / size * COORD_MAX)))


def norm_box(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> list[int]:
    """Pixel xyxy -> normalized int xyxy in [0, 999]."""
    return [_q(x1, w), _q(y1, h), _q(x2, w), _q(y2, h)]


def norm_point(x: float, y: float, w: float, h: float) -> list[int]:
    return [_q(x, w), _q(y, h)]


def denorm_box(b, w, h):
    return [b[0] / COORD_MAX * w, b[1] / COORD_MAX * h, b[2] / COORD_MAX * w, b[3] / COORD_MAX * h]


def sort_boxes(boxes):
    return sorted(boxes, key=lambda b: (b[0], b[1], b[2], b[3]))


def sort_points(points):
    return sorted(points, key=lambda p: (p[0], p[1]))


def coords_str(items) -> str:
    """[[1,2,3,4],[5,6,7,8]] with no spaces (compact, stable tokenization)."""
    return json.dumps([list(map(int, it)) for it in items], separators=(",", ":"))


def ref_box(label: str, boxes) -> str:
    return f"<ref>{label}</ref><box>{coords_str(boxes)}</box>"


def point_tag(points) -> str:
    return f"<point>{coords_str(points)}</point>"


def assistant_text(thinking: str, final: str) -> str:
    """Thinking block + concise final response, byte-identical to Qwen3.5's chat-template rendering."""
    return f"<think>\n{thinking.strip()}\n</think>\n\n{str(final).strip()}"


def _pick(key: str, options):
    import hashlib
    return options[int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % len(options)]


def count_response(n: int, singular: str, plural: str, key: str = "") -> str:
    """Concise final answer for a counting question (the paper's 'concise final response')."""
    if n == 0:
        opts = [f"There are no {plural} in the image.", f"No {plural} are visible in the image.",
                f"The image contains no {plural}."]
    elif n == 1:
        opts = [f"There is 1 {singular} in the image.", f"I count 1 {singular}.",
                f"The image contains 1 {singular}."]
    else:
        opts = [f"There are {n} {plural} in the image.", f"I count {n} {plural}.",
                f"The image contains {n} {plural}."]
    return _pick(key, opts)


_BOX_RE = re.compile(r"(?:<ref>(.*?)</ref>)?<box>(\[.*?\]\])</box>", re.S)
_POINT_RE = re.compile(r"<point>(\[.*?\]\])</point>", re.S)
_BOXED_RE = re.compile(r"\\boxed\{\s*([^{}]*?)\s*\}")


def parse_boxes(text: str):
    """-> list of (label|None, [[x1,y1,x2,y2], ...]). Raises ValueError on malformed lists."""
    out = []
    for label, arr in _BOX_RE.findall(text):
        boxes = json.loads(arr)
        if not all(isinstance(b, list) and len(b) == 4 and all(isinstance(v, int) for v in b) for b in boxes):
            raise ValueError(f"malformed box list: {arr[:80]}")
        out.append((label or None, boxes))
    return out


def parse_points(text: str):
    out = []
    for arr in _POINT_RE.findall(text):
        pts = json.loads(arr)
        if not all(isinstance(p, list) and len(p) == 2 for p in pts):
            raise ValueError(f"malformed point list: {arr[:80]}")
        out.append(pts)
    return out


def final_response(text: str) -> str:
    """Text after the last </think> (the whole text if there is no thinking block)."""
    return text.rsplit("</think>", 1)[-1].strip()


_WORD_NUM = re.compile(r"\b(" + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r")\b", re.I)


def parse_count(text: str):
    """Integer count from a model answer: \\boxed{N}, else the first number in the final response
    (digits, then number words), else 0 for 'no/none/zero ...', else None."""
    b = parse_boxed_answer(text)
    if isinstance(b, int):
        return b
    f = final_response(text)
    m = re.search(r"(?<![\d.])\d+(?!\.?\d)", f)   # "is 4." ok, "4.5" not
    if m:
        return int(m.group(0))
    m = _WORD_NUM.search(f)
    if m:
        return NUM_WORDS.index(m.group(1).lower())
    if re.search(r"\b(no|none|not any|n't see any)\b", f, re.I):
        return 0
    return None


def parse_boxed_answer(text: str):
    """Last \\boxed{...} as int if numeric, else the raw string, else None."""
    m = _BOXED_RE.findall(text)
    if not m:
        return None
    s = m[-1].strip()
    try:
        return int(s)
    except ValueError:
        return s


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_f1(pred, gt, thr: float = 0.5) -> tuple[float, float, float]:
    """Greedy one-to-one matching at IoU >= thr. Returns (precision, recall, f1)."""
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
