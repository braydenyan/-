from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from pypdf.generic import NameObject
from pypdf.xmp import XmpInformation

# =========================
# CONFIG
# =========================

# don't forget to specify these (many e-readers load from this metadata, or from title: "Book - Author")
TITLE: str = ""
AUTHOR: str = ""

# supported units: "in", "mm"
TARGET_SIZE_NAME = "A4"
TARGET_UNIT = "mm"

TARGET_W = 210.0
TARGET_H = 297.0

RATIO_TOL_REL = 0.005 # 0.5% ratio comparison tolerance
SIZE_TOL = 0.5        # TARGET_UNIT tolerance
EPS = 1e-9

# smart crop
CROP_DETECT_DPI = 150 # higher = more accurate but slower
CROP_WHITE_THRESHOLD = 245
CROP_MIN_DARK_PIXELS_PER_ROW = 2
CROP_MIN_DARK_PIXELS_PER_COL = 2
MIN_CENTER_CROP_SCALE_CHANGE = 0.002 # skips miniscule crops

# smart crop focus:
#   "center" (crop symmetrically all around)
#   "left"   (anchor left edge; crop top/bottom/right only)
#   "right"  (anchor right edge; crop top/bottom/left only)
#   "top"    (anchor top edge; crop left/right/bottom only)
#   "bottom" (anchor bottom edge; crop left/right/top only)
SMART_CROP_FOCUS_MODE = "center"

# post-smart-crop padding:
# applied only to pages that actually receive smart crop before rescaling back to target size
# this value is the VERTICAL per-side padding (top and bottom) in mm.
# horizontal per-side padding is computed automatically to preserve aspect ratio
CENTER_CROP_POST_PADDING_MM = 5.0

# optional pass AFTER smart-crop:
# restore every page back to the pre-normalization-padding dimensions (crop-only, no rescale)
# using normalization-padding provenance + smart-crop outcomes + measured whitespace
REMOVE_NORMALIZATION_PAD_RESIDUAL_AFTER_SMART_CROP = True

# post-smart-crop content repositioning (no scaling) via whitespace redistribution:
# pair modes: none / normal / greedy
# greedy side values:
#   top/bottom pair -> "top" or "bottom"
#   left/right pair -> "left" or "right"
EDGE_REPOSITION_TOP_BOTTOM_MODE = "none"
EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE = "top"

EDGE_REPOSITION_LEFT_RIGHT_MODE = "none"
EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE = "left"

EDGE_REPOSITION_DETECT_DPI = 150 # higher = more accurate but slower
EDGE_REPOSITION_WHITE_THRESHOLD = 245

# some PDFs have fragile content streams that pypdf can rewrite into a visually blank page when compressed
# enable this if you know your PDFs can be safely recompressed and want to reduce output file size
COMPRESS_PAGE_CONTENT_STREAMS = True

# pipeline stage toggles (for debugging)
ENABLE_NORMALIZATION = True
ENABLE_SMART_CROP = True

# =========================
# END CONFIG
# =========================

SUPPORTED_UNITS = {"in", "mm"}
UNIT_SCALE_FROM_INCH = {
    "in": 1.0,
    "mm": 25.4,
}

PAIR_MODES = {"none", "normal", "greedy"}
TB_SIDES = {"top", "bottom"}
LR_SIDES = {"left", "right"}
SMART_CROP_FOCUS_MODES = {"center", "left", "right", "top", "bottom"}

if TARGET_UNIT not in SUPPORTED_UNITS:
    raise ValueError(f"Unsupported TARGET_UNIT={TARGET_UNIT!r}; use one of {sorted(SUPPORTED_UNITS)}")

if TARGET_W <= 0 or TARGET_H <= 0:
    raise ValueError("TARGET_W and TARGET_H must be > 0")

if SIZE_TOL < 0:
    raise ValueError("SIZE_TOL must be >= 0")

if CROP_DETECT_DPI <= 0:
    raise ValueError("CROP_DETECT_DPI must be > 0")

if not (0 <= CROP_WHITE_THRESHOLD <= 255):
    raise ValueError("CROP_WHITE_THRESHOLD must be in [0, 255]")

if SMART_CROP_FOCUS_MODE not in SMART_CROP_FOCUS_MODES:
    raise ValueError(f"SMART_CROP_FOCUS_MODE must be one of {sorted(SMART_CROP_FOCUS_MODES)}")

if CENTER_CROP_POST_PADDING_MM < 0:
    raise ValueError("CENTER_CROP_POST_PADDING_MM must be >= 0")

if EDGE_REPOSITION_TOP_BOTTOM_MODE not in PAIR_MODES:
    raise ValueError(f"EDGE_REPOSITION_TOP_BOTTOM_MODE must be one of {sorted(PAIR_MODES)}")

if EDGE_REPOSITION_LEFT_RIGHT_MODE not in PAIR_MODES:
    raise ValueError(f"EDGE_REPOSITION_LEFT_RIGHT_MODE must be one of {sorted(PAIR_MODES)}")

if EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE not in TB_SIDES:
    raise ValueError(f"EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE must be one of {sorted(TB_SIDES)}")

if EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE not in LR_SIDES:
    raise ValueError(f"EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE must be one of {sorted(LR_SIDES)}")

if EDGE_REPOSITION_DETECT_DPI <= 0:
    raise ValueError("EDGE_REPOSITION_DETECT_DPI must be > 0")

if not (0 <= EDGE_REPOSITION_WHITE_THRESHOLD <= 255):
    raise ValueError("EDGE_REPOSITION_WHITE_THRESHOLD must be in [0, 255]")

TARGET_LONG_SHORT = max(TARGET_W, TARGET_H) / max(min(TARGET_W, TARGET_H), EPS)

@dataclass(frozen=True)
class PageTransform:
    page_num: int
    action: str
    before_size: Tuple[float, float]
    after_size: Tuple[float, float]
    note: str = ""

@dataclass(frozen=True)
class PadEquivalentTrimSides:
    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0

@dataclass(frozen=True)
class EdgeWhitespaceFractions:
    left: float
    right: float
    top: float
    bottom: float

@dataclass(frozen=True)
class SmartCropOutcome:
    page_num: int
    applied: bool
    page_possible_scale: float
    uniform_scale: float
    pre_w_cfg: float
    pre_h_cfg: float

def format_size(w: float, h: float) -> str:
    decimals = 2 if TARGET_UNIT == "in" else 1
    return f"{w:.{decimals}f}×{h:.{decimals}f} {TARGET_UNIT}"

def print_transform_report(events: List[PageTransform], out_path: Path) -> None:
    total = len(events)
    changed = [e for e in events if e.action != "none"]
    n_changed = len(changed)

    n_rotate = sum(1 for e in changed if "rotate" in e.action)
    n_scale = sum(
        1 for e in changed
        if ("scale" in e.action and "pad+scale" not in e.action and "smart-crop" not in e.action)
    )
    n_pad_scale = sum(1 for e in changed if "pad+scale" in e.action)
    n_restitution = sum(1 for e in changed if "normalization-pad-restitution" in e.action)
    n_smart_crop = sum(1 for e in changed if "smart-crop" in e.action)
    n_reposition = sum(1 for e in changed if "content-reposition" in e.action)

    print("\n=== Report ===")
    print(f"Output: {out_path}")
    print(f"Pages: {total}")

    if n_changed == 0:
        print("No transformations necessary.")
        return

    if n_rotate:
        print(f"Rotated 90° CW: {n_rotate}")
    if n_scale:
        print(f"Scaled ({TARGET_SIZE_NAME}): {n_scale}")
    if n_pad_scale:
        print(f"Padded + Scaled ({TARGET_SIZE_NAME}): {n_pad_scale}")
    if n_smart_crop:
        print(f"Smart crop ({SMART_CROP_FOCUS_MODE}): {n_smart_crop}")
    if n_restitution:
        print(f"Normalization padding restitution: {n_restitution}")
    if n_reposition:
        print(f"Whitespace redistribution: {n_reposition}")

    print("\nChanged pages:")

    cap = 12
    for e in changed[:cap]:
        note = f" — {e.note}" if e.note else ""
        print(
            f"  p{e.page_num}: {format_size(*e.before_size)} -> "
            f"{format_size(*e.after_size)} [{e.action}]{note}"
        )
    if n_changed > cap:
        print(f"  …and {n_changed - cap} more changed pages.")

def next_windows_paren_name(in_path: Path, out_suffix: str = ".pdf") -> Path:
    directory = in_path.parent
    stem = in_path.stem
    n = 1
    while True:
        candidate = directory / f"{stem} ({n}){out_suffix}"
        if not candidate.exists():
            return candidate
        n += 1

def convert_epub_to_pdf(epub_path: Path, pdf_out: Path) -> None:
    exe = shutil.which("ebook-convert")
    if not exe:
        raise RuntimeError(
            "EPUB input requires Calibre's `ebook-convert` on PATH.\n"
            "Install Calibre and ensure `ebook-convert` is available, then retry."
        )

    cmd = [exe, str(epub_path), str(pdf_out)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"EPUB->PDF conversion failed.\nCommand: {' '.join(cmd)}\n\n{stderr}") from e

def pdf_units_per_inch(user_unit: float) -> float:
    return 72.0 / float(user_unit)

def config_units_per_inch() -> float:
    return UNIT_SCALE_FROM_INCH[TARGET_UNIT]

def pdf_units_per_config_unit(user_unit: float) -> float:
    return pdf_units_per_inch(user_unit) / config_units_per_inch()

def page_size_in_config_units(page) -> Tuple[float, float]:
    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
    w_pdf_units = float(page.mediabox.width)
    h_pdf_units = float(page.mediabox.height)
    factor = (uu / 72.0) * config_units_per_inch()
    return w_pdf_units * factor, h_pdf_units * factor

def mm_to_config_units(mm: float) -> float:
    if TARGET_UNIT == "mm":
        return float(mm)
    return float(mm) / 25.4

def is_target_size(w: float, h: float) -> bool:
    portrait = (abs(w - TARGET_W) <= SIZE_TOL and abs(h - TARGET_H) <= SIZE_TOL)
    landscape = (abs(w - TARGET_H) <= SIZE_TOL and abs(h - TARGET_W) <= SIZE_TOL)
    return portrait or landscape

def ratio_matches_target(w: float, h: float) -> bool:
    long_side = max(w, h)
    short_side = max(min(w, h), EPS)
    r = long_side / short_side
    return abs(r - TARGET_LONG_SHORT) / TARGET_LONG_SHORT <= RATIO_TOL_REL

def set_all_page_boxes(page, llx: float, lly: float, urx: float, ury: float) -> None:
    page.mediabox.lower_left = (llx, lly)
    page.mediabox.upper_right = (urx, ury)
    page.cropbox.lower_left = (llx, lly)
    page.cropbox.upper_right = (urx, ury)
    page.bleedbox.lower_left = (llx, lly)
    page.bleedbox.upper_right = (urx, ury)
    page.trimbox.lower_left = (llx, lly)
    page.trimbox.upper_right = (urx, ury)
    page.artbox.lower_left = (llx, lly)
    page.artbox.upper_right = (urx, ury)

def crop_page_by_edge_fractions(
    page,
    *,
    left_frac: float = 0.0,
    right_frac: float = 0.0,
    top_frac: float = 0.0,
    bottom_frac: float = 0.0,
) -> None:
    for name, v in [
        ("left_frac", left_frac),
        ("right_frac", right_frac),
        ("top_frac", top_frac),
        ("bottom_frac", bottom_frac),
    ]:
        if v < -EPS:
            raise ValueError(f"{name} must be >= 0")

    left_frac = max(0.0, float(left_frac))
    right_frac = max(0.0, float(right_frac))
    top_frac = max(0.0, float(top_frac))
    bottom_frac = max(0.0, float(bottom_frac))

    if (left_frac + right_frac) >= (1.0 - EPS):
        raise ValueError("Horizontal crop fractions would consume page width.")
    if (top_frac + bottom_frac) >= (1.0 - EPS):
        raise ValueError("Vertical crop fractions would consume page height.")

    llx = float(page.mediabox.left)
    lly = float(page.mediabox.bottom)
    w = float(page.mediabox.width)
    h = float(page.mediabox.height)

    new_llx = llx + (w * left_frac)
    new_lly = lly + (h * bottom_frac)
    new_urx = llx + (w * (1.0 - right_frac))
    new_ury = lly + (h * (1.0 - top_frac))

    if (new_urx - new_llx) <= EPS or (new_ury - new_lly) <= EPS:
        raise ValueError("Crop produced invalid/non-positive page box.")

    set_all_page_boxes(page, new_llx, new_lly, new_urx, new_ury)

def expand_canvas_centered(page, new_w_pdf_units: float, new_h_pdf_units: float) -> None:
    old_w = float(page.mediabox.width)
    old_h = float(page.mediabox.height)
    if new_w_pdf_units + EPS < old_w or new_h_pdf_units + EPS < old_h:
        raise ValueError("expand_canvas_centered() only supports expanding.")

    llx = float(page.mediabox.left)
    lly = float(page.mediabox.bottom)
    urx = llx + float(new_w_pdf_units)
    ury = lly + float(new_h_pdf_units)

    set_all_page_boxes(page, llx, lly, urx, ury)

    tx = (float(new_w_pdf_units) - old_w) / 2.0
    ty = (float(new_h_pdf_units) - old_h) / 2.0
    if abs(tx) > EPS or abs(ty) > EPS:
        page.add_transformation(Transformation().translate(tx, ty))

def rotate_page_clockwise_to_portrait(page) -> None:
    if hasattr(page, "rotate"):
        page.rotate(90)
    elif hasattr(page, "rotate_clockwise"):
        page.rotate_clockwise(90)
    else:
        raise RuntimeError("This pypdf version does not support page rotation.")

    if hasattr(page, "transfer_rotation_to_content"):
        page.transfer_rotation_to_content()

def scale_to_target(page) -> None:
    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
    pdf_per_cfg = pdf_units_per_config_unit(uu)

    w_pdf_units = float(page.mediabox.width)
    h_pdf_units = float(page.mediabox.height)

    target_portrait_w = min(TARGET_W, TARGET_H) * pdf_per_cfg
    target_portrait_h = max(TARGET_W, TARGET_H) * pdf_per_cfg
    target_landscape_w = target_portrait_h
    target_landscape_h = target_portrait_w

    if h_pdf_units >= w_pdf_units:
        target_w = target_portrait_w
        target_h = target_portrait_h
    else:
        target_w = target_landscape_w
        target_h = target_landscape_h

    page.scale_to(target_w, target_h)


def maybe_compress_page_content_streams(page) -> None:
    if not COMPRESS_PAGE_CONTENT_STREAMS:
        return
    try:
        page.compress_content_streams()
    except Exception:
        pass

def _page_rotation_degrees(page) -> int:
    try:
        rot = int(getattr(page, "rotation", 0) or 0)
    except Exception:
        try:
            rot = int(page.get("/Rotate", 0) or 0)
        except Exception:
            rot = 0
    rot %= 360
    if rot not in (0, 90, 180, 270):
        rot = int(round(rot / 90.0) * 90) % 360
    return rot

def _rect_rotation_transfer_transform(
    llx: float,
    lly: float,
    w: float,
    h: float,
    rotation_deg_clockwise: int,
) -> Tuple[Transformation, float, float]:
    rot = int(rotation_deg_clockwise) % 360
    if rot == 0:
        return Transformation(), float(w), float(h)

    cx = float(llx) + (float(w) / 2.0)
    cy = float(lly) + (float(h) / 2.0)

    trsf = Transformation().translate(-cx, -cy).rotate(-rot)

    corners = [
        (float(llx), float(lly)),
        (float(llx) + float(w), float(lly)),
        (float(llx), float(lly) + float(h)),
        (float(llx) + float(w), float(lly) + float(h)),
    ]
    pts = [trsf.apply_on(pt) for pt in corners]
    min_x = min(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_x = max(p[0] for p in pts)
    max_y = max(p[1] for p in pts)

    trsf = trsf.translate(-min_x, -min_y)
    return trsf, float(max_x - min_x), float(max_y - min_y)

def _flatten_page_rotation_transform(page) -> Tuple[Transformation, float, float]:
    w = float(page.mediabox.width)
    h = float(page.mediabox.height)
    rot = _page_rotation_degrees(page)
    return _rect_rotation_transfer_transform(0.0, 0.0, w, h, rot)

def _target_dims_pdf_units_for_orientation(
    *,
    user_unit: float,
    current_w_pdf_units: float,
    current_h_pdf_units: float,
) -> Tuple[float, float]:
    pdf_per_cfg = pdf_units_per_config_unit(user_unit)

    target_portrait_w = min(TARGET_W, TARGET_H) * pdf_per_cfg
    target_portrait_h = max(TARGET_W, TARGET_H) * pdf_per_cfg
    target_landscape_w = target_portrait_h
    target_landscape_h = target_portrait_w

    if current_h_pdf_units >= current_w_pdf_units:
        return float(target_portrait_w), float(target_portrait_h)
    return float(target_landscape_w), float(target_landscape_h)

def _copy_user_unit_if_present(src_page, dst_page) -> None:
    try:
        if "/UserUnit" in src_page:
            dst_page[NameObject("/UserUnit")] = src_page["/UserUnit"]
    except Exception:
        pass

def _make_transformed_page_from_source(
    src_page,
    *,
    out_w_pdf_units: float,
    out_h_pdf_units: float,
    transform: Transformation,
):
    new_page = PageObject.create_blank_page(
        width=float(out_w_pdf_units),
        height=float(out_h_pdf_units),
    )
    _copy_user_unit_if_present(src_page, new_page)
    new_page.merge_transformed_page(src_page, transform, over=True, expand=False)
    return new_page

def _is_identity_transform(tr: Transformation) -> bool:
    a, b, c, d, e, f = tr.ctm
    return (
        abs(a - 1.0) <= 1e-12
        and abs(b) <= 1e-12
        and abs(c) <= 1e-12
        and abs(d - 1.0) <= 1e-12
        and abs(e) <= 1e-12
        and abs(f) <= 1e-12
    )

def _crop_rect_same_ratio_with_focus(
    w: float,
    h: float,
    scale: float,
    focus_mode: str,
) -> Tuple[float, float, float, float]:
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"Invalid crop scale: {scale}")

    w = float(w)
    h = float(h)

    if scale >= 1.0 - EPS:
        return 0.0, 0.0, w, h

    new_w = w * float(scale)
    new_h = h * float(scale)

    if focus_mode == "center":
        dx = (w - new_w) / 2.0
        dy = (h - new_h) / 2.0
        return dx, dy, dx + new_w, dy + new_h

    if focus_mode == "left":
        dy = (h - new_h) / 2.0
        return 0.0, dy, new_w, dy + new_h

    if focus_mode == "right":
        dy = (h - new_h) / 2.0
        return w - new_w, dy, w, dy + new_h

    if focus_mode == "top":
        dx = (w - new_w) / 2.0
        return dx, h - new_h, dx + new_w, h

    if focus_mode == "bottom":
        dx = (w - new_w) / 2.0
        return dx, 0.0, dx + new_w, new_h

    raise ValueError(f"Unsupported smart crop focus mode: {focus_mode}")

def normalize_page_to_target(page) -> Tuple[object, str, str, PadEquivalentTrimSides]:
    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
    factor_pdf_to_cfg = (uu / 72.0) * config_units_per_inch()

    tr = Transformation()
    flatten_tr, cur_w_pdf, cur_h_pdf = _flatten_page_rotation_transform(page)
    tr = tr.transform(flatten_tr)

    rotated_cw = False
    pad_trim = PadEquivalentTrimSides()

    w_cfg = cur_w_pdf * factor_pdf_to_cfg
    h_cfg = cur_h_pdf * factor_pdf_to_cfg

    if w_cfg > h_cfg:
        rot90_tr, cur_w_pdf, cur_h_pdf = _rect_rotation_transfer_transform(
            0.0, 0.0, cur_w_pdf, cur_h_pdf, 90
        )
        tr = tr.transform(rot90_tr)
        rotated_cw = True
        w_cfg = cur_w_pdf * factor_pdf_to_cfg
        h_cfg = cur_h_pdf * factor_pdf_to_cfg

    if is_target_size(w_cfg, h_cfg):
        base_action, base_note = "none", ""
    elif ratio_matches_target(w_cfg, h_cfg):
        target_w, target_h = _target_dims_pdf_units_for_orientation(
            user_unit=uu,
            current_w_pdf_units=cur_w_pdf,
            current_h_pdf_units=cur_h_pdf,
        )
        sx = target_w / max(cur_w_pdf, EPS)
        sy = target_h / max(cur_h_pdf, EPS)
        tr = tr.scale(sx=sx, sy=sy)
        cur_w_pdf, cur_h_pdf = target_w, target_h
        base_action, base_note = "scale", f"aspect ratio matched {TARGET_SIZE_NAME}; scaled only"
    else:
        w_pdf_units = cur_w_pdf
        h_pdf_units = cur_h_pdf

        long_pdf = max(w_pdf_units, h_pdf_units)
        short_pdf = max(min(w_pdf_units, h_pdf_units), EPS)
        r = long_pdf / short_pdf

        if r < TARGET_LONG_SHORT:
            new_w_pdf = w_pdf_units
            new_h_pdf = max(h_pdf_units, w_pdf_units * TARGET_LONG_SHORT)
            pad_note = "padded height (stout/square-ish)"
        else:
            if h_pdf_units >= w_pdf_units:
                new_h_pdf = h_pdf_units
                new_w_pdf = h_pdf_units / TARGET_LONG_SHORT
                pad_note = "padded width (skinny)"
            else:
                new_w_pdf = w_pdf_units
                new_h_pdf = w_pdf_units / TARGET_LONG_SHORT
                pad_note = "padded height (skinny landscape path)"

        new_w_pdf = max(new_w_pdf, w_pdf_units)
        new_h_pdf = max(new_h_pdf, h_pdf_units)

        frac_left = max(0.0, (new_w_pdf - w_pdf_units) / max(2.0 * new_w_pdf, EPS))
        frac_right = frac_left
        frac_top = max(0.0, (new_h_pdf - h_pdf_units) / max(2.0 * new_h_pdf, EPS))
        frac_bottom = frac_top
        pad_trim = PadEquivalentTrimSides(
            left=frac_left,
            right=frac_right,
            top=frac_top,
            bottom=frac_bottom,
        )

        old_w_cfg = w_pdf_units * factor_pdf_to_cfg
        old_h_cfg = h_pdf_units * factor_pdf_to_cfg
        new_w_cfg = new_w_pdf * factor_pdf_to_cfg
        new_h_cfg = new_h_pdf * factor_pdf_to_cfg
        dw_cfg = max(0.0, new_w_cfg - old_w_cfg)
        dh_cfg = max(0.0, new_h_cfg - old_h_cfg)

        tx = (float(new_w_pdf) - float(w_pdf_units)) / 2.0
        ty = (float(new_h_pdf) - float(h_pdf_units)) / 2.0
        if abs(tx) > EPS or abs(ty) > EPS:
            tr = tr.translate(tx=tx, ty=ty)

        cur_w_pdf, cur_h_pdf = float(new_w_pdf), float(new_h_pdf)

        target_w, target_h = _target_dims_pdf_units_for_orientation(
            user_unit=uu,
            current_w_pdf_units=cur_w_pdf,
            current_h_pdf_units=cur_h_pdf,
        )
        sx = target_w / max(cur_w_pdf, EPS)
        sy = target_h / max(cur_h_pdf, EPS)
        tr = tr.scale(sx=sx, sy=sy)
        cur_w_pdf, cur_h_pdf = target_w, target_h

        decimals = 2 if TARGET_UNIT == "in" else 1
        base_action = "pad+scale"
        base_note = (
            f"{pad_note}; added {dw_cfg:.{decimals}f}{TARGET_UNIT} W, "
            f"{dh_cfg:.{decimals}f}{TARGET_UNIT} H before scale"
        )

    if not rotated_cw:
        action = base_action
        note = base_note
    else:
        rotate_note = "rotated 90° CW to portrait first"
        if base_action == "none":
            action, note = "rotate", rotate_note
        elif base_note:
            action, note = f"rotate+{base_action}", f"{rotate_note}; {base_note}"
        else:
            action, note = f"rotate+{base_action}", rotate_note

    if _is_identity_transform(tr):
        return page, action, note, pad_trim

    new_page = _make_transformed_page_from_source(
        page,
        out_w_pdf_units=cur_w_pdf,
        out_h_pdf_units=cur_h_pdf,
        transform=tr,
    )
    return new_page, action, note, pad_trim

def _smart_crop_consumed_side_fractions(scale: float, focus_mode: str) -> PadEquivalentTrimSides:
    c = max(0.0, 1.0 - float(scale))
    half = c / 2.0

    if focus_mode == "center":
        return PadEquivalentTrimSides(left=half, right=half, top=half, bottom=half)
    if focus_mode == "left":
        return PadEquivalentTrimSides(left=0.0, right=c, top=half, bottom=half)
    if focus_mode == "right":
        return PadEquivalentTrimSides(left=c, right=0.0, top=half, bottom=half)
    if focus_mode == "top":
        return PadEquivalentTrimSides(left=half, right=half, top=0.0, bottom=c)
    if focus_mode == "bottom":
        return PadEquivalentTrimSides(left=half, right=half, top=c, bottom=0.0)

    raise ValueError(f"Unsupported smart crop focus mode: {focus_mode}")

def crop_page_same_ratio_with_focus(page, scale: float, focus_mode: str) -> None:
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"Invalid crop scale: {scale}")

    if scale >= 1.0 - EPS:
        return

    llx = float(page.mediabox.left)
    lly = float(page.mediabox.bottom)
    urx = float(page.mediabox.right)
    ury = float(page.mediabox.top)
    w = urx - llx
    h = ury - lly

    new_w = w * scale
    new_h = h * scale

    if focus_mode == "center":
        dx = (w - new_w) / 2.0
        dy = (h - new_h) / 2.0
        set_all_page_boxes(page, llx + dx, lly + dy, llx + dx + new_w, lly + dy + new_h)
        return

    if focus_mode == "left":
        dy = (h - new_h) / 2.0
        set_all_page_boxes(page, llx, lly + dy, llx + new_w, lly + dy + new_h)
        return

    if focus_mode == "right":
        dy = (h - new_h) / 2.0
        set_all_page_boxes(page, urx - new_w, lly + dy, urx, lly + dy + new_h)
        return

    if focus_mode == "top":
        dx = (w - new_w) / 2.0
        set_all_page_boxes(page, llx + dx, ury - new_h, llx + dx + new_w, ury)
        return

    if focus_mode == "bottom":
        dx = (w - new_w) / 2.0
        set_all_page_boxes(page, llx + dx, lly, llx + dx + new_w, lly + new_h)
        return

    raise ValueError(f"Unsupported smart crop focus mode: {focus_mode}")

def add_ratio_abiding_post_crop_padding_by_vertical_mm(page, vertical_padding_mm: float) -> None:
    if vertical_padding_mm < 0:
        raise ValueError("vertical_padding_mm must be >= 0")
    if vertical_padding_mm == 0:
        return

    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
    w_cfg, h_cfg = page_size_in_config_units(page)
    if w_cfg <= EPS or h_cfg <= EPS:
        raise ValueError("Invalid page size for ratio-abiding padding.")

    pad_y_cfg = mm_to_config_units(vertical_padding_mm)
    pad_x_cfg = pad_y_cfg * (w_cfg / h_cfg)

    pdf_per_cfg = pdf_units_per_config_unit(uu)
    pad_x_pdf = pad_x_cfg * pdf_per_cfg
    pad_y_pdf = pad_y_cfg * pdf_per_cfg

    llx = float(page.mediabox.left)
    lly = float(page.mediabox.bottom)
    urx = float(page.mediabox.right)
    ury = float(page.mediabox.top)

    set_all_page_boxes(
        page,
        llx - pad_x_pdf,
        lly - pad_y_pdf,
        urx + pad_x_pdf,
        ury + pad_y_pdf,
    )

@contextmanager
def suppress_mupdf_messages():
    if fitz is None or not hasattr(fitz, "TOOLS"):
        yield
        return

    prev_err = None
    prev_warn = None
    try:
        prev_err = fitz.TOOLS.mupdf_display_errors(False)
        prev_warn = fitz.TOOLS.mupdf_display_warnings(False)
        yield
    finally:
        try:
            fitz.TOOLS.mupdf_display_errors(prev_err if prev_err is not None else True)
        except Exception:
            pass
        try:
            fitz.TOOLS.mupdf_display_warnings(prev_warn if prev_warn is not None else True)
        except Exception:
            pass

def _detect_content_bounds_pixel_boxes_from_pdf(
    pdf_path: Path,
    *,
    dpi: int,
    white_threshold: int,
) -> List[Optional[Tuple[int, int, int, int, int, int]]]:
    results: List[Optional[Tuple[int, int, int, int, int, int]]] = []

    with suppress_mupdf_messages():
        doc = fitz.open(str(pdf_path))
        try:
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            for p in doc:
                pix = p.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)

                h_px = int(pix.height)
                w_px = int(pix.width)
                if w_px <= 0 or h_px <= 0:
                    results.append(None)
                    continue

                buf = np.frombuffer(pix.samples, dtype=np.uint8)
                stride = int(pix.stride)
                arr = buf.reshape(h_px, stride)[:, :w_px]

                mask = arr < white_threshold
                if not bool(mask.any()):
                    results.append(None)
                    continue

                row_counts = mask.sum(axis=1)
                col_counts = mask.sum(axis=0)

                rows = np.where(row_counts >= CROP_MIN_DARK_PIXELS_PER_ROW)[0]
                cols = np.where(col_counts >= CROP_MIN_DARK_PIXELS_PER_COL)[0]

                if rows.size == 0 or cols.size == 0:
                    results.append(None)
                    continue

                y0 = int(rows[0])
                y1 = int(rows[-1]) + 1
                x0 = int(cols[0])
                x1 = int(cols[-1]) + 1

                results.append((x0, y0, x1, y1, w_px, h_px))
        finally:
            doc.close()

    return results

def detect_smart_crop_scales_from_pdf(
    pdf_path: Path,
    *,
    dpi: int = CROP_DETECT_DPI,
    white_threshold: int = CROP_WHITE_THRESHOLD,
    focus_mode: str = SMART_CROP_FOCUS_MODE,
) -> List[float]:
    bounds = _detect_content_bounds_pixel_boxes_from_pdf(
        pdf_path,
        dpi=dpi,
        white_threshold=white_threshold,
    )

    scales: List[float] = []
    for item in bounds:
        if item is None:
            scales.append(1.0)
            continue

        x0, y0, x1, y1, w_px, h_px = item

        cx = w_px / 2.0
        cy = h_px / 2.0
        half_w = w_px / 2.0
        half_h = h_px / 2.0

        req_left_center = (cx - x0) / max(half_w, EPS)
        req_right_center = (x1 - cx) / max(half_w, EPS)
        req_top_center = (cy - y0) / max(half_h, EPS)
        req_bottom_center = (y1 - cy) / max(half_h, EPS)

        if focus_mode == "center":
            s = max(req_left_center, req_right_center, req_top_center, req_bottom_center)
        elif focus_mode == "left":
            req_right_anchor_left = x1 / max(w_px, EPS)
            s = max(req_right_anchor_left, req_top_center, req_bottom_center)
        elif focus_mode == "right":
            req_left_anchor_right = (w_px - x0) / max(w_px, EPS)
            s = max(req_left_anchor_right, req_top_center, req_bottom_center)
        elif focus_mode == "top":
            req_bottom_anchor_top = y1 / max(h_px, EPS)
            s = max(req_left_center, req_right_center, req_bottom_anchor_top)
        elif focus_mode == "bottom":
            req_top_anchor_bottom = (h_px - y0) / max(h_px, EPS)
            s = max(req_left_center, req_right_center, req_top_anchor_bottom)
        else:
            raise ValueError(f"Unsupported smart crop focus mode: {focus_mode}")

        s = min(1.0, max(0.0, float(s)))
        if (1.0 - s) < MIN_CENTER_CROP_SCALE_CHANGE:
            s = 1.0

        scales.append(s)

    return scales

def choose_uniform_smart_crop_scale(crop_scales: List[float]) -> float:
    positive_crop_amounts = [(1.0 - s) for s in crop_scales if (1.0 - s) > EPS]
    if not positive_crop_amounts:
        return 1.0
    uniform_crop_amount = min(positive_crop_amounts)
    return min(1.0, max(0.0, float(1.0 - uniform_crop_amount)))

def detect_edge_whitespace_fractions_from_pdf(
    pdf_path: Path,
    *,
    dpi: int = EDGE_REPOSITION_DETECT_DPI,
    white_threshold: int = EDGE_REPOSITION_WHITE_THRESHOLD,
) -> List[Optional[EdgeWhitespaceFractions]]:
    bounds = _detect_content_bounds_pixel_boxes_from_pdf(
        pdf_path,
        dpi=dpi,
        white_threshold=white_threshold,
    )

    results: List[Optional[EdgeWhitespaceFractions]] = []
    for item in bounds:
        if item is None:
            results.append(None)
            continue

        x0, y0, x1, y1, w_px, h_px = item
        if w_px <= 0 or h_px <= 0:
            results.append(None)
            continue

        left = max(0.0, x0 / float(w_px))
        right = max(0.0, (w_px - x1) / float(w_px))
        top = max(0.0, y0 / float(h_px))
        bottom = max(0.0, (h_px - y1) / float(h_px))

        results.append(EdgeWhitespaceFractions(left=left, right=right, top=top, bottom=bottom))

    return results

def apply_smart_crop_pass(
    input_pdf_path: Path,
    out_pdf_path: Path,
    *,
    author: str,
    title: str,
) -> Tuple[List[PageTransform], List[SmartCropOutcome]]:
    crop_scales = detect_smart_crop_scales_from_pdf(
        input_pdf_path,
        dpi=CROP_DETECT_DPI,
        white_threshold=CROP_WHITE_THRESHOLD,
        focus_mode=SMART_CROP_FOCUS_MODE,
    )
    uniform_scale = choose_uniform_smart_crop_scale(crop_scales)
    uniform_crop_amount = max(0.0, 1.0 - uniform_scale)

    reader = PdfReader(str(input_pdf_path))
    writer = PdfWriter()
    events: List[PageTransform] = []
    outcomes: List[SmartCropOutcome] = []

    if len(crop_scales) != len(reader.pages):
        raise RuntimeError(
            f"Smart-crop detector returned {len(crop_scales)} scales for {len(reader.pages)} pages."
        )

    for idx, (page, page_possible_scale) in enumerate(zip(reader.pages, crop_scales), start=1):
        before = page_size_in_config_units(page)
        pre_w_cfg, pre_h_cfg = before

        page_crop_amount = max(0.0, 1.0 - float(page_possible_scale))
        page_has_crop = page_crop_amount > EPS
        apply_uniform_crop_here = page_has_crop and (uniform_crop_amount > EPS)

        transformed_page = page

        if apply_uniform_crop_here:
            uu = float(getattr(page, "user_unit", 1.0) or 1.0)
            factor_pdf_to_cfg = (uu / 72.0) * config_units_per_inch()

            tr = Transformation()
            flatten_tr, cur_w_pdf, cur_h_pdf = _flatten_page_rotation_transform(page)
            tr = tr.transform(flatten_tr)

            crop_llx, crop_lly, crop_urx, crop_ury = _crop_rect_same_ratio_with_focus(
                cur_w_pdf,
                cur_h_pdf,
                uniform_scale,
                SMART_CROP_FOCUS_MODE,
            )
            crop_w_pdf = max(EPS, crop_urx - crop_llx)
            crop_h_pdf = max(EPS, crop_ury - crop_lly)

            tr = tr.translate(tx=-crop_llx, ty=-crop_lly)
            cur_w_pdf, cur_h_pdf = crop_w_pdf, crop_h_pdf

            if CENTER_CROP_POST_PADDING_MM > 0:
                pad_y_cfg = mm_to_config_units(CENTER_CROP_POST_PADDING_MM)
                crop_w_cfg = cur_w_pdf * factor_pdf_to_cfg
                crop_h_cfg = cur_h_pdf * factor_pdf_to_cfg
                pad_x_cfg = pad_y_cfg * (crop_w_cfg / max(crop_h_cfg, EPS))

                pad_x_pdf = pad_x_cfg / max(factor_pdf_to_cfg, EPS)
                pad_y_pdf = pad_y_cfg / max(factor_pdf_to_cfg, EPS)

                tr = tr.translate(tx=pad_x_pdf, ty=pad_y_pdf)
                cur_w_pdf += 2.0 * pad_x_pdf
                cur_h_pdf += 2.0 * pad_y_pdf

            target_w, target_h = _target_dims_pdf_units_for_orientation(
                user_unit=uu,
                current_w_pdf_units=cur_w_pdf,
                current_h_pdf_units=cur_h_pdf,
            )
            sx = target_w / max(cur_w_pdf, EPS)
            sy = target_h / max(cur_h_pdf, EPS)
            tr = tr.scale(sx=sx, sy=sy)
            cur_w_pdf, cur_h_pdf = target_w, target_h

            transformed_page = _make_transformed_page_from_source(
                page,
                out_w_pdf_units=cur_w_pdf,
                out_h_pdf_units=cur_h_pdf,
                transform=tr,
            )

            action = "smart-crop"
            note_parts = [
                f"mode={SMART_CROP_FOCUS_MODE}",
                f"uniform scale={uniform_scale:.4f}",
                f"uniform crop={uniform_crop_amount:.4f}",
                f"page possible scale={page_possible_scale:.4f}",
            ]
            if CENTER_CROP_POST_PADDING_MM > 0:
                note_parts.append(
                    f"ratio-abiding post-pad vertical={CENTER_CROP_POST_PADDING_MM:.2f}mm/side"
                )
            note_parts.append(f"rescaled to {TARGET_SIZE_NAME}")
            note = "; ".join(note_parts)
        else:
            action = "none"
            note = ""

        after = page_size_in_config_units(transformed_page)

        maybe_compress_page_content_streams(transformed_page)

        writer.add_page(transformed_page)
        events.append(PageTransform(idx, action, before, after, note))
        outcomes.append(
            SmartCropOutcome(
                page_num=idx,
                applied=bool(apply_uniform_crop_here),
                page_possible_scale=float(page_possible_scale),
                uniform_scale=float(uniform_scale),
                pre_w_cfg=float(pre_w_cfg),
                pre_h_cfg=float(pre_h_cfg),
            )
        )

    wipe_and_set_book_metadata(writer, author=author, title=title)
    with open(out_pdf_path, "wb") as f:
        writer.write(f)

    return events, outcomes

def choose_document_normalization_pad_target_sides(
    per_page_norm_pad_sides: List[PadEquivalentTrimSides],
) -> PadEquivalentTrimSides:
    if not per_page_norm_pad_sides:
        return PadEquivalentTrimSides()

    return PadEquivalentTrimSides(
        left=max((max(0.0, p.left) for p in per_page_norm_pad_sides), default=0.0),
        right=max((max(0.0, p.right) for p in per_page_norm_pad_sides), default=0.0),
        top=max((max(0.0, p.top) for p in per_page_norm_pad_sides), default=0.0),
        bottom=max((max(0.0, p.bottom) for p in per_page_norm_pad_sides), default=0.0),
    )

def compute_residual_normalization_pad_after_smart_crop(
    initial_pad_sides: List[PadEquivalentTrimSides],
    smart_crop_outcomes: List[SmartCropOutcome],
) -> List[PadEquivalentTrimSides]:
    if len(initial_pad_sides) != len(smart_crop_outcomes):
        raise RuntimeError(
            f"Normalization-pad provenance count {len(initial_pad_sides)} != smart-crop outcomes "
            f"{len(smart_crop_outcomes)}"
        )

    residuals: List[PadEquivalentTrimSides] = []
    pv_cfg = max(0.0, mm_to_config_units(CENTER_CROP_POST_PADDING_MM))

    for p, o in zip(initial_pad_sides, smart_crop_outcomes):
        if not o.applied or o.uniform_scale >= 1.0 - EPS:
            residuals.append(p)
            continue

        s = max(float(o.uniform_scale), EPS)
        consumed = _smart_crop_consumed_side_fractions(s, SMART_CROP_FOCUS_MODE)

        left = max(0.0, p.left - consumed.left) / s
        right = max(0.0, p.right - consumed.right) / s
        top = max(0.0, p.top - consumed.top) / s
        bottom = max(0.0, p.bottom - consumed.bottom) / s

        if pv_cfg > EPS and o.pre_w_cfg > EPS and o.pre_h_cfg > EPS:
            cropped_w_cfg = max(EPS, o.pre_w_cfg * s)
            cropped_h_cfg = max(EPS, o.pre_h_cfg * s)

            pad_y_cfg = pv_cfg
            pad_x_cfg = pv_cfg * (o.pre_w_cfg / o.pre_h_cfg)

            kx = (cropped_w_cfg + 2.0 * pad_x_cfg) / cropped_w_cfg
            ky = (cropped_h_cfg + 2.0 * pad_y_cfg) / cropped_h_cfg

            left /= max(kx, EPS)
            right /= max(kx, EPS)
            top /= max(ky, EPS)
            bottom /= max(ky, EPS)

        residuals.append(PadEquivalentTrimSides(left=left, right=right, top=top, bottom=bottom))

    return residuals

def _allocate_exact_axis_trims(
    *,
    target_total: float,
    residual_a: float,
    residual_b: float,
    measured_a: float,
    measured_b: float,
    axis_name: str,
) -> Tuple[float, float, str]:
    target_total = max(0.0, float(target_total))
    residual_a = max(0.0, float(residual_a))
    residual_b = max(0.0, float(residual_b))
    measured_a = max(0.0, float(measured_a))
    measured_b = max(0.0, float(measured_b))

    notes: List[str] = []

    base_sum = residual_a + residual_b
    if base_sum > target_total + EPS:
        k = target_total / max(base_sum, EPS)
        trim_a = residual_a * k
        trim_b = residual_b * k
        notes.append(f"{axis_name}: residual>target scaled")
    else:
        trim_a = residual_a
        trim_b = residual_b

    remain = max(0.0, target_total - (trim_a + trim_b))

    cap_a = max(0.0, measured_a - trim_a)
    cap_b = max(0.0, measured_b - trim_b)

    if remain > EPS:
        cap_sum = cap_a + cap_b
        if cap_sum > EPS:
            take = min(remain, cap_sum)
            trim_a += take * (cap_a / cap_sum)
            trim_b += take * (cap_b / cap_sum)
            remain -= take

    if remain > EPS:
        wsum = measured_a + measured_b
        if wsum > EPS:
            add_a = remain * (measured_a / wsum)
        else:
            add_a = remain / 2.0
        add_b = remain - add_a
        trim_a += add_a
        trim_b += add_b
        remain = 0.0
        notes.append(f"{axis_name}: clipped-to-hit-exact-size")

    current_sum = trim_a + trim_b
    delta = target_total - current_sum
    if abs(delta) > 1e-12:
        if delta > 0:
            if measured_a >= measured_b:
                trim_a += delta
            else:
                trim_b += delta
        else:
            if trim_a >= trim_b and (trim_a + delta) >= -EPS:
                trim_a += delta
            elif (trim_b + delta) >= -EPS:
                trim_b += delta
            else:
                s = max(trim_a + trim_b, EPS)
                trim_a = max(0.0, trim_a + delta * (trim_a / s))
                trim_b = max(0.0, target_total - trim_a)

    trim_a = max(0.0, trim_a)
    trim_b = max(0.0, trim_b)

    return trim_a, trim_b, "; ".join(notes)

def apply_exact_normalization_pad_restitution_pass(
    input_pdf_path: Path,
    output_pdf_path: Path,
    *,
    author: str,
    title: str,
    document_target_pad_sides: PadEquivalentTrimSides,
    residual_pad_sides_per_page: List[PadEquivalentTrimSides],
) -> List[PageTransform]:
    edge_fracs = detect_edge_whitespace_fractions_from_pdf(
        input_pdf_path,
        dpi=EDGE_REPOSITION_DETECT_DPI,
        white_threshold=EDGE_REPOSITION_WHITE_THRESHOLD,
    )

    reader = PdfReader(str(input_pdf_path))
    writer = PdfWriter()
    events: List[PageTransform] = []

    if len(edge_fracs) != len(reader.pages):
        raise RuntimeError(
            f"Edge detector returned {len(edge_fracs)} results for {len(reader.pages)} pages."
        )
    if len(residual_pad_sides_per_page) != len(reader.pages):
        raise RuntimeError(
            f"Residual normalization-pad list has {len(residual_pad_sides_per_page)} entries for "
            f"{len(reader.pages)} pages."
        )

    target_total_x = max(0.0, document_target_pad_sides.left + document_target_pad_sides.right)
    target_total_y = max(0.0, document_target_pad_sides.top + document_target_pad_sides.bottom)

    for idx, (page, edgef, residual) in enumerate(
        zip(reader.pages, edge_fracs, residual_pad_sides_per_page), start=1
    ):
        before = page_size_in_config_units(page)

        if edgef is None:
            measured = EdgeWhitespaceFractions(left=1.0, right=1.0, top=1.0, bottom=1.0)
        else:
            measured = edgef

        l_trim, r_trim, x_note = _allocate_exact_axis_trims(
            target_total=target_total_x,
            residual_a=residual.left,
            residual_b=residual.right,
            measured_a=measured.left,
            measured_b=measured.right,
            axis_name="x",
        )
        t_trim, b_trim, y_note = _allocate_exact_axis_trims(
            target_total=target_total_y,
            residual_a=residual.top,
            residual_b=residual.bottom,
            measured_a=measured.top,
            measured_b=measured.bottom,
            axis_name="y",
        )

        if (l_trim + r_trim) > EPS or (t_trim + b_trim) > EPS:
            try:
                crop_page_by_edge_fractions(
                    page,
                    left_frac=l_trim,
                    right_frac=r_trim,
                    top_frac=t_trim,
                    bottom_frac=b_trim,
                )
                action = "normalization-pad-restitution"
                note_bits = [
                    "exact-size restitution (no rescale)",
                    f"targets: X={target_total_x:.6f}w, Y={target_total_y:.6f}h",
                    f"trim: L={l_trim:.6f}w, R={r_trim:.6f}w, T={t_trim:.6f}h, B={b_trim:.6f}h",
                ]
                extra = "; ".join(b for b in [x_note, y_note] if b)
                if extra:
                    note_bits.append(extra)
                note = "; ".join(note_bits)
            except Exception as e:
                action = "none"
                note = f"normalization-pad-restitution skipped: {e}"
        else:
            action = "none"
            note = ""

        after = page_size_in_config_units(page)

        maybe_compress_page_content_streams(page)

        writer.add_page(page)
        events.append(PageTransform(idx, action, before, after, note))

    wipe_and_set_book_metadata(writer, author=author, title=title)
    with open(output_pdf_path, "wb") as f:
        writer.write(f)

    return events

def _compute_vertical_shift(
    top_ws: float,
    bottom_ws: float,
    *,
    mode: str,
    greedy_side: str,
) -> float:
    if mode == "none":
        return 0.0
    if mode == "normal":
        return (top_ws - bottom_ws) / 2.0
    if mode == "greedy":
        if greedy_side == "top":
            return top_ws
        if greedy_side == "bottom":
            return -bottom_ws
        raise ValueError(f"Invalid top/bottom greedy side: {greedy_side}")
    raise ValueError(f"Invalid top/bottom mode: {mode}")

def _compute_horizontal_shift(
    left_ws: float,
    right_ws: float,
    *,
    mode: str,
    greedy_side: str,
) -> float:
    if mode == "none":
        return 0.0
    if mode == "normal":
        return (right_ws - left_ws) / 2.0
    if mode == "greedy":
        if greedy_side == "left":
            return -left_ws
        if greedy_side == "right":
            return right_ws
        raise ValueError(f"Invalid left/right greedy side: {greedy_side}")
    raise ValueError(f"Invalid left/right mode: {mode}")

def _choose_uniform_greedy_fraction(candidates: List[float]) -> float:
    positives = [float(v) for v in candidates if float(v) > EPS]
    if not positives:
        return 0.0
    return min(positives)

def edge_reposition_enabled() -> bool:
    return (
        EDGE_REPOSITION_TOP_BOTTOM_MODE != "none"
        or EDGE_REPOSITION_LEFT_RIGHT_MODE != "none"
    )

def apply_content_reposition_pass(
    input_pdf_path: Path,
    output_pdf_path: Path,
    *,
    author: str,
    title: str,
) -> List[PageTransform]:
    edge_fracs = detect_edge_whitespace_fractions_from_pdf(input_pdf_path)
    reader = PdfReader(str(input_pdf_path))
    writer = PdfWriter()
    events: List[PageTransform] = []

    if len(edge_fracs) != len(reader.pages):
        raise RuntimeError(
            f"Edge detector returned {len(edge_fracs)} results for {len(reader.pages)} pages."
        )

    uniform_tb_greedy_frac = 0.0 # fraction of page height
    uniform_lr_greedy_frac = 0.0 # fraction of page width

    if EDGE_REPOSITION_TOP_BOTTOM_MODE == "greedy":
        tb_candidates: List[float] = []
        for ef in edge_fracs:
            if ef is None:
                continue
            tb_candidates.append(
                ef.top if EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE == "top" else ef.bottom
            )
        uniform_tb_greedy_frac = _choose_uniform_greedy_fraction(tb_candidates)

    if EDGE_REPOSITION_LEFT_RIGHT_MODE == "greedy":
        lr_candidates: List[float] = []
        for ef in edge_fracs:
            if ef is None:
                continue
            lr_candidates.append(
                ef.left if EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE == "left" else ef.right
            )
        uniform_lr_greedy_frac = _choose_uniform_greedy_fraction(lr_candidates)

    for idx, (page, edgef) in enumerate(zip(reader.pages, edge_fracs), start=1):
        before = page_size_in_config_units(page)

        action = "none"
        note = ""

        if edgef is not None and edge_reposition_enabled():
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)
            if w > EPS and h > EPS:
                left_ws = edgef.left * w
                right_ws = edgef.right * w
                top_ws = edgef.top * h
                bottom_ws = edgef.bottom * h

                if EDGE_REPOSITION_LEFT_RIGHT_MODE == "greedy":
                    cap_frac = (
                        edgef.left
                        if EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE == "left"
                        else edgef.right
                    )

                    if (
                        uniform_lr_greedy_frac > EPS
                        and cap_frac + EPS >= uniform_lr_greedy_frac
                    ):
                        if EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE == "left":
                            dx = -(uniform_lr_greedy_frac * w)
                        else:
                            dx = +(uniform_lr_greedy_frac * w)
                    else:
                        dx = 0.0
                else:
                    dx = _compute_horizontal_shift(
                        left_ws,
                        right_ws,
                        mode=EDGE_REPOSITION_LEFT_RIGHT_MODE,
                        greedy_side=EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE,
                    )

                if EDGE_REPOSITION_TOP_BOTTOM_MODE == "greedy":
                    cap_frac = (
                        edgef.top
                        if EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE == "top"
                        else edgef.bottom
                    )

                    if (
                        uniform_tb_greedy_frac > EPS
                        and cap_frac + EPS >= uniform_tb_greedy_frac
                    ):
                        if EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE == "top":
                            dy = +(uniform_tb_greedy_frac * h)
                        else:
                            dy = -(uniform_tb_greedy_frac * h)
                    else:
                        dy = 0.0
                else:
                    dy = _compute_vertical_shift(
                        top_ws,
                        bottom_ws,
                        mode=EDGE_REPOSITION_TOP_BOTTOM_MODE,
                        greedy_side=EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE,
                    )

                if abs(dx) > EPS or abs(dy) > EPS:
                    page.add_transformation(Transformation().translate(dx, dy))

                    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
                    factor_pdf_to_cfg = (uu / 72.0) * config_units_per_inch()

                    action = "content-reposition"

                    extra_bits: List[str] = []
                    if EDGE_REPOSITION_TOP_BOTTOM_MODE == "greedy":
                        extra_bits.append(f"tb-uniform={uniform_tb_greedy_frac:.6f}h")
                    if EDGE_REPOSITION_LEFT_RIGHT_MODE == "greedy":
                        extra_bits.append(f"lr-uniform={uniform_lr_greedy_frac:.6f}w")

                    note = (
                        f"dx={dx * factor_pdf_to_cfg:.3f}{TARGET_UNIT}, "
                        f"dy={dy * factor_pdf_to_cfg:.3f}{TARGET_UNIT}; "
                        f"tb={EDGE_REPOSITION_TOP_BOTTOM_MODE}"
                        + (
                            f":{EDGE_REPOSITION_TOP_BOTTOM_GREEDY_SIDE}"
                            if EDGE_REPOSITION_TOP_BOTTOM_MODE == "greedy"
                            else ""
                        )
                        + f", lr={EDGE_REPOSITION_LEFT_RIGHT_MODE}"
                        + (
                            f":{EDGE_REPOSITION_LEFT_RIGHT_GREEDY_SIDE}"
                            if EDGE_REPOSITION_LEFT_RIGHT_MODE == "greedy"
                            else ""
                        )
                        + (f"; {'; '.join(extra_bits)}" if extra_bits else "")
                    )

        after = page_size_in_config_units(page)

        maybe_compress_page_content_streams(page)

        writer.add_page(page)
        events.append(PageTransform(idx, action, before, after, note))

    wipe_and_set_book_metadata(writer, author=author, title=title)
    with open(output_pdf_path, "wb") as f:
        writer.write(f)

    return events

def merge_page_events(first_pass: List[PageTransform], second_pass: List[PageTransform]) -> List[PageTransform]:
    if len(first_pass) != len(second_pass):
        return list(first_pass) + list(second_pass)

    merged: List[PageTransform] = []
    for e1, e2 in zip(first_pass, second_pass):
        if e2.action == "none":
            merged.append(e1)
            continue
        if e1.action == "none":
            merged.append(e2)
            continue

        combo_action = f"{e1.action}+{e2.action}"
        combo_note = "; ".join(n for n in [e1.note, e2.note] if n)
        merged.append(
            PageTransform(
                page_num=e1.page_num,
                action=combo_action,
                before_size=e1.before_size,
                after_size=e2.after_size,
                note=combo_note,
            )
        )
    return merged

def wipe_and_set_book_metadata(writer: PdfWriter, *, author: str, title: str) -> None:
    writer.metadata = {
        "/Title": title,
        "/Author": author,
        "/Creator": "pdf",
        "/Producer": "pdf",
    }

    xmp = XmpInformation.create()
    xmp.dc_title = {"x-default": title}
    xmp.dc_creator = [author]
    xmp.dc_format = "application/pdf"
    xmp.xmp_creator_tool = "pdf"
    xmp.pdf_producer = "pdf"
    writer.xmp_metadata = xmp

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize PDF (or auto-convert EPUB) to configured target page size, "
            "rotate landscape pages to portrait, "
            "pad to target aspect ratio and scale to target, "
            "apply content-aware uniform smart crop (with optional ratio-abiding post-crop padding) and rescale to target, "
            "optionally restore exact pre-normalization-pad page dimensions, "
            "optionally redistribute whitespace, "
            "scrub metadata, set author."
        )
    )
    parser.add_argument("input", help="Input file path (.pdf or .epub)")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print("Error: input file does not exist.", file=sys.stderr)
        return 2

    suffix = in_path.suffix.lower()
    if suffix not in {".pdf", ".epub"}:
        print("Error: input must be .pdf or .epub", file=sys.stderr)
        return 2

    out_path = next_windows_paren_name(in_path, out_suffix=".pdf")

    pdf_to_process: Path
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None

    try:
        if suffix == ".epub":
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="epub_to_pdf_")
            tmp_pdf = Path(temp_dir_obj.name) / f"{in_path.stem}.pdf"
            convert_epub_to_pdf(in_path, tmp_pdf)
            pdf_to_process = tmp_pdf
        else:
            pdf_to_process = in_path

        reader = PdfReader(str(pdf_to_process))
        if getattr(reader, "is_encrypted", False):
            try:
                result = reader.decrypt("")
            except Exception:
                result = 0
            if not result:
                print("Error: PDF is encrypted and requires a password.", file=sys.stderr)
                return 3

        writer = PdfWriter()
        norm_events: List[PageTransform] = []
        per_page_norm_pad_sides: List[PadEquivalentTrimSides] = []

        for idx, page in enumerate(reader.pages, start=1):
            before = page_size_in_config_units(page)

            out_page = page
            if ENABLE_NORMALIZATION:
                out_page, action, note, norm_pad_sides = normalize_page_to_target(page)
            else:
                action, note, norm_pad_sides = ("none", "normalization disabled", PadEquivalentTrimSides())

            after = page_size_in_config_units(out_page)
            per_page_norm_pad_sides.append(norm_pad_sides)

            maybe_compress_page_content_streams(out_page)

            writer.add_page(out_page)
            norm_events.append(
                PageTransform(
                    page_num=idx,
                    action=action,
                    before_size=before,
                    after_size=after,
                    note=note,
                )
            )

        input_title = (getattr(reader.metadata, "title", None) if reader.metadata else None) or in_path.stem
        title = TITLE or input_title
        doc_norm_pad_target_sides = choose_document_normalization_pad_target_sides(per_page_norm_pad_sides)

        if temp_dir_obj is None:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="pdf_norm_")
        temp_root = Path(temp_dir_obj.name)

        normalized_pdf = temp_root / f"{in_path.stem}.normalized.pdf"
        wipe_and_set_book_metadata(writer, author=AUTHOR, title=title)
        with open(normalized_pdf, "wb") as f:
            writer.write(f)

        all_events = list(norm_events)

        if ENABLE_SMART_CROP:
            smart_cropped_pdf = temp_root / f"{in_path.stem}.smart_cropped.pdf"
            smart_crop_events, smart_crop_outcomes = apply_smart_crop_pass(
                normalized_pdf,
                smart_cropped_pdf,
                author=AUTHOR,
                title=title,
            )
            all_events = merge_page_events(all_events, smart_crop_events)
            current_pdf = smart_cropped_pdf
        else:
            smart_crop_outcomes = []
            current_pdf = normalized_pdf

        if (
            ENABLE_NORMALIZATION
            and ENABLE_SMART_CROP
            and REMOVE_NORMALIZATION_PAD_RESIDUAL_AFTER_SMART_CROP
        ):
            residuals = compute_residual_normalization_pad_after_smart_crop(
                per_page_norm_pad_sides,
                smart_crop_outcomes,
            )

            if (
                doc_norm_pad_target_sides.left > EPS
                or doc_norm_pad_target_sides.right > EPS
                or doc_norm_pad_target_sides.top > EPS
                or doc_norm_pad_target_sides.bottom > EPS
            ):
                restitution_pdf = temp_root / f"{in_path.stem}.norm_pad_restitution.pdf"
                restitution_events = apply_exact_normalization_pad_restitution_pass(
                    current_pdf,
                    restitution_pdf,
                    author=AUTHOR,
                    title=title,
                    document_target_pad_sides=doc_norm_pad_target_sides,
                    residual_pad_sides_per_page=residuals,
                )
                all_events = merge_page_events(all_events, restitution_events)
                current_pdf = restitution_pdf

        if edge_reposition_enabled():
            reposition_pdf = temp_root / f"{in_path.stem}.repositioned.pdf"
            reposition_events = apply_content_reposition_pass(
                current_pdf,
                reposition_pdf,
                author=AUTHOR,
                title=title,
            )
            all_events = merge_page_events(all_events, reposition_events)
            current_pdf = reposition_pdf

        current_pdf.replace(out_path)

        print(f"Wrote: {out_path}")
        print_transform_report(all_events, out_path)
        return 0

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

if __name__ == "__main__":
    raise SystemExit(main())

