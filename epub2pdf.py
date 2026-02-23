from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.xmp import XmpInformation

# =========================
# CONFIG
# =========================

AUTHOR: str = "Author" # many e-readers load author data from title: "Book - Author"

LETTER_W_IN = 8.5
LETTER_H_IN = 11.0
LETTER_LONG_SHORT = LETTER_H_IN / LETTER_W_IN # ≈ 1.29

RATIO_TOL_REL = 0.005  # 0.5%
SIZE_TOL_IN = 0.02     # +/- 0.02 in.
EPS = 1e-9

# =========================
# END CONFIG
# =========================

@dataclass(frozen=True)
class PageTransform:
    page_num: int
    action: str
    before_in: Tuple[float, float]
    after_in: Tuple[float, float]
    note: str = ""

def format_inches(w: float, h: float) -> str:
    return f"{w:.2f}×{h:.2f} in"

def print_transform_report(events: List[PageTransform], out_path: Path) -> None:
    total = len(events)
    changed = [e for e in events if e.action != "none"]
    n_changed = len(changed)
    n_scale = sum(1 for e in changed if e.action == "scale")
    n_pad_scale = sum(1 for e in changed if e.action == "pad+scale")

    print("\n=== Report ===")
    print(f"Output: {out_path}")
    print(f"Pages:  {total}")

    if n_changed == 0:
        print("No transformations necessary.")
        return
    else:
        print(f"Scaled (Letter):           {n_scale}")
        print(f"Padded + Scaled (Letter):  {n_pad_scale}")   

    print("\nChanged pages:")

    CAP = 10
    for e in changed[:CAP]:
        note = f" — {e.note}" if e.note else ""
        print(
            f"  p{e.page_num}: {format_inches(*e.before_in)} -> {format_inches(*e.after_in)} [{e.action}]{note}"
        )
    if n_changed > CAP:
        print(f"  …and {n_changed - CAP} more changed pages.")

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

def units_per_inch(user_unit: float) -> float:
    return 72.0 / float(user_unit)

def page_size_in(page) -> Tuple[float, float]:
    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
    w_units = float(page.mediabox.width)
    h_units = float(page.mediabox.height)
    w_in = (w_units * uu) / 72.0
    h_in = (h_units * uu) / 72.0
    return w_in, h_in

def is_letter_size(w_in: float, h_in: float) -> bool:
    portrait = (abs(w_in - LETTER_W_IN) <= SIZE_TOL_IN and abs(h_in - LETTER_H_IN) <= SIZE_TOL_IN)
    landscape = (abs(w_in - LETTER_H_IN) <= SIZE_TOL_IN and abs(h_in - LETTER_W_IN) <= SIZE_TOL_IN)
    return portrait or landscape

def ratio_matches_letter(w_in: float, h_in: float) -> bool:
    long_side = max(w_in, h_in)
    short_side = max(min(w_in, h_in), EPS)
    r = long_side / short_side
    return abs(r - LETTER_LONG_SHORT) / LETTER_LONG_SHORT <= RATIO_TOL_REL

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

def expand_canvas_centered(page, new_w_units: float, new_h_units: float) -> None:
    old_w = float(page.mediabox.width)
    old_h = float(page.mediabox.height)
    if new_w_units + EPS < old_w or new_h_units + EPS < old_h:
        raise ValueError("expand_canvas_centered() only supports expanding.")

    llx = float(page.mediabox.left)
    lly = float(page.mediabox.bottom)
    urx = llx + float(new_w_units)
    ury = lly + float(new_h_units)

    set_all_page_boxes(page, llx, lly, urx, ury)

    tx = (float(new_w_units) - old_w) / 2.0
    ty = (float(new_h_units) - old_h) / 2.0
    if abs(tx) > EPS or abs(ty) > EPS:
        page.add_transformation(Transformation().translate(tx, ty))

def scale_to_letter(page) -> None:
    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
    upi = units_per_inch(uu)

    w_units = float(page.mediabox.width)
    h_units = float(page.mediabox.height)

    if h_units >= w_units:
        target_w = LETTER_W_IN * upi
        target_h = LETTER_H_IN * upi
    else:
        target_w = LETTER_H_IN * upi
        target_h = LETTER_W_IN * upi

    page.scale_to(target_w, target_h)

def normalize_page_to_letter(page) -> Tuple[str, str]:
    if hasattr(page, "transfer_rotation_to_content"):
        page.transfer_rotation_to_content()

    w_in, h_in = page_size_in(page)
    if is_letter_size(w_in, h_in):
        return "none", ""

    if ratio_matches_letter(w_in, h_in):
        scale_to_letter(page)
        return "scale", "aspect ratio matched Letter; scaled only"

    uu = float(getattr(page, "user_unit", 1.0) or 1.0)
    upi = units_per_inch(uu)

    w_units = float(page.mediabox.width)
    h_units = float(page.mediabox.height)

    long_u = max(w_units, h_units)
    short_u = max(min(w_units, h_units), EPS)
    r = long_u / short_u

    if r < LETTER_LONG_SHORT:
        new_w = w_units
        new_h = max(h_units, w_units * LETTER_LONG_SHORT)
        pad_note = "padded height (square)"
    else:
        if h_units >= w_units:
            new_h = h_units
            new_w = h_units / LETTER_LONG_SHORT
            pad_note = "padded width"
        else:
            new_w = w_units
            new_h = w_units / LETTER_LONG_SHORT
            pad_note = "padded height"

    new_w = max(new_w, w_units)
    new_h = max(new_h, h_units)

    old_w_in = (w_units * uu) / 72.0
    old_h_in = (h_units * uu) / 72.0
    new_w_in = (new_w * uu) / 72.0
    new_h_in = (new_h * uu) / 72.0
    dw_in = max(0.0, new_w_in - old_w_in)
    dh_in = max(0.0, new_h_in - old_h_in)

    expand_canvas_centered(page, new_w, new_h)
    scale_to_letter(page)

    return "pad+scale", f"{pad_note}; added {dw_in:.2f}in W, {dh_in:.2f}in H before scale"

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
    parser = argparse.ArgumentParser(description="Normalize a PDF (or auto-convert ePUB) to true Letter size, scrub metadata, set author")
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
                reader.decrypt("")
            except Exception:
                print("Error: PDF is encrypted and could not be decrypted.", file=sys.stderr)
                return 3

        writer = PdfWriter()
        events: List[PageTransform] = []

        for idx, page in enumerate(reader.pages, start=1):
            before = page_size_in(page)
            action, note = normalize_page_to_letter(page)
            after = page_size_in(page)

            try:
                page.compress_content_streams()
            except Exception:
                pass

            writer.add_page(page)
            events.append(PageTransform(page_num=idx, action=action, before_in=before, after_in=after, note=note))

        title = (getattr(reader.metadata, "title", None) if reader.metadata else None) or in_path.stem
        wipe_and_set_book_metadata(writer, author=AUTHOR, title=title)

        with open(out_path, "wb") as f:
            writer.write(f)

        print(f"Wrote: {out_path}")
        print_transform_report(events, out_path)
        return 0

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

if __name__ == "__main__":
    raise SystemExit(main())