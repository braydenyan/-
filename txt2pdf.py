# Flags:
# --overwrite | overwrite existing output file
# --reverse   | .pdf -> .txt instead of .txt -> .pdf
# --nocheck   | skip scanning for consecutive empty lines

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import fitz # PyMuPDF # type: ignore

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter # font(s) of choice
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =========================
# CONFIG
# =========================

# i/o paths
# forward: TXT_PATH -> PDF_PATH
# reverse: PDF_PATH -> TXT_PATH
TXT_PATH = r"./txt.txt"
PDF_PATH = r"./pdf.pdf"

# primary font: standard font for most characters
PRIMARY_FONT_TTF_PATH = r"C:\Windows\Fonts\georgia.ttf"
PRIMARY_FONT_NAME = "PrimaryFont"

# fallback font: unicode font for unrecognized characters
FALLBACK_FONT_TTF_PATH = str(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts" / "DejaVuSerif.ttf")
FALLBACK_FONT_NAME = "FallbackFont"

# built-in fallback names (used only if ttf files are missing)
PRIMARY_BUILTIN_NAME = "Georgia"
FALLBACK_BUILTIN_NAME = "DejaVu Serif"

FONT_SIZE = 9 # in points
LINE_SPACING = 1.2 # multiplier on font size (e.g. 1.2 = 20% extra leading)

PAGE_WIDTH, PAGE_HEIGHT = (842, 595) # either set to standard ("letter") or explicit (612, 792)

# page margins (in points)
MARGIN_LEFT = 36 # half-inch
MARGIN_RIGHT = 36
MARGIN_TOP = 36
MARGIN_BOTTOM = 36

# sentinel characters
SENTINEL_WRAP = "§" # for forcibly wrapped lines
SENTINEL_BLANK_NEAR_BREAK = "¶" # for blank lines adjacent to a page break

# =========================
# FORWARD: .TXT -> .PDF
# =========================

class OutputLine:
    def __init__(self, text, forced_wrap=False, is_blank=False):
        self.text = text
        self.forced_wrap = forced_wrap
        self.is_blank = is_blank
        self.blank_adjacent_to_break = False

def register_fonts():
    # primary font
    primary_name = PRIMARY_BUILTIN_NAME
    if PRIMARY_FONT_TTF_PATH:
        primary_path = Path(PRIMARY_FONT_TTF_PATH)
        try:
            if not primary_path.is_file():
                raise FileNotFoundError(f"Primary TTF font file not found: {primary_path}")
            pdfmetrics.registerFont(TTFont(PRIMARY_FONT_NAME, str(primary_path)))
            primary_name = PRIMARY_FONT_NAME
        except Exception as e:
            print(f"[WARN] Could not register primary TTF font '{primary_path}': {e}")
            print(f"[WARN] Falling back to built-in font '{PRIMARY_BUILTIN_NAME}'.")
    else:
        print(f"[INFO] No primary TTF font specified. Using built-in '{PRIMARY_BUILTIN_NAME}'.")

    # fallback font
    fallback_name = primary_name
    if FALLBACK_FONT_TTF_PATH:
        fallback_path = Path(FALLBACK_FONT_TTF_PATH)
        try:
            if not fallback_path.is_file():
                raise FileNotFoundError(f"Fallback TTF font file not found: {fallback_path}")
            pdfmetrics.registerFont(TTFont(FALLBACK_FONT_NAME, str(fallback_path)))
            fallback_name = FALLBACK_FONT_NAME
        except Exception as e:
            print(f"[WARN] Could not register fallback TTF font '{fallback_path}': {e}")
            print(f"[WARN] Fallback will use primary font '{primary_name}'.")
    else:
        print(f"[INFO] No fallback TTF font specified. Fallback will use primary font '{primary_name}'.")

    return primary_name, fallback_name

def wrap_line_with_flags(line: str, font_name: str, font_size: int, max_width: float):
    if line.strip() == "":
        return [OutputLine("", forced_wrap=False, is_blank=True)]

    tokens = re.findall(r'\s+|\S+', line)
    segments = []
    current = ""

    def append_segment(seg_text):
        segments.append(
            OutputLine(seg_text, forced_wrap=(len(segments) > 0), is_blank=False)
        )

    for tok in tokens:
        if current == "":
            current = tok
            continue

        candidate = current + tok
        width = pdfmetrics.stringWidth(candidate, font_name, font_size)

        if width <= max_width:
            current = candidate
        else:
            append_segment(current)
            current = tok

            while pdfmetrics.stringWidth(current, font_name, font_size) > max_width:
                cut = 1
                while (
                    cut <= len(current)
                    and pdfmetrics.stringWidth(current[:cut], font_name, font_size) <= max_width
                ):
                    cut += 1
                if cut == 1:
                    cut = 2
                prefix = current[:cut - 1]
                append_segment(prefix)
                current = current[cut - 1:]

    if current:
        append_segment(current)

    return segments

def should_use_fallback(ch: str, fallback_differs: bool) -> bool:
    if not fallback_differs:
        return False
    if ch.isspace():
        return False
    return ord(ch) > 0xFF

def draw_text_with_fallback(
    c,
    text: str,
    x: float,
    y: float,
    primary_font: str,
    fallback_font: str,
    font_size: float,
    fallback_differs: bool,
):
    if not text:
        return

    segments = []
    current_font = None
    current_text = ""

    for ch in text:
        use_fallback = should_use_fallback(ch, fallback_differs)
        font = fallback_font if use_fallback else primary_font

        if current_font is None:
            current_font = font
            current_text = ch
        elif font == current_font:
            current_text += ch
        else:
            segments.append((current_font, current_text))
            current_font = font
            current_text = ch

    if current_text:
        segments.append((current_font, current_text))

    x_cursor = x
    for font, seg_text in segments:
        c.setFont(font, font_size)
        c.drawString(x_cursor, y, seg_text)
        w = pdfmetrics.stringWidth(seg_text, font, font_size)
        x_cursor += w

def txt_to_pdf(input_path: Path, output_path: Path):
    primary_font_name, fallback_font_name = register_fonts()
    fallback_differs = (fallback_font_name != primary_font_name)

    raw_text = input_path.read_text(encoding="utf-8", errors="replace")
    text = raw_text.replace("\t", "    ")

    c = canvas.Canvas(str(output_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.setTitle(input_path.name)

    available_width = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    available_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    line_height = FONT_SIZE * LINE_SPACING
    if line_height <= 0:
        raise ValueError("LINE_SPACING and FONT_SIZE must produce a positive line height.")

    max_lines_per_page = max(int(available_height // line_height), 1)

    output_lines = []
    for raw_line in text.splitlines():
        output_lines.extend(
            wrap_line_with_flags(raw_line, primary_font_name, FONT_SIZE, available_width)
        )

    total_lines = len(output_lines)

    for i, ol in enumerate(output_lines):
        if not ol.is_blank:
            continue

        if total_lines == 0:
            continue

        page_index = i // max_lines_per_page
        pos_in_page = i % max_lines_per_page

        if pos_in_page == max_lines_per_page - 1 and i < total_lines - 1:
            ol.blank_adjacent_to_break = True

        if pos_in_page == 0 and i > 0:
            ol.blank_adjacent_to_break = True

    current_y = PAGE_HEIGHT - MARGIN_TOP
    current_page_index = 0

    space_width = pdfmetrics.stringWidth(" ", primary_font_name, FONT_SIZE)

    for idx, ol in enumerate(output_lines):
        page_index = idx // max_lines_per_page

        if page_index != current_page_index:
            c.showPage()
            current_page_index = page_index
            current_y = PAGE_HEIGHT - MARGIN_TOP

        sentinel = None
        if ol.forced_wrap:
            sentinel = SENTINEL_WRAP
        elif ol.is_blank and ol.blank_adjacent_to_break:
            sentinel = SENTINEL_BLANK_NEAR_BREAK

        x_text = MARGIN_LEFT

        if sentinel:
            sentinel_width = pdfmetrics.stringWidth(sentinel, primary_font_name, FONT_SIZE)
            sentinel_x = x_text - (sentinel_width + space_width)
            if sentinel_x < 2:
                sentinel_x = 2
            c.setFont(primary_font_name, FONT_SIZE)
            c.drawString(sentinel_x, current_y, sentinel)

        if ol.text:
            draw_text_with_fallback(
                c,
                ol.text,
                x_text,
                current_y,
                primary_font_name,
                fallback_font_name,
                FONT_SIZE,
                fallback_differs,
            )

        current_y -= line_height

    if total_lines == 0:
        c.showPage()

    c.save()

# =========================
# REVERSE: .PDF -> .TXT
# =========================

@dataclass
class PhysicalLine:
    page_index: int
    y: float # vertical position
    text: str # text content with sentinels removed
    has_wrap: bool # contains SENTINEL_WRAP
    has_blank_sentinel: bool # contains SENTINEL_BLANK_NEAR_BREAK

@dataclass
class ReconstructedLine:
    text: str
    forced_wrap: bool
    is_blank: bool

def extract_physical_lines(pdf_path: str) -> List[PhysicalLine]:
    doc = fitz.open(pdf_path)
    physical_lines: List[PhysicalLine] = []

    for page_index, page in enumerate(doc):
        data = page.get_text("dict")

        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                y_vals = [span["origin"][1] for span in spans if "origin" in span]
                if not y_vals:
                    continue
                y = sum(y_vals) / len(y_vals)

                has_wrap = False
                has_blank = False
                text_chars: List[str] = []

                for span in spans:
                    t = span.get("text", "")
                    for ch in t:
                        if ch == SENTINEL_WRAP:
                            has_wrap = True
                            continue
                        if ch == SENTINEL_BLANK_NEAR_BREAK:
                            has_blank = True
                            continue
                        text_chars.append(ch)

                text = "".join(text_chars)

                if has_wrap:
                    text = text.lstrip(" ")

                if not text and not (has_wrap or has_blank):
                    continue

                physical_lines.append(
                    PhysicalLine(
                        page_index=page_index,
                        y=y,
                        text=text,
                        has_wrap=has_wrap,
                        has_blank_sentinel=has_blank,
                    )
                )

    doc.close()
    return physical_lines

def estimate_line_gap(physical_lines: List[PhysicalLine]) -> float:
    if not physical_lines:
        raise RuntimeError("No text found in PDF.")

    gaps: List[float] = []
    by_page: Dict[int, List[PhysicalLine]] = {}

    for pl in physical_lines:
        by_page.setdefault(pl.page_index, []).append(pl)

    for page_index, lines in by_page.items():
        lines_sorted = sorted(lines, key=lambda l: l.y)
        for a, b in zip(lines_sorted, lines_sorted[1:]):
            dy = abs(b.y - a.y)
            if dy > 0.1:
                gaps.append(dy)

    if not gaps:
        return 1.0

    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    return median_gap

def build_output_lines(physical_lines: List[PhysicalLine], base_gap: float) -> List[ReconstructedLine]:
    by_page: Dict[int, List[PhysicalLine]] = {}
    for pl in physical_lines:
        by_page.setdefault(pl.page_index, []).append(pl)

    output_lines: List[ReconstructedLine] = []

    for page_index in sorted(by_page.keys()):
        page_lines = by_page[page_index]
        page_lines_sorted = sorted(page_lines, key=lambda l: l.y)

        prev: PhysicalLine | None = None
        for pl in page_lines_sorted:
            if prev is not None and base_gap > 0:
                gap = abs(pl.y - prev.y)
                steps = int(round(gap / base_gap))
                if steps < 1:
                    steps = 1
                blanks_between = steps - 1

                for _ in range(blanks_between):
                    output_lines.append(
                        ReconstructedLine(text="", forced_wrap=False, is_blank=True)
                    )

            is_blank = (pl.text == "")
            output_lines.append(
                ReconstructedLine(
                    text=pl.text,
                    forced_wrap=pl.has_wrap,
                    is_blank=is_blank,
                )
            )
            prev = pl

    return output_lines

def reconstruct_text(output_lines: List[ReconstructedLine]) -> str:
    logical_lines: List[str] = []
    buffer: str | None = None

    for ol in output_lines:
        if ol.is_blank:
            if buffer is not None:
                logical_lines.append(buffer)
                buffer = None
            logical_lines.append("")
            continue

        if not ol.forced_wrap:
            if buffer is not None:
                logical_lines.append(buffer)
            buffer = ol.text
        else:
            seg = ol.text
            if buffer is None:
                buffer = seg
            else:
                left = buffer.rstrip(" ")
                right = seg.lstrip(" ")

                if left and right:
                    buffer = left + " " + right
                else:
                    buffer = left + right

    if buffer is not None:
        logical_lines.append(buffer)

    return "\n".join(logical_lines)

def pdf_to_txt(input_pdf: Path, output_txt: Path) -> None:
    physical_lines = extract_physical_lines(str(input_pdf))
    if not physical_lines:
        raise RuntimeError("No lines extracted from PDF")

    base_gap = estimate_line_gap(physical_lines)
    output_lines = build_output_lines(physical_lines, base_gap)
    text = reconstruct_text(output_lines)

    with output_txt.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)

# =========================
# CONSECUTIVE EMPTY LINES
# =========================

def report_consecutive_empty_lines(input_path: Path) -> None:
    try:
        raw_text = input_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[WARN] Could not read text file for empty-line check: {e}")
        return

    lines = raw_text.splitlines()
    consecutive_ranges = []

    start = None
    for idx, line in enumerate(lines, start=1):
        if line.strip() == "":
            if start is None:
                start = idx
        else:
            if start is not None:
                end = idx - 1
                if end - start + 1 >= 2:
                    consecutive_ranges.append((start, end))
                start = None

    if start is not None:
        end = len(lines)
        if end - start + 1 >= 2:
            consecutive_ranges.append((start, end))

    if not consecutive_ranges:
        return

    print("Consecutive empty lines detected in input text:")
    for start, end in consecutive_ranges:
        if start == end:
            print(f"  line {start}")
        else:
            print(f"  lines {start}-{end}")

# =========================
# OUTPUT PATH
# =========================

def resolve_output_path(base_path: Path, overwrite: bool) -> Path:
    if overwrite or not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 1
    while True:
        candidate = parent / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

# =========================
# MAIN
# =========================

def main():
    args = sys.argv[1:]
    reverse_mode = False
    overwrite = False
    nocheck = False
    unknown: List[str] = []

    for arg in args:
        if arg.lower() == "--reverse":
            reverse_mode = True
        elif arg.lower() == "--overwrite":
            overwrite = True
        elif arg.lower() == "--nocheck":
            nocheck = True
        else:
            unknown.append(arg)

    if unknown:
        prog = Path(sys.argv[0]).name
        print(f"[WARN] Unrecognized arguments {unknown!r} will be ignored. Usage:")
        print(f"  {prog}              # .txt -> .pdf | using configured TXT_PATH / PDF_PATH")
        print(f"  {prog} --overwrite  # .txt -> .pdf | overwrite existing PDF_PATH")
        print(f"  {prog} --nocheck    # .txt -> .pdf | skip consecutive-empty-line check")
        print(f"  {prog} --reverse    # .pdf -> .txt | using configured PDF_PATH / TXT_PATH")

    if reverse_mode:
        input_pdf = Path(PDF_PATH)
        if not input_pdf.is_file():
            print(f"Error: PDF not found for reverse mode: {input_pdf}")
            return

        base_txt_path = Path(TXT_PATH)
        output_txt = resolve_output_path(base_txt_path, overwrite)

        try:
            pdf_to_txt(input_pdf, output_txt)
            print(f"Reconstructed text written to: {output_txt}")
        except Exception as e:
            print(f"Failed to reconstruct file: {e}")
        return

    input_txt = Path(TXT_PATH)
    if not input_txt.is_file():
        print(f"Error: text file not found: {input_txt}")
        return

    if not nocheck:
        report_consecutive_empty_lines(input_txt)

    base_pdf_path = Path(PDF_PATH)
    output_pdf = resolve_output_path(base_pdf_path, overwrite)

    try:
        txt_to_pdf(input_txt, output_pdf)
        print(f"PDF written to: {output_pdf}")
    except Exception as e:
        print(f"Failed to convert file: {e}")

if __name__ == "__main__":
    main()




