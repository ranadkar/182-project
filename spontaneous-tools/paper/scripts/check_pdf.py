"""Validate the built paper and optionally rasterize every page for visual review."""
import argparse
import re
import statistics
import subprocess
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


PAPER = Path(__file__).resolve().parents[1]


def render(document, destination):
    destination.mkdir(parents=True, exist_ok=True)
    columns = 3
    tile_width, tile_height = 320, 434
    sheet = Image.new("RGB", (columns * tile_width,
                              ((len(document) + columns - 1) // columns) * tile_height), "#d7dadd")
    labels = ImageDraw.Draw(sheet)
    for index, page in enumerate(document):
        pixels = page.get_pixmap(dpi=144, alpha=False)
        name = f"page-{index + 1:02d}.png"
        pixels.save(destination / name)
        with Image.open(destination / name) as picture:
            picture.thumbnail((310, 402))
            x = (index % columns) * tile_width + 5
            y = (index // columns) * tile_height + 5
            sheet.paste(picture, (x, y))
            labels.text((x, y + 406), f"Page {index + 1}", fill="#142028")
    sheet.save(destination / "contact-sheet.png")


FRAME = pymupdf.Rect(108, 72, 504, 720)
CENTER = 306


def check_style_frame():
    # A ragged last page cannot reveal its bottom margin from glyph bounds. Fill the
    # actual available region in a separate typeset probe, using the production style.
    source = ('#import "style.typ": paper-style\n#show: paper-style\n'
              '#rect(width: 100%, height: 100%, stroke: none, fill: rgb("#0000ff"))')
    run = subprocess.run(["typst", "compile", "--root", ".", "-", "-"], cwd=PAPER,
                         input=source.encode(), stdout=subprocess.PIPE, check=True)
    with pymupdf.open(stream=run.stdout, filetype="pdf") as probe:
        assert len(probe) == 1
        (rect,) = [g["rect"] for g in probe[0].get_drawings() if g["type"] == "f" and g["items"][0][0] == "re"]
        assert all(abs(u-v) < .02 for u, v in zip(rect, FRAME)), f"Typeset frame: {rect}"
    return rect


def check_page(page, bibliography_page):
    assert page.rect == pymupdf.Rect(0, 0, 612, 792)
    number = page.number + 1
    full_lines, right_starts, ruler, footer = [], [], [], []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            content = "".join(s["text"] for s in spans).strip()
            if not content:
                continue
            x, y, right, _ = line["bbox"]
            if right < FRAME.x0:
                assert content.isdigit() and "Heros" in spans[0]["font"], (number, content)
                ruler.append(int(content))
                continue
            if y > FRAME.y1 + 10:
                footer.append(content)
                assert abs((x + right) / 2 - CENTER) < .1
                continue
            # A full line of prose must itself cross the centre. Pooling adjacent
            # columns by their y-coordinate could falsely pass a two-column PDF.
            font_size = 9 if bibliography_page else 10
            if len(content) >= 34 and "Termes" in spans[0]["font"] and abs(spans[0]["size"] - font_size) < .1:
                if x < FRAME.x0 + 37 and right >= FRAME.x1 - 6:
                    full_lines.append(spans[0]["origin"][1])
                elif x >= CENTER:
                    right_starts.append(content)
    assert ruler == list(range(54 * (number-1), 54 * number)), (number, "review ruler")
    assert footer == [str(number)], (number, "page number")
    assert len(full_lines) >= 4 and not right_starts, (number, "prose columns", right_starts)

    # PDF text boxes include ascenders and descenders outside the nominal cap box.
    # Actual words must fit the horizontal frame. Check graphics and ink as well
    # (MuPDF expands a painted path's bounding box slightly).
    words = [w for w in page.get_text("words") if w[2] > FRAME.x0 and w[1] < FRAME.y1 + 10]
    assert all(FRAME.x0 - .5 <= w[0] <= w[2] <= FRAME.x1 + .5 and
               FRAME.y0 - 3 <= w[1] <= w[3] <= FRAME.y1 + 3 for w in words), number
    for kind, box in page.get_bboxlog():
        if box[2] < FRAME.x0 or box[1] > FRAME.y1 + 10:
            assert kind in ("fill-text", "ignore-text"), (number, kind, box)
            continue
        assert (FRAME.x0-3 <= box[0] <= box[2] <= FRAME.x1+3 and
                FRAME.y0-3 <= box[1] <= box[3] <= FRAME.y1+3), (number, kind, box)
    baselines = sorted(set(full_lines))
    steps = [b-a for a, b in zip(baselines, baselines[1:]) if 8 < b-a < 14]
    return len(full_lines), steps


def check_front_matter(page):
    lines = [l for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])]
    title = [l for l in lines if abs(l["spans"][0]["size"]-17) < .01]
    assert len(title) == 2
    for line in title:
        left, _, right, _ = line["bbox"]
        assert abs((left + right)/2 - CENTER) < .1 and "Bold" in line["spans"][0]["font"]
    rules = [g["rect"] for g in page.get_drawings() if g["type"] == "f" and
             abs(g["rect"].width - FRAME.width) < .1]
    assert [round(r.height, 2) for r in rules] == [4, 1]
    (heading,) = page.search_for("Abstract")
    first = min((line for line in lines if line["bbox"][1] >= heading.y1
                 and line["bbox"][0] >= FRAME.x0), key=lambda line: line["bbox"][1])
    assert abs(first["bbox"][0] - (FRAME.x0 + 36)) < .1
    abstract = page.get_text("words", clip=pymupdf.Rect(90, first["bbox"][1] - .1, 522,
                              page.search_for("Introduction")[0].y0-1))
    assert abs(max(w[2] for w in abstract) - (FRAME.x1-36)) < .5


def check(document):
    frame = check_style_frame()
    pages = [p.get_text().replace("\u00ad", "").replace("\xa0", " ") for p in document]
    (references,) = (i for i, s in enumerate(pages) if "References" in s.splitlines())
    (appendix,) = (i for i, s in enumerate(pages) if "Appendix A. Evaluation protocol" in s.splitlines())
    assert 5 <= references <= 10 and references < appendix
    bib_count = len(re.findall(r"^@\w+\{", (PAPER / "references.bib").read_text(), re.M))
    entries = re.findall(r"^\[(\d+)\](?: |$)", "\n".join(pages[references:appendix]), re.M)
    assert sorted(map(int, entries)) == list(range(1, bib_count + 1))
    embedded = {font[0] for page in document for font in page.get_fonts()}
    assert all(document.extract_font(xref)[-1] for xref in embedded)
    body = " ".join(" ".join(pages[:references]).split())
    for heading in ["Introduction", "Background and related work", "Methodology", "Results",
                    "Discussion and limitations", "Conclusion"]:
        assert heading.lower() in body.lower(), heading
    check_front_matter(document[0])
    full_lines, steps = 0, []
    for i, page in enumerate(document):
        count, distances = check_page(page, references <= i < appendix)
        full_lines += count
        if i < references:
            steps.extend(distances)
        for link in page.get_links():
            if link["kind"] == pymupdf.LINK_GOTO:
                assert 0 <= link["page"] < len(document)
    baseline = statistics.median(steps)
    assert abs(baseline - 11) < .05, baseline
    print(f"US Letter 612 × 792 pt; one {frame.width:g} × {frame.height:g} pt body column. "
          f"Margins 108 pt on each side and 72 pt at top and bottom.\n"
          f"{full_lines} full-width prose lines; median core baseline spacing {baseline:.2f} pt. "
          "Title, abstract inset, review ruler, page numbers and font embedding verified.")
    print(f"{references} core pages, {appendix - references} reference page(s), "
          f"{len(document) - appendix} appendix pages; {bib_count} numbered references. "
          f"Total: {len(document)} pages.")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--render-dir", type=Path)
    options = args.parse_args()
    with pymupdf.open(PAPER / "paper.pdf") as pdf:
        check(pdf)
        if options.render_dir:
            render(pdf, options.render_dir)
