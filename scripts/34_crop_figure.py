"""
Crop the compiled TikZ diagram (manuscript/figures/Figure_2.tex -> .pdf) to its
drawing, so it can be supplied as a separate vector artwork file.

  pdflatex -jobname=Figure_2_full Figure_2   (in figures/)
  python 34_crop_figure.py                   -> Figure_2.pdf, the cropped page

The page is re-placed as a vector form XObject (no rasterising) with a 2 pt
margin around the union of every drawing and text span.
"""

import os
import pymupdf as fitz

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
MARGIN = 2.0


def main():
    src = fitz.open(os.path.join(FIG, "Figure_2_full.pdf"))
    out_path = os.path.join(FIG, "Figure_2.pdf")
    page = src[0]
    rect = fitz.Rect()
    for d in page.get_drawings():
        rect |= d["rect"]
    for b in page.get_text("dict")["blocks"]:
        rect |= fitz.Rect(b["bbox"])
    rect = fitz.Rect(rect.x0 - MARGIN, rect.y0 - MARGIN,
                     rect.x1 + MARGIN, rect.y1 + MARGIN) & page.rect
    out = fitz.open()
    p = out.new_page(width=rect.width, height=rect.height)
    p.show_pdf_page(p.rect, src, 0, clip=rect)
    out.save(out_path, garbage=4, deflate=True)
    out.close()
    src.close()
    print(f"{out_path}: {rect.width:.0f} x {rect.height:.0f} pt")


if __name__ == "__main__":
    main()
