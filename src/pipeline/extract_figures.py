import os
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  #PyMuPDF
from PIL import Image


@dataclass
class ExtractedFigure:
    pdf_name: str
    page_number: int        # 1-based
    image_index: int        # index within the page
    file_path: str
    width: int
    height: int

    xref: int
    bbox: Optional[Tuple[float, float, float, float]]  # (x0, y0, x1, y1) in PDF coords


def _ensure_png_compatible(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode == "CMYK":
        return pil_img.convert("RGB")
    if pil_img.mode in ("P", "PA", "YCbCr", "LAB", "HSV", "I", "F"):
        return pil_img.convert("RGB")
    return pil_img


def _pick_best_rect(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    """Pick the most likely placement rect if multiple exist (largest area)."""
    if not rects:
        return None
    return max(rects, key=lambda r: float(r.width * r.height))


def extract_figures_from_pdf(
    pdf_path: str,
    output_dir: str = "output",
    min_size: int = 50,
) -> List[ExtractedFigure]:
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")

    extracted: List[ExtractedFigure] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=False)

        for img_index, img in enumerate(images):
            xref = img[0]

            # get where this image is drawn on the page
            rects = page.get_image_rects(xref)
            best_rect = _pick_best_rect(rects)
            bbox = None
            if best_rect is not None:
                bbox = (float(best_rect.x0), float(best_rect.y0), float(best_rect.x1), float(best_rect.y1))

            # Extract embedded image bytes.
            # PyMuPDF's `extract_image()` can fail on some PDFs (e.g., alpha-channel images
            # that cannot be serialized as JPEG), so we fall back to rendering a Pixmap
            # and exporting it as PNG bytes.
            image_bytes = b""
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", b"")
            except Exception:
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # If the image has an alpha channel, drop it (convert to RGB)
                    if getattr(pix, "alpha", 0):
                        pix = fitz.Pixmap(pix, 0)
                    image_bytes = pix.tobytes("png")
                except Exception:
                    image_bytes = b""
            if not image_bytes:
                continue

            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = _ensure_png_compatible(pil_img)

            width, height = pil_img.size
            if width < min_size or height < min_size:
                continue

            filename = f"{pdf_name}_page{page_index + 1}_img{img_index}.png"
            output_path = os.path.join(output_dir, filename)
            pil_img.save(output_path, format="PNG")

            extracted.append(
                ExtractedFigure(
                    pdf_name=pdf_name,
                    page_number=page_index + 1,
                    image_index=img_index,
                    file_path=output_path,
                    width=width,
                    height=height,
                    xref=int(xref),
                    bbox=bbox,
                )
            )

    doc.close()
    return extracted
