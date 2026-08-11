"""Local OCR helpers for chat-ui: images and PDFs → plain text."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".bmp"}
PDF_SUFFIXES = {".pdf"}

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 40
MAX_OCR_CHARS = 1_500_000
# If selectable PDF text is thinner than this per page, treat as scanned.
MIN_CHARS_PER_PAGE = 40


@dataclass(frozen=True)
class OcrTooling:
    tesseract: str | None
    pdftotext: str | None
    pdftoppm: str | None

    @property
    def images_ok(self) -> bool:
        return bool(self.tesseract)

    @property
    def pdf_ok(self) -> bool:
        return bool(self.pdftotext) and bool(self.tesseract)


def discover_tools() -> OcrTooling:
    return OcrTooling(
        tesseract=shutil.which("tesseract"),
        pdftotext=shutil.which("pdftotext"),
        pdftoppm=shutil.which("pdftoppm"),
    )


def tooling_status(tools: OcrTooling | None = None) -> dict[str, object]:
    t = tools or discover_tools()
    missing: list[str] = []
    if not t.tesseract:
        missing.append("tesseract")
    if not t.pdftotext:
        missing.append("pdftotext (poppler)")
    if not t.pdftoppm:
        missing.append("pdftoppm (poppler)")
    return {
        "images": t.images_ok,
        "pdf": t.pdf_ok,
        "scanned_pdf": bool(t.tesseract and t.pdftoppm),
        "tesseract": t.tesseract,
        "pdftotext": t.pdftotext,
        "pdftoppm": t.pdftoppm,
        "missing": missing,
        "install_hint": "brew install tesseract poppler",
    }


class OcrError(RuntimeError):
    """Raised when OCR cannot produce text."""


def _run(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise OcrError(f"missing tool: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise OcrError(f"timed out running {' '.join(cmd[:2])}") from exc


def _cap_text(text: str) -> str:
    text = text.replace("\x00", "").strip()
    if len(text) > MAX_OCR_CHARS:
        return text[:MAX_OCR_CHARS] + "\n\n[truncated: OCR output exceeded size limit]"
    return text


def ocr_image_file(path: Path, tools: OcrTooling | None = None) -> str:
    t = tools or discover_tools()
    if not t.tesseract:
        raise OcrError("tesseract not found; install with: brew install tesseract")
    if not path.is_file():
        raise OcrError(f"image not found: {path.name}")
    size = path.stat().st_size
    if size <= 0:
        raise OcrError("empty image file")
    if size > MAX_IMAGE_BYTES:
        raise OcrError(f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
    proc = _run([t.tesseract, str(path), "stdout", "-l", "eng", "--psm", "3"])
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "tesseract failed").strip()
        raise OcrError(err[:500])
    text = _cap_text(proc.stdout or "")
    if not text:
        raise OcrError("OCR produced no text (empty or unreadable image)")
    return text


def _pdf_page_count(path: Path, pdftoppm: str | None, pdftotext: str | None) -> int:
    # Prefer pdfinfo if present; otherwise estimate via pdftoppm dry listing.
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        proc = _run([pdfinfo, str(path)], timeout=60)
        for line in (proc.stdout or "").splitlines():
            if line.lower().startswith("pages:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    break
    if pdftotext:
        # Force layout and count form-feed separators as a rough page count.
        proc = _run([pdftotext, "-layout", str(path), "-"], timeout=120)
        if proc.returncode == 0:
            pages = (proc.stdout or "").count("\f") + 1
            return max(1, pages)
    if pdftoppm:
        with tempfile.TemporaryDirectory(prefix="ds4-pdf-count-") as tmp:
            out_prefix = Path(tmp) / "page"
            proc = _run(
                [pdftoppm, "-png", "-f", "1", "-l", str(MAX_PDF_PAGES + 1), str(path), str(out_prefix)],
                timeout=180,
            )
            if proc.returncode == 0:
                return len(list(Path(tmp).glob("page-*.png")))
    return 1


def _ocr_pdf_pages(path: Path, tools: OcrTooling, page_count: int) -> str:
    if not tools.pdftoppm:
        raise OcrError(
            "scanned PDF needs pdftoppm; install with: brew install poppler"
        )
    if not tools.tesseract:
        raise OcrError("tesseract not found; install with: brew install tesseract")
    chunks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ds4-pdf-ocr-") as tmp:
        out_prefix = Path(tmp) / "page"
        last = min(page_count, MAX_PDF_PAGES)
        proc = _run(
            [tools.pdftoppm, "-png", "-r", "200", "-f", "1", "-l", str(last), str(path), str(out_prefix)],
            timeout=300,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "pdftoppm failed").strip()
            raise OcrError(err[:500])
        pages = sorted(Path(tmp).glob("page-*.png"))
        if not pages:
            raise OcrError("pdftoppm produced no page images")
        for i, page in enumerate(pages, start=1):
            text = ocr_image_file(page, tools)
            chunks.append(f"----- page {i} -----\n{text}")
    return _cap_text("\n\n".join(chunks))


def ocr_pdf_file(path: Path, tools: OcrTooling | None = None) -> tuple[str, str]:
    """Return (text, method) where method is 'text' or 'ocr'."""
    t = tools or discover_tools()
    if not path.is_file():
        raise OcrError(f"pdf not found: {path.name}")
    size = path.stat().st_size
    if size <= 0:
        raise OcrError("empty PDF")
    if size > MAX_PDF_BYTES:
        raise OcrError(f"PDF larger than {MAX_PDF_BYTES // (1024 * 1024)} MiB")
    if not t.pdftotext and not (t.pdftoppm and t.tesseract):
        raise OcrError(
            "PDF tools missing; install with: brew install tesseract poppler"
        )

    page_count = _pdf_page_count(path, t.pdftoppm, t.pdftotext)
    if page_count > MAX_PDF_PAGES:
        raise OcrError(f"PDF has {page_count} pages; limit is {MAX_PDF_PAGES}")

    selectable = ""
    if t.pdftotext:
        proc = _run([t.pdftotext, "-layout", str(path), "-"], timeout=180)
        if proc.returncode == 0:
            selectable = _cap_text(proc.stdout or "")

    if selectable and len(selectable) >= max(MIN_CHARS_PER_PAGE, MIN_CHARS_PER_PAGE * page_count // 2):
        return selectable, "text"

    # Scanned / image-heavy PDF.
    return _ocr_pdf_pages(path, t, page_count), "ocr"


def extract_attachment(filename: str, data: bytes) -> dict[str, object]:
    """OCR or extract text from an uploaded image/PDF payload."""
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    tools = discover_tools()

    with tempfile.TemporaryDirectory(prefix="ds4-ocr-") as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)

        if suffix in IMAGE_SUFFIXES:
            text = ocr_image_file(path, tools)
            return {
                "filename": name,
                "kind": "image",
                "method": "ocr",
                "chars": len(text),
                "text": text,
            }
        if suffix in PDF_SUFFIXES:
            text, method = ocr_pdf_file(path, tools)
            return {
                "filename": name,
                "kind": "pdf",
                "method": method,
                "chars": len(text),
                "text": text,
            }
        raise OcrError(
            f"unsupported type '{suffix or name}'; attach png/jpg/webp/gif/tiff/bmp or pdf"
        )
