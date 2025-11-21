from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Tuple, TYPE_CHECKING

import pandas as pd
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from lxml import etree
from docx import Document as DocxDocument

if TYPE_CHECKING:
    from .anthropic_client import AnthropicClient


@dataclass
class ExtractionResult:
    text: str
    ocr_pages: int = 0
    ocr_provider: str | None = None
    ocr_prompt_tokens: int = 0
    ocr_completion_tokens: int = 0


class DocumentProcessor:
    """Extracts machine-readable text from PDFs, images, XML, and Excel workbooks."""

    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    SUPPORTED_EXCEL_EXTS = {".xls", ".xlsx", ".xlsm"}
    SUPPORTED_TEXT_EXTS = {".txt", ".md", ".rtf"}
    SUPPORTED_DOC_EXTS = {".docx"}

    def __init__(
        self,
        ocr_lang: str = "eng",
        ocr_provider: str = "tesseract-ocr",
        anthropic_client: "AnthropicClient" | None = None,
    ) -> None:
        self.ocr_lang = ocr_lang
        self.ocr_provider = ocr_provider
        self.anthropic_client = anthropic_client

    def extract_text(self, path: str | Path) -> ExtractionResult:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_from_pdf(path)
        if suffix in self.SUPPORTED_IMAGE_EXTS:
            return self._extract_from_image(path)
        if suffix == ".xml":
            return self._extract_from_xml(path)
        if suffix in self.SUPPORTED_EXCEL_EXTS:
            return self._extract_from_excel(path)
        if suffix in self.SUPPORTED_DOC_EXTS:
            return self._extract_from_docx(path)
        if suffix in self.SUPPORTED_TEXT_EXTS:
            return ExtractionResult(text=path.read_text(encoding="utf-8"))
        raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_from_pdf(self, path: Path) -> ExtractionResult:
        text_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    text_parts.append(text)
        text = "\n".join(text_parts).strip()
        if text:
            return ExtractionResult(text=text)
        return self._ocr_pdf(path)

    def _ocr_pdf(self, path: Path) -> ExtractionResult:
        images = convert_from_path(str(path))
        texts: list[str] = []
        total_prompt = 0
        total_completion = 0
        for img in images:
            text, prompt_tokens, completion_tokens = self._run_ocr(img)
            texts.append(text)
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            img.close()
        return ExtractionResult(
            text="\n".join(texts),
            ocr_pages=len(images),
            ocr_provider=self.ocr_provider,
            ocr_prompt_tokens=total_prompt,
            ocr_completion_tokens=total_completion,
        )

    def _extract_from_image(self, path: Path) -> ExtractionResult:
        with Image.open(path) as img:
            text, prompt_tokens, completion_tokens = self._run_ocr(img)
            return ExtractionResult(
                text=text,
                ocr_pages=1,
                ocr_provider=self.ocr_provider,
                ocr_prompt_tokens=prompt_tokens,
                ocr_completion_tokens=completion_tokens,
            )

    def _run_ocr(self, img: Image.Image) -> Tuple[str, int, int]:
        if self.ocr_provider.lower().startswith("claude"):
            if not self.anthropic_client:
                raise EnvironmentError("Set ANTHROPIC_API_KEY to use Claude OCR.")
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            text, prompt_tokens, completion_tokens = self.anthropic_client.ocr_image(
                model=self.ocr_provider,
                image_bytes=buffer.getvalue(),
                media_type="image/png",
                prompt="Extract the textual content of this document. Respond with raw text only.",
            )
            return text, prompt_tokens, completion_tokens
        return pytesseract.image_to_string(img, lang=self.ocr_lang), 0, 0

    def _extract_from_xml(self, path: Path) -> ExtractionResult:
        tree = etree.parse(str(path))
        text = "\n".join(self._iter_xml_text(tree.getroot())).strip()
        return ExtractionResult(text=text)

    def _iter_xml_text(self, node: etree._Element) -> Iterable[str]:
        if node.text and node.text.strip():
            yield node.text.strip()
        for child in node:
            yield from self._iter_xml_text(child)
        if node.tail and node.tail.strip():
            yield node.tail.strip()

    def _extract_from_excel(self, path: Path) -> ExtractionResult:
        text_parts: list[str] = []
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            text_parts.append(f"Sheet: {sheet}")
            text_parts.append(df.to_csv(index=False))
        return ExtractionResult(text="\n".join(text_parts))

    def _extract_from_docx(self, path: Path) -> ExtractionResult:
        doc = DocxDocument(path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text and para.text.strip()]
        return ExtractionResult(text="\n".join(paragraphs))
