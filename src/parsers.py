import asyncio
import io
import logging
import os
import re
import tempfile
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import camelot
import docx
import pdfplumber
import textract

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    pass


@dataclass
class DocumentStructure:
    headers: List[Tuple[str, int]]
    sections: Dict[str, List[str]]
    tables: List[Dict[str, Any]]
    key_value_pairs: Dict[str, str]

    def __init__(self):
        self.headers = []
        self.sections = {}
        self.tables = []
        self.key_value_pairs = {}


class BaseParser:
    @abstractmethod
    def parse(
        self, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], Optional[DocumentStructure]]:
        pass


class PDFParser(BaseParser):
    def __init__(self):
        self._parser_lock = Lock()

    def parse(
        self, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], DocumentStructure]:
        working_copy = io.BytesIO(content.getvalue())

        full_text = []
        metadata = {"pages": 0, "tables": 0, "sections": 0}
        doc_structure = DocumentStructure()

        try:
            with self._parser_lock:
                # First pass - extract basic structure and text
                with pdfplumber.open(working_copy) as pdf:
                    metadata["pages"] = len(pdf.pages)
                    current_section = "Introduction"
                    section_text = []

                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text = (
                                page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                            )
                            lines = page_text.split("\n")

                            for line in lines:
                                line = line.strip()
                                if self._is_header(line):
                                    header_level = self._determine_header_level(line)
                                    doc_structure.headers.append((line, header_level))
                                    current_section = line
                                    metadata["sections"] += 1
                                    # flush previous section content
                                    if section_text:
                                        doc_structure.sections[current_section] = (
                                            section_text
                                        )
                                        section_text = []

                            section_text.append(page_text)
                            full_text.append(
                                f"--- Page {i+1} | Section: {current_section} ---\n{page_text}"
                            )

                            # key-value pairs
                            self._extract_key_value_pairs(page_text, doc_structure)

                        except Exception as page_error:
                            logger.error(
                                f"Error processing page {i+1}: {str(page_error)}"
                            )
                            continue

                    # add final section
                    if section_text:
                        doc_structure.sections[current_section] = section_text

                # second pass - tables only
                working_copy.seek(0)
                try:
                    tables = camelot.read_pdf(
                        working_copy,
                        flavor="lattice",
                        pages="all",
                        suppress_stdout=True,
                        process_background=True,
                    )
                    metadata["tables"] = len(tables)
                    self._process_tables(tables, full_text, doc_structure)
                except Exception as table_error:
                    logger.error(f"Table extraction failed: {str(table_error)}")

            with open("out.md", "w") as f:
                f.write("\n".join(full_text))
            return "\n".join(full_text), metadata, doc_structure

        except pdfplumber.PDFSyntaxError as e:
            raise DocumentProcessingError(f"Invalid PDF structure: {e}") from e
        except Exception as e:
            raise DocumentProcessingError(f"PDF processing failed: {str(e)}")
        finally:
            working_copy.close()

    def _is_header(self, text: str) -> bool:
        """Improved header detection logic"""
        if not text:
            return False
        # Check for all caps with reasonable length
        if text.isupper() and 2 < len(text) < 80:
            return True
        # Check for numbered headings (e.g., "1. Introduction")
        if re.match(r"^\d+\.\s+\w+", text):
            return True
        # Check for section headings ending with colon
        if text.endswith(":") and len(text.split()) < 8:
            return True
        return False

    def _determine_header_level(self, text: str) -> int:
        """Determine heading level based on formatting"""
        if text.isupper():
            return 1  # Highest level
        if re.match(r"^\d+\.", text):
            return 2
        return 3  # Default level

    def _extract_key_value_pairs(self, text: str, doc_structure: DocumentStructure):
        """Extract key: value pairs from text"""
        kv_pattern = re.compile(r"([A-Z][A-Za-z\s]+):\s*([^\n]+)")
        for key, value in kv_pattern.findall(text):
            clean_key = key.strip()
            clean_value = value.strip()
            if clean_key and clean_value:
                doc_structure.key_value_pairs[clean_key] = clean_value

    def _process_tables(self, tables, full_text, doc_structure):
        """Process extracted tables into structured format"""
        for table in tables:
            if table.shape[0] <= 1:  # Skip empty/single-row tables
                continue

            table_metadata = {
                "page": table.page,
                "table_id": f"table_{table.page}_{table.order}",
                "rows": table.shape[0],
                "cols": table.shape[1],
            }
            doc_structure.tables.append(table_metadata)

            # Convert to markdown
            df = table.df
            headers = [self._clean_table_cell(cell) for cell in df.iloc[0].tolist()]
            rows = [
                "| " + " | ".join(self._clean_table_cell(cell) for cell in row) + " |"
                for _, row in df[1:].iterrows()
            ]

            markdown_table = (
                "| " + " | ".join(headers) + " |\n"
                "| "
                + " | ".join(["---"] * len(headers))
                + " |\n"
                + "\n".join(rows)
                + "\n"
            )

            full_text.append(
                f"\n--- Table {table.order} on Page {table.page} ---\n{markdown_table}\n"
            )

    def _clean_table_cell(self, text: str) -> str:
        """Clean table cell content"""
        if not text or not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", str(text).strip())


class DOCXParser(BaseParser):
    def parse(
        self, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], DocumentStructure]:
        try:
            doc = docx.Document(content)
            full_text = []
            current_section = "Introduction"
            metadata = {
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables),
                "sections": len(doc.sections),
            }
            doc_structure = DocumentStructure()

            for para in doc.paragraphs:
                if para.text.strip():  # todo check errors from lsp
                    # header
                    if para.style.name.startswith("Heading") or (
                        para.text.isupper() and len(para.text) < 100
                    ):
                        level = 1
                        if "Heading" in para.style.name:
                            level = (
                                int(para.style.name.split()[-1])
                                if para.style.name.split()[-1].isdigit()
                                else 1
                            )
                        doc_structure.headers.append((para.text, level))
                        current_section = para.text

                    # add to section
                    if current_section not in doc_structure.sections:
                        doc_structure.sections[current_section] = []
                    doc_structure.sections[current_section].append(para.text)

                    full_text.append(f"[Section: {current_section}] {para.text}")

            # tables
            for i, table in enumerate(doc.tables):
                table_text = [
                    " | ".join(cell.text.strip() for cell in row.cells)
                    for row in table.rows
                ]

                table_metadata = {
                    "table_id": f"docx_table_{i+1}",
                    "rows": len(table.rows),
                    "cols": len(table.rows[0].cells) if table.rows else 0,
                }
                doc_structure.tables.append(table_metadata)
                full_text.append(
                    f"\n--- Table {i+1} ---\n" + "\n".join(table_text) + "\n"
                )

            return "\n".join(full_text), metadata, doc_structure
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse DOCX: {str(e)}") from e


class DOCParser(BaseParser):
    def parse(self, content: io.BytesIO) -> Tuple[str, Dict[str, Any], None]:
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(content.read())
                tmp.flush()
                text = textract.process(tmp.name).decode("utf-8")
            return text, {"format": "DOC"}, None
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse DOC: {str(e)}") from e


class TXTParser(BaseParser):
    def parse(self, content: io.BytesIO) -> Tuple[str, Dict[str, Any], None]:
        try:
            text = content.read().decode("utf-8", errors="ignore")
            return text, {"size": len(text)}, None
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse TXT: {str(e)}") from e


# parser mapping
PARSER_MAPPING: Dict[str, BaseParser] = {
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".doc": DOCParser(),
    ".txt": TXTParser(),
}


class DocumentParser:
    def __init__(
        self,
        parsers: Dict[str, Any],
        max_workers: int = 4,
        max_file_size: int = 30 * 1024 * 1024,  # 30MB
        timeout: float = 30.0,
    ):
        self.parsers = parsers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_file_size = max_file_size
        self.timeout = timeout

    async def parse(
        self, content: io.BytesIO, file_ext: str
    ) -> Tuple[str, Dict[str, Any], Optional[DocumentStructure]]:
        logger.info("Parsing a doc")
        # Validate input
        content.seek(0, os.SEEK_END)
        file_size = content.tell()
        content.seek(0)

        if file_size > self.max_file_size:
            raise DocumentProcessingError(
                f"File size {file_size/1024/1024:.2f}MB exceeds maximum {self.max_file_size/1024/1024:.2f}MB"
            )

        # get appropriate parser
        parser = self._get_parser(file_ext)

        try:
            # run parsing in thread pool with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor, self._parse_with_retry, parser, content
                ),
                timeout=self.timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise DocumentProcessingError(
                f"Parsing timed out after {self.timeout} seconds"
            )
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse document: {str(e)}")

    def _get_parser(self, file_ext: str) -> BaseParser:
        """Get parser instance for file extension"""
        file_ext = file_ext.lower()
        if file_ext not in self.parsers:
            if file_ext in [".rtf", ".odt"]:  # fallbacks
                return self.parsers.get(".docx", self.parsers[".txt"])
            return self.parsers[".txt"]  # default fallback
        return self.parsers[file_ext]

    def _parse_with_retry(
        self, parser: BaseParser, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], Optional[DocumentStructure]]:
        """Parse document with retries"""
        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                content_copy = io.BytesIO(content.getvalue())
                return parser.parse(content_copy)
            except Exception as e:
                last_exception = e
                logger.warning(f"Parse attempt {attempt} failed: {str(e)}")
                if attempt < max_attempts:
                    continue
                raise DocumentProcessingError(
                    f"Failed after {max_attempts} attempts: {str(last_exception)}"
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

    async def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
