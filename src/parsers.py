import io
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import docx
import fitz
import textract


class DocumentProcessingError(Exception):
    pass


# data structures
class ChunkMetadata:
    def __init__(
        self,
        chunk_id: str,
        page_num: Optional[int] = None,
        section: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        chunk_type: str = "text",
        importance_score: float = 1.0,
    ):
        self.chunk_id = chunk_id
        self.page_num = page_num
        self.section = section
        self.keywords = keywords or []
        self.chunk_type = chunk_type
        self.importance_score = importance_score


class DocumentStructure:
    def __init__(self):
        self.sections: Dict[str, List[str]] = {}
        self.headers: List[Tuple[str, int]] = []  # (header_text, level)
        self.tables: List[Dict[str, Any]] = []
        self.key_value_pairs: Dict[str, str] = {}
        self.definitions: Dict[str, str] = {}


def do_boxes_intersect(box1: fitz.Rect, box2: fitz.Rect) -> bool:
    return box1.intersects(box2)


class BaseParser:
    """Abstraction layer over parsers"""

    def parse(
        self, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], Optional[DocumentStructure]]:
        raise NotImplementedError("Parser not implemented")


class PDFParser(BaseParser):
    def parse(
        self, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], DocumentStructure]:
        full_text_parts = []
        metadata = {"pages": 0, "tables": 0, "sections": 0, "format": "PDF"}
        doc_structure = DocumentStructure()

        try:
            doc = fitz.open(stream=content.read(), filetype="pdf")
            metadata["pages"] = len(doc)
            current_section = "Introduction"

            for i, page in enumerate(doc):
                page_tables = list(page.find_tables())
                table_bboxes = [fitz.Rect(t.bbox) for t in page_tables]

                text_blocks = page.get_text("blocks")
                filtered_text_blocks = []
                for block in text_blocks:
                    block_bbox = fitz.Rect(block[:4])
                    is_in_table = any(
                        do_boxes_intersect(block_bbox, table_bbox)
                        for table_bbox in table_bboxes
                    )
                    if not is_in_table:
                        filtered_text_blocks.append(block[4])

                page_text = "".join(filtered_text_blocks)

                full_text_parts.append(
                    f"--- Page {i+1} | Section: {current_section} ---\n{page_text}"
                )

                lines = page_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if line and (
                        line.isupper()
                        or re.match(r"^[A-Z][A-Za-z\s]+:$", line)
                        or re.match(r"^\d+\.\s+[A-Z]", line)
                    ):
                        if line not in [h[0] for h in doc_structure.headers]:
                            doc_structure.headers.append((line, 1))
                            current_section = line
                            metadata["sections"] += 1

                if current_section not in doc_structure.sections:
                    doc_structure.sections[current_section] = []
                doc_structure.sections[current_section].append(page_text)

                if page_tables:
                    metadata["tables"] += len(page_tables)
                    for j, table in enumerate(page_tables):
                        table_data = table.extract()
                        if not table_data:
                            continue

                        table_metadata = {
                            "page": i + 1,
                            "table_id": f"table_{i+1}_{j+1}",
                            "rows": len(table_data),
                            "cols": len(table_data[0]) if table_data else 0,
                        }
                        doc_structure.tables.append(table_metadata)

                        if len(table_data) > 1:
                            headers = table_data[0]
                            markdown_table = (
                                "| "
                                + " | ".join(map(lambda x: str(x or ""), headers))
                                + " |\n"
                            )
                            markdown_table += "|" + "---|" * len(headers) + "\n"
                            for row in table_data[1:]:
                                markdown_table += (
                                    "| "
                                    + " | ".join(map(lambda x: str(x or ""), row))
                                    + " |\n"
                                )

                            full_text_parts.append(
                                f"\n--- Table {j+1} on Page {i+1} ---\n{markdown_table}\n"
                            )

                kv_matches = re.findall(r"([A-Z][A-Za-z\s/]+):\s*([^\n]+)", page_text)
                for key, value in kv_matches:
                    doc_structure.key_value_pairs[key.strip()] = value.strip()

        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse PDF: {str(e)}") from e

        full_text = "\n".join(full_text_parts)
        with open("o.md", "w") as f:
            f.write(full_text)
        return full_text, metadata, doc_structure


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
    def parse_doc(self, content: io.BytesIO) -> Tuple[str, Dict[str, Any], None]:
        temp_file_path = f"{uuid.uuid4()}.doc"
        try:
            with open(temp_file_path, "wb") as f:
                f.write(content.read())

            text = textract.process(temp_file_path).decode("utf-8")
            return text, {"format": "DOC"}, None
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse DOC: {str(e)}") from e
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


class TXTParser(BaseParser):
    def parse_txt(self, content: io.BytesIO) -> Tuple[str, Dict[str, Any], None]:
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
