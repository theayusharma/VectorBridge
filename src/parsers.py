import io
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import docx
import pdfplumber
import camelot
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
    def parse(
        self, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], Optional[DocumentStructure]]:
        raise NotImplementedError("Parser not implemented")


class PDFParser(BaseParser):
    def parse(
        self, content: io.BytesIO
    ) -> Tuple[str, Dict[str, Any], DocumentStructure]:
        full_text = []
        metadata = {"pages": 0, "tables": 0, "sections": 0}
        doc_structure = DocumentStructure()

        try:
            with pdfplumber.open(content) as pdf:
                metadata["pages"] = len(pdf.pages)
                current_section = "Introduction"

                # camelot for table extraction
                content.seek(0)
                tables = camelot.read_pdf(content, flavor='lattice', pages='all')

                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text(x_tolerance=2) or ""

                    # headers (lines that are likely section headers)
                    lines = page_text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and (
                            line.isupper()
                            or re.match(r"^[A-Z][A-Za-z\s]+:$", line)
                            or re.match(r"^\d+\.\s+[A-Z]", line)
                        ):
                            doc_structure.headers.append((line, 1))
                            current_section = line
                            metadata["sections"] += 1

                    # add to chunks
                    if current_section not in doc_structure.sections:
                        doc_structure.sections[current_section] = []
                    doc_structure.sections[current_section].append(page_text)

                    full_text.append(
                        f"--- Page {i+1} | Section: {current_section} ---\n{page_text}"
                    )

                    # tables (using Camelot)
                    page_tables = [t for t in tables if t.page == i + 1]
                    if page_tables:
                        metadata["tables"] += len(page_tables)
                        for j, table in enumerate(page_tables):
                            table_metadata = {
                                "page": i + 1,
                                "table_id": f"table_{i+1}_{j+1}",
                                "rows": table.shape[0],
                                "cols": table.shape[1],
                            }
                            doc_structure.tables.append(table_metadata)

                            # conversion to markdown
                            df = table.df  # Camelot table as pandas DataFrame
                            if not df.empty and len(df) > 1:
                                headers = df.iloc[0].tolist()
                                # replace newlines and multiple spaces
                                headers = [
                                    re.sub(r'\s+', ' ', str(cell).strip()) if cell else ""
                                    for cell in headers
                                ]
                                markdown_table = (
                                    "| "
                                    + " | ".join(headers)
                                    + " |\n"
                                )
                                markdown_table += "|" + "---|" * len(headers) + "\n"

                                for _, row in df[1:].iterrows():
                                    # replace newlines and multiple spaces
                                    cleaned_row = [
                                        re.sub(r'\s+', ' ', str(cell).strip()) if cell else ""
                                        for cell in row
                                    ]
                                    markdown_table += (
                                        "| "
                                        + " | ".join(cleaned_row)
                                        + " |\n"
                                    )

                                full_text.append(
                                    f"\n--- Table {j+1} on Page {i+1} ---\n{markdown_table}\n"
                                )

                    # key-value pairs
                    kv_matches = re.findall(r"([A-Z][A-Za-z\s]+):\s*([^\n]+)", page_text)
                    for key, value in kv_matches:
                        doc_structure.key_value_pairs[key.strip()] = value.strip()

        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse PDF: {str(e)}")

        # Write output to markdown
        # with open("out.md", "w") as f:
        #     f.write("\n".join(full_text))

        return "\n".join(full_text), metadata, doc_structure


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
