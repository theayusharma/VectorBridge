import io
import logging
from pathlib import Path
from typing import Tuple
from urllib.parse import unquote, urlparse

import httpx
from fastapi import HTTPException

from parsers import PARSER_MAPPING

logger = logging.getLogger(__name__)


class DocumentDownloader:
    def __init__(self, max_size: int, timeout: float, max_redirects: int = 3):
        self.max_size = max_size
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.dangerous_extensions = {
            ".exe",
            ".bat",
            ".sh",
            ".php",
            ".js",
            ".jar",
            ".py",
            ".rb",
            ".pl",
            ".cgi",
            ".bin",
        }

    def _is_dangerous_extension(self, path: str) -> bool:
        path = unquote(path).lower()
        ext = Path(path).suffix
        return ext in self.dangerous_extensions

    async def download(
        self, url: str, user_agent: str = "HackRxDocBot/1.0"
    ) -> Tuple[io.BytesIO, str]:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise HTTPException(
                status_code=400, detail="Only HTTP/HTTPS URLs are supported"
            )

        file_ext = Path(parsed_url.path).suffix.lower()
        if self._is_dangerous_extension(file_ext):
            raise HTTPException(
                status_code=400, detail=f"Dangerous file extension blocked: {file_ext}"
            )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            max_redirects=self.max_redirects,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=4),
            headers={"User-Agent": user_agent, "Accept": "application/pdf, text/*"},
        ) as client:
            try:
                buffer = io.BytesIO()
                total_bytes = 0

                async with client.stream(
                    "GET", url, timeout=httpx.Timeout(self.timeout)
                ) as response:
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "").lower()
                    if not any(
                        ct in content_type
                        for ct in ("application/pdf", "text/plain", "text/html")
                    ):
                        logger.warning(
                            f"Unexpected content-type: {content_type} for URL: {url}"
                        )

                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        total_bytes += len(chunk)
                        if total_bytes > self.max_size:
                            raise HTTPException(
                                status_code=413,
                                detail=f"File exceeds size limit of {self.max_size / 1e6:.2f}MB",
                            )
                        buffer.write(chunk)

                if total_bytes == 0:
                    raise HTTPException(status_code=400, detail="Empty file received")

                if not file_ext or file_ext not in PARSER_MAPPING:
                    if "pdf" in content_type:
                        file_ext = ".pdf"
                    elif "text" in content_type:
                        file_ext = ".txt"
                    else:
                        file_ext = ".txt"

                buffer.seek(0)
                return buffer, file_ext

            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Failed to download file: Server returned {e.response.status_code} for URL: {url}",
                )
            except httpx.RequestError as e:
                logger.error(f"Download failed for {url}: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Download failed: {e}"
                ) from e
