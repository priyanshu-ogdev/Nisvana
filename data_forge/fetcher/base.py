"""
Project AEGIS — Base Downloader with Resumption, Checksumming, and Verification
"""

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataForge.Fetcher")


@dataclass
class FetchProgress:
    total_bytes: int
    downloaded_bytes: int
    percent: float
    speed_kbps: float
    status: str


@dataclass
class DownloadResult:
    success: bool
    destination: Path
    bytes_downloaded: int
    elapsed_sec: float
    md5: str
    error: Optional[str] = None


class BaseFetcher(ABC):
    """Base class for all dataset fetchers."""

    def __init__(self, output_dir: Path, timeout: int = 60, max_retries: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProjectAEGIS/1.0 (Research Pipeline)"
        }

    @abstractmethod
    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        """
        Fetch dataset files.
        :param sample_mode: If True, downloads verified sample files for quick verification.
        :param dry_run: If True, verifies URLs and headers without writing multi-GB files.
        """
        pass

    def download_file(
        self,
        url: str,
        dest_path: Path,
        expected_md5: Optional[str] = None,
        dry_run: bool = False,
        progress_cb: Optional[Callable[[FetchProgress], None]] = None,
    ) -> DownloadResult:
        """Downloads a URL with streaming, resume support, and checksum verification."""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            logger.info("[DRY RUN] Probing URL: %s", url)
            try:
                resp = requests.head(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
                if resp.status_code >= 400:
                    # Fallback to GET with stream=True if HEAD is disallowed by endpoint
                    resp = requests.get(url, headers=self.headers, timeout=self.timeout, stream=True, allow_redirects=True)
                size = int(resp.headers.get("content-length", 0))
                logger.info("[DRY RUN] URL reachable (%s). Content-Length: %d bytes (%.2f MB)", resp.status_code, size, size / (1024 * 1024))
                return DownloadResult(
                    success=resp.status_code < 400,
                    destination=dest_path,
                    bytes_downloaded=0,
                    elapsed_sec=0.0,
                    md5="",
                    error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
                )
            except Exception as e:
                logger.error("[DRY RUN] Error probing %s: %s", url, e)
                return DownloadResult(success=False, destination=dest_path, bytes_downloaded=0, elapsed_sec=0.0, md5="", error=str(e))

        # Check if already fully downloaded and checksum matches
        if dest_path.exists():
            current_md5 = self.compute_md5(dest_path)
            if expected_md5 and current_md5 == expected_md5:
                logger.info("File %s already exists and MD5 matches. Skipping.", dest_path.name)
                return DownloadResult(
                    success=True,
                    destination=dest_path,
                    bytes_downloaded=dest_path.stat().st_size,
                    elapsed_sec=0.0,
                    md5=current_md5,
                )

        temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        initial_bytes = temp_path.stat().st_size if temp_path.exists() else 0

        req_headers = dict(self.headers)
        if initial_bytes > 0:
            req_headers["Range"] = f"bytes={initial_bytes}-"
            logger.info("Resuming %s from byte %d", dest_path.name, initial_bytes)

        start_time = time.time()
        for attempt in range(1, self.max_retries + 1):
            try:
                with requests.get(url, headers=req_headers, stream=True, timeout=self.timeout) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP error {resp.status_code} on {url}")

                    total_size = int(resp.headers.get("content-length", 0)) + initial_bytes
                    mode = "ab" if initial_bytes > 0 and resp.status_code == 206 else "wb"
                    if mode == "wb":
                        initial_bytes = 0

                    downloaded = initial_bytes
                    chunk_size = 1024 * 1024  # 1 MB chunks

                    with open(temp_path, mode) as f:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            elapsed = max(time.time() - start_time, 0.001)
                            speed = (downloaded / 1024.0) / elapsed
                            pct = (downloaded / total_size * 100.0) if total_size > 0 else 0.0

                            if progress_cb:
                                progress_cb(
                                    FetchProgress(
                                        total_bytes=total_size,
                                        downloaded_bytes=downloaded,
                                        percent=pct,
                                        speed_kbps=speed,
                                        status="downloading",
                                    )
                                )

                    # Atomically rename temp_path to dest_path
                    if dest_path.exists():
                        dest_path.unlink()
                    temp_path.rename(dest_path)

                    file_md5 = self.compute_md5(dest_path)
                    if expected_md5 and file_md5 != expected_md5:
                        raise ValueError(f"Checksum mismatch: expected {expected_md5}, got {file_md5}")

                    elapsed_total = time.time() - start_time
                    logger.info("Successfully fetched %s (%.2f MB in %.1fs)", dest_path.name, downloaded / 1024 / 1024, elapsed_total)
                    return DownloadResult(
                        success=True,
                        destination=dest_path,
                        bytes_downloaded=downloaded,
                        elapsed_sec=elapsed_total,
                        md5=file_md5,
                    )

            except Exception as e:
                logger.warning("Attempt %d/%d failed for %s: %s", attempt, self.max_retries, url, e)
                if attempt == self.max_retries:
                    return DownloadResult(
                        success=False,
                        destination=dest_path,
                        bytes_downloaded=0,
                        elapsed_sec=time.time() - start_time,
                        md5="",
                        error=str(e),
                    )
                time.sleep(2 * attempt)

        return DownloadResult(success=False, destination=dest_path, bytes_downloaded=0, elapsed_sec=0.0, md5="", error="Max retries reached")

    @staticmethod
    def compute_md5(filepath: Path) -> str:
        """Computes MD5 checksum of a file efficiently."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
