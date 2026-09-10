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
    """Base class for all dataset fetchers with TB-scale download resilience."""

    def __init__(self, output_dir: Path, timeout: int = None, max_retries: int = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read download-resilience settings from .env (via data_forge.config's loader)
        self.timeout = timeout or int(os.environ.get("DATA_FORGE_FETCH_TIMEOUT", "300"))
        self.max_retries = max_retries or int(os.environ.get("DATA_FORGE_FETCH_MAX_RETRIES", "10"))
        self.backoff_base = int(os.environ.get("DATA_FORGE_FETCH_BACKOFF_BASE", "5"))
        self.chunk_size = int(os.environ.get("DATA_FORGE_FETCH_CHUNK_BYTES", "4194304"))  # 4MB
        self.resume_enabled = os.environ.get("DATA_FORGE_FETCH_RESUME", "true").lower() == "true"
        self.verify_ssl = os.environ.get("DATA_FORGE_FETCH_VERIFY_SSL", "true").lower() == "true"
        disk_margin = int(os.environ.get("DATA_FORGE_DISK_SAFETY_MARGIN_GB", "50"))
        self._disk_safety_bytes = disk_margin * (1024 ** 3)

        # Read proxy and network configuration
        custom_ua = os.environ.get("DATA_FORGE_USER_AGENT")
        user_agent = custom_ua or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProjectAEGIS/1.0 (Research Pipeline)"
        self.headers = {"User-Agent": user_agent}

        # Connection-pooled requests.Session for reuse across files in a single fetcher
        pool_size = int(os.environ.get("DATA_FORGE_FETCH_POOL_SIZE", "10"))
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Proxies (HTTP / HTTPS / SOCKS)
        http_proxy = os.environ.get("DATA_FORGE_HTTP_PROXY") or os.environ.get("HTTP_PROXY")
        https_proxy = os.environ.get("DATA_FORGE_HTTPS_PROXY") or os.environ.get("HTTPS_PROXY")
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies["http"] = http_proxy
            if https_proxy:
                proxies["https"] = https_proxy
            self.session.proxies.update(proxies)
            logger.info("Configured proxy routing: %s", {k: v.split('@')[-1] for k, v in proxies.items()})

        adapter = requests.adapters.HTTPAdapter(
            max_retries=0,  # We handle retries ourselves with exponential backoff
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_headers_for_url(self, url: str) -> Dict[str, str]:
        """Returns request headers with appropriate authorization tokens injected for domain."""
        from urllib.parse import urlparse
        headers = dict(self.headers)
        domain = urlparse(url).netloc.lower()

        # GitHub rate-limit bypass (60 req/hr unauthenticated -> 5,000 req/hr authenticated)
        if "github.com" in domain or "githubusercontent.com" in domain:
            token = os.environ.get("DATA_FORGE_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if token and not token.startswith("your_"):
                headers["Authorization"] = f"token {token}"

        # Hugging Face auth for private/rate-limited endpoints
        elif "huggingface.co" in domain or "hf-mirror.com" in domain:
            token = os.environ.get("DATA_FORGE_HF_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if token and not token.startswith("your_"):
                headers["Authorization"] = f"Bearer {token}"

        # Data Dryad API bearer token
        elif "datadryad.org" in domain:
            token = os.environ.get("DATA_FORGE_DRYAD_API_TOKEN") or os.environ.get("DRYAD_API_TOKEN") or os.environ.get("DRYAD_TOKEN")
            if token and not token.startswith("your_"):
                headers["Authorization"] = f"Bearer {token}"

        # Harvard Dataverse API key
        elif "dataverse.harvard.edu" in domain:
            token = os.environ.get("DATA_FORGE_DATAVERSE_API_TOKEN") or os.environ.get("DATAVERSE_API_TOKEN")
            if token and not token.startswith("your_"):
                headers["X-Dataverse-key"] = token

        return headers

    @abstractmethod
    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        """
        Fetch dataset files.
        :param sample_mode: If True, downloads verified sample files for quick verification.
        :param dry_run: If True, verifies URLs and headers without writing multi-GB files.
        """
        pass

    def _check_disk_space(self, path: Path, needed_bytes: int = 0) -> bool:
        """Returns False if available disk space is below safety margin."""
        import shutil
        try:
            usage = shutil.disk_usage(path.parent if path.parent.exists() else Path.cwd())
            available = usage.free
            if available < self._disk_safety_bytes + needed_bytes:
                logger.error(
                    "DISK SPACE SAFETY: only %.1f GB free (need %.1f GB margin + %.1f GB file). "
                    "Aborting download to prevent silent corruption.",
                    available / (1024**3),
                    self._disk_safety_bytes / (1024**3),
                    needed_bytes / (1024**3),
                )
                return False
        except Exception:
            pass  # shutil.disk_usage may not work on all platforms
        return True

    def download_file(
        self,
        url: str,
        dest_path: Path,
        expected_md5: Optional[str] = None,
        dry_run: bool = False,
        progress_cb: Optional[Callable[[FetchProgress], None]] = None,
        fallback_urls: Optional[List[str]] = None,
    ) -> DownloadResult:
        """
        Downloads a URL with streaming, resume support, exponential backoff,
        disk safety checks, checksum verification, and automatic fallback mirror failover.
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        candidate_urls = [url] + [f for f in (fallback_urls or []) if f != url]
        last_result: Optional[DownloadResult] = None

        for mirror_idx, target_url in enumerate(candidate_urls):
            if mirror_idx > 0:
                logger.warning(
                    "Primary download failed. Attempting fallback mirror [%d/%d]: %s",
                    mirror_idx,
                    len(candidate_urls) - 1,
                    target_url,
                )
            result = self._download_single_url(
                target_url,
                dest_path,
                expected_md5=expected_md5,
                dry_run=dry_run,
                progress_cb=progress_cb,
            )
            if result.success:
                if mirror_idx > 0:
                    logger.info("Successfully fetched %s from fallback mirror: %s", dest_path.name, target_url)
                return result
            last_result = result

        return last_result or DownloadResult(
            success=False, destination=dest_path, bytes_downloaded=0, elapsed_sec=0.0, md5="", error="No candidate URLs provided"
        )

    def _download_single_url(
        self,
        url: str,
        dest_path: Path,
        expected_md5: Optional[str] = None,
        dry_run: bool = False,
        progress_cb: Optional[Callable[[FetchProgress], None]] = None,
    ) -> DownloadResult:
        """Internal single-URL download attempt with retries, resume, and backoff."""
        req_headers = self.get_headers_for_url(url)

        if dry_run:
            logger.info("[DRY RUN] Probing URL: %s", url)
            try:
                resp = self.session.head(url, headers=req_headers, timeout=self.timeout, allow_redirects=True, verify=self.verify_ssl)
                if resp.status_code >= 400:
                    resp = self.session.get(url, headers=req_headers, timeout=self.timeout, stream=True, allow_redirects=True, verify=self.verify_ssl)
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
        initial_bytes = temp_path.stat().st_size if (temp_path.exists() and self.resume_enabled) else 0

        start_time = time.time()
        for attempt in range(1, self.max_retries + 1):
            try:
                active_headers = dict(req_headers)
                if initial_bytes > 0:
                    active_headers["Range"] = f"bytes={initial_bytes}-"
                    logger.info("Resuming %s from byte %d (%.2f MB)", dest_path.name, initial_bytes, initial_bytes / (1024**2))

                with self.session.get(url, headers=active_headers, stream=True,
                                      timeout=self.timeout, allow_redirects=True,
                                      verify=self.verify_ssl) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP error {resp.status_code} on {url}")

                    total_size = int(resp.headers.get("content-length", 0)) + initial_bytes

                    # Disk safety check before committing to download
                    if not self._check_disk_space(dest_path, total_size):
                        return DownloadResult(
                            success=False, destination=dest_path,
                            bytes_downloaded=0, elapsed_sec=time.time() - start_time,
                            md5="", error="Insufficient disk space",
                        )

                    mode = "ab" if initial_bytes > 0 and resp.status_code == 206 else "wb"
                    if mode == "wb":
                        initial_bytes = 0

                    downloaded = initial_bytes

                    with open(temp_path, mode) as f:
                        for chunk in resp.iter_content(chunk_size=self.chunk_size):
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
                # Exponential backoff: backoff_base * 2^(attempt-1), capped at 5 minutes
                delay = min(self.backoff_base * (2 ** (attempt - 1)), 300)
                logger.info("Retrying in %.0f seconds...", delay)
                time.sleep(delay)

        return DownloadResult(success=False, destination=dest_path, bytes_downloaded=0, elapsed_sec=0.0, md5="", error="Max retries reached")

    @staticmethod
    def compute_md5(filepath: Path) -> str:
        """Computes MD5 checksum of a file efficiently."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
