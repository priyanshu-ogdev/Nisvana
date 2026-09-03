"""
Project AEGIS — Fetch Manager
Orchestrates downloading across all verified datasets with support for:
- Server Mode (Full 4TB dataset download)
- Sample Mode (Lightweight operational verification)
- Dry-Run Mode (Validates endpoints, HTTP status, and sizes)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from data_forge.config import RAW_DIR, MANIFESTS_DIR
from .base import BaseFetcher, DownloadResult, logger
from .noisex import NoisexFetcher
from .shared import SharedExplosionFetcher
from .gunshot import GunshotDryadFetcher
from .drone import DroneAudioSetFetcher
from .mad import MadFetcher
from .vctk_demand import VctkDemandFetcher
from .dns import DnsChallengeFetcher
from .aec import AecChallengeFetcher
from .sirens import SirensFetcher
from .rir import RirFetcher


class FetchManager:
    """Manages downloading across all verified datasets."""

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.fetchers: Dict[str, BaseFetcher] = {
            "noisex92": NoisexFetcher(self.raw_dir / "noisex92"),
            "shared": SharedExplosionFetcher(self.raw_dir / "shared_explosions"),
            "gunshot_dryad": GunshotDryadFetcher(self.raw_dir / "gunshot_dryad"),
            "drone_audioset": DroneAudioSetFetcher(self.raw_dir / "drone_audioset"),
            "mad": MadFetcher(self.raw_dir / "mad"),
            "vctk_demand": VctkDemandFetcher(self.raw_dir / "vctk_demand"),
            "dns_challenge": DnsChallengeFetcher(self.raw_dir / "dns_challenge"),
            "aec_challenge": AecChallengeFetcher(self.raw_dir / "aec_challenge"),
            "sirens_urban": SirensFetcher(self.raw_dir / "sirens_urban"),
            "openslr_rirs": RirFetcher(self.raw_dir / "openslr_rirs"),
        }

    def fetch_source(
        self,
        source_name: str,
        sample_mode: bool = False,
        dry_run: bool = False,
    ) -> List[DownloadResult]:
        """Fetches a specific dataset by name."""
        if source_name not in self.fetchers:
            raise ValueError(f"Unknown source: {source_name}. Available: {list(self.fetchers.keys())}")

        logger.info("Starting fetch for source '%s' (sample_mode=%s, dry_run=%s)", source_name, sample_mode, dry_run)
        fetcher = self.fetchers[source_name]
        return fetcher.fetch(sample_mode=sample_mode, dry_run=dry_run)

    def fetch_all(
        self,
        sample_mode: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, List[DownloadResult]]:
        """Fetches all datasets across the verified bibliography."""
        logger.info("=== PROJECT AEGIS DATA-FORGE: MULTI-SOURCE FETCH INITIATED ===")
        all_results = {}
        for name, fetcher in self.fetchers.items():
            logger.info(">>> Fetching: %s", name)
            try:
                results = fetcher.fetch(sample_mode=sample_mode, dry_run=dry_run)
                all_results[name] = results
            except Exception as e:
                logger.error("Error fetching %s: %s", name, e)
                all_results[name] = [DownloadResult(success=False, destination=self.raw_dir / name, bytes_downloaded=0, elapsed_sec=0.0, md5="", error=str(e))]

        self.save_fetch_summary(all_results)
        return all_results

    def save_fetch_summary(self, results: Dict[str, List[DownloadResult]]) -> None:
        """Saves a JSON summary of all fetch operations."""
        summary = {}
        total_bytes = 0
        success_count = 0
        total_files = 0

        for source, res_list in results.items():
            summary[source] = []
            for r in res_list:
                total_files += 1
                if r.success:
                    success_count += 1
                    total_bytes += r.bytes_downloaded
                summary[source].append({
                    "destination": str(r.destination),
                    "success": r.success,
                    "bytes": r.bytes_downloaded,
                    "elapsed_sec": round(r.elapsed_sec, 2),
                    "md5": r.md5,
                    "error": r.error,
                })

        manifest = {
            "total_sources": len(results),
            "total_files": total_files,
            "successful_files": success_count,
            "total_downloaded_gb": round(total_bytes / (1024**3), 3),
            "sources": summary,
        }

        out_path = MANIFESTS_DIR / "fetch_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Fetch summary written to %s", out_path)
