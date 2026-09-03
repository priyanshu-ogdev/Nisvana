"""
Project AEGIS — Auto-Generated Dataset Card

Industry-standard ML datasets ship a machine-readable card describing
provenance, licenses, and splits (HuggingFace Dataset Cards / Croissant
metadata convention) alongside the data itself. Generating this from
bibliography.py and config.py rather than hand-writing it means the card
can never silently drift out of sync with the actual fetchers — if a
source is added or its citation is corrected in bibliography.py, the
card picks it up automatically on next export.
"""

from pathlib import Path
from typing import Dict, List

from data_forge.bibliography import BIBLIOGRAPHY_ENTRIES
from data_forge.config import DATASET_PROFILES, TARGET_SAMPLE_RATE, TARGET_LUFS


def _yaml_frontmatter(branch_stats: Dict[str, int]) -> str:
    lines = [
        "---",
        "pretty_name: Project AEGIS Defence ANC Training Corpus",
        "license: other  # mixed per-source, see License Ledger below — NOT uniformly redistributable",
        "task_categories:",
        "  - audio-to-audio",
        "  - audio-classification",
        "language:",
        "  - en",
        "tags:",
        "  - speech-enhancement",
        "  - noise-cancellation",
        "  - defence-audio",
        "  - webdataset",
    ]
    if branch_stats:
        lines.append("dataset_info:")
        for branch, count in branch_stats.items():
            lines.append(f"  {branch}_samples: {count}")
    lines.append("---")
    return "\n".join(lines)


def generate_dataset_card(branch_stats: Dict[str, int] = None) -> str:
    branch_stats = branch_stats or {}
    parts: List[str] = [_yaml_frontmatter(branch_stats), ""]
    parts.append("# Project AEGIS — Defence ANC Training Corpus\n")
    parts.append(
        "Auto-generated dataset card. Regenerate with "
        "`python -m data_forge export --card` after any change to "
        "`bibliography.py` or `config.py` — do not hand-edit this file.\n"
    )

    parts.append("## Audio Standard\n")
    parts.append(f"- Sample rate: {TARGET_SAMPLE_RATE} Hz (fullband)\n- Loudness: {TARGET_LUFS} LUFS\n- Format: mono WAV, 16-bit PCM\n")

    parts.append("## Sources and Sync Tiers\n")
    parts.append("| Source | Native rate | Sync tier | Augmentation policy | License |\n|---|---|---|---|---|")
    for key, profile in DATASET_PROFILES.items():
        parts.append(
            f"| {profile.name} | {profile.native_sample_rate} Hz | "
            f"{profile.default_sync_tier.name} | {profile.augmentation_policy.value} | {profile.license} |"
        )
    parts.append("")

    parts.append("## Full Citation Ledger\n")
    for entry in BIBLIOGRAPHY_ENTRIES:
        parts.append(f"- **{entry.title}** — {entry.authors}. *{entry.venue_year}*. {entry.source_url} ({entry.license}). Role: {entry.role_in_aegis}")
    parts.append("")

    parts.append("## Known Limitations (disclosed, not synthesized around)\n")
    parts.append(
        "- Rotor/helicopter and wind classes remain thin on real recordings; "
        "no synthetic/procedural data is used anywhere in this corpus by design.\n"
        "- NOISEX-92-derived classes (armored vehicle, naval, jet) are content-authentic "
        "but duration-limited (minutes per platform, not hours).\n"
        "- CC BY-NC sources (ESC-50, portions of UrbanSound8K/FSD50K) are excluded "
        "under `--commercial-strict` preprocessing mode; included only for research/prototype use otherwise.\n"
    )

    return "\n".join(parts)


def write_dataset_card(output_path: Path, branch_stats: Dict[str, int] = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_dataset_card(branch_stats), encoding="utf-8")
