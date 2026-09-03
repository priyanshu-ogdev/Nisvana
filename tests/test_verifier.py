"""
Project AEGIS — Tests for Pipeline Auditor and Reporter
"""

import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from data_forge.config import TARGET_SAMPLE_RATE
from data_forge.verifier import AuditReporter, PipelineAuditor


def test_pipeline_auditor_compliance_check(tmp_path):
    data_dir = tmp_path / "data"
    proc_dir = data_dir / "processed" / "clean_speech"
    proc_dir.mkdir(parents=True)

    # Write a 48kHz audio file
    sr = TARGET_SAMPLE_RATE
    audio = np.zeros(sr, dtype=np.float32)
    sf.write(proc_dir / "valid_clip.wav", audio, sr)

    auditor = PipelineAuditor(data_dir=data_dir)
    summary = auditor.run_full_audit()

    assert summary.total_processed_files == 1
    assert summary.sample_rate_compliance_rate == 100.0


def test_audit_reporter_markdown_generation(tmp_path):
    data_dir = tmp_path / "data"
    auditor = PipelineAuditor(data_dir=data_dir)
    summary = auditor.run_full_audit()

    report_path = tmp_path / "report.md"
    md = AuditReporter.generate_markdown_report(summary, output_path=report_path)

    assert report_path.exists()
    assert "# Project AEGIS — Data-Forge Pipeline Audit Report" in md
    assert "Corpus Inventory Summary" in md
