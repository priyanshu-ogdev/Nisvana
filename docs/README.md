# Project AEGIS — Documentation Index & Research Repository

> **Location:** `docs/`  
> **Topic:** Master Specifications, Architecture Blueprints, Training Runbooks, and Peer-Reviewed Bibliography  

---

## 1. Documentation Overview

This directory contains the authoritative technical documentation, system specifications, runbooks, and scientific citations for Project AEGIS. All documents reflect verified code implementations, real acoustic data sources, and empirical benchmarks.

---

## 2. Document Sitemap

| Document | Purpose & Scope | Key Topics Covered |
| :--- | :--- | :--- |
| **[`Nisvana_PRD.md`](file:///d:/Nisvana/docs/Nisvana_PRD.md)** | **Master Product Requirements Document (v19)** | SIH26052 / DRDO requirements, system architecture, 5-model ensemble, Grace Blackwell co-design, real-recordings policy, dual-platform deployment, multi-user backend, compliance scorecard, and open limitations. |
| **[`ARCHITECTURE.md`](file:///d:/Nisvana/docs/ARCHITECTURE.md)** | **End-to-End System Architecture Blueprint** | Full-stack architecture covering Data Forge (5-tier storage, 10-step DSP), Training Engine (QAT, bf16 AMP, distillation), Inference Runtime (escalation router, safeguards, hybrid ANC), and Multi-User Backend. |
| **[`TRAINING_PIPELINE.md`](file:///d:/Nisvana/docs/TRAINING_PIPELINE.md)** | **ML Training Pipeline Runbook** | Model factory, SOTA perceptual loss stack (MultiRes, Speech-Presence SDR, Distillation, IS³, A-weighted), Blackwell bf16 optimization, QAT lifecycle, worst-class Pareto guard, and training CLI. |
| **[`DATA_PIPELINE.md`](file:///d:/Nisvana/docs/DATA_PIPELINE.md)** | **Data Acquisition & Synthesis Guide** | 10 real dataset fetchers, ITU-R BS.1770-4 loudness normalization, polyphase 48 kHz resampling, 3-branch mixture synthesis, and WebDataset sharding. |
| **[`PIPELINE_RUNBOOK.md`](file:///d:/Nisvana/docs/PIPELINE_RUNBOOK.md)** | **Turnkey Operations & Execution Runbook** | Step-by-step commands for environment setup, data acquisition dry-runs, end-to-end preprocessing, model training, ONNX export, and testing. |
| **[`BIBLIOGRAPHY.md`](file:///d:/Nisvana/docs/BIBLIOGRAPHY.md)** | **Peer-Reviewed Scientific Bibliography** | 20+ peer-reviewed papers (IEEE/ACM, ICASSP, INTERSPEECH, Nature Sci. Data, Harvard Dataverse) with DOI links and explicit system connections. |

---

## 3. Related Component Master Documentation

In addition to the central `docs/` directory, each primary subsystem contains a dedicated `README.md` at its package root:
- **Root Runbook**: [`README.md`](file:///d:/Nisvana/README.md)
- **Data Layer**: [`data/README.md`](file:///d:/Nisvana/data/README.md)
- **Data Forge**: [`data_forge/README.md`](file:///d:/Nisvana/data_forge/README.md)
- **Training Engine**: [`training/README.md`](file:///d:/Nisvana/training/README.md)
- **Inference Runtime**: [`inference/README.md`](file:///d:/Nisvana/inference/README.md)
- **Multi-User Backend**: [`backend/README.md`](file:///d:/Nisvana/backend/README.md)
- **Scripts & Turnkey Runners**: [`scripts/README.md`](file:///d:/Nisvana/scripts/README.md)
- **Automated Test Suite**: [`tests/README.md`](file:///d:/Nisvana/tests/README.md)
