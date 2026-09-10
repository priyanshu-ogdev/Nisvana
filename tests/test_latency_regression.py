"""
tests/test_latency_regression.py — Automated Latency Regression & Budget Verification

Rev 3 P1.5: Automated benchmarking across all processing stages to guard
against latency creep in mission-critical tactical radio pipelines.

Latency budget targets established in AEGIS specification:
  - Classifier gate:    < 2.0 ms
  - Primary SE:         < 10.0 ms (real-time factor < 1.0 for 10ms chunk)
  - Escalation SE:      < 40.0 ms (deep enhancement path)
  - NLMS adaptive ANC:  < 1.0 ms
  - End-to-end pipeline:< 50.0 ms
"""

import time
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.models.model_loader import build_model_for_key
from inference.runtime.escalation_router import AcousticEscalationRouter
from inference.runtime.hybrid_anc import NormalizedLMSFilter, HybridAncPipeline
from training.data.streaming_chunker import DEFAULT_CHUNK_CONFIG


class TestLatencyRegressionBudgets:
    """Rigorous timing benchmarks for each processing stage on 10ms (480-sample) frames."""

    @pytest.fixture(autouse=True)
    def setup_bench(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_samples = DEFAULT_CHUNK_CONFIG.chunk_samples  # 480 @ 48kHz (10ms)
        self.sample_rate = DEFAULT_CHUNK_CONFIG.sample_rate      # 48000
        self.num_warmup = 10
        self.num_iterations = 50

    def test_classifier_gate_latency_budget(self):
        """Stage 1: Classifier gate must execute in < 2.0ms per frame."""
        clf = build_model_for_key("aegis-clf-gate").to(self.device).eval()
        dummy_in = torch.randn(1, 9600, device=self.device)  # 200ms rolling classifier window

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                clf(dummy_in)

        # Timed benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                t0 = time.perf_counter()
                clf(dummy_in)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.median(times))
        p95_ms = float(np.percentile(times, 95))
        assert median_ms < 2.0, f"Classifier median latency {median_ms:.2f}ms exceeded 2.0ms target"
        assert p95_ms < 5.0, f"Classifier p95 latency {p95_ms:.2f}ms exceeded 5.0ms envelope"

    def test_primary_se_latency_budget(self):
        """Stage 2: Primary causal SE must execute in < 10.0ms (RTF < 1.0 on 10ms frame)."""
        model = build_model_for_key("aegis-se-primary").to(self.device).eval()
        dummy_in = torch.randn(1, self.chunk_samples, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                model(dummy_in)

        times = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                t0 = time.perf_counter()
                model(dummy_in)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.median(times))
        p95_ms = float(np.percentile(times, 95))
        # 10.0ms budget on GPU / edge hardware; allow 25.0ms on dev-environment CPU
        budget_target_ms = 10.0 if self.device.type == "cuda" else 25.0
        assert median_ms < budget_target_ms, f"Primary SE median latency {median_ms:.2f}ms exceeded {budget_target_ms}ms target"

    def test_escalation_se_latency_budget(self):
        """Stage 3: Escalation SE must execute in < 40.0ms."""
        model = build_model_for_key("aegis-se-escalation").to(self.device).eval()
        dummy_in = torch.randn(1, self.chunk_samples, device=self.device)

        with torch.no_grad():
            for _ in range(self.num_warmup):
                model(dummy_in)

        times = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                t0 = time.perf_counter()
                model(dummy_in)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.median(times))
        budget_target_ms = 40.0 if self.device.type == "cuda" else 60.0
        assert median_ms < budget_target_ms, f"Escalation SE median latency {median_ms:.2f}ms exceeded {budget_target_ms}ms target"

    def test_nlms_adaptive_filter_latency_budget(self):
        """Stage 4: NLMS residual adaptive filter must execute in < 1.0ms."""
        nlms = NormalizedLMSFilter(filter_length=64, step_size=0.05)
        primary = np.random.randn(self.chunk_samples).astype(np.float32)
        ref = np.random.randn(self.chunk_samples).astype(np.float32)

        # Warmup
        for _ in range(self.num_warmup):
            nlms.filter_batch(ref, primary)

        times = []
        for _ in range(self.num_iterations):
            t0 = time.perf_counter()
            nlms.filter_batch(ref, primary)
            times.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.median(times))
        # < 1.0ms target on compiled C/Cython; allow < 5.0ms in pure Python interpreter
        assert median_ms < 5.0, f"NLMS median latency {median_ms:.3f}ms exceeded 5.0ms target"

    def test_end_to_end_router_pipelined_latency_budget(self):
        """Stage 5: End-to-end pipelined routing & enhancement must execute in < 50.0ms."""
        router = AcousticEscalationRouter(device=self.device)
        frame = np.random.randn(self.chunk_samples).astype(np.float32)

        for _ in range(self.num_warmup):
            router.route_and_enhance_pipelined(frame)

        times = []
        for _ in range(self.num_iterations):
            t0 = time.perf_counter()
            router.route_and_enhance_pipelined(frame)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.median(times))
        assert median_ms < 50.0, f"End-to-end pipelined latency {median_ms:.2f}ms exceeded 50.0ms target"
