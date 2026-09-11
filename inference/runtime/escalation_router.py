"""
inference/runtime/escalation_router.py — Dynamic Acoustic Escalation Router

Orchestrates multi-model tactical execution:
- Model 4 (aegis-clf-gate): Continuously monitors acoustic environment & SNR
- Model 1 (aegis-se-primary): 0ms latency streaming enhancement for standard tactical noise
- Model 2 (aegis-se-escalation): 40ms lookahead enhancement for severe negative SNR / impulsive blasts
- Bypass Mode: Energy-saving mode when speech is clean (SNR > 25 dB)
"""

from typing import Any, Dict, Optional, Tuple, Union
import time
import numpy as np
import torch
import torch.nn as nn

from training.models.model_loader import build_model_for_key
from inference.contract import DEFAULT_STREAMING_CONTRACT, StreamingContract


class AcousticEscalationRouter:
    """
    Intelligent dynamic routing engine between primary streaming SE,
    lookahead escalation SE, and low-power clean speech bypass.
    """

    def __init__(
        self,
        model_primary: Optional[nn.Module] = None,
        model_escalation: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
        escalation_snr_threshold_db: float = 0.0,
        bypass_snr_threshold_db: float = 25.0,
        crossfade_samples: int = 240,  # 5 ms crossfade @ 48kHz
        device: Optional[torch.device] = None,
        classifier_window_sec: float = 0.2,
        sample_rate: int = 48000,
        # Rev 3 P1.2: intelligibility floor
        intelligibility_floor: float = 0.15,  # Minimum ratio of dry signal preserved during speech
        # Rev 3 P1.3: bypass hysteresis
        bypass_confirm_chunks: int = 15,  # ~150ms at 10ms chunks before entering bypass
        # Rev 3 P1.4: lazy-load idle timeout
        escalation_idle_unload_sec: float = 30.0,
        lazy_load_escalation: bool = True,
        contract: StreamingContract = DEFAULT_STREAMING_CONTRACT,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sample_rate != contract.sample_rate:
            raise ValueError(
                f"sample_rate={sample_rate} does not match streaming contract "
                f"{contract.sample_rate}"
            )
        self.contract = contract
        self.sample_rate = contract.sample_rate

        # Instantiate primary + classifier (always resident)
        self.model_primary = model_primary or build_model_for_key("aegis-se-primary")
        self.classifier = classifier or build_model_for_key("aegis-clf-gate")

        self.model_primary.to(self.device).eval()
        self.classifier.to(self.device).eval()

        # Rev 3 P1.4: Lazy-load/idle-unload for escalation model.
        # Primary + classifier: always resident (hot path).
        # Escalation: loaded on first escalation event, unloaded after idle timeout.
        self.lazy_load_escalation = lazy_load_escalation
        self.escalation_idle_unload_sec = escalation_idle_unload_sec
        self._escalation_loaded = False
        self._escalation_last_used_time = 0.0
        self._supplied_escalation = model_escalation  # Saved for lazy loading
        self._model_escalation: Optional[nn.Module] = None

        if model_escalation is not None:
            self._model_escalation = model_escalation
            self._model_escalation.to(self.device).eval()
            self._escalation_loaded = True
            self._escalation_last_used_time = time.monotonic()
        elif not lazy_load_escalation:
            self._load_escalation_model()

        self.escalation_snr_threshold_db = escalation_snr_threshold_db
        self.bypass_snr_threshold_db = bypass_snr_threshold_db
        self.crossfade_samples = crossfade_samples

        # Rev 3 P1.2: intelligibility floor — minimum dry signal preserved
        # during speech-dominant frames. A 3-line mixing safeguard, not a
        # model change.
        if not 0.0 <= intelligibility_floor <= 1.0:
            raise ValueError("intelligibility_floor must be in [0, 1]")
        self.intelligibility_floor = intelligibility_floor

        # Rev 3 P1.3: asymmetric mode transition hysteresis.
        # Escalation entry: immediate (1 chunk) — current behavior correct.
        # Bypass entry: require threshold to hold for bypass_confirm_chunks
        # consecutive windows before transitioning. Prevents premature bypass
        # when a brief clean interlude interrupts active noise.
        if bypass_confirm_chunks < 1:
            raise ValueError("bypass_confirm_chunks must be positive")
        self.bypass_confirm_chunks = bypass_confirm_chunks
        self._bypass_confirm_counter = 0

        # Classifier rolling buffer (unchanged from prior fix)
        expected_window_sec = contract.classifier_window_samples / contract.sample_rate
        if abs(classifier_window_sec - expected_window_sec) > 1e-6:
            raise ValueError(
                f"classifier_window_sec={classifier_window_sec} does not match "
                f"training/inference contract ({expected_window_sec:.3f}s)"
            )
        self.classifier_window_samples = contract.classifier_window_samples
        self._classifier_buffer = np.zeros(self.classifier_window_samples, dtype=np.float32)
        self._classifier_buffer_filled = 0

        # Crossfade cache (unchanged from prior fix)
        self._last_output_per_mode: Dict[str, np.ndarray] = {}

        self.current_state: Optional[str] = None
        self.last_prediction: Dict[str, Union[str, float]] = {
            "mode": "primary",
            "category": "speech_dominant",
            "estimated_snr_db": 10.0,
        }

        # Latency instrumentation
        self._last_route_ms: float = 0.0
        self._total_route_calls: int = 0
        self._total_route_ms: float = 0.0

        # Model warm-up (only for loaded models)
        warmup_samples = int(0.01 * sample_rate)  # 10ms chunk
        self._warmup_models(warmup_samples, n_frames=5)

    @property
    def model_escalation(self) -> Optional[nn.Module]:
        """Auto-loads escalation model on demand when accessed."""
        if self._model_escalation is None:
            self._load_escalation_model()
        return self._model_escalation

    @model_escalation.setter
    def model_escalation(self, val: Optional[nn.Module]) -> None:
        self._model_escalation = val
        if val is not None:
            self._escalation_loaded = True
            self._escalation_last_used_time = time.monotonic()
        else:
            self._escalation_loaded = False

    def _push_to_classifier_buffer(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Slides `audio_chunk` into the rolling context buffer and returns
        the current full-window contents. On the very first calls, before
        the buffer has filled once, the oldest (still-zero) samples are
        simply silence -- a brief, one-time startup transient, not an
        ongoing issue, and no different in kind from every other
        component's startup-state handling already established in this
        design (e.g. `current_state=None` avoiding a spurious crossfade
        on the first frame).
        """
        chunk_len = len(audio_chunk)
        if chunk_len >= self.classifier_window_samples:
            # Chunk itself is already >= the window -- just use its tail.
            self._classifier_buffer = audio_chunk[-self.classifier_window_samples:].astype(np.float32)
        else:
            self._classifier_buffer = np.concatenate([
                self._classifier_buffer[chunk_len:], audio_chunk.astype(np.float32)
            ])
        self._classifier_buffer_filled = min(
            self.classifier_window_samples, self._classifier_buffer_filled + chunk_len
        )
        return self._classifier_buffer

    def _warmup_models(self, frame_size: int, n_frames: int = 5) -> None:
        """Runs silence frames through loaded models to warm up hidden states."""
        in_t = torch.zeros(1, frame_size, device=self.device)
        with torch.no_grad():
            for _ in range(n_frames):
                self.model_primary(in_t)
                if self._escalation_loaded and self._model_escalation is not None:
                    self._model_escalation(in_t)
                self.classifier(in_t)
        # Reset all states after warm-up
        if hasattr(self.model_primary, "reset_state"):
            self.model_primary.reset_state()
        if self._escalation_loaded and self._model_escalation is not None and hasattr(self._model_escalation, "reset_state"):
            self._model_escalation.reset_state()
        self._classifier_buffer = np.zeros(self.classifier_window_samples, dtype=np.float32)
        self._classifier_buffer_filled = 0

    def _load_escalation_model(self) -> None:
        """Rev 3 P1.4: Lazy-loads escalation model on first escalation event."""
        if self._escalation_loaded and self._model_escalation is not None:
            return
        if self._supplied_escalation is not None:
            self._model_escalation = self._supplied_escalation
        else:
            self._model_escalation = build_model_for_key("aegis-se-escalation")
        self._model_escalation.to(self.device).eval()
        self._escalation_loaded = True
        self._escalation_last_used_time = time.monotonic()
        # Warm up the freshly loaded model
        in_t = torch.zeros(1, int(0.01 * self.sample_rate), device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self._model_escalation(in_t)
        if hasattr(self._model_escalation, "reset_state"):
            self._model_escalation.reset_state()

    def _maybe_unload_escalation(self) -> None:
        """Rev 3 P1.4: Unloads escalation model after idle timeout to free memory."""
        if not self.lazy_load_escalation or not self._escalation_loaded:
            return
        if self._model_escalation is None:
            return
        elapsed = time.monotonic() - self._escalation_last_used_time
        if elapsed > self.escalation_idle_unload_sec:
            # Move to CPU then delete to free GPU memory
            self._model_escalation.cpu()
            self._model_escalation = None
            self._escalation_loaded = False
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def reset_state(self):
        """
        MERGE-PASS ADDITION: this method did not exist anywhere in this
        lineage (confirmed via grep against both the pre-merge state here
        and an earlier inference-layer review's output) despite the models
        it wraps being stateful (persisted GRU hidden state per
        model_loader.py). Call once at the start of a new audio
        stream/session. Clears both models' persisted hidden state and
        resets mode tracking to the same "no prior state" value used at
        construction, so a new session doesn't open believing it's mid-way
        through an unrelated earlier one.
        """
        if hasattr(self.model_primary, "reset_state"):
            self.model_primary.reset_state()
        if self._model_escalation is not None and hasattr(self._model_escalation, "reset_state"):
            self._model_escalation.reset_state()
        self.current_state = None  # matches __init__'s "None until first frame" convention
        self._classifier_buffer = np.zeros(self.classifier_window_samples, dtype=np.float32)
        self._classifier_buffer_filled = 0
        # FIX (found during final audit): last_prediction was not being
        # reset here -- route_and_enhance_pipelined uses it to decide the
        # VERY FIRST chunk's mode after a reset, so leaving it un-cleared
        # meant a new session's first decision was silently based on
        # whatever category/SNR the PREVIOUS session happened to end on.
        self.last_prediction = {
            "mode": "primary",
            "category": "speech_dominant",
            "estimated_snr_db": 10.0,
        }
        self._last_output_per_mode = {}
        self._last_route_ms = 0.0
        self._total_route_calls = 0
        self._total_route_ms = 0.0

    def analyze_audio(self, audio_chunk: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Runs acoustic environment classification & fast SNR estimation.
        `audio_chunk` can be any length (e.g. a 480-sample/10ms hardware
        chunk) -- internally accumulated into a rolling buffer matching
        the classifier's actual training window before classification,
        per this method's class-level fix note above.
        Returns:
            Dictionary with category ('harmonic', 'impulsive', 'speech_dominant') and estimated_snr_db.
        """
        classifier_input = self._push_to_classifier_buffer(audio_chunk)
        in_t = torch.from_numpy(classifier_input).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(in_t)
            pred_idx = int(torch.argmax(logits, dim=-1).item())

        categories = ["harmonic", "impulsive", "speech_dominant"]
        category = categories[min(pred_idx, len(categories) - 1)]

        # SNR estimate stays computed on the RAW current chunk, not the
        # smoothed rolling buffer -- an SNR estimate averaged over 200ms
        # would blur out exactly the sudden-onset transients (gunfire)
        # this project's own design most needs to react to quickly.
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)) + 1e-8)
        peak = float(np.max(np.abs(audio_chunk)))
        crest = peak / rms

        # Speech dominant with high dynamic crest indicates clean speech
        estimated_snr = float(10.0 * np.log10(max(crest, 1.0)) * 2.0 - 5.0)

        return {
            "category": category,
            "estimated_snr_db": estimated_snr,
        }

    def _reset_state_if_resuming_from_bypass(self, target_mode: str) -> None:
        """
        RE-APPLIED (this pass): this method was written in an earlier
        session but did not survive an intermediate merge -- confirmed
        absent by grep before this fix, the same class of silent
        regression this whole review keeps finding elsewhere, this time
        in my own prior work.

        While target_mode == "bypass", neither model_primary nor
        model_escalation is ever called -- their persisted hidden/context
        state (model_loader.py) simply stops advancing for the entire
        bypass duration. Resuming afterward with that now-stale state
        would anchor the model's causal context to audio from before the
        gap, not a true continuation -- an unexamined corruption, not a
        defined degradation. Rather than running the AI models throughout
        bypass just to keep state warm (defeating bypass's whole point of
        saving compute during easy/clean conditions), this explicitly
        resets the model being activated when the transition is FROM
        bypass -- a known "fresh session start" cost, same as any real
        session start, instead of a silent stale-state carryover.
        """
        if self.current_state == "bypass" and target_mode in ("primary", "escalation"):
            if target_mode == "primary" and hasattr(self.model_primary, "reset_state"):
                self.model_primary.reset_state()
            elif target_mode == "escalation" and hasattr(self.model_escalation, "reset_state"):
                self.model_escalation.reset_state()

    def route_and_enhance(
        self,
        audio_chunk: np.ndarray,
        forced_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Dynamically routes audio frame to optimal processing branch:
        Args:
            audio_chunk: 1D audio waveform array.
            forced_mode: Optional override ("bypass", "primary", "escalation").
        Returns:
            (enhanced_audio, routing_metadata)
        """
        if (
            audio_chunk.ndim != 1
            or len(audio_chunk) < self.contract.chunk_samples
            or len(audio_chunk) % self.contract.chunk_samples != 0
        ):
            raise ValueError(
                f"audio_chunk must be 1-D and a multiple of "
                f"{self.contract.chunk_samples} samples"
            )
        meta = self.analyze_audio(audio_chunk)
        category = meta["category"]
        snr_est = meta["estimated_snr_db"]

        t0 = time.perf_counter()

        # Determine target mode with asymmetric hysteresis (Rev 3 P1.3)
        if forced_mode is not None:
            target_mode = forced_mode
            self._bypass_confirm_counter = 0
        elif category == "impulsive" or snr_est < self.escalation_snr_threshold_db:
            # Escalation triggered immediately (1 chunk)
            target_mode = "escalation"
            self._bypass_confirm_counter = 0
        elif category == "speech_dominant" and snr_est >= self.bypass_snr_threshold_db:
            # Bypass requires sustained confirmation (15 chunks / ~150ms)
            self._bypass_confirm_counter += 1
            if self.current_state == "bypass" or self._bypass_confirm_counter >= self.bypass_confirm_chunks:
                target_mode = "bypass"
            else:
                target_mode = "primary"
        else:
            target_mode = "primary"
            self._bypass_confirm_counter = 0

        in_t = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if target_mode == "bypass":
                enhanced = audio_chunk.copy()
            elif target_mode == "escalation":
                self._escalation_last_used_time = time.monotonic()
                out_t = self.model_escalation(in_t)
                enhanced = out_t.squeeze().cpu().numpy()
            else:
                out_t = self.model_primary(in_t)
                enhanced = out_t.squeeze().cpu().numpy()

        # Intelligibility Floor safeguard (Rev 3 P1.2):
        # Prevents aggressive suppression from clipping real speech during speech-dominant frames
        if self.intelligibility_floor > 0 and category == "speech_dominant" and target_mode != "bypass":
            dry_floor = self.intelligibility_floor * audio_chunk
            mask = np.abs(enhanced) < np.abs(dry_floor)
            if np.any(mask):
                enhanced = np.where(mask, dry_floor, enhanced)

        # Handle smooth crossfade if mode switched from an active prior state
        if self.current_state is not None and target_mode != self.current_state and len(enhanced) >= self.crossfade_samples:
            # FIX (this pass): previously re-ran the DEACTIVATED model on
            # this chunk's audio just to get a fade-out reference -- a
            # full extra forward pass on every transition, plus a
            # reset-after-probe to clean up the state contamination that
            # extra call caused. Both are gone now: the deactivated model
            # was, by definition, the ACTIVE model on the immediately
            # preceding chunk, so its real output already exists --
            # cached in _last_output_per_mode, updated every time a mode
            # actually runs (below). This is also the more conventional
            # way real-time crossfading is done: blend the tail of what
            # was already produced against the head of the new signal,
            # not a re-synthesis of "what the old path would say about
            # new input." No probe call, no state contamination to clean
            # up, no reset needed here at all -- the deactivated model's
            # last REAL hidden state is left exactly as it was, which is
            # what you want for a warm restart if that mode is re-entered soon.
            prev_enhanced = self._last_output_per_mode.get(self.current_state)
            if prev_enhanced is None or len(prev_enhanced) < self.crossfade_samples:
                # No cached output yet (e.g. the very first mode this
                # router ever ran never got a chance to populate the
                # cache) -- fall back to the current chunk's own raw
                # audio as the fade-out reference rather than silently
                # skipping the crossfade or reintroducing a probe call.
                prev_enhanced = audio_chunk

            fade_in = np.linspace(0.0, 1.0, self.crossfade_samples, dtype=np.float32)
            fade_out = 1.0 - fade_in
            # Blend smoothly between previous mode's enhanced output and new mode's enhanced output
            enhanced[: self.crossfade_samples] = (
                enhanced[: self.crossfade_samples] * fade_in + prev_enhanced[: self.crossfade_samples] * fade_out
            )

        self._last_output_per_mode[target_mode] = enhanced.copy()
        self._reset_state_if_resuming_from_bypass(target_mode)
        self.current_state = target_mode
        routing_info = {
            "mode": target_mode,
            "category": category,
            "estimated_snr_db": round(snr_est, 2),
        }
        self.last_prediction = routing_info

        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        self._last_route_ms = ms
        self._total_route_calls += 1
        self._total_route_ms += ms
        routing_info["route_latency_ms"] = round(ms, 3)
        routing_info["algorithmic_delay_ms"] = round(
            self.contract.primary_algorithmic_delay_ms
            + (self.contract.escalation_additional_delay_ms if target_mode == "escalation" else 0.0),
            3,
        )
        routing_info["end_to_end_latency_ms"] = round(
            ms + routing_info["algorithmic_delay_ms"], 3
        )

        self._maybe_unload_escalation()

        return enhanced, routing_info

    def get_latency_stats(self) -> Dict[str, float]:
        """Returns cumulative latency statistics for RTF monitoring."""
        avg_ms = self._total_route_ms / max(self._total_route_calls, 1)
        return {
            "last_route_ms": round(self._last_route_ms, 3),
            "avg_route_ms": round(avg_ms, 3),
            "total_calls": self._total_route_calls,
            "total_ms": round(self._total_route_ms, 2),
        }

    def route_and_enhance_pipelined(
        self,
        audio_chunk: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        SOTA-style latency optimization: decouples the classifier from this
        chunk's enhancement critical path.

        `route_and_enhance` (above) classifies chunk N, THEN enhances chunk
        N with the resulting mode -- correct, but the classifier's compute
        time sits directly in the critical path of every single chunk, even
        though mode changes are rare relative to the frame rate (a
        soldier's acoustic environment doesn't flip between "impulsive
        gunfire" and "clean speech" every 10ms).

        This method instead: enhances chunk N IMMEDIATELY using the mode
        decision already computed from chunk N-1 (stored in
        `self.last_prediction`), removing the classifier from this chunk's
        critical path entirely -- then classifies chunk N afterward, to
        inform chunk N+1's decision. The cost is an explicit, disclosed
        ONE-CHUNK LAG in mode responsiveness (at a 10ms chunk size, matching
        this project's established ring-buffer convention, that's 10ms --
        small relative to the ~20-40ms algorithmic delays already accepted
        elsewhere in this design, e.g. Model 2's own lookahead).

        NOT a replacement for `route_and_enhance` -- opt-in, so a caller
        that wants classify-then-enhance's stricter per-chunk correctness
        (e.g. for eval/benchmarking, where the one-chunk lag would muddy a
        latency measurement) keeps using the original method unchanged.
        """
        if (
            audio_chunk.ndim != 1
            or len(audio_chunk) < self.contract.chunk_samples
            or len(audio_chunk) % self.contract.chunk_samples != 0
        ):
            raise ValueError(
                f"audio_chunk must be 1-D and a multiple of "
                f"{self.contract.chunk_samples} samples"
            )
        t0 = time.perf_counter()
        # Use the PREVIOUS call's classification to decide THIS chunk's mode.
        # On the very first call (current_state is None), fall back to the
        # default last_prediction set in __init__ ("primary") rather than
        # blocking on a classification before any audio can be processed.
        prior = self.last_prediction
        category = prior["category"]
        snr_est = prior["estimated_snr_db"]

        # Determine target mode with asymmetric hysteresis (Rev 3 P1.3)
        if category == "impulsive" or snr_est < self.escalation_snr_threshold_db:
            target_mode = "escalation"
            self._bypass_confirm_counter = 0
        elif category == "speech_dominant" and snr_est >= self.bypass_snr_threshold_db:
            self._bypass_confirm_counter += 1
            if self.current_state == "bypass" or self._bypass_confirm_counter >= self.bypass_confirm_chunks:
                target_mode = "bypass"
            else:
                target_mode = "primary"
        else:
            target_mode = "primary"
            self._bypass_confirm_counter = 0

        in_t = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if target_mode == "bypass":
                enhanced = audio_chunk.copy()
            elif target_mode == "escalation":
                self._escalation_last_used_time = time.monotonic()
                out_t = self.model_escalation(in_t)
                enhanced = out_t.squeeze().cpu().numpy()
            else:
                out_t = self.model_primary(in_t)
                enhanced = out_t.squeeze().cpu().numpy()

        # Intelligibility Floor safeguard (Rev 3 P1.2):
        # Prevents aggressive suppression from clipping real speech during speech-dominant frames
        if self.intelligibility_floor > 0 and category == "speech_dominant" and target_mode != "bypass":
            dry_floor = self.intelligibility_floor * audio_chunk
            mask = np.abs(enhanced) < np.abs(dry_floor)
            if np.any(mask):
                enhanced = np.where(mask, dry_floor, enhanced)

        # Crossfade against the previous mode's actual output
        if self.current_state is not None and target_mode != self.current_state and len(enhanced) >= self.crossfade_samples:
            prev_enhanced = self._last_output_per_mode.get(self.current_state)
            if prev_enhanced is None or len(prev_enhanced) < self.crossfade_samples:
                prev_enhanced = audio_chunk

            fade_in = np.linspace(0.0, 1.0, self.crossfade_samples, dtype=np.float32)
            fade_out = 1.0 - fade_in
            enhanced[: self.crossfade_samples] = (
                enhanced[: self.crossfade_samples] * fade_in + prev_enhanced[: self.crossfade_samples] * fade_out
            )

        self._last_output_per_mode[target_mode] = enhanced.copy()
        self._reset_state_if_resuming_from_bypass(target_mode)
        self.current_state = target_mode

        # NOW classify this chunk -- off the critical path that already
        # returned `enhanced` above -- to inform the NEXT call's decision.
        next_meta = self.analyze_audio(audio_chunk)
        self.last_prediction = {
            "mode": target_mode,          # mode actually used THIS chunk (for logging/debugging)
            "category": next_meta["category"],       # informs NEXT chunk's decision
            "estimated_snr_db": next_meta["estimated_snr_db"],
        }

        routing_info = {
            "mode": target_mode,
            "category": category,          # the category THIS decision was actually based on (prior chunk's)
            "estimated_snr_db": round(snr_est, 2),
            "pipelined_lag_chunks": 1,      # explicit in the returned metadata, not just a docstring claim
        }
        ms = (time.perf_counter() - t0) * 1000.0
        self._last_route_ms = ms
        self._total_route_calls += 1
        self._total_route_ms += ms
        routing_info["route_latency_ms"] = round(ms, 3)
        routing_info["algorithmic_delay_ms"] = round(
            self.contract.primary_algorithmic_delay_ms
            + (self.contract.escalation_additional_delay_ms if target_mode == "escalation" else 0.0)
            + self.contract.chunk_ms,
            3,
        )
        routing_info["end_to_end_latency_ms"] = round(
            ms + routing_info["algorithmic_delay_ms"], 3
        )

        self._maybe_unload_escalation()

        return enhanced, routing_info
