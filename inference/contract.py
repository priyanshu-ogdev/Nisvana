"""Shared training/inference contract for AEGIS streaming deployment."""

from dataclasses import dataclass

from data_forge.config import TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class StreamingContract:
    sample_rate: int = TARGET_SAMPLE_RATE
    chunk_samples: int = 480
    classifier_window_samples: int = 9600
    primary_algorithmic_delay_ms: float = 30.0
    escalation_additional_delay_ms: float = 10.0
    intelligibility_floor: float = 0.15
    bypass_confirm_chunks: int = 15

    @property
    def chunk_ms(self) -> float:
        return self.chunk_samples * 1000.0 / self.sample_rate

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.chunk_samples <= 0:
            raise ValueError("sample_rate and chunk_samples must be positive")
        if self.classifier_window_samples < self.chunk_samples:
            raise ValueError("classifier window must contain at least one streaming chunk")
        if not 0.0 <= self.intelligibility_floor <= 1.0:
            raise ValueError("intelligibility_floor must be in [0, 1]")


DEFAULT_STREAMING_CONTRACT = StreamingContract()
