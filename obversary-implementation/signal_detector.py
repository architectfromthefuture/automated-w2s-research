"""
Signal detection for weak-to-strong supervision tasks.

Classifies training examples and research directions by structural
characteristics BEFORE routing them to models or evaluation. This is
the architectural pattern that replaces prescribed sequential pipelines
with signal-responsive routing.

Mapping to W2S findings:
- Anthropic found that removing prescribed workflows improved agent
  performance. Signal detection explains WHY: the task characteristics
  should determine the processing path, not a fixed template.
- OpenAI's Preparedness Framework structures safeguards as independent
  evaluation channels (Robustness, Monitoring, Trust, Oversight). Each
  channel responds to different signal types.

Author: Brian Moran, Obversary Studios LLC
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class SignalType(str, Enum):
    needs_decomposition = "needs_decomposition"
    needs_verification = "needs_verification"
    ambiguous_label = "ambiguous_label"
    high_confidence = "high_confidence"
    weak_strong_disagreement = "weak_strong_disagreement"
    needs_parallel_evaluation = "needs_parallel_evaluation"
    potential_shortcut = "potential_shortcut"
    distribution_shift = "distribution_shift"


@dataclass
class Signal:
    signal_type: SignalType
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


class W2SSignalDetector:
    """
    Detect structural characteristics of training examples in the
    weak-to-strong supervision setting.

    Instead of treating all examples uniformly (as the baseline methods do),
    classify each example by what kind of processing it needs. This enables
    routing: high-confidence examples go straight to training, ambiguous
    examples get multi-model verification, potential shortcuts get flagged
    for held-out validation.
    """

    def __init__(
        self,
        weak_confidence_threshold: float = 0.7,
        disagreement_threshold: float = 0.3,
    ):
        self.weak_confidence_threshold = weak_confidence_threshold
        self.disagreement_threshold = disagreement_threshold

    def detect(
        self,
        weak_label: int,
        weak_confidence: float,
        strong_zero_shot: Optional[float] = None,
        embedding_density: Optional[float] = None,
        neighbor_label_consistency: Optional[float] = None,
    ) -> List[Signal]:
        """
        Produce signals for a single training example.

        Args:
            weak_label: Binary label from weak teacher (0 or 1).
            weak_confidence: Weak teacher's confidence in its label.
            strong_zero_shot: Strong model's zero-shot prediction probability
                              (if available from base model logits).
            embedding_density: Local density in strong model's embedding space.
            neighbor_label_consistency: Fraction of k-NN neighbors sharing
                                        the same weak label.
        """
        signals: List[Signal] = []

        if weak_confidence < self.weak_confidence_threshold:
            signals.append(Signal(
                signal_type=SignalType.ambiguous_label,
                confidence=1.0 - weak_confidence,
                source="weak_teacher",
                metadata={"weak_confidence": weak_confidence},
            ))
        else:
            signals.append(Signal(
                signal_type=SignalType.high_confidence,
                confidence=weak_confidence,
                source="weak_teacher",
            ))

        if strong_zero_shot is not None:
            strong_label = 1 if strong_zero_shot > 0.5 else 0
            if strong_label != weak_label:
                disagreement_strength = abs(strong_zero_shot - 0.5)
                if disagreement_strength > self.disagreement_threshold:
                    signals.append(Signal(
                        signal_type=SignalType.weak_strong_disagreement,
                        confidence=disagreement_strength,
                        source="strong_base_model",
                        metadata={
                            "weak_label": weak_label,
                            "strong_zero_shot": strong_zero_shot,
                        },
                    ))
                    signals.append(Signal(
                        signal_type=SignalType.needs_verification,
                        confidence=disagreement_strength,
                        source="disagreement_detector",
                    ))

        if neighbor_label_consistency is not None and neighbor_label_consistency < 0.6:
            signals.append(Signal(
                signal_type=SignalType.potential_shortcut,
                confidence=1.0 - neighbor_label_consistency,
                source="embedding_geometry",
                metadata={"consistency": neighbor_label_consistency},
            ))

        if embedding_density is not None and embedding_density < 0.2:
            signals.append(Signal(
                signal_type=SignalType.distribution_shift,
                confidence=1.0 - embedding_density,
                source="embedding_geometry",
                metadata={"density": embedding_density},
            ))

        return signals

    def route(self, signals: List[Signal]) -> str:
        """
        Decide processing path based on detected signals.

        Returns one of:
            'direct_train'    — high confidence, no disagreement
            'verify_first'    — disagreement or ambiguity detected
            'flag_shortcut'   — potential dataset shortcut
            'hold_for_review' — multiple concerning signals
        """
        types = {s.signal_type for s in signals}

        concerning = types & {
            SignalType.potential_shortcut,
            SignalType.distribution_shift,
        }
        uncertain = types & {
            SignalType.ambiguous_label,
            SignalType.weak_strong_disagreement,
            SignalType.needs_verification,
        }

        if len(concerning) + len(uncertain) >= 3:
            return "hold_for_review"
        if concerning:
            return "flag_shortcut"
        if uncertain:
            return "verify_first"
        return "direct_train"
