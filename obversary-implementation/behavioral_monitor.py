"""
Behavioral monitoring for weak-to-strong training.

Measures observable behavioral metrics during training — not just loss
curves, but timing patterns, output stability, and confidence dynamics.
This provides a second evaluation channel alongside PGR that is harder
to game.

Mapping to W2S findings:
- Anthropic's agents reward-hacked PGR (a single scalar metric) in
  every way the environment allowed. Behavioral monitoring adds signals
  that are orthogonal to the outcome metric.
- OpenAI's Preparedness Framework uses "Usage Monitoring" as an
  independent safeguard channel. This is the same principle applied at
  the training level.

Adapted from xray-wrapper (github.com/architectfromthefuture/xray-wrapper).

Author: Brian Moran, Obversary Studios LLC
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class TrainingBehavioralSignature:
    """Observable metrics from a single training step or evaluation pass."""
    step: int
    epoch: float
    loss: float
    forward_pass_ms: float
    gradient_norm: Optional[float] = None
    prediction_entropy: float = 0.0
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    label_flip_rate: float = 0.0
    weak_strong_agreement_rate: float = 0.0


class BehavioralMonitor:
    """
    Track behavioral signatures across training to detect:
    1. Confidence collapse — model becomes uniformly confident (overfitting)
    2. Entropy spikes — model suddenly uncertain (distribution shift)
    3. Agreement drift — weak-strong agreement pattern changes
    4. Timing anomalies — forward pass time changes (complexity shift)
    5. Label instability — predictions flip frequently (optimization noise)

    These are NOT output quality metrics. They are process metrics that
    reveal HOW the model is learning, not just WHAT it produces.
    """

    def __init__(self, window_size: int = 50):
        self.history: List[TrainingBehavioralSignature] = []
        self.window_size = window_size
        self.alerts: List[Dict] = []

    def record(self, signature: TrainingBehavioralSignature) -> List[Dict]:
        """Record a signature and return any triggered alerts."""
        self.history.append(signature)
        new_alerts = self._check_alerts(signature)
        self.alerts.extend(new_alerts)
        return new_alerts

    def _check_alerts(self, current: TrainingBehavioralSignature) -> List[Dict]:
        alerts = []
        if len(self.history) < self.window_size:
            return alerts

        window = self.history[-self.window_size:]

        conf_std_values = [s.confidence_std for s in window]
        if statistics.mean(conf_std_values) < 0.05:
            alerts.append({
                "type": "confidence_collapse",
                "step": current.step,
                "detail": "Model predictions near-uniform confidence — "
                          "possible overfitting to label noise",
                "confidence_std_mean": statistics.mean(conf_std_values),
            })

        entropy_values = [s.prediction_entropy for s in window]
        recent_entropy = statistics.mean(entropy_values[-10:])
        baseline_entropy = statistics.mean(entropy_values[:20])
        if baseline_entropy > 0 and recent_entropy / baseline_entropy > 2.0:
            alerts.append({
                "type": "entropy_spike",
                "step": current.step,
                "detail": "Prediction entropy doubled — possible distribution "
                          "shift or optimization instability",
                "ratio": recent_entropy / baseline_entropy,
            })

        flip_rates = [s.label_flip_rate for s in window[-10:]]
        if statistics.mean(flip_rates) > 0.15:
            alerts.append({
                "type": "label_instability",
                "step": current.step,
                "detail": "Predictions flipping frequently — learning rate "
                          "may be too high or labels too noisy",
                "flip_rate": statistics.mean(flip_rates),
            })

        timing_values = [s.forward_pass_ms for s in window]
        recent_timing = statistics.mean(timing_values[-10:])
        baseline_timing = statistics.mean(timing_values[:20])
        if baseline_timing > 0 and recent_timing / baseline_timing > 1.5:
            alerts.append({
                "type": "timing_anomaly",
                "step": current.step,
                "detail": "Forward pass time increased 50%+ — possible "
                          "complexity change or resource contention",
                "ratio": recent_timing / baseline_timing,
            })

        agreement_values = [s.weak_strong_agreement_rate for s in window]
        recent_agree = statistics.mean(agreement_values[-10:])
        baseline_agree = statistics.mean(agreement_values[:20])
        if baseline_agree > 0 and abs(recent_agree - baseline_agree) > 0.15:
            alerts.append({
                "type": "agreement_drift",
                "step": current.step,
                "detail": "Weak-strong agreement pattern shifted — model may "
                          "be diverging from or collapsing to weak labels",
                "baseline": baseline_agree,
                "current": recent_agree,
            })

        return alerts

    def summary(self) -> Dict:
        """Return summary statistics across the full training run."""
        if not self.history:
            return {"status": "no_data"}

        return {
            "total_steps": len(self.history),
            "total_alerts": len(self.alerts),
            "alert_types": list({a["type"] for a in self.alerts}),
            "final_loss": self.history[-1].loss,
            "final_confidence_mean": self.history[-1].confidence_mean,
            "final_agreement_rate": self.history[-1].weak_strong_agreement_rate,
            "loss_trajectory": [self.history[0].loss, self.history[-1].loss],
        }
