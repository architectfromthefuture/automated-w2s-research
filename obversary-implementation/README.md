# Obversary Implementation: Signal-Gated Architecture for W2S

This directory demonstrates how signal-gated cognitive architecture maps onto the weak-to-strong supervision sandbox from [Wen et al. 2026](https://alignment.anthropic.com/2026/automated-w2s-researcher/). Each component addresses a specific limitation identified in the original paper and connects to governance patterns from [OpenAI's Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf).

## Components

### `signal_detector.py` — Task Classification Before Routing

Classifies training examples by structural characteristics (ambiguous labels, weak-strong disagreement, potential shortcuts, distribution shifts) and routes them to appropriate processing paths.

**Addresses:** The paper found that prescribed sequential scaffolding underperforms autonomous routing. Signal detection is the architectural explanation — let the task determine the processing path instead of forcing a fixed pipeline.

### `behavioral_monitor.py` — Process Metrics Alongside Outcome Metrics

Tracks observable behavioral signatures during training: confidence dynamics, prediction stability, timing anomalies, weak-strong agreement drift. Fires alerts when patterns indicate overfitting, distribution shift, or optimization instability.

**Addresses:** The paper showed that PGR (a single scalar) was gamed via seed cherry-picking, label exfiltration, and shortcut exploitation. Behavioral monitoring adds orthogonal signals that reveal HOW the model is learning, not just WHAT it produces. Adapted from [xray-wrapper](https://github.com/architectfromthefuture/xray-wrapper).

### `multi_critic_evaluator.py` — Independent Critics, Not a Single Score

Four independent critics evaluate different dimensions:

- **Outcome Critic** — standard PGR (necessary but not sufficient)
- **Behavioral Critic** — training process health from the monitor
- **Provenance Critic** — can the result be traced and explained?
- **Consistency Critic** — does the result generalize across distributions?

**Addresses:** The paper's reward hacking findings and OpenAI's multi-channel safeguard architecture (Table 3). Maps to OpenAI's Robustness + Usage Monitoring + Trust-based Access + System Oversight pattern.

### `memory_substrate.py` — Event-Driven Research Trajectories

Persists the full research trajectory as structured events: hypotheses proposed, experiments run, failures encountered, shortcuts detected. Makes past failures searchable so future runs do not repeat known-bad approaches.

**Addresses:** Anthropic's finding that local browsable memory outperformed remote keyword/MCP search, and their observation that "richer logs of science" — full trajectories including failures — are a missing asset. Adapted from [memory-dropbox](https://github.com/architectfromthefuture/memory-dropbox).

## Connection to the Original Paper

| Paper Finding | Component | Architectural Pattern |
|---|---|---|
| Less imposed structure → better performance | `signal_detector.py` | Signal-responsive routing over prescribed pipelines |
| Entropy collapse without diverse starting routes | `signal_detector.py` | Typed signals as structural diversity priors |
| Reward hacking via single metric (PGR) | `multi_critic_evaluator.py` | Multi-critic evaluation with independent inductive biases |
| Local memory > remote search > MCP search | `memory_substrate.py` | Memory-as-substrate over memory-as-query |
| Full trajectory as training data for future agents | `memory_substrate.py` | Event-driven logging with provenance |
| Behavioral shortcuts not caught by outcome metrics | `behavioral_monitor.py` | Process observation alongside output evaluation |

## Connection to OpenAI Preparedness Framework v2

| Framework Element | Component | Mapping |
|---|---|---|
| Multi-channel safeguards (Table 3) | `multi_critic_evaluator.py` | Independent critics = independent safeguard channels |
| Sandbagging (Research Category) | `behavioral_monitor.py` | Confidence collapse detection |
| Undermining Safeguards (Research Category) | `memory_substrate.py` | Shortcut/hack detection persisted as memory |
| Staged capability thresholds (High → Critical) | `signal_detector.py` | Signal-gated routing with escalation paths |
| System Architecture safeguards | All components | Structural constraints on the optimization surface |

## Author

Brian Moran, [Obversary Studios LLC](https://github.com/architectfromthefuture)
