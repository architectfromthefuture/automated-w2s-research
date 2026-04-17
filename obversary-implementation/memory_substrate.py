"""
Event-driven memory substrate for research trajectories.

Persists the FULL trajectory of research — what was tried, what failed,
what worked, and why — as structured events, not text logs. This makes
the trajectory searchable, auditable, and reusable.

Mapping to W2S findings:
- Anthropic noted: "One valuable resource that has been missing from
  science is the full trajectory of discoveries... every negative result,
  every dead-end hyperparameter is recorded by default."
- Their finding that local browsable memory outperformed remote keyword
  search and MCP-based retrieval validates the memory-as-substrate
  design pattern.

Adapted from memory-dropbox (github.com/architectfromthefuture/memory-dropbox).

Author: Brian Moran, Obversary Studios LLC
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path


class EventType(str, Enum):
    hypothesis_proposed = "hypothesis_proposed"
    experiment_started = "experiment_started"
    experiment_completed = "experiment_completed"
    experiment_failed = "experiment_failed"
    evaluation_submitted = "evaluation_submitted"
    evaluation_result = "evaluation_result"
    alert_triggered = "alert_triggered"
    method_modified = "method_modified"
    finding_shared = "finding_shared"
    shortcut_detected = "shortcut_detected"
    ablation_run = "ablation_run"
    direction_changed = "direction_changed"


@dataclass
class ResearchEvent:
    event_type: EventType
    timestamp: float
    agent_id: str
    data: Dict[str, Any]
    parent_event_id: Optional[str] = None
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.agent_id}_{self.event_type.value}_{int(self.timestamp * 1000)}"


class MemorySubstrate:
    """
    In-memory event store with persistence to JSONL files.

    Design principles (from memory-dropbox architecture):
    - Events are primary data. Conclusions are derived views.
    - Every event has provenance (agent_id, timestamp, parent).
    - The full trajectory is browsable, not just queryable.
    - Synchronous writes — no fire-and-forget.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.events: List[ResearchEvent] = []
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: ResearchEvent) -> str:
        """Record an event. Returns the event ID."""
        self.events.append(event)
        if self.persist_path:
            self._persist(event)
        return event.event_id

    def _persist(self, event: ResearchEvent):
        with open(self.persist_path, "a") as f:
            record = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "agent_id": event.agent_id,
                "parent_event_id": event.parent_event_id,
                "data": event.data,
            }
            f.write(json.dumps(record) + "\n")

    def query_by_type(self, event_type: EventType) -> List[ResearchEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def query_by_agent(self, agent_id: str) -> List[ResearchEvent]:
        return [e for e in self.events if e.agent_id == agent_id]

    def get_failures(self) -> List[ResearchEvent]:
        return self.query_by_type(EventType.experiment_failed)

    def get_shortcuts_detected(self) -> List[ResearchEvent]:
        return self.query_by_type(EventType.shortcut_detected)

    def trajectory(self, agent_id: str) -> List[Dict]:
        """
        Return the full research trajectory for an agent as a
        chronological list. This is the "richer log of science" that
        Anthropic identified as a missing asset.
        """
        agent_events = sorted(
            self.query_by_agent(agent_id),
            key=lambda e: e.timestamp,
        )
        return [
            {
                "event_id": e.event_id,
                "type": e.event_type.value,
                "time": e.timestamp,
                "data": e.data,
                "parent": e.parent_event_id,
            }
            for e in agent_events
        ]

    def known_failures(self) -> List[Dict]:
        """
        Return all recorded failures so future agents can check
        against them BEFORE trying the same approach.

        This is memory-persistent evaluation: past hacks and failures
        are flagged by memory match, not by rules.
        """
        return [
            {
                "agent": e.agent_id,
                "time": e.timestamp,
                "what_failed": e.data.get("method", "unknown"),
                "why": e.data.get("reason", "unknown"),
                "pgr": e.data.get("pgr"),
            }
            for e in self.get_failures()
        ]

    def summary(self) -> Dict:
        """High-level summary of the memory substrate state."""
        type_counts: Dict[str, int] = {}
        for e in self.events:
            type_counts[e.event_type.value] = type_counts.get(e.event_type.value, 0) + 1

        agents = list({e.agent_id for e in self.events})

        return {
            "total_events": len(self.events),
            "event_types": type_counts,
            "agents": agents,
            "failures": len(self.get_failures()),
            "shortcuts_detected": len(self.get_shortcuts_detected()),
        }
