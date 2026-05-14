"""
AlphaAgent — API Monitoring & Metrics

Tracks performance, latency, and success rates for all agents and API endpoints.
Provides the telemetry needed for the Observability Dashboard.
"""

import threading
import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    total_requests: int = 0
    total_errors: int = 0
    average_latency_ms: float = 0.0
    agent_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_reset: datetime = field(default_factory=datetime.utcnow)


class Monitor:
    """Central monitoring class for telemetry. All mutations are lock-protected."""

    _metrics = APIMetrics()
    _lock = threading.Lock()

    @classmethod
    def record_request(cls, latency_ms: float, success: bool):
        with cls._lock:
            cls._metrics.total_requests += 1
            if not success:
                cls._metrics.total_errors += 1
            n = cls._metrics.total_requests
            cls._metrics.average_latency_ms = (
                (cls._metrics.average_latency_ms * (n - 1) + latency_ms) / n
            )

    @classmethod
    def record_agent_run(cls, agent_name: str, latency_ms: float, success: bool):
        with cls._lock:
            if agent_name not in cls._metrics.agent_performance:
                cls._metrics.agent_performance[agent_name] = {
                    "runs": 0,
                    "errors": 0,
                    "avg_latency": 0.0,
                }
            perf = cls._metrics.agent_performance[agent_name]
            perf["runs"] += 1
            if not success:
                perf["errors"] += 1
            n = perf["runs"]
            perf["avg_latency"] = (perf["avg_latency"] * (n - 1) + latency_ms) / n

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        with cls._lock:
            return {
                "uptime_seconds": (datetime.utcnow() - cls._metrics.last_reset).total_seconds(),
                "requests": cls._metrics.total_requests,
                "errors": cls._metrics.total_errors,
                "avg_latency_ms": round(cls._metrics.average_latency_ms, 2),
                "agents": dict(cls._metrics.agent_performance),
            }

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._metrics = APIMetrics()
