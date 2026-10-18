"""Dispatch-time guard. No outcome labels or future information are inputs."""

from dataclasses import dataclass
from datetime import datetime
import math


CLUSTERS = {"华东推理集群", "华北通用算力池", "华南视频渲染池", "西南音视频节点组"}
EAST = "华东推理集群"


@dataclass(frozen=True)
class GuardDecision:
    action: str
    reason: str
    status_age_minutes: float | None


def decide(cluster_name, node_free_vram, dispatch_time, status_synced_at,
           retry_count=0, age_threshold=20.0, east_vram_threshold=4.0):
    """Return a proposed guard action, not a claim that a retry will succeed.

    Refresh/revalidation, atomic memory reservation and destination selection
    must be implemented by the scheduler before any actual dispatch.
    """
    if not isinstance(retry_count, int) or isinstance(retry_count, bool) or retry_count < 0:
        return GuardDecision("RECHECK", "invalid_retry_count", None)
    if retry_count >= 1:
        return GuardDecision("ESCALATE", "automatic_retry_budget_exhausted", None)
    if cluster_name not in CLUSTERS:
        return GuardDecision("RECHECK", "unknown_cluster", None)
    try:
        vram = float(node_free_vram)
        at = datetime.fromisoformat(str(dispatch_time).replace("Z", "+00:00"))
        synced = datetime.fromisoformat(str(status_synced_at).replace("Z", "+00:00"))
        if at.utcoffset() is None or synced.utcoffset() is None:
            raise ValueError("timezone required")
        age = (at - synced).total_seconds() / 60
        if not math.isfinite(vram) or vram < 0 or age < 0:
            raise ValueError("invalid state")
    except (ValueError, TypeError, OverflowError):
        return GuardDecision("RECHECK", "invalid_state_or_timestamp", None)
    if cluster_name == EAST and vram < east_vram_threshold:
        return GuardDecision("RECHECK_AND_REROUTE", "east_low_vram", age)
    if cluster_name != EAST and age > age_threshold:
        return GuardDecision("RECHECK_AND_REROUTE", "stale_node_status", age)
    return GuardDecision("PASS", "no_selected_risk_signal", age)
