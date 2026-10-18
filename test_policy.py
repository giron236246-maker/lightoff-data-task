"""Synthetic boundary inputs test software only, never business conclusions."""
import csv
from datetime import datetime
import json
from pathlib import Path
import unittest
from policy import EAST, decide


class GuardTests(unittest.TestCase):
    def decision(self, cluster="华北通用算力池", vram=10,
                 synced="2026-09-01T00:40:00Z", retry=0):
        return decide(cluster, vram, "2026-09-01T09:00:00+08:00", synced, retry_count=retry)

    def test_age_boundary_and_timezone(self):
        self.assertEqual(self.decision().action, "PASS")
        self.assertEqual(self.decision(synced="2026-09-01T00:39:59Z").action, "RECHECK_AND_REROUTE")

    def test_vram_boundary(self):
        self.assertEqual(self.decision(cluster=EAST, vram=4).action, "PASS")
        self.assertEqual(self.decision(cluster=EAST, vram=3.99).action, "RECHECK_AND_REROUTE")

    def test_cluster_interaction(self):
        self.assertEqual(self.decision(cluster=EAST, synced="2026-09-01T00:00:00Z").action, "PASS")
        self.assertEqual(self.decision(vram=0).action, "PASS")
        # PASS is not resource-fit approval: the existing resource check still applies.

    def test_invalid_states(self):
        self.assertEqual(self.decision(cluster="unknown").action, "RECHECK")
        for vram in [None, -1, float("nan"), float("inf"), "oops"]:
            self.assertEqual(self.decision(vram=vram).action, "RECHECK")
        for synced in ["bad", "2026-09-01T01:01:00Z", "2026-09-01T00:40:00"]:
            self.assertEqual(self.decision(synced=synced).action, "RECHECK")

    def test_retry_budget(self):
        self.assertEqual(self.decision(retry=1).action, "ESCALATE")
        self.assertEqual(self.decision(retry=2).action, "ESCALATE")
        self.assertEqual(self.decision(retry=-1).action, "RECHECK")


class SuppliedDataReconciliation(unittest.TestCase):
    def test_independent_stdlib_reconciliation(self):
        with Path("data/task_dispatches.csv").open(encoding="utf-8-sig", newline="") as source:
            raw = list(csv.DictReader(source))
        unique = {tuple(row.items()): row for row in raw}.values()
        counts = dict(known=0, pending=0, tp=0, fp=0, fn=0, tn=0, hit_pending=0)
        for row in unique:
            at = datetime.fromisoformat(row["dispatch_time"])
            synced = datetime.fromisoformat(row["status_synced_at"].replace("Z", "+00:00"))
            hit = (float(row["node_free_vram"]) < 4 if row["cluster_name"] == EAST
                   else (at-synced).total_seconds() > 1200)
            if row["is_success"] == "":
                counts["pending"] += 1
                counts["hit_pending"] += int(hit)
            else:
                counts["known"] += 1
                failed = float(row["is_success"]) == 0
                counts["tp" if hit and failed else "fp" if hit else "fn" if failed else "tn"] += 1
        self.assertEqual(len(raw), 1138)
        self.assertEqual(len(unique), 1084)
        self.assertEqual(counts, dict(known=1037, pending=47, tp=120, fp=137, fn=25, tn=755, hit_pending=8))
        summary = json.loads(Path("outputs/summary.json").read_text(encoding="utf-8"))
        actual = summary["all_days_selected"]
        for key in ["tp", "fp", "fn", "tn"]:
            self.assertEqual(actual[key], counts[key])
        self.assertEqual(actual["scenario_known_savings_minutes"], 120*.5*counts["tp"]-10*(counts["tp"]+counts["fp"]))
        self.assertEqual(summary["training_candidate_count"], 63)
        self.assertEqual(summary["selected_age_threshold_minutes_strictly_greater"], 20)
        self.assertEqual(summary["selected_east_vram_gb_strictly_less"], 4)
        with Path("outputs/dispatch_decisions.csv").open(encoding="utf-8-sig", newline="") as source:
            decisions = list(csv.DictReader(source))
        self.assertEqual(len(decisions), 1084)
        self.assertEqual(sum(row["action"] == "RECHECK_AND_REROUTE" for row in decisions), 265)


if __name__ == "__main__":
    unittest.main()
