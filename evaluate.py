"""Reproduce data audit, historical rule replay and explicit cost scenarios."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from policy import CLUSTERS, EAST, decide


AGE_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60]
VRAM_GRID = [0, 2, 4, 6, 8, 10, 12]
TRAIN_END = "2026-09-05"


def wilson(k, n):
    if not n:
        return (None, None)
    z = 1.959963984540054
    p = k / n
    middle = (p + z*z/(2*n)) / (1+z*z/n)
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return middle-width, middle+width


def load_data(source):
    raw = pd.read_csv(source, dtype={"task_id": "string"})
    required = {"task_id", "dispatch_time", "status_synced_at", "cluster_name",
                "task_type", "node_free_vram", "payload_size_mb", "is_success"}
    if set(raw.columns) != required:
        raise ValueError("Input columns do not match the task schema")
    clean = raw.drop_duplicates().copy()
    if clean.task_id.isna().any() or clean.task_id.duplicated().any():
        raise ValueError("Missing IDs or conflicting duplicate IDs require investigation")
    labels = pd.to_numeric(clean.is_success, errors="raise")
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError("Outcome must be 0, 1 or blank")
    clean["is_success"] = labels
    clean["dispatch_utc"] = pd.to_datetime(clean.dispatch_time, utc=True, errors="raise")
    clean["synced_utc"] = pd.to_datetime(clean.status_synced_at, utc=True, errors="raise")
    clean["status_age_min"] = (clean.dispatch_utc-clean.synced_utc).dt.total_seconds()/60
    clean["day"] = clean.dispatch_utc.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    for name in ["node_free_vram", "payload_size_mb"]:
        clean[name] = pd.to_numeric(clean[name], errors="raise")
        if clean[name].isna().any() or not clean[name].map(math.isfinite).all() or clean[name].lt(0).any():
            raise ValueError(f"Invalid {name}")
    if clean[["dispatch_utc", "synced_utc", "cluster_name", "task_type"]].isna().any().any():
        raise ValueError("Missing dispatch-time fields")
    if clean.status_age_min.lt(0).any() or not clean.cluster_name.isin(CLUSTERS).all():
        raise ValueError("Future sync time or unknown cluster")
    clean["failure"] = 1-clean.is_success
    clean = clean.sort_values(["dispatch_utc", "task_id"]).reset_index(drop=True)
    audit = {
        "source": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "exact_duplicate_rows_removed": len(raw)-len(clean),
        "unique_dispatches": len(clean), "labeled_dispatches": int(clean.is_success.notna().sum()),
        "successful_dispatches": int(clean.is_success.eq(1).sum()),
        "failed_dispatches": int(clean.is_success.eq(0).sum()),
        "pending_dispatches": int(clean.is_success.isna().sum()),
        "raw_missing_by_column": {k: int(v) for k, v in raw.isna().sum().items()},
        "period_start": clean.dispatch_utc.min().isoformat(),
        "period_end": clean.dispatch_utc.max().isoformat(),
        "deduplication": "Remove only rows identical across all 8 source columns; reject conflicting IDs",
        "training_end_local_date_inclusive": TRAIN_END,
    }
    return raw, clean, audit


def signal_table(frame, groups):
    rows = []
    for keys, block in frame.groupby(groups, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        known = block[block.is_success.notna()]
        n = len(known)
        k = int(known.failure.sum())
        low, high = wilson(k, n)
        rows.append(dict(zip(groups, keys)) | {
            "all_n": len(block), "known_n": n, "pending_n": len(block)-n,
            "failures": k, "failure_rate": k/n if n else None,
            "wilson_95_low": low, "wilson_95_high": high,
        })
    return pd.DataFrame(rows)


def mask_for(frame, name, age, vram):
    stale = frame.status_age_min.gt(age)
    low = frame.node_free_vram.lt(vram)
    east = frame.cluster_name.eq(EAST)
    return {
        "no_guard": lambda: pd.Series(False, index=frame.index),
        "guard_all": lambda: pd.Series(True, index=frame.index),
        "stale_only": lambda: stale,
        "global_stale_or_low_vram": lambda: stale | low,
        "cluster_specific": lambda: (east & low) | (~east & stale),
    }[name]()


def score(frame, hit, q, failure_cost, guard_cost):
    known = frame.is_success.notna()
    tp = int((hit & frame.is_success.eq(0)).sum())
    fp = int((hit & frame.is_success.eq(1)).sum())
    fn = int((~hit & frame.is_success.eq(0)).sum())
    tn = int((~hit & frame.is_success.eq(1)).sum())
    unknown_hit = int((hit & ~known).sum())
    baseline = (tp+fn)*failure_cost
    hit_known = tp+fp
    savings = failure_cost*q*tp-guard_cost*hit_known
    low, high = wilson(tp, hit_known)
    return {
        "all_n": len(frame), "known_n": int(known.sum()),
        "hit_all": int(hit.sum()), "hit_known": hit_known,
        "pending_hit": unknown_hit, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp/hit_known if hit_known else None,
        "precision_wilson_low": low, "precision_wilson_high": high,
        "recall": tp/(tp+fn) if tp+fn else None,
        "guard_rate_all": float(hit.mean()) if len(frame) else None,
        "baseline_known_minutes": baseline,
        "scenario_q": q,
        "scenario_known_minutes": baseline-savings,
        "scenario_known_savings_minutes": savings,
        "scenario_known_savings_percent": savings/baseline if baseline else None,
        "break_even_q": guard_cost*hit_known/(failure_cost*tp) if tp else None,
        "pending_all_success_savings_minutes": savings-guard_cost*unknown_hit,
        "pending_all_failure_savings_minutes": savings+(failure_cost*q-guard_cost)*unknown_hit,
    }


def chart(signals, comparison, sensitivity, out, q):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    clusters = sorted(signals.cluster_name.unique())
    for offset, stale, label, color in [(-.18, False, "状态延迟 ≤20 分钟", "#4263eb"), (.18, True, "状态延迟 >20 分钟", "#e8590c")]:
        part = signals[signals.stale.eq(stale)].set_index("cluster_name").loc[clusters]
        axes[0].bar([i+offset for i in range(len(clusters))], part.failure_rate*100, width=.34, label=label, color=color)
    axes[0].set_xticks(range(len(clusters)), [c[:2] for c in clusters])
    axes[0].set_ylabel("历史失败率（已回传，%）")
    axes[0].set_title("集群差异决定拦截方式")
    axes[0].legend(fontsize=8)
    names = ["不拦截", "全部拦截", "统一状态阈值", "统一状态或显存", "分集群规则"]
    part = comparison[comparison.period.eq("later_two_days")]
    axes[1].barh(names, part.scenario_known_minutes, color=["#adb5bd"]*4+["#4263eb"])
    axes[1].set_xlabel("情景等待损失（任务分钟）")
    axes[1].set_title(f"后两天复核：假设重试有效率 {q:.0%}")
    for i, value in enumerate(part.scenario_known_minutes):
        axes[1].text(value+25, i, f"{value:,.0f}", va="center", fontsize=9)
    axes[1].set_xlim(0, max(part.scenario_known_minutes)*1.2)
    part = sensitivity[sensitivity.period.eq("all_days")]
    axes[2].plot(part.scenario_q*100, part.scenario_known_savings_minutes, color="#4263eb", marker="o")
    axes[2].axhline(0, color="#868e96", linewidth=1)
    axes[2].set_xlabel("假设：被命中失败的重试有效率（%）")
    axes[2].set_ylabel("情景净节省（任务分钟）")
    axes[2].set_title("收益取决于重试是否真正有效")
    fig.suptitle("调度拦截历史回放；情景值不等于已实现收益", fontsize=15)
    fig.savefig(out/"findings.png", dpi=160)
    plt.close(fig)


def run(source, out, q=.5, failure_cost=120., guard_cost=10.):
    if not 0 <= q <= 1 or failure_cost <= 0 or guard_cost < 0:
        raise ValueError("Invalid scenario assumptions")
    out.mkdir(parents=True, exist_ok=True)
    raw, frame, audit = load_data(source)
    raw[raw.duplicated(keep="first")].to_csv(out/"duplicate_rows.csv", index=False, encoding="utf-8-sig")
    frame["stale"] = frame.status_age_min.gt(20)
    frame["vram_below_4"] = frame.node_free_vram.lt(4)
    frame["age_band"] = pd.cut(frame.status_age_min, [-1,5,10,15,20,25,30,40,50,70,math.inf]).astype(str)
    frame["payload_band"] = pd.cut(frame.payload_size_mb, [-1,20,40,60,80,math.inf]).astype(str)
    signals = signal_table(frame, ["cluster_name", "stale"])
    signals.to_csv(out/"cluster_staleness.csv", index=False, encoding="utf-8-sig")
    for name, groups in {
        "cluster_vram": ["cluster_name", "vram_below_4"],
        "cluster_interactions": ["cluster_name", "stale", "vram_below_4"],
        "task_type": ["task_type"], "payload": ["payload_band"],
        "age": ["age_band"], "daily": ["day"],
    }.items():
        signal_table(frame, groups).to_csv(out/f"{name}.csv", index=False, encoding="utf-8-sig")
    train = frame[frame.day.le(TRAIN_END)]
    later = frame[frame.day.gt(TRAIN_END)]
    search = []
    for age in AGE_GRID:
        for vram in VRAM_GRID:
            row = score(train, mask_for(train, "cluster_specific", age, vram), q, failure_cost, guard_cost)
            search.append({"age_threshold": age, "east_vram_threshold": vram} | row)
    search = pd.DataFrame(search).sort_values(
        ["scenario_known_savings_minutes", "hit_all", "age_threshold", "east_vram_threshold"],
        ascending=[False, True, True, True], kind="stable")
    search.to_csv(out/"threshold_search_train_only.csv", index=False)
    best = search.iloc[0]
    age, vram = int(best.age_threshold), int(best.east_vram_threshold)
    names = ["no_guard", "guard_all", "stale_only", "global_stale_or_low_vram", "cluster_specific"]
    comparisons, sensitivity = [], []
    for period, block in [("first_five_days", train), ("later_two_days", later), ("all_days", frame)]:
        for name in names:
            comparisons.append({"period": period, "policy": name} | score(block, mask_for(block, name, age, vram), q, failure_cost, guard_cost))
        for qi in range(11):
            sensitivity.append({"period": period} | score(block, mask_for(block, "cluster_specific", age, vram), qi/10, failure_cost, guard_cost))
    comparison = pd.DataFrame(comparisons)
    sensitivity = pd.DataFrame(sensitivity)
    comparison.to_csv(out/"policy_comparison.csv", index=False)
    sensitivity.to_csv(out/"retry_effectiveness_sensitivity.csv", index=False)
    decisions = [decide(row.cluster_name, row.node_free_vram, row.dispatch_time, row.status_synced_at,
                        age_threshold=age, east_vram_threshold=vram) for row in frame.itertuples()]
    frame["action"] = [d.action for d in decisions]
    frame["reason"] = [d.reason for d in decisions]
    hit = frame.action.eq("RECHECK_AND_REROUTE")
    if not hit.equals(mask_for(frame, "cluster_specific", age, vram)):
        raise AssertionError("Dispatch guard differs from replay rule")
    frame.to_csv(out/"dispatch_decisions.csv", index=False, encoding="utf-8-sig")
    daily = []
    for day, block in frame.groupby("day"):
        daily.append({"day": day} | score(block, mask_for(block, "cluster_specific", age, vram), q, failure_cost, guard_cost))
    pd.DataFrame(daily).to_csv(out/"daily_policy_replay.csv", index=False)
    summary = audit | {
        "selected_age_threshold_minutes_strictly_greater": age,
        "selected_east_vram_gb_strictly_less": vram,
        "training_candidate_count": len(search),
        "failure_cost_minutes": failure_cost, "guard_cost_minutes": guard_cost,
        "assumed_retry_effectiveness": q,
        "all_days_selected": comparison[(comparison.period.eq("all_days")) & comparison.policy.eq("cluster_specific")].iloc[0].to_dict(),
        "later_two_days_selected": comparison[(comparison.period.eq("later_two_days")) & comparison.policy.eq("cluster_specific")].iloc[0].to_dict(),
        "validation_note": "Retrospective temporal stability check; whole-dataset exploration preceded design, so not a pristine untouched test or causal experiment.",
    }
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    chart(signals, comparison, sensitivity, out, q)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/task_dispatches.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--retry-effectiveness", type=float, default=.5)
    parser.add_argument("--failure-cost", type=float, default=120.)
    parser.add_argument("--guard-cost", type=float, default=10.)
    args = parser.parse_args()
    run(args.csv, args.out, args.retry_effectiveness, args.failure_cost, args.guard_cost)
