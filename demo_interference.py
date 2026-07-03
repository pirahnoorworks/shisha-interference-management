#!/usr/bin/env python3
"""Interference-management benchmark: SHISHA scheduler vs baseline schedulers.

This demo is inspired by the code dump's paper-3 style interference mode:
- SHISHA seed generation merges adjacent stage hints.
- Baselines use static best-width and minimal-width strategies.
- The scheduler compares throughput, latency, SLO misses, and utilization.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


@dataclass
class Job:
    job_id: int
    model: str
    arrival: float
    batch: int
    priority: int
    slo_target: int


@dataclass
class RunningJob:
    job: Job
    width: int
    start_time: float
    end_time: float
    allowed_latency: float


@dataclass
class CompletedJob:
    job: Job
    width: int
    start_time: float
    end_time: float
    latency: float
    allowed_latency: float


MODELS: Dict[str, Dict[str, List[float]]] = {
    "resnet50": {
        "hints": [1.2, 0.8, 1.1, 0.9, 1.5, 0.7, 1.0, 1.4],
        "layer_ms": [2.5, 2.1, 2.3, 2.0, 2.7, 1.9, 2.2, 2.6],
    },
    "vgg16": {
        "hints": [1.0, 1.3, 1.2, 1.4, 1.1, 1.5, 1.2, 1.3],
        "layer_ms": [3.0, 3.4, 3.1, 3.2, 2.9, 3.6, 3.0, 3.1],
    },
    "resnet152": {
        "hints": [1.6, 1.4, 1.5, 1.3, 1.7, 1.6, 1.2, 1.8],
        "layer_ms": [4.2, 3.8, 4.1, 3.9, 4.4, 4.0, 3.7, 4.3],
    },
}

# Pair-wise conflict penalties. These mimic model co-location interference.
INTERFERENCE_TABLE: Dict[Tuple[str, str], float] = {
    ("resnet50", "resnet50"): 0.08,
    ("resnet50", "vgg16"): 0.11,
    ("resnet50", "resnet152"): 0.14,
    ("vgg16", "vgg16"): 0.10,
    ("vgg16", "resnet152"): 0.16,
    ("resnet152", "resnet152"): 0.13,
}


def shisha_seed_partition(hints: List[float], width: int) -> List[int]:
    """Generate a stage partition using adjacent-merge SHISHA seed logic."""
    if width <= 0:
        raise ValueError("width must be positive")
    if width >= len(hints):
        return [1] * len(hints)

    conf = [1] * len(hints)
    weights = hints[:]
    while len(conf) > width:
        best_idx = 0
        best_merge = float("inf")
        for i in range(len(weights) - 1):
            merged = weights[i] + weights[i + 1]
            if merged < best_merge:
                best_merge = merged
                best_idx = i
        conf[best_idx] += conf[best_idx + 1]
        del conf[best_idx + 1]
        weights[best_idx] += weights[best_idx + 1]
        del weights[best_idx + 1]
    return conf


def partition_stage_times(layer_times: List[float], partition: List[int]) -> List[float]:
    stage_times: List[float] = []
    ptr = 0
    for chunk in partition:
        stage_times.append(sum(layer_times[ptr : ptr + chunk]))
        ptr += chunk
    return stage_times


def pipeline_latency_ms(model: str, width: int, batch: int) -> float:
    hints = MODELS[model]["hints"]
    layer_ms = MODELS[model]["layer_ms"]
    partition = shisha_seed_partition(hints, width)
    stages = partition_stage_times(layer_ms, partition)
    bottleneck = max(stages)
    fill_drain = sum(stages)
    return fill_drain + bottleneck * max(0, batch - 1)


def best_width_for_model(model: str, max_width: int) -> int:
    best_width = 1
    best = float("inf")
    for width in range(1, max_width + 1):
        t = pipeline_latency_ms(model, width, batch=2)
        if t < best:
            best = t
            best_width = width
    return best_width


def acceptable_widths(job: Job, max_width: int) -> List[int]:
    best_latency = min(pipeline_latency_ms(job.model, w, job.batch) for w in range(1, max_width + 1))
    slowdown_budget = 100 - job.slo_target
    allowed = best_latency * (1.0 + slowdown_budget / 100.0)
    widths = []
    for w in range(1, max_width + 1):
        if pipeline_latency_ms(job.model, w, job.batch) <= allowed:
            widths.append(w)
    return widths or [best_width_for_model(job.model, max_width)]


def interference_multiplier(candidate: str, running_models: List[str]) -> float:
    if not running_models:
        return 1.0
    penalty = 0.0
    for m in running_models:
        pair = tuple(sorted((candidate, m)))
        penalty += INTERFERENCE_TABLE.get(pair, 0.09)
    # A small global contention bump for occupancy.
    return 1.0 + penalty + 0.03 * max(0, len(running_models) - 1)


def build_jobs(count: int, seed: int, arrival_rate: float) -> List[Job]:
    rnd = random.Random(seed)
    jobs: List[Job] = []
    t = 0.0
    models = list(MODELS.keys())
    for i in range(count):
        t += rnd.expovariate(arrival_rate)
        jobs.append(
            Job(
                job_id=i,
                model=rnd.choice(models),
                arrival=t,
                batch=rnd.randint(1, 5),
                priority=rnd.randint(1, 5),
                slo_target=rnd.randint(70, 95),
            )
        )
    return jobs


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def pick_candidate(
    policy: str,
    waiting: List[Job],
    free_eps: int,
    running: List[RunningJob],
    max_width: int,
    now: float,
) -> Optional[Tuple[int, int, float]]:
    if free_eps <= 0 or not waiting:
        return None

    running_models = [r.job.model for r in running]

    if policy == "best_fit":
        for idx, job in enumerate(waiting):
            width = best_width_for_model(job.model, max_width)
            if width <= free_eps:
                base = pipeline_latency_ms(job.model, width, job.batch)
                lat = base * interference_multiplier(job.model, running_models)
                return idx, width, lat
        return None

    if policy == "min_ep":
        for idx, job in enumerate(waiting):
            choices = [w for w in acceptable_widths(job, max_width) if w <= free_eps]
            if choices:
                width = min(choices)
                base = pipeline_latency_ms(job.model, width, job.batch)
                lat = base * interference_multiplier(job.model, running_models)
                return idx, width, lat
        return None

    if policy == "shisha_interference":
        # Prioritize urgent, higher-priority jobs.
        ranked = list(enumerate(waiting))
        ranked.sort(
            key=lambda x: (
                x[1].priority,
                -((now - x[1].arrival) / max(1.0, x[1].slo_target)),
                x[1].arrival,
            )
        )

        best_choice: Optional[Tuple[int, int, float, float]] = None
        for idx, job in ranked:
            widths = [w for w in acceptable_widths(job, max_width) if w <= free_eps]
            if not widths:
                continue
            for width in widths:
                base = pipeline_latency_ms(job.model, width, job.batch)
                lat = base * interference_multiplier(job.model, running_models)
                # Cost combines latency, queue pressure, and EP footprint.
                urgency = (now - job.arrival) / max(1.0, job.slo_target)
                cost = lat + 1.5 * width - 4.0 * urgency
                if best_choice is None or cost < best_choice[3]:
                    best_choice = (idx, width, lat, cost)

        if best_choice is None:
            return None
        return best_choice[0], best_choice[1], best_choice[2]

    raise ValueError(f"Unknown policy: {policy}")


def simulate(policy: str, jobs: List[Job], total_eps: int, max_width: int) -> Dict[str, float]:
    waiting: List[Job] = []
    running: List[RunningJob] = []
    done: List[CompletedJob] = []

    i = 0
    now = 0.0
    busy_area = 0.0

    while i < len(jobs) or waiting or running:
        next_arrival = jobs[i].arrival if i < len(jobs) else float("inf")
        next_finish = min((r.end_time for r in running), default=float("inf"))
        next_time = min(next_arrival, next_finish)

        if next_time == float("inf"):
            break

        in_use = sum(r.width for r in running)
        busy_area += in_use * max(0.0, next_time - now)
        now = next_time

        completed_now = [r for r in running if abs(r.end_time - now) < 1e-9]
        if completed_now:
            for r in completed_now:
                done.append(
                    CompletedJob(
                        job=r.job,
                        width=r.width,
                        start_time=r.start_time,
                        end_time=r.end_time,
                        latency=r.end_time - r.job.arrival,
                        allowed_latency=r.allowed_latency,
                    )
                )
            running = [r for r in running if r not in completed_now]

        while i < len(jobs) and abs(jobs[i].arrival - now) < 1e-9:
            waiting.append(jobs[i])
            i += 1

        # Keep launching jobs while resources are available.
        while True:
            free_eps = total_eps - sum(r.width for r in running)
            choice = pick_candidate(policy, waiting, free_eps, running, max_width, now)
            if choice is None:
                break

            idx, width, pred_latency = choice
            job = waiting.pop(idx)

            best_latency = min(pipeline_latency_ms(job.model, w, job.batch) for w in range(1, max_width + 1))
            allowed_latency = best_latency * (1.0 + (100 - job.slo_target) / 100.0)
            running.append(
                RunningJob(
                    job=job,
                    width=width,
                    start_time=now,
                    end_time=now + pred_latency,
                    allowed_latency=allowed_latency,
                )
            )

    latencies = sorted([d.latency for d in done])
    slo_misses = sum(1 for d in done if d.latency > d.allowed_latency)
    makespan = max((d.end_time for d in done), default=1.0)

    return {
        "completed": float(len(done)),
        "throughput_jobs_per_sec": len(done) / makespan,
        "avg_latency_sec": mean(latencies) if latencies else 0.0,
        "p95_latency_sec": percentile(latencies, 0.95),
        "slo_miss_rate": (slo_misses / len(done)) if done else 0.0,
        "utilization": busy_area / (total_eps * makespan),
    }


def write_results_csv(path: Path, results: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for policy, metrics in results.items():
        row = {"scheduler": policy}
        row.update(metrics)
        rows.append(row)

    fields = list(rows[0].keys()) if rows else ["scheduler"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_chart(path: Path, results: Dict[str, Dict[str, float]]) -> None:
    if plt is None:
        return

    policies = list(results.keys())
    throughput = [results[p]["throughput_jobs_per_sec"] for p in policies]
    miss = [results[p]["slo_miss_rate"] * 100.0 for p in policies]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = range(len(policies))
    ax1.bar([i - 0.18 for i in x], throughput, width=0.36, label="Throughput (jobs/s)", color="#1f77b4")
    ax2.bar([i + 0.18 for i in x], miss, width=0.36, label="SLO Miss Rate (%)", color="#ff7f0e")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(policies)
    ax1.set_ylabel("Throughput (jobs/s)")
    ax2.set_ylabel("SLO Miss Rate (%)")
    ax1.set_title("Interference Benchmark: SHISHA vs Baselines")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interference-management scheduler demo")
    parser.add_argument("--jobs", type=int, default=300, help="Number of arriving inference jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--arrival-rate", type=float, default=1.35, help="Poisson arrival rate")
    parser.add_argument("--eps", type=int, default=16, help="Total execution processors")
    parser.add_argument("--max-width", type=int, default=8, help="Max pipeline width")
    parser.add_argument("--plot", action="store_true", help="Render PNG chart if matplotlib is installed")
    args = parser.parse_args()

    jobs = build_jobs(count=args.jobs, seed=args.seed, arrival_rate=args.arrival_rate)
    results: Dict[str, Dict[str, float]] = {}

    for policy in ["best_fit", "min_ep", "shisha_interference"]:
        results[policy] = simulate(policy, jobs, total_eps=args.eps, max_width=args.max_width)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "interference_results.csv"
    write_results_csv(csv_path, results)

    print("\n=== Interference Management Results ===")
    for policy, metrics in results.items():
        print(f"\n[{policy}]")
        for k, v in metrics.items():
            if k in {"completed"}:
                print(f"  {k:24s}: {int(v)}")
            else:
                print(f"  {k:24s}: {v:.4f}")

    print(f"\nSaved: {csv_path}")

    if args.plot:
        png_path = out_dir / "interference_benchmark.png"
        render_chart(png_path, results)
        if png_path.exists():
            print(f"Saved: {png_path}")
        elif plt is None:
            print("Plot skipped: matplotlib is not installed.")


if __name__ == "__main__":
    main()
