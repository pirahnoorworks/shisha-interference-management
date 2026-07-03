# SHISHA Interference Management Lab

A clean, interview-ready project that demonstrates **interference-aware scheduling** for ML inference workloads.

This project is derived from the ideas in your dump's SHISHA runtime and `INTERFERENCE_MODE` flow (paper-3 direction), but rewritten as a compact, reproducible Python benchmark.

## Why hiring teams care

- Shows practical understanding of **tail latency**, **SLO compliance**, and **throughput under contention**.
- Compares a custom scheduler against clear baselines with deterministic simulation.
- Produces artifacts (CSV + optional chart) suitable for portfolio screenshots and discussion.

## What is benchmarked

Schedulers compared:

1. `best_fit`: baseline that always prefers the best-throughput pipeline width.
2. `min_ep`: baseline that minimizes EP footprint if SLO is still met.
3. `shisha_interference`: SHISHA-inspired policy with urgency + interference-aware cost.

Metrics reported:

- Completed jobs
- Throughput (jobs/sec)
- Average latency
- P95 latency
- SLO miss rate
- EP utilization

## How to run

```bash
python demo_interference.py --jobs 300 --seed 42 --plot
```

PowerShell shortcut:

```powershell
./run_demo.ps1
```

Outputs:

- `results/interference_results.csv`
- `results/interference_benchmark.png` (if `matplotlib` installed and `--plot` used)

## Install dependencies

```bash
pip install -r requirements.txt
```

## Suggested talking points for interviews

- Why static best-throughput scheduling can fail under interference.
- How SHISHA seed partitioning and adaptive width selection reduce SLO misses.
- Trade-offs between fairness, utilization, and tail latency.

> Disclaimer: This demo was created for portfolio purposes with assistance from GitHub Copilot.
