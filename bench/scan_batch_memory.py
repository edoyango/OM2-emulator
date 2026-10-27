"""
Scan batch size x rollout memory setting on one GPU: the cost of a training
batch at every rollout stage of the schedule, its peak memory, and the
estimated training time of the whole schedule. The data and the model are
loaded once; batch size and rollout checkpointing are changed in place, and an
out-of-memory setting is recorded instead of ending the scan.

Usage, from the repo root:

  .venv/bin/python bench/scan_batch_memory.py --out runs/bench/scan.csv
  .venv/bin/python bench/scan_batch_memory.py --batches 32 64 --modes none 4 --steps 1 12

--modes: "none" = no rollout checkpointing; k = checkpointing with the last k
steps kept in memory (train.rollout_steps_in_memory), so 0 recomputes every
step. The estimate takes each epoch's rollout from window.rollout_schedule
and scales the measured cost per sample to the training windows, so it leaves
out validation, checkpointing and the first epoch's cuDNN autotuning.
"""
import argparse
import gc
import math
import statistics

import pandas as pd
import torch

from common import parse_overrides, notebook_config
from step_bench import Bench, parse_steps

from Experiment.config import rollout_spec  # noqa: E402 (src/ is on the path via common)
from Experiment.training import RolloutSchedule  # noqa: E402


def configure(bench, batch_size, mode):
    bench.train_t0 = bench.data.train_indices[:batch_size].cuda()
    module = bench.module
    if mode == "none":
        module.checkpoint_steps = False
    else:
        module.checkpoint_steps, module.steps_in_memory = True, int(mode)


def free_after_oom(bench):
    bench.optimizer.zero_grad(set_to_none=True)
    for term in bench.losses:
        term.pop_running()
    bench.module.errors["train"].pop_rmse()
    gc.collect()
    torch.cuda.empty_cache()


def scan(bench, batches, modes, steps, warmup, repeats):
    rows = []
    for batch_size in batches:
        for mode in modes:
            configure(bench, batch_size, mode)
            for text in steps:
                stage, spec = parse_steps(text, bench.module)
                row = {"batch": batch_size, "mode": mode, "stage": text}
                try:
                    times, peak, reserved = bench.time(stage, spec, warmup, repeats)
                except torch.OutOfMemoryError:
                    free_after_oom(bench)
                    rows.append({**row, "status": "OOM"})
                    print(f"batch {batch_size:4d} | mode {mode:>4s} | {text:6s} | out of memory", flush=True)
                    continue
                median = statistics.median(times)
                rows.append({**row, "status": "ok", "ms_per_batch": median * 1e3,
                             "ms_per_sample": median * 1e3 / batch_size,
                             "peak_gib": peak / 2**30, "reserved_gib": reserved / 2**30})
                print(f"batch {batch_size:4d} | mode {mode:>4s} | {text:6s} | {median * 1e3:9.1f} ms/batch | "
                      f"{median * 1e3 / batch_size:7.2f} ms/sample | reserved {reserved / 2**30:6.2f} GiB", flush=True)
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def schedule_estimate(results, cfg, n_train):
    """Estimated training time of the schedule (s), optimizer updates and peak memory per batch x mode."""
    w = cfg.window
    schedule = RolloutSchedule(w.rollout_schedule) if w.rollout_schedule is not None else None
    stages = [schedule.steps_for(e) if schedule else rollout_spec(w.rollout_steps) for e in range(cfg.train.max_epochs)]
    stage_text = ["+".join(map(str, s)) if s[0] else str(s[1]) for s in stages]
    rows = []
    for (batch_size, mode), group in results.groupby(["batch", "mode"], sort=False):
        per_sample = dict(zip(group.stage, group.ms_per_sample.where(group.status == "ok")))
        cost = [per_sample.get(t, math.nan) for t in stage_text]
        rows.append({
            "batch": batch_size, "mode": mode,
            "train_s": sum(cost) * n_train / 1e3,
            "updates": cfg.train.max_epochs * math.ceil(n_train / batch_size),
            "max_reserved_gib": group.reserved_gib.max() if "reserved_gib" in group else math.nan,
            "fits": bool((group.status == "ok").all()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64, 128, 186, 372])
    parser.add_argument("--modes", nargs="+", default=["none", "8", "6", "4", "2", "0"])
    parser.add_argument("--steps", nargs="+", default=["1", "4", "12", "24+12"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE", help="config override")
    parser.add_argument("--out", help="CSV of every measurement; the estimate goes next to it (_estimate.csv)")
    args = parser.parse_args()

    cfg = notebook_config(parse_overrides(args.set))
    bench = Bench(cfg)
    n_train = len(bench.data.train_indices)
    batches = [b for b in args.batches if b <= n_train]
    print(f"GPU {torch.cuda.get_device_name()} | {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB | "
          f"torch {torch.__version__} | precision {bench.precision} | {n_train} training windows", flush=True)

    results = scan(bench, batches, args.modes, args.steps, args.warmup, args.repeats)
    estimate = schedule_estimate(results, cfg, n_train)
    if args.out:
        results.to_csv(args.out, index=False)
        estimate.to_csv(args.out.removesuffix(".csv") + "_estimate.csv", index=False)

    ok = results[results.status == "ok"]
    print("\nms per sample by stage (rows: batch, mode):")
    print(ok.pivot_table(index=["batch", "mode"], columns="stage", values="ms_per_sample", sort=False).round(2).to_string())
    print(f"\nEstimated training time of the schedule ({cfg.train.max_epochs} epochs, {n_train} windows; "
          "no validation or checkpoint saving), fastest first:")
    print(estimate.sort_values("train_s").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
