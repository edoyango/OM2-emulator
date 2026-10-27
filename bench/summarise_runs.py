"""
Side-by-side summary of run_timed.py results: training time, median epoch
time per rollout stage, final validation RMSE and skill-test metrics.

Usage: .venv/bin/python bench/summarise_runs.py runs/bench/baseline runs/bench/optimised ...
"""
import json
import sys
from pathlib import Path

import pandas as pd


def main(paths):
    runs = {Path(p).name: Path(p) for p in paths if (Path(p) / "summary.json").exists()}
    if not runs:
        sys.exit("No run_timed.py results (summary.json) in " + " ".join(paths))
    summary = {name: json.loads((path / "summary.json").read_text()) for name, path in runs.items()}
    epochs = {name: pd.read_csv(path / "epoch_times.csv") for name, path in runs.items()}
    skill = {name: pd.read_csv(path / "skill_by_lead.csv") for name, path in runs.items()}
    tables = {
        "Training (s), optimizer updates, growth factor": pd.DataFrame({
            name: {"fit_s": s["fit_s"], "updates": s["optimizer_steps"], "growth": s["leading_growth_factor"]}
            for name, s in summary.items()
        }),
        "Median epoch time (s) by rollout, epoch 0 excluded": pd.DataFrame({
            name: e.iloc[1:].groupby("rollout").total_s.median() for name, e in epochs.items()
        }),
        "Final validation RMSE (normalised)": pd.DataFrame({name: s["final_val_rmse"] for name, s in summary.items()}),
        "Skill test: mean ACC over all leads": pd.DataFrame({
            name: k.groupby("variable").acc_emulator.mean() for name, k in skill.items()
        }),
        "Skill test: mean RMSE / climatology RMSE (< 1 beats climatology)": pd.DataFrame({
            name: k.groupby("variable").rmse_emulator.mean() / k.groupby("variable").rmse_climatology.mean()
            for name, k in skill.items()
        }),
    }
    for title, table in tables.items():
        print(f"\n{title}\n{table.round(3).to_string()}")


if __name__ == "__main__":
    main(sys.argv[1:])
