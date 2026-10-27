"""
Train the notebook's experiment end to end with per-epoch timings, then run the
skill test, so a change can be judged on both cost and skill.

Usage, from the repo root:

  .venv/bin/python bench/run_timed.py --name baseline
  .venv/bin/python bench/run_timed.py --name batch16 --set train.batch_size=16 --set train.checkpoint_rollout_steps=False

Writes to runs/bench/<name>/ (Lightning's config.json, metrics.csv and
checkpoints, plus):
  epoch_times.csv    per epoch: rollout, training, validation and remaining
                     (checkpointing, logging) seconds
  rmse_history.csv   per-variable train/validation RMSE per epoch
  skill_by_lead.csv  skill test per variable and lead month: RMSE (emulator,
                     persistence, climatology), ACC, anomaly amplitude
  skill_summary.csv  the notebook's skill summary table
  summary.json       wall times, leading growth factor, final validation RMSE
"""
import argparse
import json
import time

import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

from common import REPO, build, notebook_config, parse_overrides  # also puts src/ on the path

import Experiment  # noqa: E402


class EpochTimer(Callback):
    """Wall time of each epoch's training batches, validation and the rest (checkpoint saving, logging)."""

    def __init__(self):
        self.rows, self.marks = [], {}

    def _mark(self, name):
        torch.cuda.synchronize()
        self.marks[name] = time.perf_counter()

    def on_train_epoch_start(self, trainer, module):
        if "epoch_start" in self.marks:
            self._close(trainer)
        self.marks = {}
        self._mark("epoch_start")

    def on_validation_start(self, trainer, module):
        self._mark("val_start")

    def on_validation_end(self, trainer, module):
        self._mark("val_end")
        self.steps = list(module.train_steps)

    def on_train_end(self, trainer, module):
        self._close(trainer)

    def _close(self, trainer):
        self._mark("end")
        m = self.marks
        val_start, val_end = m.get("val_start", m["end"]), m.get("val_end", m["end"])
        self.rows.append({
            "epoch": len(self.rows),
            "rollout": "+".join(map(str, getattr(self, "steps", []))),
            "train_s": val_start - m["epoch_start"],
            "val_s": val_end - val_start,
            "other_s": m["end"] - val_end,
            "total_s": m["end"] - m["epoch_start"],
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="run name: results go to runs/bench/<name>")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE", help="config override")
    args = parser.parse_args()

    out = REPO / "runs" / "bench" / args.name
    overrides = {"train.run_dir": str(out), **parse_overrides(args.set)}
    cfg = notebook_config(overrides)
    start = time.perf_counter()
    data, model, losses, module = build(cfg, verbose=True)
    trainer = Experiment.build_trainer(cfg)
    timer = EpochTimer()
    trainer.callbacks.append(timer)
    fit_start = time.perf_counter()
    trainer.fit(module, data.train_dl, data.valid_dl)
    fit_s = time.perf_counter() - fit_start

    epochs = pd.DataFrame(timer.rows)
    epochs.to_csv(out / "epoch_times.csv", index=False)
    print(epochs.round(1).to_string(index=False))
    module.rmse_history.to_csv(out / "rmse_history.csv", index=False)

    skill = Experiment.run_skill_test(cfg, model, data)
    by_lead = []
    for variable in cfg.data.prognostic:
        d = Experiment.skill_diagnostics(skill, variable).to_dataframe()
        by_lead.append(d.assign(variable=variable, lead=range(1, len(d) + 1)))
    pd.concat(by_lead).to_csv(out / "skill_by_lead.csv")
    table = Experiment.skill_summary(skill, cfg.data.prognostic)
    table.to_csv(out / "skill_summary.csv")
    print(table.to_string())
    growth = Experiment.leading_growth_mode(model, data, cfg.time.test[0])["growth"]

    history = module.rmse_history
    final_val = history[(history.stage == "val") & (history.epoch == history.epoch.max())]
    summary = {
        "overrides": {k: str(v) for k, v in overrides.items()},
        "fit_s": fit_s,
        "total_s": time.perf_counter() - start,
        "optimizer_steps": trainer.global_step,
        "leading_growth_factor": float(growth),
        "final_val_rmse": dict(zip(final_val.variable, final_val.rmse)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
