# Training benchmarks

Scripts for measuring the cost of the emulator's training. Both read the
experiment config from the config cell of `notebooks/Ocean_Emulator.ipynb`, so
they measure the settings that are actually run, and take `--set
SECTION.FIELD=VALUE` overrides (e.g. `--set train.batch_size=16`). The
normalised-fields cache goes to `emulator_cache/` in the repo, and results to
`runs/bench/` (git-ignored).

Run from the repo root on a GPU node, with nothing else on the GPU:

- `bench/step_bench.py`: one training or validation batch outside Lightning
  (same step, precision and optimizer).
  - Default: time per batch and peak memory at rollouts of 1, 4, 12 and 24+12
    steps, and one validation batch.
  - `--profile`: GPU time by kind of operation (convolution, data movement,
    elementwise), CPU-GPU syncs, and the top ops.
  - `--save-reference` / `--check-reference`: loss terms and gradients for a
    fixed batch and fixed weights, to check that a change leaves the maths
    unchanged. Check unchanged code against its own reference first: the land
    fill's atomic adds make gradients differ from run to run at about 1e-6.
- `bench/run_timed.py --name NAME`: the full training run with per-epoch
  timings, then the skill test, written to `runs/bench/NAME/`. Use it to
  compare changes that alter training (batch size, learning rate) on skill as
  well as time.
- `bench/summarise_runs.py runs/bench/A runs/bench/B ...`: side-by-side
  training time, epoch time per rollout stage, final validation RMSE and
  skill-test metrics of `run_timed.py` results.
- `bench/gadi_compare.pbs`: a Gadi batch job comparing the original code
  (`1dcbae8`) with a given commit on one GPU, per batch and end to end:
  `qsub -q gpuhopper -P tm70 -v OPTIMISED=$(git rev-parse HEAD) bench/gadi_compare.pbs`
  (from the repo root; results in `runs/bench/<queue>_<job id>/`).
