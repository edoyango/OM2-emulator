"""
Time one training (or validation) batch of the emulator on the real data and
the notebook's config, outside Lightning but with the same training step:
EmulatorModule._step under the configured precision (fp16 autocast plus a
GradScaler for "16-mixed"), backward, Adam step, cuDNN benchmark mode.

Usage, from the repo root:

  .venv/bin/python bench/step_bench.py                        # time 1, 4, 12, 24+12 steps and validation
  .venv/bin/python bench/step_bench.py --steps 12 --profile   # where the GPU time of one batch goes
  .venv/bin/python bench/step_bench.py --save-reference runs/bench/reference.pt
  .venv/bin/python bench/step_bench.py --check-reference runs/bench/reference.pt
  .venv/bin/python bench/step_bench.py --set train.batch_size=16 --set train.checkpoint_rollout_steps=False

--steps takes a training rollout as n or n_free+n_trained, or "val" for one
validation batch (window.valid_rollout_steps, no gradients). The cost of one
loss term is the difference against a run with that term switched off, e.g.
--set "loss.terms={'local_mse': 1.0, 'spectral': 2.0, 'heat_closure': 0, 'freshwater_closure': 0}".

The reference mode saves the loss terms and every parameter gradient for a
fixed batch and fixed initial weights (fp32 by default), so a change that
should not alter the maths can be checked against it. Gradients go through
index_add (atomic adds) in the land fill, so even unchanged code differs from
run to run at rounding level: check the unchanged code against its own
reference first to see that floor.
"""
import argparse
import contextlib
import statistics
import time
from collections import Counter

import torch

from common import build, notebook_config, parse_overrides

DEFAULT_STEPS = ["1", "4", "12", "24+12", "val"]
REFERENCE_CASES = {"train_4": "4", "train_2+4": "2+4", "val": "val"}


def parse_steps(text, module):
    """'12' -> ('train', (0, 12)), '24+12' -> ('train', (24, 12)), 'val' -> ('val', valid steps)."""
    if text == "val":
        return "val", module.valid_steps
    n_free, _, n_trained = text.rpartition("+")
    return "train", (int(n_free or 0), int(n_trained))


def precision_context(precision):
    if precision == "16-mixed":
        return torch.autocast("cuda", dtype=torch.float16)
    if precision != "32-true":
        raise ValueError(f"step_bench supports 16-mixed and 32-true, not {precision!r}")
    return contextlib.nullcontext()


class Bench:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.backends.cudnn.benchmark = True  # as build_trainer (benchmark=True)
        self.data, self.model, self.losses, self.module = build(cfg)
        self.module.cuda()
        # Outside a Trainer, self.log has nowhere to go: keep the values instead
        # (as tensors: converting here would add a GPU sync per logged value).
        self.logged = {}
        self.module.log = lambda name, value, **_: self.logged.__setitem__(name, value)
        self.optimizer = self.module.configure_optimizers()
        if isinstance(self.optimizer, dict):
            self.optimizer = self.optimizer["optimizer"]
        self.precision = cfg.train.precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "16-mixed")
        batch_size = cfg.train.batch_size
        self.train_t0 = self.data.train_indices[:batch_size].cuda()
        self.valid_t0 = self.data.valid_indices[:batch_size].cuda()

    def run(self, stage, steps, optimizer_step=True):
        """One batch; returns the loss (a tensor, not synchronised)."""
        module = self.module
        if stage == "val":
            module.eval()
            with torch.no_grad(), precision_context(self.precision):
                loss = module._step(self.valid_t0, steps, "val")
        else:
            module.train()
            self.optimizer.zero_grad(set_to_none=True)
            with precision_context(self.precision):
                loss = module._step(self.train_t0, steps, "train")
            self.scaler.scale(loss).backward()
            if optimizer_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
        module.errors[stage].pop_rmse()
        return loss

    def time(self, stage, steps, warmup, repeats):
        for _ in range(warmup):  # the first calls also pick the cuDNN algorithms
            self.run(stage, steps)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            self.run(stage, steps)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        return times, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


# =============================================================================
# Profile: GPU time by kind of operation
# =============================================================================

def op_category(name):
    """Kind of work an aten op launches, by name: the innermost op owns the kernel time."""
    name = name.lower()
    if "conv" in name or "cudnn" in name:
        return "convolution"
    if "upsample" in name:
        return "interpolate"
    movers = ("copy", "cat", "index", "gather", "scatter", "flip", "fill", "zero", "pad", "clone", "contiguous",
              "slice", "select_backward", "narrow", "stack", "repeat", "expand", "roll", "where")
    if any(m in name for m in movers):
        return "data movement (copy/cat/pad/index/fill/where)"
    if "fft" in name:
        return "fft (spectral loss)"
    if any(m in name for m in ("sum", "mean", "norm", "max", "min", "amax", "reduce")):
        return "reductions"
    return "elementwise / other"


def profile(bench, stage, steps, warmup, trace=None):
    from torch.profiler import ProfilerActivity, profile as torch_profile

    for _ in range(warmup):
        bench.run(stage, steps)
    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start = time.perf_counter()
        bench.run(stage, steps)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
    if trace:
        prof.export_chrome_trace(trace)
        print(f"Chrome trace: {trace}")

    events = prof.key_averages()
    device_time = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    by_category, total = Counter(), 0.0
    for e in events:
        if e.key.startswith("aten::") and device_time(e) > 0:
            by_category[op_category(e.key[6:])] += device_time(e)
    # All kernel time, including kernels not under an aten op (e.g. the optimizer's foreach kernels).
    total = sum(device_time(e) for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
    print(f"\nWall time {wall * 1e3:.1f} ms | GPU kernel time {total / 1e3:.1f} ms "
          f"({100 * total / 1e3 / (wall * 1e3):.0f}% of wall; the rest is the GPU waiting on the CPU)")
    print("GPU time by aten op category:")
    attributed = sum(by_category.values())
    for category, t in by_category.most_common():
        print(f"  {category:48s} {t / 1e3:9.1f} ms  {100 * t / attributed:5.1f}%")

    calls = {e.key: e.count for e in events}
    syncs = {k: calls.get(k, 0) for k in ("aten::_local_scalar_dense", "cudaStreamSynchronize",
                                          "cudaDeviceSynchronize", "cudaMemcpyAsync")}
    print("CPU-GPU synchronisation:", ", ".join(f"{k} x{v}" for k, v in syncs.items()))
    sort_key = "self_device_time_total" if hasattr(events[0], "self_device_time_total") else "self_cuda_time_total"
    print(events.table(sort_by=sort_key, row_limit=30, max_name_column_width=60))


# =============================================================================
# Reference: loss terms and gradients for a fixed batch and fixed weights
# =============================================================================

def reference_case(bench, initial_state, text):
    bench.model.load_state_dict(initial_state)
    stage, steps = parse_steps(text, bench.module)
    bench.logged.clear()
    loss = bench.run(stage, steps, optimizer_step=False)
    # As in training, an fp16 overflow means the loss scale is too high: halve
    # it and redo the batch (GradScaler would skip the step and back off).
    while stage == "train" and bench.scaler.is_enabled() and not all(
        torch.isfinite(p.grad).all() for p in bench.model.parameters() if p.grad is not None
    ):
        bench.scaler.update(bench.scaler.get_scale() / 2)
        bench.logged.clear()
        loss = bench.run(stage, steps, optimizer_step=False)
    scale = bench.scaler.get_scale() if bench.scaler.is_enabled() else 1.0
    grads = {} if stage == "val" else {
        name: (p.grad.float() / scale).cpu() for name, p in bench.model.named_parameters() if p.grad is not None
    }
    return {"loss": float(loss.detach()), "terms": {k: float(v) for k, v in bench.logged.items()}, "grads": grads}


def save_reference(bench, path):
    initial_state = {k: v.detach().cpu().clone() for k, v in bench.model.state_dict().items()}
    cases = {name: reference_case(bench, initial_state, text) for name, text in REFERENCE_CASES.items()}
    torch.save({"precision": bench.precision, "model_state": initial_state, "cases": cases}, path)
    for name, case in cases.items():
        print(f"{name:10s} loss {case['loss']:.6g} | " + ", ".join(f"{k} {v:.6g}" for k, v in case["terms"].items()))
    print(f"Saved reference: {path}")


def check_reference(bench, path):
    ref = torch.load(path, weights_only=False)
    if ref["precision"] != bench.precision:
        print(f"Warning: reference precision {ref['precision']}, this run {bench.precision}")
    for name, text in REFERENCE_CASES.items():
        case, expected = reference_case(bench, ref["model_state"], text), ref["cases"][name]
        rel = lambda a, b: abs(a - b) / max(abs(b), 1e-30)
        terms = {k: rel(case["terms"].get(k, float("nan")), v) for k, v in expected["terms"].items()}
        worst_term = max(terms, key=terms.get)
        line = f"{name:10s} loss rel diff {rel(case['loss'], expected['loss']):.2e} | worst term {worst_term} {terms[worst_term]:.2e}"
        if expected["grads"]:
            if case["grads"].keys() != expected["grads"].keys():
                raise ValueError(f"{name}: the gradients are for different parameters")
            grad_rel = {k: ((case["grads"][k] - g).norm() / g.norm().clamp_min(1e-30)).item()
                        for k, g in expected["grads"].items()}
            worst = max(grad_rel, key=grad_rel.get)
            line += f" | grad rel diff (norm) max {grad_rel[worst]:.2e} ({worst}), median {statistics.median(grad_rel.values()):.2e}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", default=None, help="n, n_free+n_trained or val (default: %(default)s)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE", help="config override")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="profile one batch per --steps")
    parser.add_argument("--trace", help="with --profile: write a Chrome trace here")
    parser.add_argument("--save-reference", metavar="PATH")
    parser.add_argument("--check-reference", metavar="PATH")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    if args.save_reference or args.check_reference:
        overrides.setdefault("train.precision", "32-true")
    cfg = notebook_config(overrides)
    bench = Bench(cfg)
    print(f"GPU {torch.cuda.get_device_name()} | torch {torch.__version__} | precision {bench.precision} | "
          f"batch {cfg.train.batch_size} | checkpoint_rollout_steps {cfg.train.checkpoint_rollout_steps} "
          f"(in memory {cfg.train.rollout_steps_in_memory}) | compile {cfg.train.compile_model}")
    if overrides:
        print("Overrides:", overrides)

    if args.save_reference:
        return save_reference(bench, args.save_reference)
    if args.check_reference:
        return check_reference(bench, args.check_reference)

    for text in args.steps or DEFAULT_STEPS:
        stage, steps = parse_steps(text, bench.module)
        if args.profile:
            print(f"\n===== {stage} {text}")
            profile(bench, stage, steps, args.warmup, args.trace)
            continue
        times, peak, reserved = bench.time(stage, steps, args.warmup, args.repeats)
        n_free, n_scored = steps
        batch = len(bench.valid_t0 if stage == "val" else bench.train_t0)
        median = statistics.median(times)
        print(f"{stage:5s} {text:6s} batch {batch:3d} | {median * 1e3:8.1f} ms/batch (min {min(times) * 1e3:.1f}) | "
              f"{median * 1e3 / n_scored:6.1f} ms per scored step | peak {peak / 2**30:5.2f} GiB "
              f"(reserved {reserved / 2**30:5.2f})")


if __name__ == "__main__":
    main()
