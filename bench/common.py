"""
Shared setup for the training benchmarks.

The experiment config is read from the notebook's config cell (the code cell
containing ``cfg = ExperimentConfig(``), so the benchmarks measure the settings
that are actually run. Overrides are "section.field" paths, e.g.
{"train.batch_size": 16, "window.rollout_steps": [24, 12]}.
"""
import ast
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import Experiment  # noqa: E402

NOTEBOOK = REPO / "notebooks" / "Ocean_Emulator.ipynb"
# The notebook's default cache lives next to the data, which may not be
# writable; the benchmarks keep their own in the repo (git-excluded).
DEFAULT_OVERRIDES = {"data.cache_dir": str(REPO / "emulator_cache"), "train.run_dir": None}


def notebook_config(overrides=None, notebook=NOTEBOOK):
    """The notebook's ExperimentConfig with DEFAULT_OVERRIDES, then ``overrides``, applied."""
    cells = json.loads(Path(notebook).read_text())["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    source = next((s for s in sources if "cfg = ExperimentConfig(" in s), None)
    if source is None:
        raise ValueError(f"No cell with 'cfg = ExperimentConfig(' in {notebook}")
    namespace = {**vars(Experiment), "SRC": REPO / "src"}
    exec(source, namespace)
    cfg = namespace["cfg"]
    for path, value in {**DEFAULT_OVERRIDES, **(overrides or {})}.items():
        cfg = _replace(cfg, path, value)
    return cfg


def _replace(obj, path, value):
    """dataclasses.replace along a dotted path (re-validates the ExperimentConfig)."""
    head, *rest = path.split(".", 1)
    if not hasattr(obj, head):
        raise KeyError(f"{type(obj).__name__} has no field {head!r}")
    new = _replace(getattr(obj, head), rest[0], value) if rest else value
    return dataclasses.replace(obj, **{head: new})


def parse_overrides(items):
    """['train.batch_size=16', 'model.output_head=None'] -> {path: value} (Python literals, else strings)."""
    overrides = {}
    for item in items or []:
        path, _, text = item.partition("=")
        try:
            overrides[path] = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            overrides[path] = text
    return overrides


def build(cfg, verbose=False):
    """Data, model, loss terms and Lightning module, built as the notebook builds them."""
    data = Experiment.build_data(cfg, verbose=verbose)
    model = Experiment.build_model(cfg, data)
    losses = Experiment.build_losses(cfg, data)
    module = Experiment.build_module(cfg, model, losses, data)
    return data, model, losses, module
