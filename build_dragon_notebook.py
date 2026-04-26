"""
Builds run_dragondiffusion.ipynb — a standalone Colab notebook that runs
DragonDiffusion in isolation against the diffusers/transformers/hf_hub
versions the repo was developed against (~mid-2023). The trick:

  - We install OLD diffusers/transformers/hf_hub into /content/dragon_pkgs
    using `pip install --target=... --no-deps`, so they sit beside the
    main site-packages without overwriting torch/numpy.
  - A worker script prepends /content/dragon_pkgs to sys.path before any
    `import diffusers`, so DragonDiffusion sees the API it expects.
  - The notebook subprocesses the worker once per experiment, passing
    inputs/outputs through JSON + PNG files.

The main run_all_experiments.ipynb is unaffected.
"""
import json
import nbformat  # type: ignore

nb = nbformat.v4.new_notebook()
cells = []

def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
def code(text): cells.append(nbformat.v4.new_code_cell(text))


md("""# DragonDiffusion (ICLR 2024) — Standalone Failure Analysis

The main literature-survey notebook (`run_all_experiments.ipynb`) covers MasaCtrl and PnP, but DragonDiffusion was written against **diffusers 0.16.x** (mid-2023). Running it on Colab's modern stack triggers a chain of import-time errors because the diffusers internal API has been reorganized several times since.

This notebook isolates DragonDiffusion in its own dependency sandbox:

1. Old `diffusers`, `transformers`, `huggingface_hub` are installed into `/content/dragon_pkgs/` via `pip install --target=...`.
2. A worker script prepends that directory to `sys.path` before importing diffusers, so DragonDiffusion sees the legacy API.
3. The main kernel keeps using its current diffusers (good — MasaCtrl/PnP need it). DragonDiffusion runs in subprocesses.

Inputs come from `/content/test_images/` (the same shared test images as the main notebook). Outputs land in `/content/results/dragon/`, identical to the layout the main notebook would have used, so the visualization / HTML-table cells in `run_all_experiments.ipynb` will pick them up automatically if you run both notebooks in the same session.
""")

# ----------------------------------------------------------------- setup
md("## 1. Setup")

code('''import os, sys, subprocess, json, time, traceback
from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

ROOT         = Path("/content") if Path("/content").exists() else Path.cwd()
TEST_DIR     = ROOT / "test_images"
RESULTS_DIR  = ROOT / "results" / "dragon"
REPOS_DIR    = ROOT / "repos"
DRAGON_DIR   = REPOS_DIR / "DragonDiffusion"
DRAGON_PKGS  = ROOT / "dragon_pkgs"
WORKER_PATH  = ROOT / "dragon_worker.py"
SPEC_DIR     = ROOT / "dragon_specs"
for d in (TEST_DIR, RESULTS_DIR, REPOS_DIR, SPEC_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("Dragon repo dir:", DRAGON_DIR)
print("Old-diffusers dir:", DRAGON_PKGS)
''')

# ----------------------------------------------------------------- clone repo
md("## 2. Clone DragonDiffusion")

code('''if not DRAGON_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/MC-E/DragonDiffusion.git", str(DRAGON_DIR)],
        check=False,
    )
print("Repo present:", DRAGON_DIR.exists())
''')

# ----------------------------------------------------------------- isolate
md("""## 3. Install legacy diffusers stack into an isolated directory

We use `pip install --target=...` so the legacy versions live in their own
folder and don't replace the kernel's current diffusers (which MasaCtrl/PnP
in the main notebook depend on).

`--no-deps` is critical: it stops pip from upgrading numpy / torch / etc.
We pin the small handful of transitive deps that DragonDiffusion's old
diffusers actually needs at runtime.""")

code('''def pip_target_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "--target", str(DRAGON_PKGS), "--no-deps", *pkgs]
    return subprocess.run(cmd, check=False)

if not (DRAGON_PKGS / "diffusers").exists():
    DRAGON_PKGS.mkdir(exist_ok=True)
    # The three packages whose API DragonDiffusion is hard-coded against.
    pip_target_install([
        "diffusers==0.16.1",
        "transformers==4.25.1",
        "huggingface_hub==0.16.4",
        "accelerate==0.20.3",
        # Transitive deps that the old releases need but newer ones moved.
        "tokenizers==0.13.3",
        "safetensors==0.3.1",
    ])
    print("Legacy diffusers stack installed at:", DRAGON_PKGS)
else:
    print("Legacy diffusers stack already present.")

# Other libs DragonDiffusion needs. basicsr\'s dep tree may pull numpy<2,
# breaking the ABI with Colab\'s torch (built against numpy 2.x). We
# install, then force numpy back to 2.x.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "basicsr==1.4.2", "einops==0.7.0", "pytorch_lightning==2.1.3",
     "gradio==3.50.2"],
    check=False,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "numpy>=2.0,<3"],
    check=False,
)
print("Kernel-side deps installed; numpy held at 2.x.")
print("NOTE: if you see a numpy ABI error in the next cell, "
      "Runtime > Restart session and re-run.")
''')

code('''# Patch basicsr WITHOUT importing it (importing pulls numpy and may
# trigger an ABI error before we get to the patch). We resolve the
# install path via `pip show`, then edit the file directly.
_show = subprocess.run(
    [sys.executable, "-m", "pip", "show", "basicsr"],
    capture_output=True, text=True,
)
_loc = next((l.split(": ", 1)[1] for l in _show.stdout.splitlines()
             if l.startswith("Location:")), None)
if _loc:
    _f = Path(_loc) / "basicsr" / "data" / "degradations.py"
    if _f.exists():
        _s = _f.read_text()
        _n = _s.replace(
            "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
            "from torchvision.transforms.functional import rgb_to_grayscale",
        )
        if _n != _s:
            _f.write_text(_n)
            print("Patched", _f)
        else:
            print("basicsr already patched.")
    else:
        print("WARNING: degradations.py not found at", _f)
else:
    print("WARNING: basicsr install location not found via pip show.")
''')

# ----------------------------------------------------------------- prompts
md("""## 4. Experiment prompts

We embed the four DragonDiffusion experiments here. The same definitions live in `failure_prompts.py` in the repo root — keeping them inline makes this notebook self-contained.""")

code('''DRAGON_EXPERIMENTS = [
    {
        "id": "dragon_face_stretch",
        "failure_type": "unnatural_prompt",
        "source_image_key": "portrait",
        "prompt": "a portrait of a person",
        "handle_points": [(0.50, 0.40)],
        "target_points": [(0.50, 0.10)],
        "mask_box": (0.30, 0.20, 0.70, 0.60),
        "hypothesis": "Pulling the nose far above the face forces feature correspondence to extrapolate; expect smearing or a duplicated nose artifact.",
        "seed": 10,
    },
    {
        "id": "dragon_hair_and_shoulder",
        "failure_type": "attribute_conflict",
        "source_image_key": "portrait",
        "prompt": "a portrait of a person",
        "handle_points": [(0.40, 0.30), (0.70, 0.75)],
        "target_points": [(0.35, 0.20), (0.80, 0.70)],
        "mask_box": (0.20, 0.10, 0.90, 0.90),
        "hypothesis": "Two simultaneous drags share the same latent optimization; background between them is expected to warp even though it was not selected.",
        "seed": 11,
    },
    {
        "id": "dragon_move_person_in_crowd",
        "failure_type": "spatial_complex",
        "source_image_key": "street",
        "prompt": "a crowded street scene",
        "handle_points": [(0.50, 0.60)],
        "target_points": [(0.25, 0.60)],
        "mask_box": (0.10, 0.40, 0.60, 0.90),
        "hypothesis": "Dragging a subject across overlapping people should cause identity merging at the destination and a hollow figure at origin.",
        "seed": 12,
    },
    {
        "id": "dragon_move_object_indoor",
        "failure_type": "spatial_complex",
        "source_image_key": "indoor",
        "prompt": "a living room",
        "handle_points": [(0.30, 0.70)],
        "target_points": [(0.70, 0.40)],
        "mask_box": (0.10, 0.30, 0.90, 0.90),
        "hypothesis": "Moving an object through clutter should leave a hole or ghost in the origin region and bleed texture into the destination.",
        "seed": 13,
    },
]

# Resolve test images. If the test-images cell in the main notebook hasn\'t
# been run, download three Wikimedia images here too so this notebook is
# fully standalone.
import requests
from PIL import Image
from io import BytesIO
TEST_URLS = {
    "portrait": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Elon_Musk_Royal_Society_%28crop2%29.jpg/480px-Elon_Musk_Royal_Society_%28crop2%29.jpg",
    "street":   "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Times_Square%2C_New_York_City_%28HDR%29.jpg/640px-Times_Square%2C_New_York_City_%28HDR%29.jpg",
    "indoor":   "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Modern_living_room.jpg/640px-Modern_living_room.jpg",
}
TEST_IMAGES = {}
for name, url in TEST_URLS.items():
    dst = TEST_DIR / f"{name}.jpg"
    if not dst.exists():
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "litsurvey/1.0"})
            r.raise_for_status()
            Image.open(BytesIO(r.content)).convert("RGB").resize((512, 512), Image.LANCZOS).save(dst, quality=95)
        except Exception as e:
            print(f"download failed for {name}: {e}")
            Image.new("RGB", (512, 512), (128, 128, 128)).save(dst)
    TEST_IMAGES[name] = dst
print("Test images:", {k: str(v) for k, v in TEST_IMAGES.items()})
''')

# ----------------------------------------------------------------- worker
md("""## 5. Worker script

This script runs in a fresh Python subprocess. Its `sys.path` is reordered so the legacy diffusers in `/content/dragon_pkgs` is found first. It imports `DragonModels`, runs one experiment, saves the output PNG, and exits.

Reading inputs/outputs through files is what makes the isolation work: nothing is shared with the kernel except the JSON spec on disk.""")

code('''WORKER_SRC = r"""
import sys, os, json, argparse, types
from pathlib import Path

# IMPORTANT: legacy diffusers stack must come BEFORE the system one.
DRAGON_PKGS = os.environ["DRAGON_PKGS"]
DRAGON_DIR  = os.environ["DRAGON_DIR"]
sys.path.insert(0, DRAGON_PKGS)
sys.path.insert(0, DRAGON_DIR)

# Stub xformers (DragonDiffusion does a hard `import xformers`).
import torch
import torch.nn.functional as F
def _mea(q, k, v, attn_bias=None, p=0.0, scale=None):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=p, scale=scale,
    )
_xf = types.ModuleType("xformers")
_xf_ops = types.ModuleType("xformers.ops")
_xf_ops.memory_efficient_attention = _mea
_xf_ops.MemoryEfficientAttentionFlashAttentionOp = None
_xf.ops = _xf_ops
sys.modules["xformers"] = _xf
sys.modules["xformers.ops"] = _xf_ops

import numpy as np
from PIL import Image, ImageDraw

def _mask_from_box(box, size=512):
    m = Image.new("L", (size, size), 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(m).rectangle(
        [int(x0*size), int(y0*size), int(x1*size), int(y1*size)],
        fill=255,
    )
    return m

def _px(pt, size=512):
    return [int(pt[0]*size), int(pt[1]*size)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--out_input", required=True)
    p.add_argument("--out_output", required=True)
    args = p.parse_args()

    spec = json.loads(Path(args.spec).read_text())

    # Sanity print: confirm we are using the legacy diffusers.
    import diffusers
    print("worker diffusers:", diffusers.__version__, "from", diffusers.__file__)

    from src.demo.model import DragonModels
    dragon = DragonModels(pretrained_model_path=None)

    src = Image.open(spec["source_image"]).convert("RGB").resize((512, 512))
    src.save(args.out_input)

    mask = _mask_from_box(spec["mask_box"])
    handles = [_px(pt) for pt in spec["handle_points"]]
    targets = [_px(pt) for pt in spec["target_points"]]

    edited = None
    try:
        edited = dragon.run_drag_content(
            original_image=np.array(src),
            mask=np.array(mask),
            prompt=spec["prompt"],
            selected_points=handles + targets,
            guidance_scale=7.5,
            energy_scale=0.5,
            max_resolution=512,
            SDE_strength=0.4,
            ip_scale=0.1,
        )
    except TypeError:
        edited = dragon.run_drag_content(
            np.array(src), np.array(mask),
            spec["prompt"], handles + targets,
            7.5, 0.5, 512, 0.4, 0.1,
        )
    if isinstance(edited, (list, tuple)):
        edited = edited[0]
    if isinstance(edited, np.ndarray):
        edited = Image.fromarray(edited.astype("uint8"))
    edited.save(args.out_output)
    print("OK", args.out_output)

if __name__ == "__main__":
    main()
"""

WORKER_PATH.write_text(WORKER_SRC)
print("Wrote worker:", WORKER_PATH)
''')

# ----------------------------------------------------------------- run
md("""## 6. Run experiments

For each experiment we serialize a small JSON spec, then call the worker as a subprocess. The kernel never imports DragonDiffusion or its old diffusers — that all stays in the worker process and is freed when it exits.""")

code('''STATUS = {}
TIMING_TOTAL = 0.0

env = os.environ.copy()
env["DRAGON_PKGS"] = str(DRAGON_PKGS)
env["DRAGON_DIR"]  = str(DRAGON_DIR)
# Make sure the worker doesn\'t accidentally fall back to the kernel\'s
# diffusers because of inherited PYTHONPATH.
env["PYTHONPATH"]  = ""

for exp in DRAGON_EXPERIMENTS:
    spec = {
        "source_image": str(TEST_IMAGES[exp["source_image_key"]]),
        "prompt": exp["prompt"],
        "mask_box": list(exp["mask_box"]),
        "handle_points": [list(p) for p in exp["handle_points"]],
        "target_points": [list(p) for p in exp["target_points"]],
        "seed": exp["seed"],
    }
    spec_path = SPEC_DIR / f"{exp[\'id\']}.json"
    spec_path.write_text(json.dumps(spec))

    out_input  = RESULTS_DIR / f"{exp[\'id\']}_input.png"
    out_output = RESULTS_DIR / f"{exp[\'id\']}_output.png"

    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(WORKER_PATH),
         "--spec", str(spec_path),
         "--out_input", str(out_input),
         "--out_output", str(out_output)],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    dt = time.time() - t0
    TIMING_TOTAL += dt

    if r.returncode == 0 and out_output.exists():
        STATUS[exp["id"]] = "ok"
        print(f"  [ok]   {exp[\'id\']:35s} ({dt:.1f}s)")
    else:
        STATUS[exp["id"]] = f"error (rc={r.returncode})"
        print(f"  [fail] {exp[\'id\']:35s} ({dt:.1f}s)")
        # Last 40 lines of stderr for diagnosis.
        tail = "\\n".join(r.stderr.strip().splitlines()[-40:])
        print(textwrap_indent(tail, "         | ") if False else tail)

print(f"\\nTotal wall clock: {TIMING_TOTAL:.1f}s")
''')

# ----------------------------------------------------------------- viz
md("## 7. Visualize results")

code('''import matplotlib.pyplot as plt
from PIL import Image

rows = [e for e in DRAGON_EXPERIMENTS
        if (RESULTS_DIR / f"{e[\'id\']}_output.png").exists()]

if not rows:
    print("No successful experiments to visualize. Check the [fail] lines above.")
else:
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax_row, exp in zip(axes, rows):
        inp = Image.open(RESULTS_DIR / f"{exp[\'id\']}_input.png")
        out = Image.open(RESULTS_DIR / f"{exp[\'id\']}_output.png")
        ax_row[0].imshow(inp); ax_row[0].set_title(f"{exp[\'id\']} — input"); ax_row[0].axis("off")
        ax_row[1].imshow(out); ax_row[1].set_title(f"{exp[\'failure_type\']}"); ax_row[1].axis("off")
    fig.tight_layout()
    plt.show()
''')

md("## 8. Summary")
code('''ok   = sum(1 for v in STATUS.values() if v == "ok")
fail = len(STATUS) - ok
print("=" * 50)
print(f"DragonDiffusion: {ok} ok / {fail} failed / {len(STATUS)} total")
print(f"Total wall-clock: {TIMING_TOTAL:.1f}s")
print(f"Outputs in: {RESULTS_DIR}")
print("=" * 50)
for k, v in STATUS.items():
    tag = "OK  " if v == "ok" else "FAIL"
    print(f"  [{tag}] {k:40s} {v if v != \'ok\' else \'\'}")
''')

# ------------------------------------------------------------------ write
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}
out = "/Users/melikekara/Documents/GitHub/Litreture-Survey/run_dragondiffusion.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)
print("Wrote", out, "with", len(cells), "cells")
