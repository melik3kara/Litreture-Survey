"""
Builds run_ledits.ipynb — failure analysis using LEDITS++ (CVPR 2024).

Why LEDITS++ instead of DragonDiffusion:
  - Officially integrated into diffusers (LEditsPPPipelineStableDiffusion)
  - Public weights, no auth, no version pinning
  - Same kernel-side install, no subprocess isolation
  - Three knobs that map cleanly to our failure-mode taxonomy:
      editing_prompt, reverse_editing_direction, edit_guidance_scale, edit_threshold
"""
import nbformat  # type: ignore

nb = nbformat.v4.new_notebook()
cells = []

def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
def code(text): cells.append(nbformat.v4.new_code_cell(text))


md("""# LEDITS++ — Failure Analysis on Challenging Edits

**Paper:** *LEDITS++: Limitless Image Editing using Text-to-Image Models* (Brack et al., CVPR 2024 highlight).

**Idea.** A real input image is inverted with DPM-Solver++; during the reverse pass, semantic edits are applied as additive shifts in noise space, gated by a spatial threshold over per-concept attention maps. Multiple edits can be composed: each concept gets its own guidance scale, threshold, and direction (add or remove).

**Why we picked it.** Three knobs map directly to our three failure-mode buckets:
- `editing_prompt`: textual concept(s) to add/remove → tests **unnatural prompts**
- `reverse_editing_direction`: True = remove, False = add → multi-concept lists test **attribute conflicts**
- `edit_threshold`: per-concept spatial gate → tests **spatial complexity in dense scenes**

**Why this is easier than DragonDiffusion.** LEDITS++ is officially integrated into `diffusers` as `LEditsPPPipelineStableDiffusion`. No custom repo, no patched imports, no isolated venv — just `from diffusers import LEditsPPPipelineStableDiffusion`.
""")

# --------------------------------------------------------------- setup
md("## 1. Setup")
code('''import os, sys, subprocess, time, traceback
from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0),
          "{:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory/1e9))

ROOT        = Path("/content") if Path("/content").exists() else Path.cwd()
TEST_DIR    = ROOT / "test_images"
RESULTS_DIR = ROOT / "results" / "ledits"
for d in (TEST_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("results ->", RESULTS_DIR)
''')

# --------------------------------------------------------------- install
md("""## 2. Install

LEDITS++ is in `diffusers >= 0.27`. We pin a recent stable version that ships the pipeline. Everything else (`transformers`, `accelerate`) goes with the latest compatible release.""")
code('''pkgs = [
    "diffusers>=0.30,<0.40",
    "transformers>=4.40",
    "accelerate>=0.30",
    "safetensors",
    "Pillow",
    "matplotlib",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
import diffusers, transformers
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
from diffusers import LEditsPPPipelineStableDiffusion
print("LEditsPPPipelineStableDiffusion is importable")
''')

# --------------------------------------------------------------- images
md("""## 3. Test images

Either drag-and-drop your own three images named `portrait.jpg`, `street.jpg`, `indoor.jpg` into `/content/test_images/`, or run this cell to download three Wikimedia images automatically.""")
code('''import requests
from io import BytesIO
from PIL import Image

URLS = {
    "portrait": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Elon_Musk_Royal_Society_%28crop2%29.jpg/480px-Elon_Musk_Royal_Society_%28crop2%29.jpg",
    "street":   "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Times_Square%2C_New_York_City_%28HDR%29.jpg/640px-Times_Square%2C_New_York_City_%28HDR%29.jpg",
    "indoor":   "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Modern_living_room.jpg/640px-Modern_living_room.jpg",
}

# If a file already exists in /content/ root (drag-and-drop), copy it over.
import shutil
for name in URLS:
    rooted = Path("/content") / f"{name}.jpg"
    target = TEST_DIR / f"{name}.jpg"
    if rooted.exists() and not target.exists():
        shutil.copy(rooted, target)
        print("moved drag-and-drop:", rooted, "->", target)

TEST_IMAGES = {}
for name, url in URLS.items():
    dst = TEST_DIR / f"{name}.jpg"
    if not dst.exists():
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "litsurvey/1.0"})
            r.raise_for_status()
            Image.open(BytesIO(r.content)).convert("RGB").resize(
                (512, 512), Image.LANCZOS).save(dst, quality=95)
            print("downloaded", name)
        except Exception as e:
            print(f"download failed for {name}: {e}")
            Image.new("RGB", (512, 512), (128, 128, 128)).save(dst)
    else:
        # Make sure it\'s exactly 512x512
        Image.open(dst).convert("RGB").resize((512, 512), Image.LANCZOS).save(dst, quality=95)
    TEST_IMAGES[name] = dst
print("test images ->", {k: str(v) for k, v in TEST_IMAGES.items()})
''')

# --------------------------------------------------------------- experiments
md("""## 4. Failure-mode experiments

LEDITS++ takes lists of concepts. Each list-position is one parallel edit; we compose multiple concepts to expose conflicts.

The `hypothesis` field on each experiment is what we expect to observe in the report.""")
code('''LEDITS_EXPERIMENTS = [
    # ----- unnatural / impossible additions -----
    {
        "id": "ledits_three_eyes",
        "failure_type": "unnatural_prompt",
        "source_image_key": "portrait",
        "editing_prompt":            ["three eyes on the face"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale":       [10.0],
        "edit_threshold":            [0.85],
        "hypothesis": (
            "The inverted source\'s facial structure dominates; we expect the third "
            "eye to either fail to materialise or to appear as a textural ghost on "
            "the forehead, not as new geometry."
        ),
        "seed": 42,
    },
    {
        "id": "ledits_floating_cars",
        "failure_type": "unnatural_prompt",
        "source_image_key": "street",
        "editing_prompt":            ["cars floating above the buildings"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale":       [10.0],
        "edit_threshold":            [0.80],
        "hypothesis": (
            "The threshold gate localises the edit to attention regions matching "
            "\'cars floating\'; with no such region in the source, we expect either "
            "no change or a sky-region hallucination disconnected from real cars."
        ),
        "seed": 7,
    },

    # ----- multi-concept attribute conflicts -----
    {
        "id": "ledits_hair_and_glasses",
        "failure_type": "attribute_conflict",
        "source_image_key": "portrait",
        "editing_prompt":            ["long blonde hair", "round glasses"],
        "reverse_editing_direction": [False, False],
        "edit_guidance_scale":       [8.0, 8.0],
        "edit_threshold":            [0.85, 0.90],
        "hypothesis": (
            "Two simultaneous additive edits operate on overlapping attention maps "
            "(hair around the head, glasses around the eyes). We expect one edit "
            "to dominate when guidance scales are equal, and a partial fusion "
            "artefact (e.g. hair texture inside the glasses region)."
        ),
        "seed": 123,
    },
    {
        "id": "ledits_remove_smile_add_beard",
        "failure_type": "attribute_conflict",
        "source_image_key": "portrait",
        "editing_prompt":            ["smile", "thick beard"],
        "reverse_editing_direction": [True, False],
        "edit_guidance_scale":       [7.0, 9.0],
        "edit_threshold":            [0.90, 0.85],
        "hypothesis": (
            "Removing one local feature while adding another in the lower face "
            "competes for the same attention region. Expect the beard to mask "
            "incomplete smile removal, or a malformed mouth boundary."
        ),
        "seed": 17,
    },

    # ----- spatial complexity in dense scenes -----
    {
        "id": "ledits_crowd_to_robots",
        "failure_type": "spatial_complex",
        "source_image_key": "street",
        "editing_prompt":            ["robots instead of people"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale":       [9.0],
        "edit_threshold":            [0.80],
        "hypothesis": (
            "Many small subjects share a noisy attention map; the threshold gate "
            "produces inconsistent per-person conversions, with some pedestrians "
            "fully transformed and neighbours unchanged."
        ),
        "seed": 2024,
    },
    {
        "id": "ledits_indoor_to_forest",
        "failure_type": "spatial_complex",
        "source_image_key": "indoor",
        "editing_prompt":            ["dense forest with tall trees"],
        "reverse_editing_direction": [False],
        "edit_guidance_scale":       [10.0],
        "edit_threshold":            [0.75],
        "hypothesis": (
            "The interior scene\'s spatial features (wall and floor boundaries) "
            "leak through the inversion. Expect tree trunks to align with wall "
            "edges and foliage to inherit furniture silhouettes."
        ),
        "seed": 9,
    },
]

print(f"{len(LEDITS_EXPERIMENTS)} experiments defined")
''')

# --------------------------------------------------------------- pipeline
md("""## 5. Load the LEDITS++ pipeline

We use `stable-diffusion-v1-5/stable-diffusion-v1-5` as the base model — it\'s a public community-maintained mirror after Runway pulled the original. Falls back to `CompVis/stable-diffusion-v1-4` if the mirror is unavailable.""")
code('''from diffusers import LEditsPPPipelineStableDiffusion

PIPE = None
last_err = None
for model_id in [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
]:
    try:
        print(f"loading {model_id} ...")
        PIPE = LEditsPPPipelineStableDiffusion.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
            safety_checker=None,
        ).to(DEVICE)
        PIPE.enable_attention_slicing()
        print("loaded:", model_id)
        break
    except Exception as e:
        print(f"  failed: {str(e)[:200]}")
        last_err = e

if PIPE is None:
    raise last_err
''')

# --------------------------------------------------------------- run
md("""## 6. Run experiments

For each experiment we (a) invert the source image with `pipe.invert(...)`, (b) call the pipeline with the multi-concept edit lists. Inputs and outputs are saved to `/content/results/ledits/<id>_input.png` and `_output.png`.""")
code('''STATUS = {}
TIMING = {}

for exp in LEDITS_EXPERIMENTS:
    print(f"\\n=== {exp[\'id\']} ({exp[\'failure_type\']}) ===")
    t0 = time.time()
    try:
        src_path = TEST_IMAGES[exp["source_image_key"]]
        src_img  = Image.open(src_path).convert("RGB").resize((512, 512), Image.LANCZOS)

        # 1) invert
        torch.manual_seed(exp["seed"])
        _ = PIPE.invert(
            image=src_img,
            num_inversion_steps=50,
            skip=0.1,
        )

        # 2) edit
        out = PIPE(
            editing_prompt            = exp["editing_prompt"],
            reverse_editing_direction = exp["reverse_editing_direction"],
            edit_guidance_scale       = exp["edit_guidance_scale"],
            edit_threshold            = exp["edit_threshold"],
        ).images[0]

        src_img.save(RESULTS_DIR / f"{exp[\'id\']}_input.png")
        out.save(    RESULTS_DIR / f"{exp[\'id\']}_output.png")
        STATUS[exp["id"]] = "ok"
        print(f"  [ok]   ({time.time()-t0:.1f}s)")
    except Exception as e:
        STATUS[exp["id"]] = f"error: {type(e).__name__}: {e}"
        print(f"  [fail] ({time.time()-t0:.1f}s): {e}")
        traceback.print_exc()
    finally:
        TIMING[exp["id"]] = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print()
print("=" * 50)
ok   = sum(1 for v in STATUS.values() if v == "ok")
fail = len(STATUS) - ok
print(f"Done. {ok} ok / {fail} failed / {len(STATUS)} total")
''')

# --------------------------------------------------------------- viz
md("## 7. Visualize")
code('''import matplotlib.pyplot as plt

rows = [e for e in LEDITS_EXPERIMENTS
        if (RESULTS_DIR / f"{e[\'id\']}_output.png").exists()]
if not rows:
    print("no successful experiments to display.")
else:
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 4.5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax_row, exp in zip(axes, rows):
        inp = Image.open(RESULTS_DIR / f"{exp[\'id\']}_input.png")
        out = Image.open(RESULTS_DIR / f"{exp[\'id\']}_output.png")
        concept = ", ".join(
            f"{\'-\' if rev else \'+\'}{c}"
            for c, rev in zip(exp["editing_prompt"], exp["reverse_editing_direction"])
        )
        ax_row[0].imshow(inp); ax_row[0].set_title(f"{exp[\'id\']}\\ninput"); ax_row[0].axis("off")
        ax_row[1].imshow(out); ax_row[1].set_title(f"{exp[\'failure_type\']}\\n{concept}"); ax_row[1].axis("off")
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / "_grid.png", dpi=110, bbox_inches="tight")
    plt.show()
    print("grid saved ->", RESULTS_DIR / "_grid.png")
''')

# --------------------------------------------------------------- table
md("## 8. HTML comparison table (for the report)")
code('''import base64
from IPython.display import HTML, display

def _b64(path):
    if not Path(path).exists(): return ""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()

rows = []
for exp in LEDITS_EXPERIMENTS:
    inp = _b64(RESULTS_DIR / f"{exp[\'id\']}_input.png")
    out = _b64(RESULTS_DIR / f"{exp[\'id\']}_output.png")
    concept = ", ".join(
        f"{\'-\' if rev else \'+\'}{c}"
        for c, rev in zip(exp["editing_prompt"], exp["reverse_editing_direction"])
    )
    status = STATUS.get(exp["id"], "not run")
    rows.append(f"""
    <tr>
      <td>{exp[\'id\']}</td>
      <td>{exp[\'failure_type\']}</td>
      <td>{\'<img src=\"\' + inp + \'\" width=180>\' if inp else \'—\'}</td>
      <td>{\'<img src=\"\' + out + \'\" width=180>\' if out else \'—\'}</td>
      <td style=\"max-width:240px;font-size:12px\">{concept}</td>
      <td style=\"max-width:280px;font-size:12px\">{exp[\'hypothesis\']}</td>
      <td>{status}</td>
    </tr>""")

html = f"""
<style>
  table.ls {{ border-collapse: collapse; font-family: sans-serif; }}
  table.ls th, table.ls td {{ border: 1px solid #aaa; padding: 6px; vertical-align: top; }}
  table.ls th {{ background: #eee; }}
</style>
<table class=\'ls\'>
  <tr><th>ID</th><th>Type</th><th>Input</th><th>Output</th><th>Edit concepts</th><th>Hypothesis</th><th>Status</th></tr>
  {"".join(rows)}
</table>
"""
display(HTML(html))
(ROOT / "ledits_comparison.html").write_text(html)
print("saved ->", ROOT / "ledits_comparison.html")
''')

# --------------------------------------------------------------- summary
md("## 9. Summary")
code('''print("=" * 60)
print("LEDITS++ EXPERIMENT SUMMARY")
print("=" * 60)
ok = fail = 0
for exp_id, status in STATUS.items():
    t = TIMING.get(exp_id, 0.0)
    if status == "ok":
        ok += 1; tag = "OK  "
    else:
        fail += 1; tag = "FAIL"
    print(f"  [{tag}] {exp_id:35s} {t:6.1f}s   {status if status != \'ok\' else \'\'}")
print("-" * 60)
print(f"TOTAL: {ok+fail}   OK: {ok}   FAIL: {fail}")
print(f"Outputs: {RESULTS_DIR}")
print(f"HTML:    {ROOT / \'ledits_comparison.html\'}")
''')

# ----------------------------------------------------------- write
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}
out_path = "/Users/melikekara/Documents/GitHub/Litreture-Survey/ledits_experiments.ipynb"
with open(out_path, "w") as f:
    nbformat.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
