"""
Builds run_all_experiments.ipynb from structured cell definitions.
Kept as a separate script so cells stay readable in plain Python.
"""
import json
import nbformat  # type: ignore

nb = nbformat.v4.new_notebook()
cells = []

def md(text: str):
    cells.append(nbformat.v4.new_markdown_cell(text))

def code(text: str):
    cells.append(nbformat.v4.new_code_cell(text))


# ---------------------------------------------------------------- header
md("""# Diffusion-Based Image Editing — Failure Analysis

**Literature survey**: MasaCtrl (ICCV 2023), Plug-and-Play Diffusion (CVPR 2023), DragonDiffusion (ICLR 2024).

This notebook runs inference for all three methods on a shared set of challenging inputs designed to expose characteristic failure modes:

1. **Unnatural / impossible prompts** — inputs that violate physical or anatomical priors (e.g. *"a person with three eyes"*, *"a car floating above buildings"*).
2. **Multi-attribute editing conflicts** — simultaneous edits that compete for the same latent structure (e.g. change hair *and* add glasses).
3. **Spatial manipulation in complex scenes** — edits inside crowded or cluttered contexts (e.g. moving a person in a busy street).

Each method section is independently runnable. A top-level `try/except` guards every section so that a failure in one method does not block the others. At the end, a visualization cell builds a side-by-side comparison and an HTML table for the report.

> Target runtime: **Google Colab, T4 GPU, free tier**. Expect the full notebook to take 30–45 minutes end-to-end on T4.
""")

# ---------------------------------------------------------------- setup
md("## 1. Setup: GPU check, directories, and dependency base stack")

code('''import os, sys, time, subprocess, json, textwrap, traceback
from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch  :", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
    print("VRAM   : {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print("WARNING: no GPU detected, falling back to CPU (will be very slow).")

ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
TEST_DIR    = ROOT / "test_images"
RESULTS_DIR = ROOT / "results"
REPOS_DIR   = ROOT / "repos"
for d in (TEST_DIR, RESULTS_DIR, REPOS_DIR,
          RESULTS_DIR / "masactrl",
          RESULTS_DIR / "pnp",
          RESULTS_DIR / "dragon"):
    d.mkdir(parents=True, exist_ok=True)
print("Root        :", ROOT)
print("Test images :", TEST_DIR)
print("Results     :", RESULTS_DIR)
print("Repos       :", REPOS_DIR)

# Tracks per-experiment status across the notebook.
STATUS = {"masactrl": {}, "pnp": {}, "dragon": {}}
TIMING = {"masactrl": None, "pnp": None, "dragon": None}
''')

code('''# Install the shared base stack. Each method may add more on top.
# We install quietly; errors will surface when the pipelines fail to import.
BASE_PACKAGES = [
    "diffusers==0.21.4",
    "huggingface_hub==0.25.2",
    "transformers==4.34.1",
    "accelerate==0.24.1",
    "safetensors==0.4.1",
    "tokenizers==0.14.1",
    "einops==0.7.0",
    "omegaconf==2.3.0",
    "opencv-python==4.8.1.78",
    "matplotlib==3.8.2",
    "Pillow==10.2.0",
    "ftfy==6.1.3",
    "regex==2023.12.25",
    "numpy==1.26.4",
]

def pip_install(pkgs, extra_flags=()):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *extra_flags, *pkgs]
    print(">>", " ".join(cmd[:6]), "..." if len(pkgs) > 3 else "")
    return subprocess.run(cmd, check=False)

pip_install(BASE_PACKAGES)
print("Base stack install finished.")
''')

# ---------------------------------------------------------------- test imgs
md("""## 2. Test images

Three Creative Commons / public-domain images covering a portrait, a street scene, and an indoor scene. We host-download from Wikimedia so the cell is reproducible on Colab with no external credentials.
""")

code('''import requests
from PIL import Image
from io import BytesIO

TEST_URLS = {
    # Portrait: public-domain portrait photograph (Wikimedia)
    "portrait": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Elon_Musk_Royal_Society_%28crop2%29.jpg/480px-Elon_Musk_Royal_Society_%28crop2%29.jpg",
    # Street scene: busy urban street (Wikimedia, CC)
    "street":   "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Times_Square%2C_New_York_City_%28HDR%29.jpg/640px-Times_Square%2C_New_York_City_%28HDR%29.jpg",
    # Indoor scene: living room (Wikimedia, CC)
    "indoor":   "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Modern_living_room.jpg/640px-Modern_living_room.jpg",
}

TEST_IMAGES = {}
for name, url in TEST_URLS.items():
    dst = TEST_DIR / f"{name}.jpg"
    if not dst.exists():
        try:
            r = requests.get(url, timeout=30,
                             headers={"User-Agent": "litsurvey/1.0"})
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(dst, quality=95)
            print(f"Downloaded {name:8s} -> {dst.name}")
        except Exception as e:
            print(f"FAILED to download {name}: {e}")
            # Fallback: create a solid gradient so downstream code still runs.
            Image.new("RGB", (512, 512), (128, 128, 128)).save(dst)
    else:
        print(f"Already present: {dst.name}")
    TEST_IMAGES[name] = dst

print("Test images:", {k: str(v) for k, v in TEST_IMAGES.items()})
''')

# ---------------------------------------------------------------- prompts file
md("""## 3. Experiment prompts

We embed `failure_prompts.py` inline so the notebook is self-contained on Colab. The same content is also available as a standalone file in the repository.
""")

import base64 as _b64
with open("/Users/melikekara/Documents/GitHub/Litreture-Survey/failure_prompts.py") as _f:
    _failure_prompts_src = _f.read()
_b64_prompts = _b64.b64encode(_failure_prompts_src.encode()).decode()

code(f'''import base64
PROMPTS_B64 = "{_b64_prompts}"
PROMPTS_PY  = base64.b64decode(PROMPTS_B64).decode()

prompts_path = ROOT / "failure_prompts.py"
prompts_path.write_text(PROMPTS_PY)
sys.path.insert(0, str(ROOT))

import importlib, failure_prompts
importlib.reload(failure_prompts)
from failure_prompts import MASACTRL_EXPERIMENTS, PNP_EXPERIMENTS, DRAGON_EXPERIMENTS

print(f"MasaCtrl experiments : {{len(MASACTRL_EXPERIMENTS)}}")
print(f"PnP experiments      : {{len(PNP_EXPERIMENTS)}}")
print(f"Dragon experiments   : {{len(DRAGON_EXPERIMENTS)}}")
''')

# Shared utilities
md("## 4. Shared utilities")

code('''from PIL import Image
import matplotlib.pyplot as plt

def save_image(img, path):
    """Save a PIL image or HxWx3 np array or torch tensor to `path`."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.save(path)
    elif isinstance(img, np.ndarray):
        Image.fromarray(img.astype("uint8")).save(path)
    elif torch.is_tensor(img):
        t = img.detach().cpu()
        if t.ndim == 4: t = t[0]
        if t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
            t = t.permute(1, 2, 0)
        t = t.clamp(0, 1) * 255 if t.dtype.is_floating_point else t
        Image.fromarray(t.numpy().astype("uint8")).save(path)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    return path

def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return img

class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        print(f"[timing] {self.label}: {self.elapsed:.1f}s")
''')

# ---------------------------------------------------------------- MasaCtrl
md("""## 5. MasaCtrl (ICCV 2023) — Mutual Self-Attention Control

**Idea.** During denoising of an edited prompt, replace the *self-attention keys and values* in certain UNet layers/steps with those computed from the source prompt's denoising trajectory. This preserves the subject's appearance while allowing the text prompt to change non-rigid structure (pose, action).

**Why we expect failures on our inputs.**
- *Unnatural prompts*: mutual-attention pins the source's facial/object topology, so injected anatomical changes (e.g. *three eyes*) collide with preserved structure.
- *Attribute conflicts*: two independent edits both need to re-route self-attention in overlapping regions; at most one tends to succeed cleanly.
- *Complex scenes*: distractors in the background provide spurious attention matches, smearing the edit onto non-target pixels.
""")

code('''# Clone MasaCtrl (official TencentARC repo — already diffusers-based).
MASACTRL_DIR = REPOS_DIR / "MasaCtrl"
if not MASACTRL_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/TencentARC/MasaCtrl.git", str(MASACTRL_DIR)],
        check=False,
    )
print("MasaCtrl at:", MASACTRL_DIR, "exists:", MASACTRL_DIR.exists())
if str(MASACTRL_DIR) not in sys.path:
    sys.path.insert(0, str(MASACTRL_DIR))
''')

code('''# MasaCtrl inference: consistent-synthesis mode.
# The source and target prompt are denoised in parallel; the target uses
# MutualSelfAttentionControl to reuse the source's K/V after step START_STEP
# in layers >= START_LAYER.
try:
    with Timer("MasaCtrl total") as timer:
        from diffusers import DDIMScheduler
        from masactrl.diffuser_utils import MasaCtrlPipeline
        from masactrl.masactrl_utils import regiter_attention_editor_diffusers
        from masactrl.masactrl import MutualSelfAttentionControl

        MODEL_ID = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False,
        )
        pipe = MasaCtrlPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        ).to(DEVICE)
        pipe.enable_attention_slicing()

        out_dir = RESULTS_DIR / "masactrl"

        for exp in MASACTRL_EXPERIMENTS:
            try:
                torch.manual_seed(exp["seed"])
                prompts = [exp["source_prompt"], exp["target_prompt"]]
                start_code = torch.randn(
                    [1, 4, 64, 64], device=DEVICE,
                    dtype=pipe.unet.dtype,
                ).expand(2, -1, -1, -1)

                # 1) Source-only generation for reference.
                from masactrl.masactrl_utils import AttentionBase
                regiter_attention_editor_diffusers(pipe, AttentionBase())
                imgs_src = pipe(prompts, latents=start_code,
                                guidance_scale=7.5,
                                num_inference_steps=40)
                src_img = imgs_src[0]

                # 2) Edited generation with MutualSelfAttentionControl.
                editor = MutualSelfAttentionControl(start_step=4, start_layer=10)
                regiter_attention_editor_diffusers(pipe, editor)
                imgs_edit = pipe(prompts, latents=start_code,
                                 guidance_scale=7.5,
                                 num_inference_steps=40)
                edit_img = imgs_edit[-1]

                # Convert tensor images to PIL.
                def _to_pil(t):
                    import numpy as np
                    arr = (t.detach().float().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
                    if arr.ndim == 3 and arr.shape[0] in (1, 3):
                        arr = arr.transpose(1, 2, 0)
                    return Image.fromarray(arr)

                save_image(_to_pil(src_img),  out_dir / f"{exp['id']}_input.png")
                save_image(_to_pil(edit_img), out_dir / f"{exp['id']}_output.png")
                STATUS["masactrl"][exp["id"]] = "ok"
                print(f"  [ok]   {exp['id']}")
            except Exception as ex:
                STATUS["masactrl"][exp["id"]] = f"error: {ex}"
                print(f"  [fail] {exp['id']}: {ex}")

        del pipe
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    TIMING["masactrl"] = timer.elapsed
except Exception as outer:
    print("MasaCtrl section failed entirely:")
    traceback.print_exc()
    TIMING["masactrl"] = None
''')

# ---------------------------------------------------------------- PnP
md("""## 6. Plug-and-Play Diffusion (CVPR 2023)

**Idea.** Given a *real* source image, run DDIM inversion to recover its latent trajectory, then during generation with the target prompt **inject** the source's spatial features (output of a chosen decoder ResBlock) and self-attention maps at specific timesteps. Structure is preserved by the injected features; appearance is overwritten by the new prompt.

**Why we expect failures on our inputs.**
- *Unnatural prompts*: injected features enforce the source's layout, so semantics that need new geometry (extra eyes, swimming fish) appear as texture on top of the old shape.
- *Attribute conflicts*: when the target prompt changes multiple global properties (scene type, illumination), the preserved structure conflicts with all of them.
- *Complex scenes*: small repeated subjects (crowds) share feature channels, causing per-subject identity bleed.
""")

code('''# Clone pnp-diffusers (Michal Geyer\'s diffusers port of PnP).
PNP_DIR = REPOS_DIR / "pnp-diffusers"
if not PNP_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/MichalGeyer/pnp-diffusers.git", str(PNP_DIR)],
        check=False,
    )
print("PnP at:", PNP_DIR, "exists:", PNP_DIR.exists())

# The repo ships its own requirements; we keep our pinned diffusers stack
# and only add what PnP strictly needs beyond it.
pip_install(["pyyaml>=6.0"])
if str(PNP_DIR) not in sys.path:
    sys.path.insert(0, str(PNP_DIR))
''')

code('''# PnP inference. The official workflow has two steps:
#   (a) preprocess.py  -- DDIM-invert the source image, save per-step latents.
#   (b) pnp.py         -- generate with feature+attention injection.
# We invoke them as subprocesses so any import-time bugs in the repo stay
# isolated from the notebook kernel.
try:
    with Timer("PnP total") as timer:
        import yaml
        out_dir = RESULTS_DIR / "pnp"
        latents_root = PNP_DIR / "latents_forward"
        latents_root.mkdir(exist_ok=True)

        for exp in PNP_EXPERIMENTS:
            try:
                src_path = TEST_IMAGES[exp["source_image_key"]]
                # Copy/resize source to PnP\'s expected input location.
                local_src = PNP_DIR / f"input_{exp[\'id\']}.png"
                Image.open(src_path).convert("RGB").resize((512, 512)).save(local_src)

                # (a) Preprocess: DDIM inversion.
                pre = subprocess.run(
                    [sys.executable, "preprocess.py",
                     "--data_path", str(local_src),
                     "--inversion_prompt", exp["source_prompt"],
                     "--save_steps", "50",
                     "--steps", "50"],
                    cwd=str(PNP_DIR),
                    capture_output=True, text=True, timeout=900,
                )
                if pre.returncode != 0:
                    raise RuntimeError(f"preprocess failed: {pre.stderr[-400:]}")

                # (b) Generation: write a temporary config YAML.
                cfg = {
                    "seed": exp["seed"],
                    "device": "cuda" if DEVICE.type == "cuda" else "cpu",
                    "output_path": str(out_dir / exp["id"]),
                    "image_path": str(local_src),
                    "latents_path": str(latents_root / local_src.stem),
                    "prompt": exp["target_prompt"],
                    "negative_prompt": "ugly, blurry, low quality",
                    "guidance_scale": 7.5,
                    "n_timesteps": 50,
                    "pnp_attn_t": 0.5,
                    "pnp_f_t": 0.8,
                }
                cfg_path = PNP_DIR / f"cfg_{exp[\'id\']}.yaml"
                with open(cfg_path, "w") as f:
                    yaml.safe_dump(cfg, f)

                gen = subprocess.run(
                    [sys.executable, "pnp.py", "--config_path", str(cfg_path)],
                    cwd=str(PNP_DIR),
                    capture_output=True, text=True, timeout=900,
                )
                if gen.returncode != 0:
                    raise RuntimeError(f"pnp.py failed: {gen.stderr[-400:]}")

                # Save copies under results/pnp/.
                save_image(Image.open(local_src), out_dir / f"{exp[\'id\']}_input.png")
                # PnP writes output(s) into cfg["output_path"].
                out_candidates = sorted(Path(cfg["output_path"]).glob("*.png"))
                if out_candidates:
                    save_image(Image.open(out_candidates[-1]),
                               out_dir / f"{exp[\'id\']}_output.png")
                else:
                    raise RuntimeError("no output image produced by pnp.py")

                STATUS["pnp"][exp["id"]] = "ok"
                print(f"  [ok]   {exp[\'id\']}")
            except Exception as ex:
                STATUS["pnp"][exp["id"]] = f"error: {ex}"
                print(f"  [fail] {exp[\'id\']}: {ex}")

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    TIMING["pnp"] = timer.elapsed
except Exception:
    print("PnP section failed entirely:")
    traceback.print_exc()
    TIMING["pnp"] = None
''')

# ---------------------------------------------------------------- DragonDiffusion
md("""## 7. DragonDiffusion (ICLR 2024)

**Idea.** Learn an editing energy over pre-trained Stable Diffusion features: given user-specified handle points and target points (with an optional edit mask), optimize the latent at each denoising step so that *features at the handle locations migrate to the target locations* while features outside the mask stay fixed. No extra training required — purely feature correspondence.

**Why we expect failures on our inputs.**
- *Unnatural prompts / large drags*: feature correspondence is local; pulling a handle far outside its feature neighborhood forces extrapolation and produces smears.
- *Attribute conflicts*: multiple drags share the same latent optimization; regions *between* the drags tend to warp even though they were not selected.
- *Complex scenes*: dragging through overlapping subjects causes identity merging at the destination and a hollowed-out origin.
""")

code('''# Clone DragonDiffusion. It ships its own requirements.txt.
DRAGON_DIR = REPOS_DIR / "DragonDiffusion"
if not DRAGON_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/MC-E/DragonDiffusion.git", str(DRAGON_DIR)],
        check=False,
    )
print("Dragon at:", DRAGON_DIR, "exists:", DRAGON_DIR.exists())

# Extra deps from DragonDiffusion that are not in our base stack.
pip_install(["gradio==3.50.2", "basicsr==1.4.2", "einops==0.7.0"])

if str(DRAGON_DIR) not in sys.path:
    sys.path.insert(0, str(DRAGON_DIR))
''')

code('''# DragonDiffusion inference via its drag-content editor.
# We call the repo\'s DragonModels class directly. We convert our normalized
# (x, y) coordinates in failure_prompts.py into pixel coordinates at 512x512.
try:
    with Timer("DragonDiffusion total") as timer:
        import numpy as np
        from PIL import ImageDraw

        out_dir = RESULTS_DIR / "dragon"

        # Lazy import: DragonDiffusion has heavy side effects on import.
        try:
            from src.demo.model import DragonModels  # type: ignore
        except Exception as imp_ex:
            raise RuntimeError(f"cannot import DragonModels: {imp_ex}")

        # pretrained_model_path=None lets DragonModels use its default SD v1.5.
        dragon = DragonModels(pretrained_model_path=None)

        def _px(pt, size=512):
            return [int(pt[0] * size), int(pt[1] * size)]

        def _mask_from_box(box, size=512):
            x0, y0, x1, y1 = box
            m = Image.new("L", (size, size), 0)
            ImageDraw.Draw(m).rectangle(
                [int(x0*size), int(y0*size), int(x1*size), int(y1*size)],
                fill=255,
            )
            return m

        for exp in DRAGON_EXPERIMENTS:
            try:
                src_path = TEST_IMAGES[exp["source_image_key"]]
                src_img  = Image.open(src_path).convert("RGB").resize((512, 512))
                mask     = _mask_from_box(exp["mask_box"])

                handles  = [_px(p) for p in exp["handle_points"]]
                targets  = [_px(p) for p in exp["target_points"]]

                # DragonModels exposes run_drag_content(...) -> edited PIL image.
                # Different commits of the repo use slightly different signatures;
                # we try the most common one and fall back.
                edited = None
                try:
                    edited = dragon.run_drag_content(
                        original_image=np.array(src_img),
                        mask=np.array(mask),
                        prompt=exp["prompt"],
                        selected_points=handles + targets,
                        guidance_scale=7.5,
                        energy_scale=0.5,
                        max_resolution=512,
                        SDE_strength=0.4,
                        ip_scale=0.1,
                    )
                except TypeError:
                    # Older signature: positional args only.
                    edited = dragon.run_drag_content(
                        np.array(src_img), np.array(mask),
                        exp["prompt"], handles + targets,
                        7.5, 0.5, 512, 0.4, 0.1,
                    )

                if isinstance(edited, (list, tuple)):
                    edited = edited[0]
                if isinstance(edited, np.ndarray):
                    edited = Image.fromarray(edited.astype("uint8"))

                save_image(src_img, out_dir / f"{exp[\'id\']}_input.png")
                save_image(edited,  out_dir / f"{exp[\'id\']}_output.png")
                STATUS["dragon"][exp["id"]] = "ok"
                print(f"  [ok]   {exp[\'id\']}")
            except Exception as ex:
                STATUS["dragon"][exp["id"]] = f"error: {ex}"
                print(f"  [fail] {exp[\'id\']}: {ex}")

        del dragon
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    TIMING["dragon"] = timer.elapsed
except Exception:
    print("DragonDiffusion section failed entirely:")
    traceback.print_exc()
    TIMING["dragon"] = None
''')

# ---------------------------------------------------------------- visualization
md("""## 8. Side-by-side visualization

For each method, show every experiment's input and output next to each other. This is the matrix we'll screenshot for the report.
""")

code('''import matplotlib.pyplot as plt

def show_method(method, experiments):
    rows = [e for e in experiments
            if (RESULTS_DIR / method / f"{e[\'id\']}_input.png").exists()
            and (RESULTS_DIR / method / f"{e[\'id\']}_output.png").exists()]
    if not rows:
        print(f"[{method}] no successful experiments to display.")
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax_row, exp in zip(axes, rows):
        inp = Image.open(RESULTS_DIR / method / f"{exp[\'id\']}_input.png")
        out = Image.open(RESULTS_DIR / method / f"{exp[\'id\']}_output.png")
        ax_row[0].imshow(inp); ax_row[0].set_title(f"{exp[\'id\']} — input"); ax_row[0].axis("off")
        ax_row[1].imshow(out); ax_row[1].set_title(f"{exp[\'id\']} — output ({exp[\'failure_type\']})"); ax_row[1].axis("off")
    fig.suptitle(f"{method.upper()} results", fontsize=14, y=1.0)
    fig.tight_layout()
    plt.show()

show_method("masactrl", MASACTRL_EXPERIMENTS)
show_method("pnp",      PNP_EXPERIMENTS)
show_method("dragon",   DRAGON_EXPERIMENTS)
''')

# ---------------------------------------------------------------- HTML table
md("""## 9. HTML comparison table

Renders a single HTML table — input, output, failure type, hypothesis — across all methods. Right-click → *Save as image* (or screenshot) for the report figure.
""")

code('''from IPython.display import HTML, display
import base64

def _b64(path):
    if not Path(path).exists(): return ""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()

rows_html = []
for method, exps in [("masactrl", MASACTRL_EXPERIMENTS),
                     ("pnp", PNP_EXPERIMENTS),
                     ("dragon", DRAGON_EXPERIMENTS)]:
    for exp in exps:
        inp = _b64(RESULTS_DIR / method / f"{exp[\'id\']}_input.png")
        out = _b64(RESULTS_DIR / method / f"{exp[\'id\']}_output.png")
        status = STATUS[method].get(exp["id"], "not run")
        rows_html.append(f"""
        <tr>
          <td><b>{method}</b></td>
          <td>{exp[\'id\']}</td>
          <td>{exp[\'failure_type\']}</td>
          <td>{\'<img src=\"\' + inp + \'\" width=180>\' if inp else \'—\'}</td>
          <td>{\'<img src=\"\' + out + \'\" width=180>\' if out else \'—\'}</td>
          <td style=\"max-width:280px;font-size:12px\">{exp.get(\'target_prompt\') or exp.get(\'prompt\',\'\')}</td>
          <td style=\"max-width:280px;font-size:12px\">{exp[\'hypothesis\']}</td>
          <td>{status}</td>
        </tr>""")

html = f"""
<style>
  table.ls {{ border-collapse: collapse; font-family: sans-serif; }}
  table.ls th, table.ls td {{ border: 1px solid #aaa; padding: 6px; vertical-align: top; }}
  table.ls th {{ background: #eee; }}
</style>
<table class='ls'>
  <tr>
    <th>Method</th><th>ID</th><th>Failure type</th>
    <th>Input</th><th>Output</th>
    <th>Prompt / target</th><th>Hypothesis</th><th>Status</th>
  </tr>
  {"".join(rows_html)}
</table>
"""
display(HTML(html))

# Also write to disk so you can open it outside Colab.
(Path(ROOT) / "comparison_table.html").write_text(html)
print("Saved:", ROOT / "comparison_table.html")
''')

# ---------------------------------------------------------------- summary
md("## 10. Summary")

code('''total = ok = fail = 0
print("=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
for method in ("masactrl", "pnp", "dragon"):
    print(f"\\n[{method.upper()}]  (total wall-clock: "
          f"{TIMING[method]:.1f}s" if TIMING[method] else f"\\n[{method.upper()}]  (did not run)")
    for exp_id, status in STATUS[method].items():
        total += 1
        if status == "ok":
            ok += 1; tag = "OK  "
        else:
            fail += 1; tag = "FAIL"
        print(f"   [{tag}] {exp_id:40s} {status if status != \'ok\' else \'\'}")

print("\\n" + "=" * 60)
print(f"TOTAL: {total}   OK: {ok}   FAIL: {fail}")
print("=" * 60)
print(f"Outputs:     {RESULTS_DIR}")
print(f"HTML table:  {ROOT / \'comparison_table.html\'}")
''')

# ------------------------------------------------------------------ write
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}

out_path = "/Users/melikekara/Documents/GitHub/Litreture-Survey/run_all_experiments.ipynb"
with open(out_path, "w") as f:
    nbformat.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
