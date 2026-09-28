# okr_inference

Server-ready evaluator for Artemis bushbean OKRs: compare **GPT-6 Astra**
(Roboflow Autolabel boxes) against **your custom Roboflow models** on GT
benchmark datasets.

- Select **one, two, or all** tasks (`plant_stand`, `flowers`, `pods`)
- Run a **seeded sample** or **all** images
- Optional **playground-style** visualization (detections | match map | mAP card)
- Writes **per-task CSVs** + a **summary leaderboard**

Match-map colors: green = only GT, red = only model, **yellow = both**.
Photo overlay: **pink = ground truth**, **thick cyan = predictions**.

## Setup (server)

```bash
cd /Users/mutuvi/Documents/artemis/okr_inference   # or your server path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional editable install for `okr-inference` console script:
# pip install -e .

export ROBOFLOW_API_KEY="your_private_api_key"
```

Copy `.env.example` if you prefer a local reminder file — the CLI reads
`$ROBOFLOW_API_KEY` (or `--api-key`).

## Tasks

Defined in [`config/tasks.yaml`](config/tasks.yaml):

| Task | GT project | Classes | Custom model |
|------|------------|---------|--------------|
| `plant_stand` | `artemis_2_bushbean_bb_benchmark-hxtfg` / 3 | `bushbean` | `beanbush_plantstand_bb_344_2/6` |
| `flowers` | `artemis_2_bushbean_bb_benchmark2` / 3 | flower stages + `Plant-Bean` | `dup_merged_1_to_15_flower_inst_seg-mhpnh/44` |
| `pods` | `artemis_2_bushbean_pod_benchmark` / 1 | `Fruit_pod`, `Plant_Bean` | `artemis2_pod_segmentation_batch3-gd8ng/53` |

Edit that YAML to add/change tasks or models. List names:

```bash
python run.py --list-tasks
```

## Usage

### Flowers + pods, 20 images each (default-ish)

```bash
python run.py --tasks flowers,pods --sample 20
```

### Only Astra, or only custom

```bash
python run.py --tasks flowers,pods --models astra --sample 20
python run.py --tasks pods --models custom --sample 20 --visualize
python run.py --tasks pods --models custom --sample 20 --visualize 5
# aliases also work: --models autolabel  |  --models roboflow
```

### One task, all images

```bash
python run.py --tasks pods --all
```

### Three tasks, sample 10, with visualizations

```bash
python run.py --tasks plant_stand,flowers,pods --sample 10 --visualize 2
```

### Full dataset, custom only, with resume after a crash

```bash
# start (writes checkpoints after each image)
python run.py --tasks flowers,pods --models custom --all --visualize 5

# if it dies mid-run, continue into the same output folder:
python run.py --tasks flowers,pods --models custom --all --visualize 5 \
  --resume outputs/20260928_044301_flowers-pods
```

### Module form

```bash
python -m okr_inference --tasks flowers --sample 5 --visualize
```

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--tasks` | `flowers,pods` | Comma list or `all` |
| `--models` | `all` | `astra`, `custom`, `astra,custom`, or `all` |
| `--sample N` | `20` | Seeded sample size per task |
| `--all` | off | Use every image (overrides `--sample`) |
| `--visualize [N]` | off (`0`) | Save playground panels for N images/model (`--visualize` alone ⇒ 1) |
| `--resume RUN_DIR` | off | Continue an interrupted run (skip checkpointed images) |
| `--seed` | `42` (YAML) | Sampling seed |
| `--custom-conf` | `0.40` | Confidence floor for custom serverless models |
| `--match-iou` | `0.50` | IoU for match-map / micro precision-recall |
| `--force-redownload` | off | Wipe cached GT folders and download again |
| `--output-dir` | `outputs/` | CSV + viz root |
| `--download-dir` | `roboflow_datasets/` | GT cache (reused across runs; skip re-download if present) |
| `--api-key` | `$ROBOFLOW_API_KEY` | Roboflow key |

## Outputs

Each run creates `outputs/<timestamp>_<tasks>/`:

| File | Contents |
|------|----------|
| `{task}_per_image.csv` | `image_name`, GT counts/classes, per-model pred/match stats |
| `{task}_model_metrics.csv` | Full COCO AP/AR suite + expected/predicted/matched |
| `summary_model_performance.csv` | All tasks × models |
| `summary_leaderboard.csv` | Compact AP/AR comparison |
| `visualizations/{task}/{model}__{image}.png` | Only if `--visualize` |

## Notes

- Metrics are **bbox** COCO (`pycocotools`). Custom flower/pod models may be
  trained as instance segmentation; serverless still returns boxes used here.
- Astra uses Autolabel preview (`gpt-6-astra-boxes`) with an identity ontology
  from GT class names (includes `Plant-Bean` / `Plant_Bean` where configured).
- Class name matching is normalized (`Plant_Bean` ≈ `Plant-Bean`).
- No `inference-sdk` dependency (avoids Pillow conflicts).

## Layout

```
okr_inference/
  config/tasks.yaml      # task / model registry
  okr_inference/         # package
    cli.py
    evaluate.py
    infer.py
    metrics.py
    viz.py
    ...
  run.py                 # convenience entrypoint
  requirements.txt
```
