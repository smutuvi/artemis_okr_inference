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

cp .env.example .env   # then put your key in .env
```

The CLI reads the key from `--api-key`, then `$ROBOFLOW_API_KEY`, then
`ROBOFLOW_API_KEY=...` in `.env` at the project root. `.env` is git-ignored.

## Tasks

Defined in [`config/tasks.yaml`](config/tasks.yaml):

| Task | GT project | Classes | Custom model |
|------|------------|---------|--------------|
| `plant_stand` | `artemis_2_bushbean_bb_benchmark-hxtfg` / 3 | `bushbean` | `cgiar-workspace/bushbean_stand-row_count_diversity-2-rfdetr-medium-t4` |
| `flowers` | `artemis_2_bushbean_bb_benchmark2` / 3 | flower stages + `Plant-Bean` | `dup_merged_1_to_15_flower_inst_seg-mhpnh/8` |
| `pods` | `artemis_2_bushbean_pod_benchmark` / 2 | `Fruit_pod` only | `artemis2_pod_segmentation_batch3-gd8ng/53` |

Edit that YAML to add/change tasks or models. List names:

```bash
python run.py --list-tasks
```

## Usage

### Pack large JSONLs for Jupyter download (not for GitHub)

Mask/checkpoint JSONLs often exceed GitHub’s 100 MB limit. Zip them to your
home directory (stdlib `zipfile`, no system `zip` needed), then download via
Jupyter:

```bash
# after a run
python -m okr_inference.pack_outputs --run-dir outputs/20261124_okr_4_3
# → ~/20261124_okr_4_3_jsonl.zip

# or only mask preds (no checkpoints)
python -m okr_inference.pack_outputs --run-dir outputs/20261124_okr_4_3 --no-checkpoints

# or enable at the end of eval
python run.py --okr-csv --all --pack-jsonl --run-name 20261124_okr_4_3
```

```python
from IPython.display import FileLink
FileLink("/home/smutuvi/20261124_okr_4_3_jsonl.zip")
```

When pushing results to GitHub, still force-add only CSVs + `visualizations/`
(skip `checkpoints/` and `*_predictions_with_masks.jsonl`).

### Re-score an existing run at other confidence floors (no re-infer)

```bash
python -m okr_inference.rescore \
  --run-dir outputs/20261124_okr_4_3_ready \
  --task cowpea_plant_stand \
  --okr-csv \
  --split train \
  --conf-sweep 0.01,0.10,0.25,0.40,0.50
```

### OKR CSV (cowpea / sorghum model_path + data_path)

```bash
# list ready rows parsed from config/OKR_4.3.csv
python run.py --okr-csv --list-tasks

# run all ready OKR traits (custom only, conf 0.01 from YAML)
python run.py --okr-csv --all --visualize 5 --split valid \
  --run-name 20261124_okr_4_3_ready

# one trait only
python run.py --okr-csv --tasks cowpea_pods --all --visualize 5 --split valid
```

### Bushbean tasks.yaml, custom only, 20 images each

```bash
python run.py --tasks plant_stand,flowers,pods --sample 20
```

### One bushbean task

```bash
python run.py --tasks pods --sample 20 --visualize
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
| `--tasks` | `all` | Comma list or `all` (YAML names, or OKR `crop_trait` slugs) |
| `--okr-csv [PATH]` | off | Load tasks from OKR CSV (`model_path` + `data_path`); default file `config/OKR_4.3.csv` |
| `--okr-include-non-ready` | off | Include CSV rows whose `dataset_status` ≠ `ready` |
| `--model` / `--models` | `custom` | `custom` (default), `astra`, lists, or raw Autolabel type |
| `--sample N` | `20` | Seeded sample size per task |
| `--all` | off | Use every image (overrides `--sample`) |
| `--visualize [N]` | off (`0`) | Save playground panels for N images/model (`--visualize` alone ⇒ 1) |
| `--resume RUN_DIR` | off | Continue an interrupted run (skip checkpointed images) |
| `--seed` | `42` (YAML) | Sampling seed |
| `--custom-conf` | `0.01` | Confidence floor for custom serverless models |
| `--match-iou` | `0.50` | IoU for match-map / micro precision-recall |
| `--force-redownload` | off | Wipe cached GT folders and download again |
| `--output-dir` | `outputs/` | CSV + viz root |
| `--run-name NAME` | auto timestamp | Custom folder under `outputs/` (e.g. `my_plant_stand_run`) |
| `--split` | `test` | GT COCO split: `test` (default), `valid`, or `train` |
| `--task-classes TASK=A,B` | YAML classes | Override scored GT classes (e.g. `pods=Fruit_pod,Plant_Bean`) |
| `--download-dir` | `roboflow_datasets/` | GT cache (reused across runs; skip re-download if present) |
| `--api-key` | `$ROBOFLOW_API_KEY` | Roboflow key |

## Outputs

Each run creates `outputs/<timestamp>_<tasks>/`:

| File | Contents |
|------|----------|
| `{task}_per_image.csv` | `image_name`, GT counts/classes, per-model pred/match stats |
| `{task}_model_metrics.csv` | Full COCO AP/AR suite + expected/predicted/matched |
| `{task}_predictions_with_masks.jsonl` | Flowers/pods only: per-image preds with `points` / `segmentation` / `mask` when returned |
| `summary_model_performance.csv` | All tasks × models |
| `summary_leaderboard.csv` | Compact AP/AR comparison |
| `visualizations/{task}/{model}__{image}.png` | Only if `--visualize` |

## Notes

- Metrics: **bbox** COCO for all tasks. Flowers/pods also get **mask mAP**
  (`AP_*_segm` via `COCOeval` `segm`) when GT and preds have polygons/masks.
  Full preds with masks are in `{task}_predictions_with_masks.jsonl`.
  Plant-stand stays bbox-only (RF-DETR detection).
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
