#!/usr/bin/env bash
# Smoke sorghum panicle/pearl with max_detections=2000 (Roboflow default is 300).
set -euo pipefail
cd "$(dirname "$0")/.."

python run.py \
  --okr-csv \
  --tasks sorghum_panicle_and_pearl \
  --model custom \
  --sample 5 \
  --visualize 2 \
  --split valid \
  --custom-conf 0.01 \
  --max-detections 2000 \
  --run-name 20261124_sorghum_panicle_maxdet2000_smoke

echo
echo "Compare predicted_objects / micro_recall to the earlier smoke (~295 preds/image)."
echo "Full trait run:"
echo "  python run.py --okr-csv --tasks sorghum_panicle_and_pearl --model custom --all \\"
echo "    --visualize 5 --split valid --custom-conf 0.01 --max-detections 2000 \\"
echo "    --run-name 20261124_sorghum_panicle_maxdet2000"
