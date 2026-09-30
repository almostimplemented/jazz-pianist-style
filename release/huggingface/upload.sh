#!/bin/bash
# Upload the staged bundles (see stage_bundles.py) to the Hugging Face Hub.
# Requires: huggingface_hub >= 1.0 (the `hf` CLI) and `hf auth login`
set -euo pipefail
BUNDLES="$(cd "$(dirname "$0")" && pwd)/bundles"
OWNER=drewbie

for name in jazz-pianist-style-generator \
            jazz-pianist-style-classifier \
            jazz-pianist-style-synthetic-classifier; do
  echo "== $name"
  # creates the repo (public) on first upload
  hf upload "$OWNER/$name" "$BUNDLES/$name" . --repo-type model --no-private \
     --commit-message "Release checkpoint"
done
echo "done: https://huggingface.co/$OWNER"
