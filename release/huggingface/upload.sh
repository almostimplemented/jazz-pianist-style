#!/bin/bash
# Upload the staged bundles (see stage_bundles.py) to the Hugging Face Hub.
# Requires: hf auth login   (or HF_TOKEN in the environment)
set -euo pipefail
BUNDLES="$(cd "$(dirname "$0")" && pwd)/bundles"
OWNER=almostimplemented

for name in jazz-pianist-style-generator \
            jazz-pianist-style-classifier \
            jazz-pianist-style-synthetic-classifier; do
  echo "== $name"
  hf repo create "$OWNER/$name" --repo-type model -y || true
  hf upload "$OWNER/$name" "$BUNDLES/$name" . --repo-type model
done
echo "done: https://huggingface.co/$OWNER"
