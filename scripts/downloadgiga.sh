#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/gigapixel}"

mkdir -p "${DATA_DIR}"
wget --continue -O "${DATA_DIR}/girl_20000x23466.jpg" \
  "https://commons.wikimedia.org/wiki/Special:FilePath/21%20Gigapixel%20Total%20Renovation%20of%20Girl%20with%20a%20Pearl%20Earring-Digital%20Profoundism-Demo.jpg"

echo "Girl image saved under ${DATA_DIR}."
echo "Obtain Pluto from the ACORN dataset and Tokyo from its original benchmark source."
