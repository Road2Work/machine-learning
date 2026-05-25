#!/usr/bin/env bash
set -euo pipefail
if [ -d "data_science_resources/.git" ]; then
  git submodule update --remote --merge data_science_resources
else
  echo "data_science_resources is not a git submodule. Skipping git update."
fi
curl -X POST http://localhost:8000/v1/admin/reload-ds-assets || true
