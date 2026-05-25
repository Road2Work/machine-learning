#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tensorboard --logdir models/logs --port 6006
