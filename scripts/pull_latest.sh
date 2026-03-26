#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BRANCH="${1:-main}"

cd "${PROJECT_ROOT}"
git fetch origin "${BRANCH}"
git pull --ff-only origin "${BRANCH}"
