#!/usr/bin/env bash
set -euo pipefail

BRANCH="${2:-main}"
MESSAGE="${1:-sync: update inference runtime}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit. Pushing current HEAD to origin/${BRANCH}."
else
  git commit -m "${MESSAGE}"
fi

git push origin "${BRANCH}"
