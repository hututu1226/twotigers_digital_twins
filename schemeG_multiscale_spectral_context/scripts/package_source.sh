#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="schemeG_source_${STAMP}.tar.gz"

tar --exclude='__pycache__' --exclude='*.pyc' -czf "$ARCHIVE" \
  configs docs scheme_g scripts tests README.md pyproject.toml requirements.txt \
  .gitignore .gitattributes
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf 'Created %s\n' "$ARCHIVE"
cat "${ARCHIVE}.sha256"
