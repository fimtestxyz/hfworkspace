#!/usr/bin/env bash
# Build gemma_search.py into a standalone executable with PyInstaller.
# Mirrors the CI configuration from .github/workflows/release-gemma-search.yml
# so local and release builds behave the same.
set -euo pipefail

SCRIPT="gemma_search.py"
BIN_NAME="gemma-search"

if [ ! -f "$SCRIPT" ]; then
  echo "error: $SCRIPT not found in $(pwd)" >&2
  exit 1
fi

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "PyInstaller not found; installing with runtime deps..."
  python -m pip install --upgrade pip
  python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch transformers requests markdownify duckduckgo-search pyinstaller
fi

pyinstaller \
  --onefile \
  --console \
  --clean \
  --name "$BIN_NAME" \
  --hidden-import=duckduckgo_search \
  --hidden-import=duckduckgo_search.DDGS \
  --hidden-import=markdownify \
  --hidden-import=requests \
  --hidden-import=transformers \
  --hidden-import=transformers.models \
  --hidden-import=transformers.models.auto \
  --hidden-import=torch.nn \
  --hidden-import=torch.nn.functional \
  --hidden-import=torch.cuda \
  --hidden-import=torch.backends.mps \
  --collect-submodules=duckduckgo_search \
  --collect-submodules=markdownify \
  --collect-submodules=transformers.utils \
  --collect-all=torch \
  "$SCRIPT"

ls -lh "dist/$BIN_NAME"
echo "Built: dist/$BIN_NAME"
echo "Smoke test:"
"dist/$BIN_NAME" --help
