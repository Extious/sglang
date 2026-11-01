#!/bin/bash
# Usage: pipget.sh <pip install args...>
# Resolve with `pip --dry-run --report`, download every wheel with resumable curl into
# ~/aegis/wheels, then install offline. The login node drops long downloads at about
# 160 MB and pip cannot resume them.
set -uo pipefail
export TMPDIR=$HOME/aegis/tmp PIP_CACHE_DIR=$HOME/aegis/pipcache; mkdir -p $TMPDIR
W=$HOME/aegis; PY=$W/env/bin/python; D=$W/wheels; mkdir -p $D
REPORT=$(mktemp)
$PY -m pip install --dry-run --ignore-installed --quiet --report $REPORT \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple --find-links $D "$@" || { echo RESOLVE_FAILED; exit 1; }
$PY - "$REPORT" > $D/.urls <<'PY'
import json, sys
for item in json.load(open(sys.argv[1]))["install"]:
    url = item["download_info"]["url"]
    if url.startswith("http"):
        print(url)
PY
while read -r url; do
  f=$D/$(basename "${url%%#*}" | python3 -c "import sys,urllib.parse;print(urllib.parse.unquote(sys.stdin.read().strip()))")
  for i in $(seq 1 60); do
    curl -sS -L -C - --retry 3 -o "$f" "$url" && python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" "$f" 2>/dev/null && break
    case "$f" in *.tar.gz) [ -s "$f" ] && break;; esac
    sleep 2
  done
done < $D/.urls
$PY -m pip install --no-index --find-links $D "$@" && echo PIPGET_DONE
