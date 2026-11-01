#!/bin/bash
# Usage: hfget.sh <repo> <dest>; download every file of a model repo through
# hf-mirror with resumable curl.
set -uo pipefail
REPO=$1; DEST=$2; mkdir -p $DEST
FILES=$(curl -sSL --retry 5 "https://hf-mirror.com/api/models/$REPO" | python3 -c "import json,sys;[print(s['rfilename']) for s in json.load(sys.stdin)['siblings']]")
for f in $FILES; do
  mkdir -p "$DEST/$(dirname $f)"
  for i in $(seq 1 100); do
    curl -sS -L -C - --retry 3 -o "$DEST/$f" "https://hf-mirror.com/$REPO/resolve/main/$f" && break
    sleep 3
  done
  echo "got $f $(stat -c %s $DEST/$f)"
done
echo HFGET_DONE
