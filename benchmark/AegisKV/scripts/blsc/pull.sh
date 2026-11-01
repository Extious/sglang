#!/bin/bash
# Usage: pull.sh <checkout dir> <branch>; GitHub access is intermittent, so retry.
cd "$1" || exit 1
for i in $(seq 1 90); do git fetch -q --depth 1 origin "$2" && git reset -q --hard FETCH_HEAD && git log --oneline -1 && echo PULL_DONE && exit 0; sleep 20; done
echo PULL_FAILED; exit 1
