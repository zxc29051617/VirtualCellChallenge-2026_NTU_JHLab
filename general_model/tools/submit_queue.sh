#!/usr/bin/env bash
# Submit prepared .vcc files one after another once the daily allowance resets (UTC midnight).
#   submit_queue.sh <not-before UTC, e.g. 2026-10-12T00:02:00Z> <file.vcc>:<model name>:<description> ...
# Uses the credentials stored by `vcc login`; waits for each scoring to finish before the next upload.
set -u
cd /home/zxc29051617/VCC/vccsub
NB=$(date -d "$1" +%s); shift
while [ "$(date +%s)" -lt "$NB" ]; do sleep 60; done
for spec in "$@"; do
  f=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}; desc=${rest#*:}
  until [ -f "$f" ]; do echo "$(date +%T) waiting for $f"; sleep 60; done
  echo "=== $(date '+%m-%d %T') submit $f as $name"
  ~/.local/bin/vcc submit "$f" -m "$name" -d "$desc" --wait --wait-timeout 7200 2>&1 | tr '\r' '\n' | grep -v "^\s*\[" | sed -E 's/vcc_pat_[A-Za-z0-9_-]+/vcc_pat_<redacted>/g'
  echo "=== $(date '+%m-%d %T') done $name"
done
echo "=== SUBMIT QUEUE DONE"
