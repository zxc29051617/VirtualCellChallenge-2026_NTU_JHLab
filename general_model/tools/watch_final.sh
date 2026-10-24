#!/usr/bin/env bash
# Wait until the challenge's dataset listing changes (the final-phase D/E/F controls are expected 2026-10-22), print the
# new listing and exit, so the download + final build (tools/final_pipeline.sh, ~15 min) can start right away.
#   tools/watch_final.sh [poll seconds, default 900]
# Final build once downloaded (2026-10-20 best settings = G2):
#   RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5 tools/final_pipeline.sh <unzipped dir> final_g2 D E F
set -u
POLL=${1:-900}
list() { ~/.local/bin/vcc datasets list 2>&1 | sed -E 's/(vcc_pat_|bak-)[A-Za-z0-9_\-]+/<redacted>/g'; }
BASE=$(list)
echo "=== $(date '+%m-%d %T') baseline listing:"; echo "$BASE"
while true; do
  sleep "$POLL"; NOW=$(list)
  if [ -n "$NOW" ] && [ "$NOW" != "$BASE" ] && ! echo "$NOW" | grep -qiE "error|traceback|timed out"; then
    echo "=== $(date '+%m-%d %T') listing changed:"; echo "$NOW"; exit 0
  fi
done
