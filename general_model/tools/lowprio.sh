#!/usr/bin/env bash
# Run a command so that it can never hurt the training run: lowest CPU/IO priority, and first
# in line for the kernel OOM killer (oom_score_adj 1000).
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
exec nice -n 19 ionice -c3 "$@"
