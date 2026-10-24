#!/usr/bin/env bash
# Serial job runner: runs the lines appended to data/stage3/jobs.txt one at a time, in order.
# Heavy jobs (scorer ~6 GB, model fits ~3-5 GB, data builds) must never overlap on this 15 GB machine.
#   echo "<shell command>" >> /home/zxc29051617/VCC/data/stage3/jobs.txt
Q=/home/zxc29051617/VCC/data/stage3/jobs.txt; DONE=$Q.done; touch $Q $DONE
cd /home/zxc29051617/VCC
while true; do
  n=$(wc -l < $DONE); line=$(sed -n "$((n + 1))p" $Q)
  if [ -z "$line" ]; then sleep 10; continue; fi
  echo "=== $(date '+%m-%d %T') START $line"; bash -c "$line"; rc=$?
  echo "=== $(date '+%m-%d %T') END rc=$rc $line"; echo "$line" >> $DONE
done
