#!/usr/bin/env bash
# Final-phase pipeline: challenge controls -> effects -> 400 cells/target -> .vcc, all local and low-memory.
#   tools/final_pipeline.sh <data_dir> <name> <context> [<context> ...]
# <data_dir> holds context_<X>.h5ad, gene_names.csv, pert_counts.csv (unzipped `vcc datasets download controls`).
# Output: vccsub/out/<name>.vcc (validated). Uploading is a separate, deliberate step:
#   ~/.local/bin/vcc submit vccsub/out/<name>.vcc -m <name> -d "<description>" --wait
# Model: general model v1 with every training context (incl. VIPK562), per_celltype=False, lam 3, kcap 1, alpha 3.
# Cells (default RENDER=v6): cells follow the model's effects (trans x3); each gene's total over a target's 400 cells is
# set exactly to the effects with the panel's mean and top-10 shared trans components removed (scDFM/vcc/render.py mode
# "dual"; leaderboard 2026-10-19: gm_v1_vip_v6 0.1808, gm_v1_vip_exact 0.1608, earlier rendering 0.1482).
# RENDER=v8: also remove each context's top-50 control-cell axes from the totals (tools/make_bulk_effects.py);
# RENDER=v9: v8 totals x 0.5; RENDER=f8 / f9: v8 / v9 totals with V10 cells (shared trans part doubled in the cells);
# RENDER=g1 (2026-10-20): cells as v6 (V1); totals = own + cis + s_c x trans from a second export with depth-weighted
# sources (n0=200), shared + control axes removed, per-target sharpening gamma 3, s_c per context at the scorer's
# budget = claim point (tools/make_bulk_effects.py --gamma 3 --match-claim; repro/next_20261019/run.log).
# Recommended for D/E/F (2026-10-20 evening): RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5 CELLAGREE=1 TWPOW=0.5 ENTRYAGREE=1 (cells own + cis +
# trans x4.5 x per-gene source agreement, added counts in the top 2 % of a target's cells; totals with the per-target
# factor (agreement x n_sources)^0.5); each part checked on the H1 / KOLF200 / HCT116 hold-outs (repro/next_20261019).
# RENDER=exact: exact totals without the removal; RENDER=current: the earlier rendering. Effects do not depend on the
# context (sim_tau 0), so one export is shared by all contexts; the shared components are computed across the panel.
set -euo pipefail
DATA=$(readlink -f "$1"); NAME=$2; shift 2; CTX=("$@")
ROOT=/home/zxc29051617/VCC
log() { echo "=== $(date '+%m-%d %T') $*"; }
for c in "${CTX[@]}"; do [ -f "$DATA/context_$c.h5ad" ] || { echo "missing $DATA/context_$c.h5ad"; exit 1; }; done
for f in gene_names.csv pert_counts.csv; do [ -f "$DATA/$f" ] || { echo "missing $DATA/$f"; exit 1; }; done
head -1 "$DATA/pert_counts.csv" | grep -q target_gene || { echo "pert_counts.csv has no target_gene column"; exit 1; }
if [ "$(head -1 "$DATA/pert_counts.csv" | tr ',' '\n' | wc -l)" -gt 1 ]; then
  echo "NOTE: pert_counts.csv has extra columns ($(head -1 "$DATA/pert_counts.csv")); check per-context targets / cell counts before submitting"
fi
EFF=$ROOT/data/stage3/effects/${NAME}.npz; EFF_BULK=$ROOT/data/stage3/effects/${NAME}_bulk.npz; EFF_OC=$ROOT/data/stage3/effects/${NAME}_oc.npz
RENDER=${RENDER:-v6}
TOPFRAC=${TOPFRAC:-}            # dual render: share of a target's cells that receive added counts (empty = renderer default 0.1)
CELLAMP=${CELLAMP:-}            # cells = own + cis + CELLAMP x trans of the default export (empty = 1, i.e. trans x3)
CELLAGREE=${CELLAGREE:-}        # cells x (source sign agreement / median)^CELLAGREE (tools/cell_effects.py; empty = off)
TWPOW=${TWPOW:-}                # g1 totals: per-target factor (source agreement x number of sources)^TWPOW (empty = none)
ENTRYAGREE=${ENTRYAGREE:-}      # g1 totals: trans x (per-gene source sign agreement / median)^ENTRYAGREE before the removals
CISSCALE=${CISSCALE:-}          # g1 totals: own + CISSCALE x cis neighbours + trans (empty = 1)
COMMON=${COMMON:-}              # npz of predicted common responses per context (log2); added to totals (x COMMONSCALE) and cells (x CELLCOMMON)
COMMONSCALE=${COMMONSCALE:-1}; CELLCOMMON=${CELLCOMMON:-0}
PRECOMP=${PRECOMP:-}            # path prefix of precomputed effects (<p>.npz, <p>_d200.npz, <p>_oc.npz), e.g. from a trained model
if [ -n "$PRECOMP" ]; then
  log "precomputed effects from $PRECOMP"
  cp "$PRECOMP.npz" "$EFF"; cp "${PRECOMP}_oc.npz" "$EFF_OC"
  if [ -f "${PRECOMP}_d200_${CTX[0]}.npz" ]; then          # one totals input per context (a trained model sees each context's controls)
    for c in "${CTX[@]}"; do cp "${PRECOMP}_d200_$c.npz" "${EFF%.npz}_d200_$c.npz"; done; D200_PER_CTX=1
  else
    cp "${PRECOMP}_d200.npz" "${EFF%.npz}_d200.npz"
  fi
else
log "export effects ($(tail -n +2 "$DATA/pert_counts.csv" | wc -l) targets)"
(cd $ROOT/scDFM && VCC_DATA_DIR="$DATA" .venv/bin/python -m vcc.gm export "${CTX[0]}" vcc "$EFF" \
   per_celltype=False lam=3.0 kcap=1.0 alpha=3.0 bulk_out="$EFF_BULK" shared_k=10 oc_out="$EFF_OC")
fi
CELLS=$EFF
case "$RENDER" in
  g1)    EFF_D=${EFF%.npz}_d200.npz; [ -n "${D200_PER_CTX:-}" ] && EFF_D="${EFF%.npz}_d200_{ctx}.npz"
         if [ -z "$PRECOMP" ]; then
         log "export depth-weighted effects (n0=200) for the totals"
         (cd $ROOT/scDFM && VCC_DATA_DIR="$DATA" .venv/bin/python -m vcc.gm export "${CTX[0]}" vcc "$EFF_D" \
            per_celltype=False lam=3.0 kcap=1.0 alpha=3.0 n0=200.0)
         fi
         TWARG=""
         if [ -n "$TWPOW" ]; then
           (cd $ROOT/scDFM && .venv/bin/python -I $ROOT/tools/target_weights.py "$EFF" "${EFF%.npz}_tw.npz" "$TWPOW"); TWARG="--target-weights ${EFF%.npz}_tw.npz"
         fi
         SHAREDK=${SHAREDK:-10}; GAMMA=${GAMMA:-3}      # totals: panel shared components removed, per-target sharpening
         CTRLAXES=${CTRLAXES:-50}                       # totals: top control-cell axes removed (0 = keep)
         log "per-context totals (g1: shared $SHAREDK, control axes $CTRLAXES, gamma $GAMMA, budget = claim${TWPOW:+, target weights ^$TWPOW})"
         (cd $ROOT/scDFM && .venv/bin/python -I $ROOT/tools/make_bulk_effects.py --full "$EFF_D" --oc "$EFF_OC" --data "$DATA" \
            --contexts "${CTX[@]}" --shared-k $SHAREDK --ctrl-axes $CTRLAXES --gamma $GAMMA --match-claim $TWARG ${ENTRYAGREE:+--entry-agree $ENTRYAGREE} ${COMMON:+--common $COMMON --common-scale $COMMONSCALE} ${CISSCALE:+--cis-scale $CISSCALE} --out "${EFF_BULK%.npz}_g1")
         BULK_ARGS="--render dual --effects-bulk $(for c in "${CTX[@]}"; do printf '%s=%s ' "$c" "${EFF_BULK%.npz}_g1_$c.npz"; done)";;
  v8|v9|f8|f9) SC=1.0; case "$RENDER" in v9|f9) SC=0.5;; esac; CA=""
         case "$RENDER" in f8|f9) CELLS=${EFF%.npz}_v10cells.npz; CA="--cells-out $CELLS";; esac
         log "per-context totals ($RENDER: control axes 50, scale $SC)"
         (cd $ROOT/scDFM && .venv/bin/python -I $ROOT/tools/make_bulk_effects.py --full "$EFF" --oc "$EFF_OC" --data "$DATA" \
            --contexts "${CTX[@]}" --shared-k 10 --ctrl-axes 50 --scale $SC --out "${EFF_BULK%.npz}_$RENDER" $CA)
         BULK_ARGS="--render dual --effects-bulk $(for c in "${CTX[@]}"; do printf '%s=%s ' "$c" "${EFF_BULK%.npz}_${RENDER}_$c.npz"; done)";;
  v6)    BULK_ARGS="--render dual --effects-bulk $EFF_BULK";;
  *)     BULK_ARGS="--render $RENDER";;
esac
if [ -n "$CELLAGREE" ]; then
  CELLS_A=${EFF%.npz}_cells_agree.npz
  (cd $ROOT/scDFM && .venv/bin/python -I $ROOT/tools/cell_effects.py "$CELLS" "$EFF_OC" "$CELLS_A" "${CELLAMP:-1.5}" "$CELLAGREE")
  CELLS=$CELLS_A
elif [ -n "$CELLAMP" ]; then
  CELLS_A=${EFF%.npz}_cells_x$CELLAMP.npz
  (cd $ROOT/scDFM && .venv/bin/python -I -c "
import numpy as np
z = np.load('$CELLS', allow_pickle=True); o = np.load('$EFF_OC', allow_pickle=True)
assert list(z['targets']) == list(o['targets'])
np.savez_compressed('$CELLS_A', targets=z['targets'], lfc=(o['lfc'] + $CELLAMP * (z['lfc'] - o['lfc'])).astype(np.float32))
print('cells: own + cis + $CELLAMP x trans')")
  CELLS=$CELLS_A
fi
CELLSPEC="$CELLS"
if [ -n "$COMMON" ] && [ "$CELLCOMMON" != "0" ]; then     # cells: + CELLCOMMON x the context's predicted common response
  CELLSPEC=""
  for c in "${CTX[@]}"; do
    (cd $ROOT/scDFM && .venv/bin/python -I -c "
import numpy as np
z = np.load('$CELLS', allow_pickle=True); cv = np.load('$COMMON')['$c'].astype(np.float64)
np.savez_compressed('${CELLS%.npz}_common_$c.npz', targets=z['targets'], lfc=(z['lfc'] + $CELLCOMMON * cv[None]).astype(np.float32))
print('cells $c: + $CELLCOMMON x common response')")
    CELLSPEC="$CELLSPEC $c=${CELLS%.npz}_common_$c.npz"
  done
fi
FREE=$(df --output=avail -B1G "$ROOT" | tail -n 1 | tr -d ' ')
if [ "$FREE" -lt 40 ]; then echo "only ${FREE} GB free: building and packing needs ~37 GB at its peak (h5ad + temporary copy + .vcc)"; exit 1; fi
log "build cells"
(cd $ROOT/vccsub && $ROOT/scDFM/.venv/bin/python -I make_effect_submission.py --data "$DATA" --effects $CELLSPEC \
   --contexts "${CTX[@]}" $BULK_ARGS ${TOPFRAC:+--top-frac $TOPFRAC} \
   --out out/$NAME.h5ad)
if [ -n "${NOPACK:-}" ]; then log "done: $ROOT/vccsub/out/$NAME.h5ad (NOPACK: not packed)"; exit 0; fi
log "pack"
(cd $ROOT/vccsub && ~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/$NAME.h5ad out/$NAME.vcc.part \
   --genes "$DATA/gene_names.csv" --perts "$DATA/pert_counts.csv" --contexts "${CTX[@]}" && mv out/$NAME.vcc.part out/$NAME.vcc \
   && rm -f out/$NAME.h5ad)
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta, contexts_from_vcc
p = '$ROOT/vccsub/out/$NAME.vcc'; validate_vcc(p); print('validated', p, read_vcc_meta(p))
assert set(contexts_from_vcc(p) or ()) == set('${CTX[*]}'.split()), 'contexts not recorded (vcc 0.3.0 meta): the upload would fail at scoring'")
log "done: $ROOT/vccsub/out/$NAME.vcc (not submitted)"
