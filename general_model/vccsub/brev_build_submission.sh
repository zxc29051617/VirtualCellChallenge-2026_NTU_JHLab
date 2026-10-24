#!/usr/bin/env bash
# Build + package a VCC submission on a Brev instance (needs ~35 GB RAM for `vcc prep`).
# Run on THIS machine:
#   vccsub/brev_build_submission.sh <instance> <name> <effects.npz>                       (same effects for A/B/C)
#   vccsub/brev_build_submission.sh <instance> <name> A=<a.npz> B=<b.npz> C=<c.npz>       (per-context effects)
# It uploads the inputs (~650 MB the first time), builds out/<name>.h5ad and out/<name>.vcc there, and STOPS before
# submitting. Run it once per submission on the same instance; the controls and environment are reused.
# Submitting is a separate, explicit step, e.g. with a token that is not stored anywhere:
#   ssh -F ~/.brev/ssh_config <instance> 'cd ~/vcc && VCC_TOKEN=... ~/.local/bin/vcc submit out/<name>.vcc -m "<name>" --wait'
set -euo pipefail
INST=$1; NAME=$2; shift 2
SSH="ssh -F $HOME/.brev/ssh_config -o ServerAliveInterval=30 -T $INST"
SCP="scp -F $HOME/.brev/ssh_config -q"
HERE="$(cd "$(dirname "$0")" && pwd)"
FILES=(); SPEC=()
for e in "$@"; do
  if [[ $e == *=* ]]; then f=${e#*=}; FILES+=("$f"); SPEC+=("${e%%=*}=$(basename "$f")")
  else FILES+=("$e"); SPEC+=("$(basename "$e")"); fi
done
$SSH 'mkdir -p ~/vcc/data ~/vcc/out'
$SCP "$HERE/writer.py" "$HERE/make_effect_submission.py" "${FILES[@]}" $INST:vcc/
rsync -a --info=progress2 -e "ssh -F $HOME/.brev/ssh_config" /home/zxc29051617/VCC/data/vcc2026/{context_A,context_B,context_C}.h5ad \
      /home/zxc29051617/VCC/data/vcc2026/{gene_names.csv,pert_counts.csv,manifest.json} $INST:vcc/data/
$SSH "cd ~/vcc && command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\$HOME/.local/bin:\$PATH
[ -x env/bin/python ] || { uv venv -q -p 3.11 env && uv pip install -q -p env/bin/python anndata scipy numpy h5py pandas; }
command -v vcc >/dev/null || uv tool install -q vcc-cli
env/bin/python make_effect_submission.py --data data --effects ${SPEC[*]} --out out/$NAME.h5ad
/usr/bin/time -f 'vcc prep: %e s, peak %M KB' vcc prep out/$NAME.h5ad -g data/gene_names.csv --perts data/pert_counts.csv -o out/$NAME.vcc
rm -f out/$NAME.h5ad
ls -la out/"
echo "built $NAME.vcc on $INST -- not submitted"
