#!/bin/bash
# SCPN Reactor Systems — build the H69 + all-data static X-specialist maps and assemble the DIII-D archive
# usage: chain_candidate.sh <list of 8 specialists, sha256 path per line> <name of the output directory>
set -u
Y=$FEC_RUNS/ml350/runs/d43_deploy_20261012
P=$HOME/fec_mast_v10/venv/bin/python
LIST=$1; OUT=$Y/$2
cd $Y/src
sha256sum -c SHA256SUMS_chain || exit 10
[ "$(wc -l < "$LIST")" = 8 ] || { echo "the list must name eight specialists"; exit 11; }
$P build_h69x_v2.py --x-list "$LIST" --x-share 0.5 --out "$OUT/build" --workers 16 || exit 12
S=$(sha256sum $Y/d36_diii_d_public_test.npz | cut -d' ' -f1)
[ "$S" = ba467d733135080ab6a0841117352705101efbd1331a2fea7ac7f1ff7ddf1744 ] || exit 13
$P assemble_submission.py --maps "$OUT/build/diii_d_public_test.npz" --scalars $Y/d36_diii_d_public_test.npz --scalars-sha256 $S --out "$OUT/diii_d_public_test.npz" || exit 14
echo "candidate assembled: $OUT/diii_d_public_test.npz"
