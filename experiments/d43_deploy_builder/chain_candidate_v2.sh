#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — build the H69 + all-data static X-specialist maps and assemble the DIII-D archive (v2)
# usage: chain_candidate_v2.sh <READY list of 8 specialists, sha256 path per line> <name of the output directory>
# v2 against v1 (a494d0e8…): canonical header; guard on the pinned imported builder, the D36 scalar archive and the
# public-test inputs; refuses a list whose name does not end in _READY.sha; share fixed at 0.5.
set -u
Y=$FEC_RUNS/ml350/runs/d43_deploy_20261012
P=$HOME/fec_mast_v10/venv/bin/python
LIST=$1; OUT=$Y/$2
case "$LIST" in *_READY.sha) ;; *) echo "only a released READY list is built"; exit 9;; esac
cd $Y/src
sha256sum -c SHA256SUMS_chain_v2 || exit 10
[ "$(sha256sum $HOME/fec_first_agent_d16/deploy_build_subframe.py | cut -d' ' -f1)" = 72e9929a6de689e9420673a6826bcab319282b566263ccbde809accff03e3dec ] || { echo "pinned builder drifted"; exit 15; }
[ "$(wc -l < "$LIST")" = 8 ] || { echo "the list must name eight specialists"; exit 11; }
S=$(sha256sum $Y/d36_diii_d_public_test.npz | cut -d' ' -f1)
[ "$S" = ba467d733135080ab6a0841117352705101efbd1331a2fea7ac7f1ff7ddf1744 ] || exit 13
[ -e "$OUT" ] && { echo "output directory exists; nothing is overwritten"; exit 16; }
$P build_h69x_v2.py --x-list "$LIST" --x-share 0.5 --out "$OUT/build" --workers 16 || exit 12
$P assemble_submission.py --maps "$OUT/build/diii_d_public_test.npz" --scalars $Y/d36_diii_d_public_test.npz --scalars-sha256 $S --out "$OUT/diii_d_public_test.npz" || exit 14
sha256sum "$LIST" "$OUT/build/diii_d_public_test.npz" "$OUT/diii_d_public_test.npz" > "$OUT/SHA256SUMS_candidate"
echo "candidate assembled: $OUT/diii_d_public_test.npz"
