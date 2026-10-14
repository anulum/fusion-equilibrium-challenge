#!/bin/bash
# SCPN Reactor Systems — D39 round 1 after amendment 2: priority variants, then the rest, on the screening part
set -u
cd $FEC_RUNS/ml350/runs/d39_finisher_20261011/src
P=$HOME/fec_mast_v10/venv/bin/python
$P d39_eval.py --variants d39_variants_a2.json --workers 22 --stride 3 --offsets 0 --only base cub c100 cub_c100 c000 orc128 orc128_cub || exit 11
$P d39_eval.py --variants d39_variants_a2.json --workers 22 --stride 3 --offsets 0 --only base cub c100 cub_c100 c000 orc128 orc128_cub c075 c100r2 cub_c100r2 x075 sa1 sa2 sf1 orc128_c000 || exit 12
$P d39_eval.py --variants d39_variants_a2.json --workers 22 --stride 3 --offsets 0 || exit 13
