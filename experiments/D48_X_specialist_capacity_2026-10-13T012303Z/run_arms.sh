#!/bin/bash
# SCPN Reactor Systems — D48: the three frozen arms, one after another on one card
set -u
cd $FEC_EXPERIMENTS/D48_X_specialist_capacity_2026-10-13T012303Z
S="700 701 702 703 704 705 706 707"
$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python -B -u train_x_specialists.py --rows eligible --seeds $S --epochs 180 --width 512 --name x --out $FEC_EXPERIMENTS/D48_X_specialist_capacity_2026-10-13T012303Z/E180 || exit 11
$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python -B -u train_x_specialists.py --rows eligible --seeds $S --epochs 60 --width 512 --name x --out $FEC_EXPERIMENTS/D48_X_specialist_capacity_2026-10-13T012303Z/E60 || exit 12
$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python -B -u train_x_specialists.py --rows eligible --seeds $S --epochs 180 --width 1024 --name x --out $FEC_EXPERIMENTS/D48_X_specialist_capacity_2026-10-13T012303Z/W1024_E180 || exit 13
echo "D48 arms complete"
