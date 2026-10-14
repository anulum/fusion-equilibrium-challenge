#!/bin/bash
# SCPN Reactor Systems — D57: the eight frozen seeds of one arm, one after another on one card
set -u
cd "$(dirname "$0")" && sha256sum -c SHA256SUMS >/dev/null || exit 10
$HOME/fusion-equilibrium-challenge/fec_adapter/.venv/bin/python -B -u train_temporal.py --arm "$1" --seeds 800 801 802 803 804 805 806 807 --out "$PWD/fits" || exit 11
echo "D57 arm $1 complete"
