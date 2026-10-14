#!/bin/bash
# SCPN Reactor Systems — D39 operational health line every 20 minutes; counts only, no scores
X=$FEC_RUNS/ml350/runs/d39_finisher_20261011
now=$(date -u +%FT%TZ); deadline=$(cat $X/health_deadline_epoch)
states=""; any=0
for u in fec-d39-round1b fec-d39-orx fec-d39-round2; do s=$(systemctl --user is-active $u 2>/dev/null); states="$states \"$u\":\"$s\","; [ "$s" = active ] && any=1; done
records=$(find $X/variants -name '*.json' ! -name '*.partial.json' | wc -l)
mem=$(free -g | awk 'NR==2{print $7}'); load=$(cut -d' ' -f1 /proc/loadavg)
oom=$(journalctl --user --since "-25 min" 2>/dev/null | grep -ci "oom\|killed process")
echo "{\"UTC\":\"$now\",$states \"records\":$records,\"available_GiB\":$mem,\"load1\":$load,\"oom_lines_25min\":$oom}" >> $X/health.jsonl
if [ "$any" = 0 ] || [ "$(date +%s)" -gt "$deadline" ]; then
  echo "{\"UTC\":\"$now\",\"health_timer\":\"stopped\",\"reason\":\"$([ $any = 0 ] && echo no_active_unit || echo deadline)\"}" >> $X/health.jsonl
  systemctl --user stop fec-d39-health.timer
fi
