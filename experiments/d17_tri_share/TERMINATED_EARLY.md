# D17 terminated early for futility (deviation from the frozen single-stage rule), 2026-10-06T03:15:41Z

The harness stopped the job at its 2 h background limit after 3 of 10 builds (31 min per variant and fold on 3
workers). I did not restart it. The three finished builds, each against the R8 V0 rows of the same fold:

| build | S | V0 | dS | flux R2 | D_LCFS | Consistency | tri_top | tri_bot | kappa |
|---|---|---|---|---|---|---|---|---|---|
| T1 fold 0 | .995676 | .995733 | −5.7e-5 | .999866 (.999893) | .009370 (.008728) | .984742 (.984634) | .9954 (.9951) | .9887 (.9877) | .9793 (.9797) |
| T2 fold 0 | .995685 | .995733 | −4.8e-5 | .999865 | .009373 | .984794 | .9955 | .9888 | .9797 |
| T1 fold 1 | .995096 | .995169 | −7.3e-5 | .999810 (.999833) | .009060 (.008393) | .981908 (.981877) | .9944 (.9936) | .9948 (.9939) | .9770 (.9781) |

The mechanism is the same in all three: the stronger bend raises the triangularity R2 by only .0003–.0010 (the
linear estimate promised about .002), while the boundary distance grows by 7–8 % and the flux R2 falls by 2–3e-5.
The protocol's own caution (the edit moves the boundary) is what happened. Seven more builds (about 3.5 h of three
workstation cores) would decide nothing: a variant would need the remaining folds to average above +5e-5 to pass.

Status: NOT completed as frozen; reported as an early termination, not as a pass or a fail under the rule.
Triangularity and elongation shares are closed on this evidence.
