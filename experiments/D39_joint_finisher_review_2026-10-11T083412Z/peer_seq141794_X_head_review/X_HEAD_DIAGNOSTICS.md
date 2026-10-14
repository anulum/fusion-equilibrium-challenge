# Head-only X-point diagnostics on the 120 screening shots (first_agent, 2026-10-11)

Held-out 69 networks of D38, every second frame (13 329 frames), truth = saddle of the true map by the native
extractor. No finisher variant, no score. Scripts and log beside this file (`xmembers.py`, `xmap.py`, `xmembers.log`).

| Estimator | lower X: mean / median / p90 / p99 mm | upper X: mean / median / p90 / p99 mm |
|---|---|---|
| mean of 69 | 5.55 / 3.34 / 11.7 / 32 | 5.69 / 3.58 / 10.8 / 32 |
| median of 69 | 5.55 / 3.32 / 11.8 / 32 | 5.68 / 3.59 / 10.8 / 32 |
| 20 % trimmed mean | 5.54 / 3.32 / 11.7 / 32 | 5.68 / 3.58 / 10.7 / 32 |
| hx only (21) | 5.68 | 5.62 |
| hc only (48) | 5.62 | 5.80 |
| one member (mean of 10) | 7.71 | 7.16 |
| saddle of the raw decoded map (45 coordinates) | 6.94 / 4.68 / 13.9 | 7.47 / 5.62 / 12.9 |
| head .75 + map saddle .25 | 5.35 | 5.47 |

- Robust averaging changes nothing; going from one member to 69 takes 7.7 → 5.55 mm, so about 5.4 mm of the error is
  common to all members. More members cannot remove it.
- Member spread predicts the error (Spearman .65 lower, .54 upper); the tenth of frames with the widest spread carries a third of the summed error.
- The error is not a transient effect: frames at 20–80 % of the shot carry 52 % of it at 4.8–5.0 mm; frames above 90 % of
  peak current carry 69 %. Start and end of a shot are worse per frame (7–10 mm) but few.
- 18–29 % of the squared error is a constant offset per shot; the lag-40 ms autocorrelation within a shot is .39–.54.
- The head beats the raw map's own saddle (5.5 against 6.9–7.5 mm); their errors correlate at .37–.60; the best fixed blend gains 2–3 %.
- Earlier ledger rows on the same question: 2.30 (residual correctors of heads fail on 5936 shots), 2.50 (trees help x_lower_r at weight .17, x_lower_z at .05, nothing from sub-frame inputs).

Reading: the X-point position error of the present model class is a broad, mostly shared 5.5 mm, not a tail problem of a
few frames and not member noise. With the second agent's factorial (true positions +626 µS, CI99 [487, 811], on these 120 shots),
each 10 % of X-position error is worth roughly 60 µS if the relation is near linear — an assumption, not a measurement.
