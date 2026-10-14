# D31 — DIII-D learning rate stability

Canonical protocol: [protocol.json](protocol.json). Frozen code/dependencies: [frozen_sources.json](frozen_sources.json).

30 fresh fits: width1024, fixed60epochs, seeds200/201, fivehashfolds, peakLR .001/.0005/.00025. No candidate/epoch/seed selected from partial scores. Qualification distinct from candidates. All declared training/readout stages precede common full400shot/89584frame scoring; final paired statistics plus training diagnostics. Positive exploratory result requires fresh reserved confirmation before publication.

- Resource qualification: qualification/qualification.json; separate unit fec-d31-wide-qualify-v1.
- Controller: state.json, terminal_receipt.json; unit fec-d31-diiid-stability-v1.
- Epoch evidence: fits/<arm>_seed<seed>_k<fold>/passes.jsonl, optimizer checkpoint.npz, final network.npz/fit_report.json.
- Geometry readouts: readouts/<name>/network.npz, receipt.json.
- Evaluation: evaluation/summary.json, *_parts.json, complete prediction archives.
- Descriptive training evidence: training_diagnostic.json, all30runs; no seed/epoch omission.

Activation session $LAB/the internal record handover $LAB/the internal record Research journal $FEC_WORK/second_agent/equilibrium_takeover/RESEARCH_JOURNAL.md. All outputs Samsung; old inputs read-only. No automatic HF/Codabench submission.
