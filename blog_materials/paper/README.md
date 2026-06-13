# Paper draft

Academic write-up of the blog post (`../POST.md`), same numbers and claims, paper register.

- `main.tex` — single-column article, self-contained (manual `thebibliography`, no `.bib`).
- `main.pdf` — compiled output (13 pp).

Build: `tectonic main.tex` (downloads pgfplots on first run), or `pdflatex main.tex` twice.

## Structure

Intro → Setup → Coverage curve (Fig. 1) → Cache answers deleted content → Mechanism →
Boundary → Policies → Discussion (rolling agent context) → **Related work (placed late,
since no prior work measures this axis)** → Conclusion → Appendix A (experimental
protocol: identity gate, pairing/McNemar, verbosity control) → Appendix B (what we
ruled out) → References.

## Citation verification

All references checked against OpenAlex and arXiv (June 2026). Both 2026 entries are
real and now carry correct titles/authors:
- RelayCaching (arXiv:2603.13289) — Geng, Gao, Wu, G. Liu, J. Liu. PDF claim confirmed:
  its "influence-based selection" scores tokens by attention received from subsequent
  positions = our downstream-attention quantity.
- Make Each Token Count (arXiv:2605.09649) — Bui, Nguyen, Cohan, Ying. Abstract
  confirms "learnable eviction can surpass the full cache."
TurboRAG / ReKV / MEDA author lists expanded from OpenAlex. Venue lines (SOSP/EuroSys/
NeurIPS/ICLR/…) are from memory — spot-check before any camera-ready submission.
