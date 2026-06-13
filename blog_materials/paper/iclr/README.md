# ICLR 2026 style (PREPRINT)

The paper typeset in the official ICLR 2026 template, used as a **preprint** —
this paper is **not** accepted or under review; we only borrow the style.
`\iclrfinalcopy` is enabled solely to un-anonymize (show the author + non-blind
title block), and the running header is overridden to **"Preprint"** right after
`\maketitle`, so it does *not* claim "Published" or "Under review". Same content
as `../main.tex` (article class); only formatting and citation style differ.

## Files
- `main.tex` — the paper (uses `iclr2026_conference.sty`, author-year natbib).
- `refs.bib` — BibTeX database, all entries verified against OpenAlex/arXiv.
- `main.pdf` — compiled output (12 pp incl. references + appendices).
- `iclr2026_conference.{sty,bst}`, `fancyhdr.sty`, `natbib.sty`, `math_commands.tex`
  — unmodified template files from the ICLR Master-Template.

## Build
```
tectonic main.tex          # runs BibTeX automatically
# or: pdflatex; bibtex; pdflatex; pdflatex
```

## Notes
- Citations are author-year (`\citep` / `\citet`) per ICLR house style.
- Main body runs ~9 pages (intro→conclusion); references and the two appendices
  (A: experimental protocol; B: ruled-out table) follow. If submitting, check the
  current ICLR main-text page limit — appendices/references are exempt.
- Header modes: **preprint** (current) = `\iclrfinalcopy` + `\lhead{Preprint}`;
  anonymous submission = comment out `\iclrfinalcopy` (gives "Under review" +
  hides author); official camera-ready = `\iclrfinalcopy` and remove the
  `\lhead{Preprint}` line (only legitimate if actually accepted).
