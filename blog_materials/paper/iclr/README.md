# ICLR 2026 version (final / camera-ready style)

The paper typeset in the official ICLR 2026 template, in **final mode**
(`\iclrfinalcopy` is enabled → author shown, header reads "Published as a
conference paper at ICLR 2026"). This is the same content as `../main.tex`
(article class); only the formatting and citation style differ.

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
- To revert to anonymous submission style, comment out `\iclrfinalcopy`.
