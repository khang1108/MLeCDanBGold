# What this is

LaTeX manuscript skeleton for HCMAI's MMM 2027 Video Browser Showdown submission.
The text and screenshot are explicit placeholders, ready for collaborative writing.
The five authors and shared affiliation are set; email account names are pending.

# Running it

Use TeX Live with pdfLaTeX, BibTeX, `latexmk`, and GNU Make. The initial build was
verified with TeX Live 2026 and latexmk 4.87. Python is not needed for this build;
keep using the repository's `aic` environment for any later analysis scripts.

On a fresh Ubuntu/Debian machine, install the prerequisites:

```sh
sudo apt-get update
sudo apt-get install latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra make
```

The fonts package is large; it supplies the `newtx` fonts used by the official template.
This workspace already has the required tools and packages installed.
Author names use Latin letters without diacritics in given-name, middle-name,
family-name order. All text uses the template's font setup.
From the repository root:

```sh
make -C paper
```

Open `paper/build/main.pdf`. For automatic rebuilds, run `make -C paper watch`;
stop with Ctrl+C. Clean generated output with `make -C paper clean`.
Without Make, run `cd paper` and then `latexmk main.tex`.

For Overleaf, upload the contents of `paper/` without `build/`, select `main.tex`
as the main document, and use pdfLaTeX with a recent TeX Live version.
The local build is verified; Overleaf has not been tested in this session.

# Running the tests

The relevant validation is a clean LaTeX and BibTeX build:

```sh
make -C paper clean
make -C paper
```

A healthy result exits successfully and produces `paper/build/main.pdf` with
resolved citations and figure references. Inspect `paper/build/main.log` for
undefined references and overfull boxes, and inspect the PDF after layout changes.
The initial skeleton is two pages; this is not a final submission compliance check.
The verified build has two non-fatal warnings: an `amsmath` warning about
redefining `\vec` with the publisher's class/font combination, and an underfull
bibliography line. There are no undefined references or overfull boxes.

# How it's built

`main.tex` owns metadata, packages, section order, and bibliography setup.
`sections/` contains the abstract, introduction, related work, system overview,
retrieval, interactive search, evaluation, and conclusion. Edit these independently;
their root comments point editors back to `main.tex`.
`references.bib` stores BibTeX entries; `figures/` stores graphics.
`llncs.cls` and `splncs04.bst` are unchanged publisher files. `.latexmkrc` puts
generated output in the ignored `build/` directory. Publisher provenance is in
`vendor/lncs/PROVENANCE.md`.

# Decisions

The format is **LNCS**, not “LINC.” As checked on September 21, 2026, the
[VBS 2027 call](https://videobrowsershowdown.org/call-for-papers/)
specifies an extended demo paper with six content pages plus two pages available
for references. It requires a system description, a screenshot, and an explanation
of interactive search. Submit through the MMM submission system's Video Browser
Showdown track. The [published dates](https://videobrowsershowdown.org/call-for-papers/important-dates/)
list September 22, 2026 as the extended submission deadline; check the submission
system for its precise cutoff and timezone.

We use the [official Springer proceedings template](https://link.springer.com/series/558/information-for-authors-and-editors),
downloaded September 21, 2026, including its September 3, 2026 class and font setup.
Do not alter margins, font sizes, spacing, or publisher style files to fit more text.
The section outline is editable and is not mandated by VBS. No results or claims
about supported VBS tasks are asserted by these placeholders.

Before submission, replace all placeholders, confirm authors and affiliations,
verify the track's current anonymity requirements, add the real screenshot,
and manually check the six-page content and eight-page total limits.
The VBS call consulted here does not specify an anonymity policy; the skeleton's
placeholder author block does not establish one. No research decisions were made,
so `KNOWLEDGE.md` was not changed.
