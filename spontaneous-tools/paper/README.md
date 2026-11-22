# From Observation to Action

[Read the peer-review PDF](paper.pdf). The anonymous submission has eight core pages (1–8), one reference page (9) and three technical appendix pages (10–12). The abstract, introduction, and methods are in [main.typ](main.typ), with separate files for [related work](background.typ), [results and discussion](findings.typ), [the appendices](appendix.typ), [references](references.bib), and [typesetting](style.typ). [Review notes for V2](review_notes.md) record open scientific questions.

## Build and reproduce the figures

From the repository directory containing `qwen/`, run the following commands. The build used Typst 0.14.2, TeX Gyre Termes and TeX Gyre Termes Math, and Python 3.12. The Python environment is local to `paper/.venv`. Rendering the saved paper requires neither a GPU nor a model download.

```sh
cd paper
uv sync --frozen --python 3.12
uv run --frozen python scripts/figures.py
typst compile --creation-timestamp 1795305600 main.typ paper.pdf
uv run --frozen python scripts/check_pdf.py --render-dir ../.cache/paper-review/four-condition-final
```

The figure command regenerates [numbers.json](numbers.json) and all four vector figures. [prepare.py](scripts/prepare.py) checks 15,948 first-turn records against the archived outputs, simulator labels and ledger states. It verifies the three prompt envelopes for all 108 final discrepancy cases and checks that removing the factual envelope leaves identical prompt text across deliveries. It reconstructs the absolute intervals in Figures 2 and 3, the two paired training effects and the two within-continuation delivery contrasts in the text. The checks use the recorded domain and seed resampling scheme and compare against the archived estimates. The fitting table is checked against configuration, adapter metadata and step logs for all 12 full fits.

[figures.py](scripts/figures.py) then draws the training and evaluation example, the independent-test comparison, the three-format delivery plot and the fitting curves. Typst sets the four tables from the regenerated JSON. To rebuild only the PDF, the Typst command suffices. The fixed timestamp controls PDF metadata. On a machine with fonts in a private directory, pass `--font-path /path/to/fonts` to Typst.

[check_pdf.py](scripts/check_pdf.py) measures the available text frame by typesetting a full-area rectangle with the production style. It separately inspects the actual paper for full-width prose on every page, a second column of prose, out-of-bounds text or graphics, baseline spacing, title rules, abstract indentation, review rulers, footers, embedded fonts and numbered references. The render command writes one 144-dpi PNG per page and a contact sheet. The vector labels inside the figures are checked visually.

## Evidence map

| Paper content | Primary repository artifacts |
| --- | --- |
| Public trajectory sample, predecessor context and authorization labels | [Sample method](../scripts/sample.py), [annotated decisions](../data/annotations.tsv), [example records](../results/examples.jsonl), [audit totals](../results/training-summary.json) |
| Olmo and Nemotron native actions, control outcomes and notebook trace | [Typed benchmark](../expanded/scenarios.json), [native parser](../expanded/native.py), [Olmo outputs](../expanded/results/olmo-main.jsonl), [Nemotron outputs](../expanded/results/nemotron-main.jsonl), [rates](../expanded/results/rates.csv) |
| Four treatments, shared targets and diagram | [Training rows](../qwen/v1/data/training.jsonl), [generation and checks](../qwen/v1/training_data.py), [token audit](../qwen/v1/data/training_audit.json). Figure 1 pairs fitting row `training-00-000` with the unmodified model’s [external test trace](../qwen/v1/raw/base.final.jsonl) for `kite-remove`. |
| Fitting settings, loss and calibration | [Pinned configuration](../qwen/v1/config.json), [optimizer and masking](../qwen/train.py), [fitting entry point](../qwen/v1/fit.py), [fitting records](../qwen/v1/results/fits.csv), [per-step loss](../qwen/v1/results/loss_curves.csv), [verification](../qwen/v1/results/verification.json) |
| Split, envelopes, controls, scoring and analysis plan | [Design](../qwen/v1/design.md), [suite](../qwen/v1/suite.py), [development amendment](../qwen/v1/data/amendment.json), [final lock](../qwen/v1/data/final_lock.json), [leakage checks](../qwen/v1/results/leakage.json), [serialized evaluation prompts](../qwen/v1/data/evaluation_prompts.jsonl) |
| Figures 2 and 3, seed and control tables, uncertainty, and longer-decoding sensitivity | [Individual trials](../qwen/v1/results/trials.csv), [raw outputs](../qwen/v1/raw/), [clustered estimates](../qwen/v1/results/estimates.csv), [seed breakdown](../qwen/v1/results/by_seed.csv), [analysis implementation](../qwen/v1/analyze.py) |
| Authorized and unsolicited outcomes on the same case | [cross-task counts](../qwen/v1/results/coupled_changes.csv), [annotated trace pairs](../qwen/v1/examples.md) |

For example, the primary trial filter is `suite=final`, `evidence=discrepancy`, `delivery=environment`. Sum `target_match` within each full fitting arm. To estimate an arm difference, pair the same case and seed before resampling. The named-write and authorized-continuation controls have distinct evidence labels. All fitted-arm comparisons in the paper use the same three seeds. The [experiment report](../qwen/v1/report.md) and its [paper notes](../qwen/v1/paper_notes.md) provide the broader experimental record.

## NIPS 2015 formatting

[style.typ](style.typ) follows the 2015 anonymous submission style. The official LaTeX2e style, legacy style, LaTeX shell and specimen PDF were downloaded from the conference site. We compared the rendered pages with both the supplied review specimen and a published single-column paper, *Spatial Transformer Networks*. The PDF served under the 2015 specimen URL retains an older year on its title page. The 2015 style file and shell govern the settings here. [Official style files](https://neurips.cc/Conferences/2015/PaperInformation/StyleFiles) · [LaTeX2e style](https://media.nips.cc/Conferences/2015/Styles/nips15submit_e.sty) · [Formatting instructions](https://media.nips.cc/Conferences/2015/Styles/nips2015.tex) · [Review specimen](https://media.nips.cc/Conferences/2015/Styles/nips2015.pdf) · [Published example](https://proceedings.neurips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf)

| Measurement | Verified output |
| --- | --- |
| Page and body frame | US Letter, 612 × 792 PDF points. One 396 × 648 pt (5.5 × 9 in) text column extending from (108, 72) to (504, 720), measured from the upper-left corner. |
| Margins | Left and right 108 pt (1.5 in). Top and bottom 72 pt (1 in). The gray review ruler and centered page number occupy the margins. |
| Body and abstract | 10 pt Times-compatible type with 11.00 pt median baseline spacing, unindented paragraphs and approximately half a line of additional paragraph spacing. The abstract is inset 36 pt on each side. |
| Title and headings | Centered 17 pt bold title between 4 pt and 1 pt rules. The abstract heading and top-level sections use 12 pt bold, subsections use 10 pt bold. |
| Illustrations and references | Centered tables with captions above, vector figures with captions below, and centered display equations. Tables and captions use 10 pt. References use the permitted 9 pt size and consecutive bracketed citations. |

**Differences from the LaTeX version.** Typst uses TeX Gyre Termes and its companion math font. Font metrics, line and page breaks, and figure placement can differ from TeX’s Times fonts, flexible vertical spacing and float algorithm. The fixed 12 pt review ruler is aligned to the requested one-inch top margin. The anonymous title block omits the sample affiliation, address and email placeholders. The course PDF includes its three appendix pages after the references. The 2015 conference submission limit covered eight core pages plus one references-only page.
