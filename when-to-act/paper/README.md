# The Cost of Restraint

Anonymous UC Berkeley CS 182 final-project paper for Fall 2026. The [compiled paper](paper.pdf) has eight pages of core content, one page of references and three technical appendix pages. Related work runs from the start of page 2 to the beginning of page 4.

## Compile the paper

Use Typst 0.14.2. From this directory, run

```sh
typst compile --ignore-system-fonts --font-path fonts main.typ paper.pdf
```

The source, bibliography, generated table data, four vector figures and Times-compatible TeX Gyre Termes fonts are included. This command works with a standalone copy of this directory and requires no network access, LaTeX installation, Python environment or model weights. The font distribution notices are in [fonts](fonts). The main text uses the 2015 single-column layout on US Letter, a 5.5 by 9 inch text area, 10-point body type, 11-point baseline spacing, a 17-point title, indented abstract and gray review ruler. The page allowance follows the CS 182 assignment. The historical [NIPS 2015 style and example](https://nips.cc/Conferences/2015/PaperInformation/StyleFiles) give the formatting reference.

## Regenerate the tables and figures

From the **experiment root**, run

```sh
source env.sh
uv sync --frozen --python /usr/bin/python3
sha256sum --check --quiet results/prediction-and-loss.sha256
uv run --frozen python paper/build_figures.py
typst compile --ignore-system-fonts --font-path paper/fonts paper/main.typ paper/paper.pdf
```

The figure builder reads the four training manifests, nine models’ official generation and candidate-ranking outputs, saved cluster-bootstrap summaries, eight loss histories and the supplementary summary. It checks the call-rate, decision-recall and exact-recall estimates against the prediction records. The script writes only [tables.json](tables.json) and the SVGs in [figures](figures). The [experiment README](../README.md) describes how to reconstruct the bootstrap summaries from the saved predictions. Neither figure nor summary regeneration requires inference or training.

The [project repository](https://github.com/ranadkar/182-project) contains the experiment under `when-to-act/`. The PDF refers to the accompanying code by its directory name to preserve anonymous review.
