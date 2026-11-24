# The Cost of Restraint

Anonymous UC Berkeley CS 182 final-project paper for Fall 2026. The paper follows the CS 182 single-column format, with a core-text target of 5–10 pages. Page counts may change slightly after the final editorial typesetting pass; confirm with a fresh Typst compilation. Related work remains a substantial section.

## Compile the paper

Use Typst 0.14.2. From this directory, run

```sh
typst compile --ignore-system-fonts --font-path fonts main.typ paper.pdf
```

The revised source bundle includes the bibliography, cached tables and five vector figures. Keep the existing `fonts/` directory from your original project, which contains the Times-compatible TeX Gyre Termes fonts; the edit bundle does not redistribute font files. With that directory in place, compiling requires no network access, LaTeX installation, Python environment or model weights. The font distribution notices are in [fonts](fonts). The main text uses the 2015 single-column layout on US Letter, a 5.5 by 9 inch text area, 10-point body type, 11-point baseline spacing, a 17-point title, indented abstract and gray review ruler. The page allowance follows the CS 182 assignment. The historical [NIPS 2015 style and example](https://nips.cc/Conferences/2015/PaperInformation/StyleFiles) give the formatting reference.

## Regenerate the tables and figures

From the **experiment root**, run

```sh
source env.sh
uv sync --frozen --python /usr/bin/python3
sha256sum --check --quiet results/prediction-and-loss.sha256
sha256sum --check --quiet data/followup/outputs.sha256
uv run --frozen python paper/build_figures.py
typst compile --ignore-system-fonts --font-path paper/fonts paper/main.typ paper/paper.pdf
```

The figure builder reads the four training manifests and eight loss histories, plus the saved predictions and bootstrap summaries for all nine models in [expanded generation](../results/expanded-generation/), [matched candidate scoring](../results/matched-protocols/) and the [authored controls](../results/direct-answer-controls/). It checks the shared question IDs, gold classes, clusters, confusion counts, exact calls and protocol discordances against the raw records. The script writes [tables.json](tables.json) and the five SVGs in [figures](figures). The [analysis protocol](../data/followup/protocol.md) gives the offline commands to reconstruct the summaries and intervals from predictions. The [experiment README](../README.md) documents the original training setup.


The [project repository](https://github.com/ranadkar/182-project) contains the experiment under `when-to-act/`. The PDF refers to the accompanying code by its directory name to preserve anonymous review.

## Editorial revisions

The manuscript has been edited for concision and transparency. The forced break after the introduction was removed, an illustrative related-work paragraph was cut, the main-table F1 now averages the three supported gold classes, and the 54-output review is explicitly described as model-assisted. Original experiments, sample sizes, confidence intervals and figures are unchanged. Recompile the PDF locally with the command above before submission.
