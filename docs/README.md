# Featransform — Pipeline Diagram

`featransform_architecture.drawio` shows the Featransform 2.4.0 pipeline as a data-processing walkthrough. Each stage pairs its steps with the table it produces, so you can follow one dataset from raw input to the selected feature set.

## Files

- `featransform_architecture.drawio` — the editable diagram source.
- `featransform_architecture.drawio.png` — the same diagram exported as PNG with the source embedded. This is the image the project [README](../README.md) shows.
- `build_diagram.py` — generates the `.drawio` file (and an SVG preview) from one layout definition.
- `example_run.py` — the worked example behind every table in the diagram.

## What the diagram shows

One worked example runs through all four stages. It has 400 rows, an id column, a few missing values, a categorical column, a datetime column and a binary target. It uses the `standard` preset with `random_state = 42`, and every value in the tables is real Featransform 2.4.0 output.

| # | Stage | What the table shows | Source modules |
|---|---|---|---|
| ① | Preprocessing | the processed frame: imputed cells, label codes, 12 calendar columns derived from `signup` | `pipeline.py`, `processing/datetime_features.py`, `processing/imputation.py`, `processing/encoding.py` |
| ② | Feature Engineering | the 13 columns the anomaly, clustering and dimensionality ensembles append, and which columns each model emits | `processing/scaling.py`, `features/anomaly.py`, `features/clustering.py`, `features/dimensionality.py` |
| ③ | Feature Selection | the importance ranking with threshold cut lines, the threshold sweep with scores and the tie rule, and the 11 kept columns grouped by origin | `optimization/selector.py`, `optimization/evaluator.py` |
| ④ | Output | the `fit_transform` result, the `transform` replay steps and the fitted-pipeline API | `pipeline.py`, `utils/serializer.py` |

Colour key used in the tables:

- **Red cell:** missing value in the input.
- **Amber cell:** imputed value.
- **Blue header:** new column created during preprocessing.
- **Green header:** engineered column.
- **Pink cell:** an anomaly flag that fired.
- **Purple row:** the selected subset.

## Updating the diagram

1. Edit the `.drawio` directly, or change `build_diagram.py` and regenerate:

   ```
   python docs/build_diagram.py docs/featransform_architecture.drawio /tmp/preview.svg
   ```

2. Re-export the PNG with draw.io desktop installed:

   ```
   drawio --export --format png --embed-diagram --border 2 \
     --output docs/featransform_architecture.drawio.png docs/featransform_architecture.drawio
   ```

   You can also open the file at https://app.diagrams.net, choose File → Export as → PNG, and tick "Include a copy of my diagram".

The project README loads the image from `https://raw.githubusercontent.com/TsLu1s/Featransform/main/docs/featransform_architecture.drawio.png`. The absolute URL means it also renders on PyPI, but it only resolves once this `docs/` folder is committed and pushed to `main`.
