# Scripts and Demo Files

## Demo Files

The repository includes demo files in the root directory that serve as example launchers and templates:

- `demo.py` - Simple script demonstrating the complete workflow
- `demo.ipynb` - Interactive Jupyter notebook version of the demo

These are meant to be **copied and customized** by users who want to use DIM as a Python library or create their own custom workflows.

```bash
python demo.py --dir assets/example_cyprus --pipeline superpoint+lightglue
```

## Utility Scripts

You can use the scripts located inside the `scripts` folder for advanced usage of the Deep-Image-Matching pipeline.

Please note that the following scripts are not part of the main pipeline and are not guaranteed to work in all environments. They are provided as a starting point for advanced users who want to customize the pipeline to their needs.

Documentation for the scripts is in progress. If you have any questions, please open an issue.

## Check Matches

To visualize the geometrically verified matches, you have two options:

1. Use COLMAP to visualize the matches (see [COLMAP instructions](./colmap.md#colmap))
2. Use the `show_matches.py` script

### show_matches.py

To visualize the results you can use the `show_matches.py` script. Pass to the `--images` argument the names of the images (e.g. "img01.jpg img02.jpg") or their ids (e.g. "1 2") and choose accordingly the `--type` between `names` if you specify the name of the image with the extension, or `ids` if you specifiy the image `id`. In COLMAP image `ids` starts from 1 and not from 0.

```bash
python3 ./scripts/show_matches.py \
  --images    "1 2" \
  --type      ids \
  --database  ./assets/example_cyprus/results_superpoint+lightglue_matching_lowres_quality_high/database.db \
  --imgsdir   ./assets/example_cyprus/images \
  --output    ./assets/example_cyprus/matches.png
```

or

```bash
python3 ./scripts/show_matches.py \
  --images    "img01.jpg img02.jpg" \
  --type      names \
  --database  ./assets/example_cyprus/results_superpoint+lightglue_matching_lowres_quality_high/database.db \
  --imgsdir   ./assets/example_cyprus/images \
  --output    ./assets/example_cyprus/matches.png
```

The matches are shown in matches.png.
