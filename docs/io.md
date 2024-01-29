# Input/output

**Page under construction...**

## Check matches
To visualize the geometrically verified matches, you have two options: using  the COLMAP GUI or the script show_matches.py.

### COLMAP GUI
To install COLMAP with the GUI, please look at the COLMAP installation page (https://colmap.github.io/install.html). 

Setup a COLMAP projet:
- run COLMAP GUI
- `File` tab > `New project`
- Click Database `Open` and select `database.db` in the result folder
- Click Images `Select` to select the `images` folder inside your `project` directory
- Click `Save`

Check the matches:
- `Processing` tab > `Database Management`
- Choose an image > click on the top right `Overlapping images`
- Select the `Two-view geometries` tab
- Choose the `id` of an overlapping image and click on `Show matches`


### show_matches.py script
To visualize the results you can use the `show_matches.py` script. Pass to the `--images` argument the names of the images (e.g. "img01.jpg img02.jpg") or their ids (e.g. "1 2") and choose accordingly the `--type` between `names` if you specify the name of the image with the extension, or `ids` if you specifiy the image `id`. In COLMAP image `ids` starts from 1 and not from 0.

```
python3 ./deep-image-matching/show_matches.py \
  --images    "1 2" \
  --type      ids \
  --database  ./deep-image-matching/assets/example_cyprus/results_superpoint+lightglue_matching_lowres_quality_high/database.db \
  --imgsdir   ./deep-image-matching/assets/example_cyprus/images \
  --output    ./deep-image-matching/assets/example_cyprus/matches.png
```
or
```
python3 ./deep-image-matching/show_matches.py \
  --images    "img01.jpg img02.jpg" \
  --type      names \
  --database  ./deep-image-matching/assets/example_cyprus/results_superpoint+lightglue_matching_lowres_quality_high/database.db \
  --imgsdir   ./deep-image-matching/assets/example_cyprus/images \
  --output    ./deep-image-matching/assets/example_cyprus/matches.png
```
The matches are shown matches.png.