# Usage with MICMAC

**Please, note that the export to MICMAC is still under development.**

To import the matches found by Deep-Image-Matching into MICMAC, you must first run the matching according to the [getting started guide](./getting_started.md).

The extracted features and matches are stored in two hdf5 files in the output directory. You need those file to convert the matches to the format required by MICMAC.


Then you can use the [h5_to_micmac](https://github.com/3DOM-FBK/deep-image-matching/blob/dev/src/deep_image_matching/io/h5_to_micmac.py) script to convert the matches to the format required by MICMAC:

To run the script, please check first the help message:

```bash
python -m deep_image_matching.io.h5_to_micmac --help
```

The script requires the following arguments:

- `--image_dir`: the directory containing the images
- `--features_h5`: the path to the features hdf5 file
- `--matches_h5`: the path to the matches hdf5 file
- `--output_dir`: the directory where the output files will be saved
- `--img_ext`: the extension of the images

```bash
python -m deep_image_matching.io.h5_to_micmac --image_dir /path/to/images --features_h5 /path/to/features.h5 --matches_h5 /path/to/matches.h5 --output_dir /path/to/output --img_ext JPG
```

If you want to directly run TAPAS with the matches found by Deep-Image-Matching, you can add the `--run_Tapas` flag. In this case, you must also specify the path to the MICMAC executables with the `--micmac_path` flag

- `--run_Tapas`: if present, the script will run TAPAS with the matches found by Deep-Image-Matching
- `--micmac_path`: the path to the MICMAC executables

```bash
python -m deep_image_matching.io.h5_to_micmac --image_dir /path/to/images --features_h5 /path/to/features.h5 --matches_h5 /path/to/matches.h5 --output_dir /path/to/output --img_ext JPG --run_Tapas --micmac_path
```
