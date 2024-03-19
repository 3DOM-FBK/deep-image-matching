# Use with OpenMVG

DIM can export matches in OpenMVG-compatible format and execute a comprehensive SfM reconstruction. All intermediate and final OpenMVG data are stored in the project folder specified with the `--dir` option, within the results folder for the current pipeline, under the openmvg subfolder.

To run OpenMVG processing pass to `--openmvg` option the configuration file that containes path to the binaries and run for instance:

```
python ./main.py --dir ./assets/example_cyprus --pipeline superpoint+lightglue --openmvg ./config/openmvg_win.yaml
```

An example of openmvg configuration file for windows:
```
general:
  path_to_binaries: path\to\OpenMVG\ReleaseV1.6.Halibut.WindowsBinaries_VS2017
  openmvg_database: path\to\ReleaseV1.6.Halibut.WindowsBinaries_VS2017\sensor_width_camera_database.txt
```

An example of openmvg configuration file for linux:
```
general:
  path_to_binaries: null # If None, the binaries are assumed to be in the PATH
  openmvg_database: null # If None, it will be downloaded from the openMVG repository
```

Camera model for openmvg can be specified in `config/cameras.yaml`:
```
general:
  camera_model: "pinhole" # ["simple-pinhole", "pinhole", "simple-radial", "opencv"]
  openmvg_camera_model: "pinhole_radial_k3" # ["pinhole", "pinhole_radial_k3", "pinhole_brown_t2"]
  single_camera: False
```
Cameras can be shared between all the images (`single_camera == True`), or each camera can have a different camera model (`single_camera == False`).