# Camera model options

For the COLMAP database, by default, DIM assigns camera models to images based on the options loaded from the `config/cameras.yaml` file, unless otherwise specified.

For images not assigned to specific `cam<x>` camera groups, the options specified under `general` are applied. The `camera_model` can be selected from `["simple-pinhole", "pinhole", "simple-radial", "opencv"]`. It's worth noting that it's easily possible to extend this to include all the classical COLMAP camera models. Cameras can either be shared among all images (`single_camera == True`), or each camera can have a different camera model (`single_camera == False`).

A subset of images can share intrinsics using `cam<x>` key, by specifying the `camera_model` along with the names of the images separated by commas, and the `intrinsics` corresponding to the `camera_model`.

Note that you must specify the full image name, including the extension. Image name supports globbing, so you can use `*` to match multiple images.

If you want to read intrinsics from the EXIF data, you can set the `intrinsics` to `~` (null).

For instance:

```python
cam0:
  camera_model: "pinhole"
  images: "DSC_64*.jpg,DSC_65*.jpg"
  intrinsics: [481.14, 478.43, 481.44, 383.72]
```

There's no limit to the number of `cam<x>` entries you can use, just add them following the provided format.

A comprehensive example of a `cameras.yaml` file can be found in the `config` folder:

```python
general:
  camera_model: "pinhole" # ["simple-pinhole", "pinhole", "simple-radial", "opencv"]
  openmvg_camera_model: "pinhole_radial_k3" # ["pinhole", "pinhole_radial_k3", "pinhole_brown_t2"]
  single_camera: True
  intrinsics: ~

cam0:
  camera_model: "pinhole"
  intrinsics: [
    481.14, 478.43, 481.44, 383.72
  ]
  images : "cam0_*.jpg"

cam1:
  camera_model: "opencv"
  intrinsics: [
    481.91, 482.20, 482.70, 384.33,
    0.0, 0.0, 0.0, 0.0
  ]
  images : "cam1_*.jpg"
```

For OpenMVG and MICMAC, refer to their respective sections.