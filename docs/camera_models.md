# Camera model options

For the COLMAP database, by default, DIM assigns camera models to images based on the options loaded from the `config/cameras.yaml` file, unless otherwise specified.

For images not assigned to specific `cam<x>` camera groups, the options specified under `general` are applied. The `camera_model` can be selected from `["simple-pinhole", "pinhole", "simple-radial", "opencv"]`. It's worth noting that it's easily possible to extend this to include all the classical COLMAP camera models. Cameras can either be shared among all images (`single_camera == True`), or each camera can have a different camera model (`single_camera == False`).

A subset of images can share intrinsics using `cam<x>` key, by specifying the `camera_model` along with the names of the images separated by commas. For instance:

```
cam0:
  camera_model: "pinhole"
  images: "DSC_6468.jpg,DSC_6469.jpg"
```

There's no limit to the number of `cam<x>` entries you can use, just add them following the provided format.

A comprehensive example of a `cameras.yaml` file can be found in the `config` folder:

```
general:
  camera_model: "pinhole" # ["simple-pinhole", "pinhole", "simple-radial", "opencv"]
  openmvg_camera_model: "pinhole_radial_k3" # ["pinhole", "pinhole_radial_k3", "pinhole_brown_t2"]
  single_camera: True

cam0:
  camera_model: "pinhole"
  images : ""

cam1:
  camera_model: "pinhole"
  images : "DSC_6468.jpg,DSC_6468.jpg"
```

For OpenMVG and MICMAC, refer to their respective sections.