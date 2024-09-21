# Managing rotations

The majority of DL local features do not manage rotations. To manage this kind of datasets, the optional arg `--upright` followed by `custom`, `2clusters` or `exif` can be used. `Exif` apply rotations as written in the exif metadata (sperimental). If rotations respect a default position (e.g. horizonatl) are known, can be specified passing `cusom` option. Rotations must be declared in `./config/rotations.txt` in the format:

```
image0.txt 0
image1.txt 0
image2.txt 90
image3.txt 180
image4.txt 270
```