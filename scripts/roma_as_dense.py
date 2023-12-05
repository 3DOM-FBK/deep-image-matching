# 1. run matching with a feature-based approach (e.g., superpoint+lightglue)
# 2. run reconstruction with COLMAP or pycolmap
# 3. Export reconstruction as text file with COLMAP


with open(r"./sparse/images.txt", "r") as inp, open(
    r"./cleaned/images.txt", "w"
) as out:
    lines = inp.readlines()
    for i, line in enumerate(lines):
        if i % 2 == 0:
            out.write(line)
            out.write("\n")
