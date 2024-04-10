import os

image_dir = r"./images"

images = os.listdir(image_dir)
out_string = ""
for image in images:
    out_string = out_string + f"{image},"

print(out_string)
