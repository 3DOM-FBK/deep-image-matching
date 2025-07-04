import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_txt', type=Path, required=True)
    parser.add_argument('-o', '--output_dir', type=Path, required=True)
    args = parser.parse_args()

    input_file = args.images_txt
    output_file = args.output_dir / "images.txt"

    if os.path.exists(output_file):
        print(f"Output directory {args.output_dir} already exists. Exiting.")
        quit()

    with open(input_file, 'r') as f, open(output_file, 'w') as out_f:
        lines = f.readlines()
        c = 0
        for line in lines:
            if line.startswith("#"):
                continue
            if c % 2 == 0:
                out_f.write(line)
            else:
                out_f.write("\n")
            c += 1