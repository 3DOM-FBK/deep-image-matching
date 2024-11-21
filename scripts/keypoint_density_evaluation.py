import sqlite3
import argparse
from pathlib import Path

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def ExportMatches(
    database_path: Path, 
    min_num_matches: int, 
    output_path: Path,
    width: int,
    height: int,
) -> None:
    
    if not output_path.exists():
        print("Output folder does not exist")
        quit()

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    images = {}
    pairs = []

    cursor.execute("SELECT image_id, name FROM images")
    for row in cursor:
        image_id = row[0]
        name = row[1]
        images[image_id] = name

    cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
    for row in cursor:
        pair_id = row[0]
        n_matches = row[1]
        id_img1, id_img2 = pair_id_to_image_ids(pair_id)
        id_img1, id_img2 = int(id_img1), int(id_img2)
        img1 = images[id_img1]
        img2 = images[id_img2]

        if n_matches >= min_num_matches:
            pairs.append((img1, img2))
    
    with open(output_path / "pairs.txt", "w") as f:
        n_pairs = len(pairs)
        n_brute = 0
        N = len(images.keys())
        for x in range(1, N):
            n_brute += N-x
        print(f"n_pairs/n_brute: {n_pairs} / {n_brute}")
        
        for pair in pairs:
            f.write(f"{pair[0]} {pair[1]}\n")

    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export pairs in a txt file from a COLMAP database."
    )
    parser.add_argument("-d", "--database", type=Path, required=True, help="Path to COLMAP database.")
    parser.add_argument("-m", "--min_n_matches", type=int, required=True, help="Min number of matches that a pair should have after geometric verification.")
    parser.add_argument("-w", "--width", type=int, required=True, help="Image width.")
    parser.add_argument("-e", "--height", type=int, required=True, help="Image height.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to output folder.")
    args = parser.parse_args()

    ExportMatches(
        database_path=args.database,
        min_num_matches=args.min_n_matches,
        output_path=args.output,
        width=args.width,
        height=args.height,
    )