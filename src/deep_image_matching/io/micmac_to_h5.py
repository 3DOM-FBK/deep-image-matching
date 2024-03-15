from pathlib import Path

from deep_image_matching.io.h5_to_micmac import read_Homol_matches


def micmac_to_h5(dir: Path):

    dir = Path(dir)
    if not dir.exists():
        raise FileNotFoundError(f"Homol directory {dir} does not exist")

    images = sorted([i.name.replace("Pastis", "") for i in dir.glob("*")])

    for i in images:
        print(i)
    read_Homol_matches()

    pass


if __name__ == "__main__":

    homol_path = Path("datasets/cyprus_micmac2/micmac/Homol")

    micmac_to_h5(homol_path)
    print("Done!")
