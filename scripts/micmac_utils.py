import multiprocessing as mp
from itertools import combinations
from pathlib import Path

from deep_image_matching.io import h5_to_micmac as mm


def show_micmac_matches(project_dir: Path, img_pattern: str) -> None:

    # Load the data
    project_path = Path(project_dir)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path {project_path} does not exist")
    homol_path = project_path / "Homol"
    if not homol_path.exists():
        raise FileNotFoundError(f"Homol path {homol_path} does not exist")

    images = sorted(project_path.glob(img_pattern))
    matches_dir = project_path / "match_figs"
    matches_dir.mkdir(exist_ok=True, parents=True)

    for i0, i1 in combinations(images, 2):
        matches_path = homol_path / f"Pastis{i0.name}" / f"{i1.name}.txt"
        if not matches_path.exists():
            raise FileNotFoundError(f"Matches file {matches_path} does not exist")
        mm.show_micmac_matches(matches_path, project_path, matches_dir / f"matches_{i0.name}-{i1.name}.png")


    with mp.Pool() as p:
        p.starmap(
            mm.show_micmac_matches,
            [
                (
                    homol_path / f"Pastis{i0.name}" / f"{i1.name}.txt",
                    project_path,
                    matches_dir / f"matches_{i0.name}-{i1.name}.png",
                )
                for i0, i1 in combinations(images, 2)
            ],
        )
    return None


if __name__ == "__main__":
    project_path = Path("/home/francesco/phd/stereosat/pleiades_forni/micmac")
    img_pattern = "*.tif"
    show_micmac_matches(project_path, img_pattern)
