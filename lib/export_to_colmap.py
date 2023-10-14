from pathlib import Path

def ExportToColmap(keypoints : dict, correspondences : dict, output_dir : Path) -> None:
    
    # Export keypoints
    for img in keypoints:
        with open(output_dir / f"{img}.txt", 'w') as file:
            kpts = keypoints[img]
            file.write(f"{kpts.shape[0]} 128\n")
            for row in range(kpts.shape[0]):
                file.write(f"{kpts[row, 0]} {kpts[row, 1]} 0.00 0.00\n")
    
    # Export tie points
    with open(output_dir / "colmap_matches.txt", 'w') as file:
        for pair in correspondences:
            im0 = pair[0].name
            im1 = pair[1].name
            file.write(f"{im0} {im1}\n")
            pair_tie_points = correspondences[pair]
            for row in range(pair_tie_points.shape[0]):
                file.write("{} {}\n".format(pair_tie_points[row, 0], pair_tie_points[row, 0]))
            file.write("\n")