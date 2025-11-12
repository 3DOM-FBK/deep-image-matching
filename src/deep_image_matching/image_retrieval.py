import os
import shutil

from .thirdparty.hloc import extract_features, pairs_from_retrieval


def ImageRetrieval(imgs_dir, outs_dir, retrieval_option, sfm_pairs):
    max_track_length = 10  # Increase this number to increase the number of pairs
    if outs_dir.exists():
        shutil.rmtree(outs_dir)
    os.mkdir(outs_dir)

    number_imgs = len(os.listdir(imgs_dir))
    retrieval_conf = extract_features.confs[retrieval_option]
    retrieval_path = extract_features.main(retrieval_conf, imgs_dir, outs_dir)

    try:
        pairs_from_retrieval.main(
            retrieval_path, sfm_pairs, num_matched=max_track_length
        )
    except Exception as e:
        print("retrieval_path", retrieval_path)
        print("sfm_pairs", sfm_pairs)
        print("number_imgs", number_imgs)
        print(f"Error: {e}")
        quit()

    img_pairs = []
    with open(outs_dir / "retrieval_pairs.txt", "r") as f:
        for line in f:
            im1, im2 = line.strip().split(" ", 1)
            img_pairs.append((im1, im2))

    unique_pairs = set()
    pairs = []
    with open(outs_dir / "pairs_no_duplicates.txt", "w") as final_pairs:
        for im1, im2 in img_pairs:
            pair_key = tuple(sorted((im1, im2)))
            if pair_key not in unique_pairs:
                unique_pairs.add(pair_key)
                final_pairs.write(f"{im1} {im2}\n")
                pairs.append((imgs_dir / im1, imgs_dir / im2))

    return pairs