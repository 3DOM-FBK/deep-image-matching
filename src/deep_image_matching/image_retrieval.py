import os
import shutil

from .thirdparty.hloc import extract_features, pairs_from_retrieval


def ImageRetrieval(imgs_dir, outs_dir, retrieval_option, sfm_pairs):
    if outs_dir.exists():
        shutil.rmtree(outs_dir)
        os.mkdir(outs_dir)
    else:
        os.mkdir(outs_dir)

    number_imgs = len(os.listdir(imgs_dir))
    retrieval_conf = extract_features.confs[retrieval_option]
    retrieval_path = extract_features.main(retrieval_conf, imgs_dir, outs_dir)

    try:
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=number_imgs)
    except Exception as e:
        print("retrieval_path", retrieval_path)
        print("sfm_pairs", sfm_pairs)
        print("number_imgs", number_imgs)
        print(f"Error: {e}")
        quit()

    img_pairs = []
    with open(outs_dir / "retrieval_pairs.txt", "r") as pairs:
        lines = pairs.readlines()
        for line in lines:
            im1, im2 = line.strip().split(" ", 1)
            img_pairs.append((im1, im2))

    index_duplicate_pairs = []
    for i in range(len(img_pairs) - 1):
        pair1 = img_pairs[i]
        im1 = pair1[0]
        im2 = pair1[1]
        for j in range(i + 1, len(img_pairs)):
            pair2 = img_pairs[j]
            im3 = pair2[0]
            im4 = pair2[1]
            if im3 == im1 and im4 == im2:
                index_duplicate_pairs.append(j)
                # print('discarded', im1, im2, im3, im4)
            elif im3 == im2 and im4 == im1:
                index_duplicate_pairs.append(j)
                # print('discarded', im1, im2, im3, im4)
            else:
                pass

    pairs = []
    with open(outs_dir / "pairs_no_duplicates.txt", "w") as final_pairs:
        for i in range(len(img_pairs) - 1):
            if i not in index_duplicate_pairs:
                final_pairs.write(f"{img_pairs[i][0]} {img_pairs[i][1]}\n")
                pairs.append((imgs_dir / img_pairs[i][0], imgs_dir / img_pairs[i][1]))

    return pairs
