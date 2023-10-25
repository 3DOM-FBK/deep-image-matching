import os
import shutil

from hloc import extract_features, pairs_from_retrieval


def ImageRetrieval(imgs_dir, outs_dir, retrieval_option):
    if outs_dir.exists():
        shutil.rmtree(outs_dir)
        os.mkdir(outs_dir)
    else:
        os.mkdir(outs_dir)

    number_imgs = len(os.listdir(imgs_dir))

    sfm_pairs = outs_dir / "pairs.txt"
    retrieval_conf = extract_features.confs[retrieval_option]
    retrieval_path = extract_features.main(retrieval_conf, imgs_dir, outs_dir)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=number_imgs)

    return
