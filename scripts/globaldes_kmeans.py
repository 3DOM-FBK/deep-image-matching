# When working with huge datasets, using DL global descriptors and KMeans clustering, rextract keyframes from the dataset

from pathlib import Path
from src.deep_image_matching.thirdparty.hloc import extract_features, pairs_from_retrieval
from sklearn.cluster import KMeans
import h5py
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, help='Path to the directory containing the images')
    parser.add_argument('--outs_dir', type=str, help='Path to the output directory')
    parser.add_argument('--retrieval_conf', type=str, default='netvlad', help='Configuration for retrieval, e.g. netvlad')

    args = parser.parse_args()

    imgs_dir = Path(args.imgs_dir)
    outs_dir = Path(args.outs_dir)

    retrieval_conf = extract_features.confs[args.retrieval_conf]
    retrieval_path = extract_features.main(retrieval_conf, imgs_dir, outs_dir)

    gfeat_path = outs_dir / "global-feats-netvlad.h5"
    with h5py.File(gfeat_path, 'r') as f:
        images = list(f.keys())
        first_key = list(f.keys())[0]
        des_len = len(f[first_key]['global_descriptor'][:])
        total_vectors = len(f.keys())
        matrix = np.empty((total_vectors, des_len))
        for i, key in enumerate(f.keys()):
            matrix[i, :] = f[key]['global_descriptor'][:]
        #print(matrix.shape)
        #print(images);quit()

    X = 1000  # Number of clusters you want
    kmeans = KMeans(n_clusters=X, random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_

    keyframes = {}
    for index, cluster in enumerate(labels):
        keyframes[cluster] = images[index]
    
    with open(outs_dir / "keyframes.txt", "w") as f:
        for kfrm in keyframes.values():
            f.write(f"left/{kfrm}\n")
            f.write(f"right/{kfrm}\n")

    

    ## Get the cluster centers (centroids)
    #centroids = kmeans.cluster_centers_

