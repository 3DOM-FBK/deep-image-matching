from .core import adalam_core
from .utils import dist_matrix
from tqdm import tqdm
import torch
import numpy as np
import os
import sqlite3


class AdalamFilter:
    DEFAULT_CONFIG = {
        'area_ratio': 100,  # Ratio between seed circle area and image area. Higher values produce more seeds with smaller neighborhoods.
        'search_expansion': 4,  # Expansion factor of the seed circle radius for the purpose of collecting neighborhoods. Increases neighborhood radius without changing seed distribution
        'ransac_iters': 128,  # Fixed number of inner GPU-RANSAC iterations
        'min_inliers': 6,  # Minimum number of inliers required to accept inliers coming from a neighborhood
        'min_confidence': 200,  # Threshold used by the confidence-based GPU-RANSAC
        'orientation_difference_threshold': 30,  # Maximum difference in orientations for a point to be accepted in a neighborhood. Set to None to disable the use of keypoint orientations.
        'scale_rate_threshold': 1.5,  # Maximum difference (ratio) in scales for a point to be accepted in a neighborhood. Set to None to disable the use of keypoint scales.
        'detected_scale_rate_threshold': 5,  # Prior on maximum possible scale change detectable in image couples. Affinities with higher scale changes are regarded as outliers.
        'refit': True,  # Whether to perform refitting at the end of the RANSACs. Generally improves accuracy at the cost of runtime.
        'force_seed_mnn': True,  # Whether to consider only MNN for the purpose of selecting seeds. Generally improves accuracy at the cost of runtime. You can provide a MNN mask in input to skip MNN computation and still get the improvement.
        'device': torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu'),  # Device to be used for running AdaLAM. Use GPU if available.
#       'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),  # Device to be used for running AdaLAM. Use GPU if available.
        'th': 0.8 **2
    }

    def __init__(self, custom_config: dict=None):
        """
        This class acts as a wrapper to the method AdaLAM for outlier filtering.

        init args:
            custom_config: dictionary overriding the default configuration. Missing parameters are kept as default.
                           See documentation of DEFAULT_CONFIG for specific explanations on the accepted parameters.
        """
        self.config = AdalamFilter.DEFAULT_CONFIG.copy()

        if custom_config is not None:
            for key, val in custom_config.items():
                if key not in self.config.keys():
                    print("WARNING: custom configuration contains a key which is not recognized ({key}). "
                          "Known configurations are {list(self.config.keys())}.")
                    continue
                self.config[key] = val

    def filter_matches(self, k1: torch.Tensor, k2: torch.Tensor,
                       putative_matches: torch.Tensor,
                       scores: torch.Tensor, mnn: torch.Tensor=None,
                       im1shape: tuple=None, im2shape: tuple=None,
                       o1: torch.Tensor=None, o2: torch.Tensor=None,
                       s1: torch.Tensor=None, s2: torch.Tensor=None):
        """
            Call the core functionality of AdaLAM, i.e. just outlier filtering. No sanity check is performed on the inputs.

            Inputs:
                k1: keypoint locations in the source image, in pixel coordinates.
                    Expected a float32 tensor with shape (num_keypoints_in_source_image, 2).
                k2: keypoint locations in the destination image, in pixel coordinates.
                    Expected a float32 tensor with shape (num_keypoints_in_destination_image, 2).
                putative_matches: Initial set of putative matches to be filtered.
                                  The current implementation assumes that these are unfiltered nearest neighbor matches,
                                  so it requires this to be a list of indices a_i such that the source keypoint i is associated to the destination keypoint a_i.
                                  For now to use AdaLAM on different inputs a workaround on the input format is required.
                                  Expected a long tensor with shape (num_keypoints_in_source_image,).
                mnn: A mask indicating which putative matches are also mutual nearest neighbors. See documentation on 'force_seed_mnn' in the DEFAULT_CONFIG.
                     If None, it disables the mutual nearest neighbor filtering on seed point selection.
                     Expected a bool tensor with shape (num_keypoints_in_source_image,)
                im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of wasted runtime. So please provide it.
                          Expected a tuple with (width, height) or (height, width) of source image
                im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost of wasted runtime. So please provide it.
                          Expected a tuple with (width, height) or (height, width) of destination image
                o1/o2: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config is set to None.
                       See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
                       Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)
                s1/s2: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
                       See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
                       Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)

            Returns:
                Filtered putative matches.
                A long tensor with shape (num_filtered_matches, 2) with indices of corresponding keypoints in k1 and k2.
        """

        with torch.no_grad():
            return adalam_core(k1, k2, fnn12=putative_matches,
                                scores1=scores, mnn=mnn,
                                im1shape=im1shape,
                                im2shape=im2shape,
                                o1=o1, o2=o2,
                                s1=s1, s2=s2,
                                config=self.config)


    def match_and_filter(self, k1, k2, d1, d2, im1shape=None, im2shape=None,
                o1=None, o2=None, s1=None, s2=None, putative_matches=None, scores=None, mnn=None):
#               o1=None, o2=None, s1=None, s2=None):

        """
            Standard matching and filtering with AdaLAM.
            This function:
                - performs some elementary sanity check on the inputs;
                - wraps input arrays into torch tensors and loads to GPU if necessary;
                - extracts nearest neighbors;
                - finds mutual nearest neighbors if required;
                - finally calls AdaLAM filtering.

            Inputs:
                k1: keypoint locations in the source image, in pixel coordinates.
                    Expected an array with shape (num_keypoints_in_source_image, 2).
                k2: keypoint locations in the destination image, in pixel coordinates.
                    Expected an array with shape (num_keypoints_in_destination_image, 2).
                d1: descriptors in the source image.
                    Expected an array with shape (num_keypoints_in_source_image, descriptor_size).
                d2: descriptors in the destination image.
                    Expected an array with shape (num_keypoints_in_destination_image, descriptor_size).
                im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of wasted runtime. So please provide it.
                          Expected a tuple with (width, height) or (height, width) of source image
                im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost of wasted runtime. So please provide it.
                          Expected a tuple with (width, height) or (height, width) of destination image
                o1/o2: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config is set to None.
                       See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
                       Expected an array with shape (num_keypoints_in_source/destination_image,)
                s1/s2: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
                       See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
                       Expected an array with shape (num_keypoints_in_source/destination_image,)

            Returns:
                Filtered putative matches.
                A long tensor with shape (num_filtered_matches, 2) with indices of corresponding keypoints in k1 and k2.
        """
        if s1 is None or s2 is None:
            if self.config['scale_rate_threshold'] is not None:
                raise AttributeError("Current configuration considers keypoint scales for filtering, but scales have not been provided.\n"
                                     "Please either provide scales or set 'scale_rate_threshold' to None to disable scale filtering")
        if o1 is None or o2 is None:
            if self.config['orientation_difference_threshold'] is not None:
                raise AttributeError(
                    "Current configuration considers keypoint orientations for filtering, but orientations have not been provided.\n"
                    "Please either provide orientations or set 'orientation_difference_threshold' to None to disable orientations filtering")
        k1, k2, d1, d2, o1, o2, s1, s2 = self.__to_torch(k1, k2, d1, d2, o1, o2, s1, s2)
        o1=torch.squeeze(o1)
        o2=torch.squeeze(o2)
        s1=torch.squeeze(s1)
        s2=torch.squeeze(s2)

#       distmat = dist_matrix(d1, d2, is_normalized=False)
#       dd12, nn12 = torch.topk(distmat, k=2, dim=1, largest=False)  # (n1, 2)

        putative_matches = torch.squeeze(torch.from_numpy(putative_matches)).to(self.config['device'])
#       putative_matches = nn12[:, 0]
        scores = torch.squeeze(torch.from_numpy(scores)).to(self.config['device'])
#       scores = dd12[:, 0] / dd12[:, 1].clamp_min_(1e-3)
        if self.config['force_seed_mnn']:
#           dd21, nn21 = torch.min(distmat, dim=0)  # (n2,)
#           mnn = nn21[putative_matches] == torch.arange(k1.shape[0], device=self.config['device'])
            mnn = torch.squeeze(torch.from_numpy(mnn)).to(self.config['device'])
        else:
            mnn = None

        return self.filter_matches(k1, k2, putative_matches, scores, mnn,
                                   im1shape, im2shape, o1, o2, s1, s2)


    def __to_torch(self, *args):
        return (a if a is None or torch.is_tensor(a) else
                torch.tensor(a, device=self.config['device'], dtype=torch.float32) for a in args)

    def match_colmap_database(self, database_path: str, image_pairs_path: str, use_rootsift=True):
        """
            Match images inside a colmap database.

            Inputs:
                database_path: path to colmap database. This must exist.
                image_pairs_path: path to a text file containing a list of space-separated pairs of image names to be matched.
                                  If the file does not exist, exhaustive matching is assumed and the file is created accordingly.
                                  This argument is required as you will need it anyway to import matches in colmap afterwards!
                use_rootsift: whether to apply the rootsift transformation to descriptors found in colmap.
                              You may want to disable it if you are using non-sift descriptors.

            Returns:
                None

            Side effects:
                The matches table in the specified colmap database is populated with matches from AdaLAM.
                The two_view_geometries table is NOT affected, thus you will need to run the colmap matches_importer.
                If the image_pairs file does not exist, it is created and populated with exhaustive matching pairs.

            Known bugs:
                If the colmap database contains image names with spaces,
                    these mess with the spacing in the image_pairs file and makes this method to crash.
                    Workaround: sanitize names of your images and update these names in the colmap database before running.
        """
        def blob_to_array(blob, dtype, shape=(-1,)):
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)

        def image_ids_to_pair_id(image_id1, image_id2):
            if image_id1 > image_id2:
                return 2147483647 * image_id2 + image_id1
            else:
                return 2147483647 * image_id1 + image_id2

        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        images = dict(
                (image_name, image_id)
                for image_id, image_name in cursor.execute(
                    "SELECT image_id, name FROM images"))
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (rows, cols)))
            for image_id, rows, cols, data in cursor.execute(
                "SELECT * FROM keypoints"))
        descriptors = dict(
            (image_id, blob_to_array(data, np.uint8, (rows, cols)))
            for image_id, rows, cols, data in cursor.execute(
                "SELECT * FROM descriptors"))

        kpshape = keypoints.values().__iter__().__next__().shape[1]
        if kpshape == 4:
            kps = {imid: (imkp[:, :2], imkp[:, 2], imkp[:, 3]) for imid, imkp in keypoints.items()}
        elif kpshape == 6:
            kps = {imid: (imkp[:, :2],
                          (np.sqrt(imkp[:, 2] ** 2 + imkp[:, 4] ** 2) + np.sqrt(imkp[:, 3] ** 2 + imkp[:, 5] ** 2)) / 2,
                          np.arctan2(imkp[:, 2], imkp[:, 4]) * 180 / np.pi) for imid, imkp in keypoints.items()}
        else:
            raise TypeError()

        if os.path.exists(image_pairs_path):
            with open(image_pairs_path, 'r') as f:
                raw_pairs = f.readlines()
        else:
            raw_pairs = []
            images_list = sorted(list(images.keys()))
            for i1 in range(len(images_list)):
                for i2 in range(i1+1, len(images_list)):
                    raw_pairs.append(f"{images_list[i1]} {images_list[i2]}")
            with open(image_pairs_path, 'w') as f:
                for p in raw_pairs:
                    f.write(p + "\n")

        def rootsift(desc):
            l1d = np.linalg.norm(desc, axis=-1, ord=1, keepdims=True)
            sd = np.sign(desc)
            return np.sqrt(sd * desc / l1d) * sd

        image_pair_ids = set()
        for raw_pair in tqdm(raw_pairs, total=len(raw_pairs)):
            image_name1, image_name2 = raw_pair.strip('\n').split(' ')
            image_id1, image_id2 = images[image_name1], images[image_name2]
            image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
            if image_pair_id in image_pair_ids:
                continue

            desc1 = rootsift(descriptors[image_id1]) if use_rootsift else descriptors[image_id1]
            desc2 = rootsift(descriptors[image_id2]) if use_rootsift else descriptors[image_id2]
            k1, s1, o1 = kps[image_id1]
            k2, s2, o2 = kps[image_id2]

            im1shape = cursor.execute(
                "SELECT height, width FROM images as i JOIN cameras as c on c.camera_id = i.camera_id WHERE i.image_id = ?",
                (image_id1,)).fetchone()
            im2shape = cursor.execute(
                "SELECT height, width FROM images as i JOIN cameras as c on c.camera_id = i.camera_id WHERE i.image_id = ?",
                (image_id2,)).fetchone()


            matches = self.match_and_filter(k1=k1, k2=k2,
                                            o1=o1, o2=o2,
                                            s1=s1, s2=s2,
                                            d1=desc1, d2=desc2,
                                            im1shape=im1shape, im2shape=im2shape)
            matches = matches.cpu().numpy()
            image_pair_ids.add(image_pair_id)

            if image_id1 > image_id2:
                matches = matches[:, [1, 0]]

            matches_str = matches.astype(np.uint32).tostring()
            cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                           (image_pair_id, matches.shape[0], matches.shape[1], matches_str))
        connection.commit()

        # Close the connection to the database.
        cursor.close()
        connection.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Match a colmap database with AdaLAM")
    p.add_argument("--database_path", "-d", required=True)
    p.add_argument("--image_pairs_path", "-i", required=True)
    opt = p.parse_args()
    matcher = AdalamFilter()
    matcher.match_colmap_database(database_path=opt.database_path, image_pairs_path=opt.image_pairs_path)

