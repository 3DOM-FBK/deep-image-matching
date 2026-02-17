# based on: https://github.com/uoip/monoVO-python

import numpy as np
import cv2
import logging
import glob

def create_dataloader(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        loader = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return loader

"""
针孔相机模型类：用于定义针孔相机的内参
fx,fy:焦距
cx,cy:光心位置
k1,k2,p1,p2,p3:畸变参数
"""
class PinholeCamera(object):
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

class KITTILoader(object):
    default_config = {
        "root_path": "../test_imgs",
        "sequence": "00",
        "start": 0
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("KITTI Dataset config: ")
        logging.info(self.config)

        if self.config["sequence"] in ["00", "01", "02"]:
            self.cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
        elif self.config["sequence"] in ["03"]:
            self.cam = PinholeCamera(1242.0, 375.0, 721.5377, 721.5377, 609.5593, 172.854)
        elif self.config["sequence"] in ["04", "05", "06", "07", "08", "09", "10"]:
            self.cam = PinholeCamera(1226.0, 370.0, 707.0912, 707.0912, 601.8873, 183.1104)
        else:
            raise ValueError(f"Unknown sequence number: {self.config['sequence']}")

        # read ground truth pose
        self.pose_path = self.config["root_path"] + "/poses/" + self.config["sequence"] + ".txt"
        self.gt_poses = []
        with open(self.pose_path) as f:
            lines = f.readlines()
            for line in lines:
                ss = line.strip().split()
                pose = np.zeros((1, len(ss)))
                for i in range(len(ss)):
                    pose[0, i] = float(ss[i])

                pose.resize([3, 4])
                self.gt_poses.append(pose)

        # image id
        self.img_id = self.config["start"]
        self.img_N = len(glob.glob(pathname=self.config["root_path"] + "/sequences/" \
                                            + self.config["sequence"] + "/image_0/*.png"))
        
    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]

    def __getitem__(self, item):
        file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] \
                    + "/image_0/" + str(item).zfill(6) + ".png"
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] \
                        + "/image_0/" + str(self.img_id).zfill(6) + ".png"
            img = cv2.imread(file_name)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]
    

def create_detector(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        detector = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return detector


def create_matcher(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        matcher = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return matcher

class FrameByFrameMatcher(object):
    default_config = {
        "type": "FLANN",
        "KNN": {
            "HAMMING": True,  # For ORB Binary descriptor, only can use hamming matching
            "first_N": 300,  # For hamming matching, use first N min matches
        },
        "FLANN": {
            "kdTrees": 5,
            "searchChecks": 50
        },
        "distance_ratio": 0.75
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Frame by frame matcher config: ")
        logging.info(self.config)

        if self.config["type"] == "KNN":
            logging.info("creating brutal force matcher...")
            if self.config["KNN"]["HAMMING"]:
                logging.info("brutal force with hamming norm.")
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher()
        elif self.config["type"] == "FLANN":
            logging.info("creating FLANN matcher...")
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.config["FLANN"]["kdTrees"])
            search_params = dict(checks=self.config["FLANN"]["searchChecks"])  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher_type}")

    def match(self, kptdescs):
        self.good = []
        # get shape of the descriptor
        self.descriptor_shape = kptdescs["ref"]["descriptors"].shape[1]

        if self.config["type"] == "KNN" and self.config["KNN"]["HAMMING"]:
            logging.debug("KNN keypoints matching...")
            matches = self.matcher.match(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"])
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            # self.good = matches[:self.config["KNN"]["first_N"]]
            for i in range(self.config["KNN"]["first_N"]):
                self.good.append([matches[i]])
        else:
            logging.debug("FLANN keypoints matching...")
            matches = self.matcher.knnMatch(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"], k=2)
            # Apply ratio test
            for m, n in matches:
                if m.distance < self.config["distance_ratio"] * n.distance:
                    self.good.append([m])
            # Sort them in the order of their distance.
            self.good = sorted(self.good, key=lambda x: x[0].distance)
        return self.good

    def get_good_keypoints(self, kptdescs):
        logging.debug("getting matched keypoints...")
        kp_ref = np.zeros([len(self.good), 2])
        kp_cur = np.zeros([len(self.good), 2])
        match_dist = np.zeros([len(self.good)])
        for i, m in enumerate(self.good):
            kp_ref[i, :] = kptdescs["ref"]["keypoints"][m[0].queryIdx]
            kp_cur[i, :] = kptdescs["cur"]["keypoints"][m[0].trainIdx]
            match_dist[i] = m[0].distance

        ret_dict = {
            "ref_keypoints": kp_ref,
            "cur_keypoints": kp_cur,
            "match_score": self.normalised_matching_scores(match_dist)
        }
        return ret_dict

    def __call__(self, kptdescs):
        self.match(kptdescs)
        return self.get_good_keypoints(kptdescs)

    def normalised_matching_scores(self, match_dist):

        if self.config["type"] == "KNN" and self.config["KNN"]["HAMMING"]:
            # ORB Hamming distance
            best, worst = 0, self.descriptor_shape * 8  # min and max hamming distance
            worst = worst / 4  # scale
        else:
            # for non-normalized descriptor
            if match_dist.max() > 1:
                best, worst = 0, self.descriptor_shape * 2  # estimated range
            else:
                best, worst = 0, 1

        # normalise the score!
        match_scores = match_dist / worst
        # range constraint
        match_scores[match_scores > 1] = 1
        match_scores[match_scores < 0] = 0
        # 1: for best match, 0: for worst match
        match_scores = 1 - match_scores

        return match_scores

    def draw_matched(self, img0, img1):
        pass

# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts):
    kpts = np.round(kpts).astype(int)
    for x, y in kpts:
        cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image

class VisualOdometry(object):
    """
    A simple frame by frame visual odometry
    """

    def __init__(self, detector, matcher, cam):
        """
        :param detector: a feature detector can detect keypoints their descriptors
        :param matcher: a keypoints matcher matching keypoints between two frames
        :param cam: camera parameters
        """
        # feature detector and keypoints matcher
        self.detector = detector
        self.matcher = matcher

        # camera parameters
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        # frame index counter
        self.index = 0

        # keypoints and descriptors
        self.kptdescs = {}

        # match points
        self.match_kps = {}

        # pose of current frame
        self.cur_R = None
        self.cur_t = None

    def update(self, image, absolute_scale=1):
        """
        update a new image to visual odometry, and compute the pose
        :param image: input image
        :param absolute_scale: the absolute scale between current frame and last frame
        :return: R and t of current frame
        """
        predict_data = self.detector.extract(image)
        kptdesc = {
            "keypoints": predict_data["keypoints"].cpu().detach().numpy(),
            "descriptors": predict_data["descriptors"].cpu().detach().numpy()
        }

        # first frame
        if self.index == 0:
            # save keypoints and descriptors
            self.kptdescs["cur"] = kptdesc

            # start point
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
        else:
            # update keypoints and descriptors
            self.kptdescs["cur"] = kptdesc

            # match keypoints
            matches = self.matcher(self.kptdescs)
            self.match_kps = {"cur":matches['cur_keypoints'], "ref":matches['ref_keypoints']}

            # compute relative R,t between ref and cur frame
            E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],
                                           focal=self.focal, pp=self.pp,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                            focal=self.focal, pp=self.pp)

            # get absolute pose based on absolute_scale
            if (absolute_scale > 0.1):
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)

        self.kptdescs["ref"] = self.kptdescs["cur"]

        self.index += 1
        return self.cur_R, self.cur_t

# 计算当前帧和上一帧的绝对位移，用于调整相机的平移向量
class AbosluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose

        scale = 1.0
        if self.count != 0:
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) * (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) * (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) * (self.cur_pose[2, 3] - self.prev_pose[2, 3]))

        self.count += 1
        self.prev_pose = self.cur_pose
        return scale



