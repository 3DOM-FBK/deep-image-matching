import numpy as np
import cv2
import argparse
import yaml
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.VisualOdometry import VisualOdometry, AbosluteScaleComputer, create_dataloader, \
    plot_keypoints, create_detector, create_matcher
from models.liftfeat_wrapper import LiftFeat,MODEL_PATH


vo_config = {
    'dataset': {
        'name': 'KITTILoader',
        'root_path': '/home/yepeng_liu/code_python/dataset/visual_odometry/kitty/gray',
        'sequence': '10',
        'start': 0
    },
    'detector': {
        'name': 'LiftFeatDetector',
        'descriptor_dim': 64,
        'nms_radius': 5,
        'keypoint_threshold': 0.005,
        'max_keypoints': 4096,
        'remove_borders': 4,
        'cuda': 1
    },
    'matcher': {
        'name': 'FrameByFrameMatcher',
        'type': 'FLANN',
        'FLANN': {
            'kdTrees': 5,
            'searchChecks': 50
        },
        'distance_ratio': 0.75
    }
}

# 可视化当前frame的关键点
def keypoints_plot(img, vo, img_id, path2):
    img_ = cv2.imread(path2+str(img_id-1).zfill(6)+".png")
  
    if not vo.match_kps:
        img_ = plot_keypoints(img_, vo.kptdescs["cur"]["keypoints"])
    else:
        for index in range(vo.match_kps["ref"].shape[0]):
            ref_point = tuple(map(int, vo.match_kps['ref'][index,:]))  # 将关键点转换为整数元组
            cur_point = tuple(map(int, vo.match_kps['cur'][index,:]))
            cv2.line(img_, ref_point, cur_point, (0, 255, 0), 2)  # Draw green line
            cv2.circle(img_, cur_point, 3, (0, 0, 255), -1)  # Draw red circle at current keypoint

    return img_

# 负责绘制相机的轨迹并计算估计轨迹与真实轨迹的误差。
class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((800, 800, 3), dtype=np.uint8)
        pass

    def update(self, est_xyz, gt_xyz):
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]
        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)
        error = np.linalg.norm(est - gt)
        self.errors.append(error)
        avg_error = np.mean(np.array(self.errors))
        # === drawer ==================================
        # each point
        draw_x, draw_y = int(x) + 80, int(z) + 230
        true_x, true_y = int(gt_x) + 80, int(gt_z) + 230

        # draw trajectory
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 0, 255), 1)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 255, 0), 2)
        cv2.rectangle(self.traj, (10, 5), (450, 120), (0, 0, 0), -1)

        # draw text
        text = "[AvgError] %2.4fm" % (avg_error)
        print(text)
        cv2.putText(self.traj, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        note = "Green: GT, Red: Predict"
        cv2.putText(self.traj, note, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return self.traj

def run_video(args):
    # create dataloader
    vo_config["dataset"]['root_path'] = args.path1
    vo_config["dataset"]['sequence'] = args.id
    loader = create_dataloader(vo_config["dataset"])
    # create detector
    liftfeat=LiftFeat(weight=MODEL_PATH, detect_threshold=0.25)
    # create matcher
    matcher = create_matcher(vo_config["matcher"])

    absscale = AbosluteScaleComputer()
    traj_plotter = TrajPlotter()

  
    if not os.path.exists('./output'):
        os.makedirs('./output')
    fname = "kitti_liftfeat_flannmatch"
    log_fopen = open("output/" + fname + ".txt", mode='a')

    vo = VisualOdometry(liftfeat, matcher, loader.cam)

    # Initialize video writer for keypoints and trajectory videos
    keypoints_video_path = "output/" + fname + "_keypoints_liftfeat.avi"
    trajectory_video_path = "output/" + fname + "_trajectory_liftfeat.avi"

    # Set up video writer: choose codec and set FPS and frame size
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10  # Adjust the FPS according to your input data
    frame_size = (1200, 400)  # Get frame size from first image

    # Video writers for keypoints and trajectory
    keypoints_writer = cv2.VideoWriter(keypoints_video_path, fourcc, fps, frame_size)
    trajectory_writer = cv2.VideoWriter(trajectory_video_path, fourcc, fps, (800, 800))
    
    for i, img in enumerate(loader):
        img_id = loader.img_id
        gt_pose = loader.get_cur_pose()
       
        R, t = vo.update(img, absscale.update(gt_pose))
        
        # === log writer ==============================
        print(i, t[0, 0], t[1, 0], t[2, 0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3], file=log_fopen)

        # === drawer ==================================
        img1 = keypoints_plot(img, vo, img_id, args.path2)
        img1 = cv2.resize(img1, (1200, 400))
        img2 = traj_plotter.update(t, gt_pose[:, 3])

        # Write frames to videos
        keypoints_writer.write(img1)
        trajectory_writer.write(img2)

    # Release the video writers
    keypoints_writer.release()
    trajectory_writer.release()
    print(f"Videos saved as {keypoints_video_path} and {trajectory_video_path}")
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
    parser.add_argument('--path1', type=str, default='/home/yepeng_liu/code_python/dataset/visual_odometry/kitty/gray',
                        help='config file')
    parser.add_argument('--path2', type=str, default="/home/yepeng_liu/code_python/dataset/visual_odometry/kitty/color/sequences/03/image_2/",
                        help='config file')
    parser.add_argument('--id', type=str, default="03",
                        help='config file')
   

    args = parser.parse_args()
    
    run_video(args)
