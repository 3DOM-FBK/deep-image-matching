import os
import cv2
import glob
import logging
import argparse
import numpy as np
from nets.aliked import ALIKED
from copy import deepcopy

class ImageLoader(object):
    def __init__(self, filepath: str):
        self.images = glob.glob(os.path.join(filepath, '*.png')) + \
                      glob.glob(os.path.join(filepath, '*.jpg')) + \
                      glob.glob(os.path.join(filepath, '*.ppm'))
        self.images.sort()
        self.N = len(self.images)
        logging.info(f'Loading {self.N} images')
        self.mode = 'images'

    def __getitem__(self, item):
        filename = self.images[item]
        img = cv2.imread(filename)   
        return img

    def __len__(self):
        return self.N

def mnn_mather(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.75] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()

def plot_keypoints(image, kpts, radius=2, color=(0, 0, 255)):
    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        x0, y0 = kpt
        cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)
    return out

def plot_matches(image0,
                 image1,
                 kpts0,
                 kpts1,
                 matches,
                 radius=2,
                 color=(255, 0, 0),
                 mcolor=(0, 255, 0)):

    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[:H1, W0:, :] = out1

    mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)

    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1

        cv2.line(out, (x0, y0), (x1 + W0, y1),
                     color=mcolor,
                     thickness=1,
                     lineType=cv2.LINE_AA)

    return out
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALIKED image pair Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory.')
    parser.add_argument('--model', choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'], default="aliked-n16rot",
                        help="The model configuration")
    parser.add_argument('--device', type=str, default='cuda', help="Running device (default: cuda).")
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)')
    parser.add_argument('--scores_th', type=float, default=0.2,
                        help='Detector score threshold (default: 0.2).')
    parser.add_argument('--n_limit', type=int, default=5000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    image_loader = ImageLoader(args.input)
    model = ALIKED(model_name=args.model,
                  device=args.device,
                  top_k=args.top_k,
                  scores_th=args.scores_th,
                  n_limit=args.n_limit)
    
    logging.info("Press 'space' to start. \nPress 'q' or 'ESC' to stop!")        
    
    img_ref = image_loader[0]
    img_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    pred_ref = model.run(img_rgb)
    kpts_ref = pred_ref['keypoints']
    desc_ref = pred_ref['descriptors']
    
    for i in range(1, len(image_loader)):
        img = image_loader[i]
        if img is None:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = model.run(img_rgb)
        kpts = pred['keypoints']
        desc = pred['descriptors']

        matches = mnn_mather(desc_ref, desc)
        status = f"matches/keypoints: {len(matches)}/{len(kpts)}"
        
        vis_img = plot_matches(img_ref, img, kpts_ref, kpts, matches)
        
        cv2.namedWindow(args.model)
        cv2.setWindowTitle(args.model, args.model + ': ' + status)
        cv2.putText(vis_img, "Press 'q' or 'ESC' to stop.", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2, cv2.LINE_AA)
        cv2.imshow(args.model, vis_img)
        c = cv2.waitKey()
        if c == ord('q') or c == 27:
            break

    logging.info('Finished!')
    logging.info('Press any key to exit!')
    cv2.putText(vis_img, "Finished! Press any key to exit.", (10,70), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2, cv2.LINE_AA)
    cv2.imshow(args.model, vis_img)
    cv2.waitKey()