import cv2
import torch

from fcgnn import GNN as fcgnn
from sift import SIFT
from utils.draw import draw_matches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sift_matcher = SIFT()
fcgnn_refiner = fcgnn().to(device)

img1_path = './assets/img1.jpg'
img2_path = './assets/img2.jpg'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

matches = sift_matcher(img1, img2, device=device)

draw_matches(img1, img2, matches.detach().cpu().numpy(), filename='matches_before.png')
print("'./matches_before.png' has been created")

img1_ = torch.tensor(img1.astype('float32') / 255.)[None, None].to(device)
img2_ = torch.tensor(img2.astype('float32') / 255.)[None, None].to(device)
matches = matches.to(device)

matches_refined = fcgnn_refiner.optimize_matches(img1_, img2_, matches, thd=0.999, min_matches=10)[0]

draw_matches(img1, img2, matches_refined.detach().cpu().numpy(), filename='matches_after.png')
print("'./matches_after.png' has been created")
