import math
import numpy as np
from PIL import Image
import cv2


THRESHOLD_FACTOR = 6

ROTATION_PATTERNS = [
    [1, 2, 3,
     4, 5, 6,
     7, 8, 9],

    [4, 1, 2,
     7, 5, 3,
     8, 9, 6],

    [7, 4, 1,
     8, 5, 2,
     9, 6, 3],

    [8, 7, 4,
     9, 5, 1,
     6, 3, 2],

    [9, 8, 7,
     6, 5, 4,
     3, 2, 1],

    [6, 9, 8,
     3, 5, 7,
     2, 1, 4],

    [3, 6, 9,
     2, 5, 8,
     1, 4, 7],

    [2, 3, 6,
     1, 5, 9,
     4, 7, 8]]


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class GmsMatcher:
    def __init__(self, kp1, kp2, m12):
        self.kp1=kp1
        self.kp2=kp2
        self.m12=m12

        self.scale_ratios = [1.0, 1.0 / 2, 1.0 / math.sqrt(2.0), math.sqrt(2.0), 2.0]
        
        # Normalized vectors of 2D points
        self.normalized_points1 = []
        self.normalized_points2 = []
        # Matches - list of pairs representing numbers
        self.matches = []
        self.matches_number = 0
        # Grid Size
        self.grid_size_right = Size(0, 0)
        self.grid_number_right = 0
        # x      : left grid idx
        # y      :  right grid idx
        # value  : how many matches from idx_left to idx_right
        self.motion_statistics = []

        self.number_of_points_per_cell_left = []
        # Inldex  : grid_idx_left
        # Value   : grid_idx_right
        self.cell_pairs = []

        # Every Matches has a cell-pair
        # first  : grid_idx_left
        # second : grid_idx_right
        self.match_pairs = []

        # Inlier Mask for output
        self.inlier_mask = []
        self.grid_neighbor_right = []

        # Grid initialize
        self.grid_size_left = Size(20, 20)
        self.grid_number_left = self.grid_size_left.width * self.grid_size_left.height

        # Initialize the neihbor of left grid
        self.grid_neighbor_left = np.zeros((self.grid_number_left, 9))

#       self.descriptor = descriptor
#       self.matcher = matcher
        self.gms_matches = []
        self.keypoints_image1 = []
        self.keypoints_image2 = []

    def empty_matches(self):
        self.normalized_points1 = []
        self.normalized_points2 = []
        self.matches = []
        self.gms_matches = []

#   def compute_matches(self, img1, img2):
    def compute_matches(self, sz1r, sz1c, sz2r, sz2c):
        self.keypoints_image1=self.kp1
        self.keypoints_image2=self.kp2

#       self.keypoints_image1, descriptors_image1 = self.descriptor.detectAndCompute(img1, np.array([]))
#       self.keypoints_image2, descriptors_image2 = self.descriptor.detectAndCompute(img2, np.array([]))
                        
#       size1 = Size(img1.shape[1], img1.shape[0])
#       size2 = Size(img2.shape[1], img2.shape[0])

        size1 = Size(sz1c, sz1r)
        size2 = Size(sz2c, sz2r)

        if self.gms_matches:
            self.empty_matches()

        all_matches=self.m12
#       all_matches = self.matcher.match(descriptors_image1, descriptors_image2)
                
        self.normalize_points(self.keypoints_image1, size1, self.normalized_points1)
        self.normalize_points(self.keypoints_image2, size2, self.normalized_points2)
        self.matches_number = len(all_matches)
        self.convert_matches(all_matches, self.matches)
                
        self.initialize_neighbours(self.grid_neighbor_left, self.grid_size_left)
        
        mask, num_inliers = self.get_inlier_mask(False, False)
        # print('Found', num_inliers, 'matches')

        for i in range(len(mask)):
            if mask[i]:
                self.gms_matches.append(all_matches[i])
        return self.gms_matches, mask

    # Normalize Key points to range (0-1)
    def normalize_points(self, kp, size, npts):
        for keypoint in kp:
            npts.append((keypoint.pt[0] / size.width, keypoint.pt[1] / size.height))

    # Convert OpenCV match to list of tuples
    def convert_matches(self, vd_matches, v_matches):
        for match in vd_matches:
            v_matches.append((match.queryIdx, match.trainIdx))

    def initialize_neighbours(self, neighbor, grid_size):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_size)

    def get_nb9(self, idx, grid_size):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_size.width
        idx_y = idx // grid_size.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_size.width or idx_yy < 0 or idx_yy >= grid_size.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_size.width

        return nb9

    def get_inlier_mask(self, with_scale, with_rotation):
        max_inlier = 0
        self.set_scale(0)

        if not with_scale and not with_rotation:
            max_inlier = self.run(1)
            return self.inlier_mask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                for rotation_type in range(1, 9):
                    num_inlier = self.run(rotation_type)
                    if num_inlier > max_inlier:
                        vb_inliers = self.inlier_mask
                        max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        elif with_rotation and not with_scale:
            vb_inliers = []
            for rotation_type in range(1, 9):
                num_inlier = self.run(rotation_type)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        else:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                num_inlier = self.run(1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier

    def set_scale(self, scale):
        self.grid_size_right.width = self.grid_size_left.width * self.scale_ratios[scale]
        self.grid_size_right.height = self.grid_size_left.height * self.scale_ratios[scale]
        self.grid_number_right = self.grid_size_right.width * self.grid_size_right.height

        # Initialize the neighbour of right grid
        self.grid_neighbor_right = np.zeros((int(self.grid_number_right), 9))
        self.initialize_neighbours(self.grid_neighbor_right, self.grid_size_right)

    def run(self, rotation_type):
        self.inlier_mask = [False for _ in range(self.matches_number)]

        # Initialize motion statistics
        self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
        self.match_pairs = [[0, 0] for _ in range(self.matches_number)]

        for GridType in range(1, 5):
            self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
            self.cell_pairs = [-1 for _ in range(self.grid_number_left)]
            self.number_of_points_per_cell_left = [0 for _ in range(self.grid_number_left)]

            self.assign_match_pairs(GridType)
            self.verify_cell_pairs(rotation_type)

            # Mark inliers
            for i in range(self.matches_number):
                if self.cell_pairs[int(self.match_pairs[i][0])] == self.match_pairs[i][1]:
                    self.inlier_mask[i] = True

        return sum(self.inlier_mask)

    def assign_match_pairs(self, grid_type):
        for i in range(self.matches_number):
            lp = self.normalized_points1[self.matches[i][0]]
            rp = self.normalized_points2[self.matches[i][1]]
            lgidx = self.match_pairs[i][0] = self.get_grid_index_left(lp, grid_type)

            if grid_type == 1:
                rgidx = self.match_pairs[i][1] = self.get_grid_index_right(rp)
            else:
                rgidx = self.match_pairs[i][1]

            if lgidx < 0 or rgidx < 0:
                continue
            self.motion_statistics[int(lgidx)][int(rgidx)] += 1
            self.number_of_points_per_cell_left[int(lgidx)] += 1

    def get_grid_index_left(self, pt, type_of_grid):
        x = pt[0] * self.grid_size_left.width
        y = pt[1] * self.grid_size_left.height

        if type_of_grid == 2:
            x += 0.5
        elif type_of_grid == 3:
            y += 0.5
        elif type_of_grid == 4:
            x += 0.5
            y += 0.5

        x = math.floor(x)
        y = math.floor(y)

        if x >= self.grid_size_left.width or y >= self.grid_size_left.height:
            return -1
        return x + y * self.grid_size_left.width

    def get_grid_index_right(self, pt):
        x = int(math.floor(pt[0] * self.grid_size_right.width))
        y = int(math.floor(pt[1] * self.grid_size_right.height))
        return x + y * self.grid_size_right.width

    def verify_cell_pairs(self, rotation_type):
        current_rotation_pattern = ROTATION_PATTERNS[rotation_type - 1]

        for i in range(self.grid_number_left):
            if sum(self.motion_statistics[i]) == 0:
                self.cell_pairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.grid_number_right)):
                value = self.motion_statistics[i]
                if value[j] > max_number:
                    self.cell_pairs[i] = j
                    max_number = value[j]

            idx_grid_rt = self.cell_pairs[i]
            nb9_lt = self.grid_neighbor_left[i]
            nb9_rt = self.grid_neighbor_right[idx_grid_rt]
            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = nb9_lt[j]
                rr = nb9_rt[current_rotation_pattern[j] - 1]
                if ll == -1 or rr == -1:
                    continue

                score += self.motion_statistics[int(ll), int(rr)]
                thresh += self.number_of_points_per_cell_left[int(ll)]
                numpair += 1

            thresh = THRESHOLD_FACTOR * math.sqrt(thresh/numpair)
            if score < thresh:
                self.cell_pairs[i] = -2


class gms_module:
    def __init__(self, **args):
              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('gms').lower()

        
    def run(self, **args):  
        pt1 = np.ascontiguousarray(args['pt1'].detach().cpu())
        pt2 = np.ascontiguousarray(args['pt2'].detach().cpu())

        sz1 = Image.open(args['im1']).size
        sz2 = Image.open(args['im2']).size        
                
        l = pt1.shape[0]
        
        if l > 0:    
            kpt1 = [cv2.KeyPoint(pt1[i, 0], pt1[i, 1], 1) for i in range(l)]
            kpt2 = [cv2.KeyPoint(pt2[i, 0], pt2[i, 1], 1) for i in range(l)]
            m12 = [cv2.DMatch(i, i, 0) for i in range(l)]    
    
            gms = GmsMatcher(kpt1, kpt2, m12)
            _, mask = gms.compute_matches(sz1[1], sz1[0], sz2[1], sz2[0]);
        
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]            
            Hs = args['Hs'][mask]            
        else:
            pt1 = args['pt1']
            pt2 = args['pt2']           
            Hs = args['Hs']   
            mask = []
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
