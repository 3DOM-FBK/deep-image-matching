import os.path
from os import path
from urllib.request import urlopen
from zipfile import ZipFile
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import cv2
import csv
from munkres import Munkres  # https://pypi.org/project/munkres/

def get_soft_assignment(correspondences, homographies, inlier_threshold):
    model_number = int(homographies.shape[0] / 3)
    point_number = correspondences.shape[0]
    assignments = np.zeros((point_number, model_number))

    for model_idx in range(int(model_number)):
        # Get the current model parameters
        homography = homographies[model_idx * 3 : (model_idx + 1) * 3, :]
        
        # Iterate through the points and get the inliers of the current homography 
        for point_idx in range(point_number):
            # Project the point in the source image by the estimated homography
            pt1 = np.array([correspondences[point_idx, 0], correspondences[point_idx, 1], 1])
            pt2 = np.array([correspondences[point_idx, 2], correspondences[point_idx, 3], 1])
            pt1_transformed = homography @ pt1
            pt1_transformed /= pt1_transformed[2]
            
            # Calculate the point-to-model residual
            residual = np.linalg.norm(pt1_transformed - pt2)
            
            # Store the inlier probabilities
            if residual < inlier_threshold:
                probability = math.exp(-1.0 / 2.0 * (residual / inlier_threshold)**2)
                assignments[point_idx, model_idx] = probability
    return assignments

def model_type(type_name):
    if type_name == "homography":
        return 0
    if type_name == "two_view_motion":
        return 1
    if type_name == "vanishing_point":
        return 2
    if type_name == "plane":
        return 3
    if type_name == "motion":
        return 4
    return -1

def convert_to_number (s):
    return int.from_bytes(s.encode(), 'little')

def convert_from_number (n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

def load_data(dataset):
    files = os.listdir(dataset)
    data = {}

    for f in files:
        try:
            M = np.loadtxt(f"{dataset}/{f}/{f}.txt", delimiter=" ")
            data[f] = {}
            data[f]['corrs'] = np.concatenate((M[:,:2], M[:,3:5]), axis=1)
            data[f]['labels'] = M[:,-1]
        except:
            print(f"Error when loading scene {f}")
    return data

def load_motion_data(dataset):
    files = os.listdir(dataset)
    data = {}

    for f in files:
        with open(f"{dataset}/{f}/{f}.txt") as file:
            reader = csv.reader(file, delimiter=' ')
            line_count = 0
            M = []
            labels = []
            for row in reader:
                if line_count == 0:
                    M = np.zeros((int(row[1]), 2 * int(row[0])))
                    labels = np.zeros(int(row[1]))
                    line_count += 1
                else:
                    numbers = [ float(x) for x in row[1:-1] ]
                    M[line_count - 1, :] = numbers 
                    labels[line_count - 1] = int(row[-1]) 
                    line_count += 1
                    
            data[f] = {}
            data[f]['corrs'] = M
            data[f]['labels'] = labels
    return data

def download_datasets(url_base, datasets):
    # Download the dataset if needed
    for dataset in datasets:
        if not path.exists(dataset):
            url = f'{url_base}{dataset}.zip'
            # Download the file from the URL
            print(f"Beginning file download '{url}'")
            zipresp = urlopen(url)
            # Create a new file on the hard drive
            tempzip = open("/tmp/tempfile.zip", "wb")
             # Write the contents of the downloaded file into the new file
            tempzip.write(zipresp.read())
                # Close the newly-created file
            tempzip.close()
                # Re-open the newly-created file with ZipFile()
            zf = ZipFile("/tmp/tempfile.zip")
            # Extract its contents into <extraction_path>
            # note that extractall will automatically create the path
            zf.extractall(path = '')
            # close the ZipFile instance
            zf.close()

def misclassification(segmentation, ref_segmentation):
    n = int(max(ref_segmentation)) + 1
    indices = np.array(range(n))
    n_labels = len(segmentation)
    miss = []

    for p in multiset_permutations(indices):
        tmp_ref_segmentation = np.zeros((n_labels))

        for i in range(n):
            indices = ref_segmentation == i
            tmp_ref_segmentation[indices] = p[i]

        misclassified_points = np.sum(tmp_ref_segmentation != segmentation)
        miss.append(misclassified_points)
   
    return np.min(miss) / n_labels

def misclassification_hungarian(cluster_mask_est, cluster_mask_gt):
    """
    :param cluster_mask_est: estimated mask
    :param cluster_mask_gt: ground truth mask
    :return: number of error between the ground-truth and estimated cluster masks
    """

    # Get the list of different labels for the ground truth mask and the estimated one.
    label_list_gt = list(set(cluster_mask_gt))
    label_list_est = list(set(cluster_mask_est))

    # Initialize the cost matrix
    cost_matrix = np.empty(shape=(len(label_list_gt), len(label_list_est)))

    # Fill the cost matrix for Munkres (Hungarian algorithm)
    for i, gt_lbl in enumerate(label_list_gt):
        # set as 1 only points with the current gt label
        tmp_gt = np.where(cluster_mask_gt == gt_lbl, 1, 0)
        for j, pred_lbl in enumerate(label_list_est):
            # set as 1 only points with the current pred label
            tmp_est = np.where(cluster_mask_est == pred_lbl, 1, 0)
            cost_matrix[i, j] = np.count_nonzero(
                np.where(tmp_gt + tmp_est == 1, 1, 0))  # rationale below

    # Rationale: we need to fill a cost matrix, thus we need to count the number of errors,
    # namely False Positives (FP) and False Negatives (FN).
    # Both FP and FN are such that gt_lbl != pred_lbl and gt_lbl + pred_lbl == 1
    # [(either gt_lbl == 0 and pred_lbl == 1) or  (either gt_lbl == 1 and pred_lbl == 0)].
    # Note, we don't care about True Positives (TP), where gt_lbl == pred_lbl == 1,
    # and True Negatives (TN), where gt_lbl == pred_lbl == 0, since they are not errors.

    # Run Munkres algorithm
    mnkrs = Munkres()
    # Remark: the cost matrix must have num_of_rows <= num_of_cols (see documentation)
    if len(label_list_gt) <= len(label_list_est):
        match_pairs = mnkrs.compute(np.array(cost_matrix))
        lbl_pred2gt_dict = dict([(j, i) for (i, j) in match_pairs])
    else:  # otherwise, we compute invert rows and columns
        cost_matrix = np.transpose(cost_matrix)
        match_pairs = mnkrs.compute(np.array(cost_matrix))
        lbl_pred2gt_dict = dict(match_pairs)

    lbl_pred2gt_dict = {label_list_est[k]: label_list_gt[v]
                        for k, v in lbl_pred2gt_dict.items()}

    # Relabel cluster_mask_est according to the correct label mapping found by Munkres
    clusters_mask_relabeled = []
    for cl_idx in cluster_mask_est:
        if cl_idx in lbl_pred2gt_dict:
            clusters_mask_relabeled.append(lbl_pred2gt_dict[cl_idx])
        else:
            clusters_mask_relabeled.append(-1)
    clusters_mask_relabeled = np.array(clusters_mask_relabeled)

    # Compute the number of errors as the difference of the gt and relabelled arrays
    errors = np.count_nonzero(cluster_mask_gt - clusters_mask_relabeled)

    return errors / len(cluster_mask_gt)

def random_color(label = None):
    if label is not None:
        if label == 0:
            return (255, 0, 0)
        elif label == 1:
            return (0, 255, 0)
        elif label == 2:
            return (0, 0, 255)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def resize_image(image, scale):
    """
    Resize an image based on the specified scale percentage.
    """
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def draw_soft_assignment(img1, img2, assignments, correspondences, radius=4, figsize=(12,8)):
    point_number = assignments.shape[0]
    model_number = assignments.shape[1]
    shared_color = (255, 0, 0)
    colors = []
    for model_idx in range(model_number):
        colors.append(random_color())
        
    img1_cpy = img1.copy()
    img2_cpy = img2.copy()

    for point_idx in range(point_number):
        model_indices = assignments[point_idx, :] > 0

        if np.sum(model_indices) > 1:
            cv2.circle(img1_cpy, (round(correspondences[point_idx][0]), round(correspondences[point_idx][1])), radius, shared_color, -1)
            cv2.circle(img2_cpy, (round(correspondences[point_idx][2]), round(correspondences[point_idx][3])), radius, shared_color, -1)
        elif np.sum(model_indices) == 1:
            idx = np.where(model_indices)[0][0]
            cv2.circle(img1_cpy, (round(correspondences[point_idx][0]), round(correspondences[point_idx][1])), radius, colors[idx], -1)
            cv2.circle(img2_cpy, (round(correspondences[point_idx][2]), round(correspondences[point_idx][3])), radius, colors[idx], -1)

    img1_cpy = resize_image(img1_cpy, 0.5)
    img2_cpy = resize_image(img2_cpy, 0.5)

    # Plot the two images side by side using matplotlib.pyplot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax1.imshow(img1_cpy)
    ax2.imshow(img2_cpy)
    fig.text(0.5, 0.72, 
         'Soft assignment (red points are shared by multiple models)', 
         fontsize = 16,
         horizontalalignment = "center")
    plt.show()

def draw_labeling(img1, img2, labeling, correspondences, radius=4, figsize=(12,8)):
    img1_cpy = img1.copy()
    img2_cpy = img2.copy()
    
    for label in range(int(max(labeling))):
        mask = labeling == label
        color = random_color(label)

        for i in range(len(labeling)):
            if mask[i]:
                cv2.circle(img1_cpy, (round(correspondences[i][0]), round(correspondences[i][1])), radius, color, -1)
                cv2.circle(img2_cpy, (round(correspondences[i][2]), round(correspondences[i][3])), radius, color, -1)

    img1_cpy = resize_image(img1_cpy, 0.5)
    img2_cpy = resize_image(img2_cpy, 0.5)
    
    # Plot the two images side by side using matplotlib.pyplot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(figsize[0], figsize[1])
    ax1.imshow(img1_cpy)
    ax2.imshow(img2_cpy)
    fig.text(0.5, 0.72, 
         'Hard assignment by PEARL', 
         fontsize = 16,
         horizontalalignment = "center")
    plt.show()