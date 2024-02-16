import os
import subprocess
import sys

def get_parent_dir(directory):
    return os.path.dirname(directory)

OPENMVG_SFM_BIN = "/home/threedom/Desktop/github_lcmrl/openMVG/openMVG_Build/Linux-x86_64-RELEASE" # Indicate the openMVG binary directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/threedom/Desktop/github_lcmrl/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database" # Indicate the openMVG camera sensor width directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
input_eval_dir = os.path.abspath("/home/threedom/Desktop/github_3dom/deep-image-matching/assets/pytest/results_superpoint+lightglue_bruteforce_quality_high/openmvg/")

output_dir = input_eval_dir
matches_dir = os.path.join(output_dir, "matches")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")


reconstruction_dir = os.path.join(output_dir,"reconstruction_sequential")
print ("3. Do Incremental/Sequential reconstruction") #set manually the initial pair to avoid the prompt question
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"), "--sfm_engine", "INCREMENTAL", "--input_file", matches_dir+"/sfm_data.json", "--match_dir", matches_dir, "--output_dir", reconstruction_dir] )
pRecons.wait()