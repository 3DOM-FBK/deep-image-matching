import sys
import sqlite3
import numpy as np
import os


IS_PYTHON3 = sys.version_info[0] >= 3


#def array_to_blob(array):
#    if IS_PYTHON3:
#        return array.tostring()
#    else:
#        return np.getbuffer(array)

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def dbReturnMatches(database_path, min_num_matches):
    print(database_path)
    if os.path.exists(database_path):
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        images = {}
        matches = {}
        cursor.execute("SELECT image_id, camera_id, name FROM images;")
        for row in cursor:
            image_id = row[0]
            image_name = row[2]
            images[image_id] = image_name

        cursor.execute("SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;", (min_num_matches,))

        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_name1 = images[image_id1]
            image_name2 = images[image_id2]
            matches["{} {}".format(image_name1, image_name2)] = inlier_matches

        cursor.close()
        connection.close()

        return images, matches

    else:
        print("Database does not exist")
        quit()


def dbReturnKeypoints(database_path):
    print(database_path)
    if os.path.exists(database_path):
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        images = {}
        keypoints = {}
        cursor.execute("SELECT image_id, camera_id, name FROM images;")
        for row in cursor:
            image_id = row[0]
            image_name = row[2]
            images[image_id] = image_name
        
        cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
        for row in cursor:
            image_id = row[0]
            #kpts = np.fromstring(row[3], np.float32).reshape(-1, 6)
            kpts = np.fromstring(row[3], np.float32).reshape(-1, 2)
            keypoints[image_id] = kpts
    
    return images, keypoints



if __name__ == "__main__":
    database_path = r"D:\mapping\2023-10-05\Seq3\sglue+rsift\ransac1px\raw.db"
    out_kpts_dir = r"./kpts"
    matches_file = r"./matches.txt"
    images, keypoints = dbReturnKeypoints(database_path)
    images, matches = dbReturnMatches(database_path, min_num_matches=15)

    for id in keypoints:
        print('id', id)
        kpts = keypoints[id]
        kpt_file = open(f"{out_kpts_dir}/{images[id]}.txt", 'w')
        kpt_file.write(f"{kpts.shape[0]} 128\n")
        for row in range(kpts.shape[0]):
            #kpt_file.write(f"{kpts[row][0]} {kpts[row][1]} {kpts[row][2]} {kpts[row][3]}\n")
            kpt_file.write(f"{kpts[row][0]} {kpts[row][1]} 1.0 0.0\n")
        kpt_file.close()
    
    out_matches_file = open(matches_file, 'w')
    for pair in matches:
        out_matches_file.write(f"{pair}\n")
        tie_points = matches[pair]
        for row in range(tie_points.shape[0]):
            out_matches_file.write(f"{tie_points[row][0]} {tie_points[row][1]}\n")
        out_matches_file.write("\n")
    out_matches_file.close()
